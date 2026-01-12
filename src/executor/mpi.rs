// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! MPI-based distributed executor for parallel computation across processes.
//!
//! This module provides [`MpiExecutor`], which distributes work items across
//! MPI processes using a work-stealing pattern. The root process (rank 0)
//! lazily iterates work items and assigns them to workers on demand.

use bincode::serde::{decode_from_slice, encode_to_vec};
use mpi::{
    topology::SimpleCommunicator,
    traits::{Communicator, Destination, Root, Source},
};
use serde::{Serialize, de::DeserializeOwned};
use tracing::{debug, info, trace, warn};

use super::{DistributedExecutor, Executor};
use crate::{executor::SequentialExecutor, logging::ProgressTracker};

/// Message tags for MPI communication protocol.
///
/// These tags distinguish different message types in the work-stealing
/// protocol.
#[derive(Clone, Copy)]
#[repr(i32)]
enum MpiTag {
    /// Worker requesting a new work item from root.
    WorkRequest = 1,
    /// Root sending a work item to a worker.
    WorkAssignment = 2,
    /// Worker sending computed results back to root.
    ResultSubmission = 3,
    /// Root signaling to worker that no more work is available.
    Shutdown = 4,
}

/// MPI-based executor that distributes work across processes.
///
/// Uses a work-stealing pattern where the root process (rank 0) lazily assigns
/// work items to workers as they request them. Results are gathered at the root
/// and flattened into a single output vector.
///
/// # Work-Stealing Protocol
///
/// 1. Workers send work requests to the root process.
/// 2. Root process iterates to the next work item and assigns it to the worker.
/// 3. Worker computes results and submits the results to the root process.
/// 4. Once the iterator is exhausted, the root process sends a shutdown signal
///    to the workers
///
/// # Panics
///
/// The executor panics on serialization errors or MPI communication failures.
/// Work items cannot be skipped in the current implementation.
pub struct MpiExecutor {
    comm: SimpleCommunicator,
}

impl MpiExecutor {
    /// Create a new MPI executor by duplicating the given communicator.
    ///
    /// The executor owns a duplicate of the communicator, allowing the
    /// original to be used elsewhere. Any type implementing [`Communicator`]
    /// can be passed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use chomp3rs::executor::MpiExecutor;
    /// use mpi::traits::Communicator;
    ///
    /// let universe = mpi::initialize().unwrap();
    /// let executor = MpiExecutor::new(&universe.world());
    /// ```
    #[must_use]
    pub fn new(comm: &impl Communicator) -> Self {
        Self {
            comm: comm.duplicate(),
        }
    }

    /// Create an MPI executor from the world communicator.
    ///
    /// Initializes the MPI universe if not already initialized. Returns `None`
    /// if MPI initialization fails (e.g., if already initialized elsewhere).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use chomp3rs::executor::MpiExecutor;
    ///
    /// if let Some(executor) = MpiExecutor::from_world() {
    ///     // Use executor for distributed computation
    /// }
    /// ```
    #[must_use]
    pub fn from_world() -> Option<Self> {
        mpi::initialize().map(|universe| Self::new(&universe.world()))
    }

    /// Returns the rank of this process in the communicator.
    #[must_use]
    pub fn rank(&self) -> i32 {
        self.comm.rank()
    }

    /// Returns the total number of processes in the communicator.
    #[must_use]
    pub fn size(&self) -> i32 {
        self.comm.size()
    }

    /// Returns whether this process is the root (rank 0).
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.comm.rank() == 0
    }

    /// Root process: distribute work items to workers and collect results.
    ///
    /// Lazily iterates work items, assigning them to workers as they request
    /// work. Collects and flattens all results.
    fn run_as_root<I, O, R, F>(&mut self, label: &'static str, work_items: I, compute: F) -> Vec<R>
    where
        I: ExactSizeIterator,
        I::Item: Serialize,
        O: IntoIterator<Item = R> + DeserializeOwned,
        F: Fn(I::Item) -> O,
    {
        let total_work_items = work_items.len();

        // Communicator::size always returns nonnegative value
        #[allow(clippy::cast_sign_loss)]
        let mut active_workers = (self.comm.size() - 1) as usize;

        info!(
            "Distributing {total_work_items} work items across {active_workers} MPI worker \
             processes",
        );

        // If no workers, execute locally
        if active_workers == 0 {
            warn!("No MPI worker processes available, executing locally");
            return <Self as Executor>::execute(self, label, work_items, compute);
        }

        let mut work_iter = work_items.peekable();
        let mut results = Vec::new();
        let mut progress =
            ProgressTracker::new(format!("{label} (MPI Distributed)"), total_work_items);

        // Main dispatch loop
        while active_workers > 0 {
            // Wait for any worker to send a message (either request or result)
            let (msg_bytes, status): (Vec<u8>, _) = self.comm.any_process().receive_vec();
            let source_rank = status.source_rank();
            let tag = status.tag();

            if tag == MpiTag::WorkRequest as i32 {
                trace!("Received work request from MPI rank {source_rank}");
                self.assign_work_or_shutdown(&mut work_iter, source_rank, &mut active_workers);
            } else if tag == MpiTag::ResultSubmission as i32 {
                trace!("Received results from MPI rank {source_rank}");

                // Decode and collect results
                let worker_results: O = decode_from_slice(&msg_bytes, bincode::config::standard())
                    .expect("failed to decode results from MPI worker process")
                    .0;
                results.extend(worker_results);
                progress.increment();

                // Assign next work item or shutdown
                self.assign_work_or_shutdown(&mut work_iter, source_rank, &mut active_workers);
            } else {
                panic!("root received unexpected message tag {tag} from MPI rank {source_rank}",);
            }
        }

        progress.finish();
        results
    }

    /// Helper to assign work to a worker or send shutdown signal.
    fn assign_work_or_shutdown<I>(
        &self,
        work_iter: &mut std::iter::Peekable<I>,
        worker_rank: i32,
        active_workers: &mut usize,
    ) where
        I: Iterator,
        I::Item: Serialize,
    {
        if let Some(work_item) = work_iter.next() {
            let encoded = encode_to_vec(&work_item, bincode::config::standard())
                .expect("failed to encode work item");
            self.comm
                .process_at_rank(worker_rank)
                .send_with_tag(&encoded, MpiTag::WorkAssignment as i32);
            trace!("Assigned work to rank MPI {worker_rank}",);
        } else {
            // No more work; shutdown this worker
            self.comm
                .process_at_rank(worker_rank)
                .send_with_tag(&Vec::<u8>::new(), MpiTag::Shutdown as i32);
            *active_workers -= 1;
            trace!(
                "Sent shutdown to rank MPI {worker_rank}, {} MPI worker processes remaining",
                *active_workers
            );
        }
    }

    /// Worker process: request work, compute, and submit results.
    ///
    /// Loops until receiving a shutdown signal from root.
    fn run_as_worker<T, O, R, F>(&self, compute: F)
    where
        T: DeserializeOwned,
        O: IntoIterator<Item = R> + Serialize,
        F: Fn(T) -> O,
    {
        let rank = self.comm.rank();
        debug!("MPI rank {rank} worker process starting");

        // Send initial work request
        self.comm
            .process_at_rank(0)
            .send_with_tag(&Vec::<u8>::new(), MpiTag::WorkRequest as i32);

        loop {
            // Receive work assignment or shutdown
            let (msg_bytes, status): (Vec<u8>, _) = self.comm.process_at_rank(0).receive_vec();
            let tag = status.tag();

            if tag == MpiTag::Shutdown as i32 {
                debug!("MPI worker process rank {rank} received shutdown");
                break;
            }

            assert!(
                tag == MpiTag::WorkAssignment as i32,
                "MPI worker process rank {rank} received unexpected message tag {tag}"
            );

            // Decode work item
            let work_item: T = decode_from_slice(&msg_bytes, bincode::config::standard())
                .expect("failed to decode work item")
                .0;

            // Compute results
            let results = compute(work_item);

            // Send results back to root
            let encoded_results = encode_to_vec(&results, bincode::config::standard())
                .expect("failed to encode results");
            self.comm
                .process_at_rank(0)
                .send_with_tag(&encoded_results, MpiTag::ResultSubmission as i32);
        }

        debug!("MPI worker process rank {rank} finished");
    }

    /// Broadcast data from root to all processes.
    ///
    /// The root process (rank 0) sends its data to all other processes.
    /// Non-root processes receive the data and return it, ignoring their
    /// input.
    ///
    /// # Panics
    ///
    /// If serialization or deserialization fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use chomp3rs::executor::MpiExecutor;
    ///
    /// let mut executor = MpiExecutor::from_world().unwrap();
    ///
    /// // Root has data, workers have empty placeholder
    /// let my_data = if executor.is_root() {
    ///     vec![1, 2, 3]
    /// } else {
    ///     Vec::new()
    /// };
    ///
    /// // After broadcast, all processes have the same data
    /// let shared_data = executor.broadcast(&my_data);
    /// assert_eq!(shared_data, vec![1, 2, 3]);
    /// ```
    pub fn broadcast<T>(&self, data: &T) -> T
    where
        T: Serialize + DeserializeOwned,
    {
        // Encode data on root
        let encoded = if self.is_root() {
            encode_to_vec(data, bincode::config::standard())
                .expect("failed to encode for broadcast")
        } else {
            Vec::new()
        };

        // Broadcast the length first
        let mut len = encoded.len() as i32;
        self.comm.process_at_rank(0).broadcast_into(&mut len);

        // Prepare buffer and broadcast data
        //
        // Sent `len` is originally a usize, so cast signed -> unsigned is fine
        #[allow(clippy::cast_sign_loss)]
        let mut buffer = if self.is_root() {
            encoded
        } else {
            vec![0u8; len as usize]
        };
        self.comm.process_at_rank(0).broadcast_into(&mut buffer);

        // Decode on all processes
        decode_from_slice(&buffer, bincode::config::standard())
            .expect("failed to decode broadcast data")
            .0
    }
}

impl Executor for MpiExecutor {
    /// Execute computation locally on this process only.
    ///
    /// This implementation runs sequentially on the calling process without
    /// any MPI communication. For distributed execution across MPI processes,
    /// use [`DistributedExecutor::execute_distributed`] instead.
    ///
    /// This method exists to satisfy the `Executor` supertrait requirement,
    /// allowing `MpiExecutor` to be used in contexts which only require local
    /// execution.
    fn execute<I, O, R, F>(&mut self, label: &'static str, work_items: I, compute: F) -> Vec<R>
    where
        I: ExactSizeIterator,
        O: IntoIterator<Item = R>,
        F: Fn(I::Item) -> O,
    {
        // Local execution only - no MPI communication
        SequentialExecutor.execute(label, work_items, compute)
    }
}

impl DistributedExecutor for MpiExecutor {
    /// Execute computation distributed across MPI processes.
    ///
    /// The root process (rank 0) distributes work items to workers and
    /// collects results. Worker processes compute assigned items and return
    /// an empty vector (results are gathered at root).
    ///
    /// # Type Requirements
    ///
    /// - `I::Item`: Must be serializable for sending work items to workers
    /// - `O`: Must be serializable for sending results back to root
    fn execute_distributed<I, O, R, F>(
        &mut self,
        label: &'static str,
        work_items: I,
        compute: F,
    ) -> Vec<R>
    where
        I: ExactSizeIterator,
        I::Item: Serialize + DeserializeOwned,
        O: IntoIterator<Item = R> + Serialize + DeserializeOwned,
        F: Fn(I::Item) -> O,
    {
        if self.is_root() {
            self.run_as_root(label, work_items, compute)
        } else {
            self.run_as_worker(compute);
            Vec::new()
        }
    }
}
