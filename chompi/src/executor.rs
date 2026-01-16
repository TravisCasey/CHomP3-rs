// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! MPI-based distributed executor implementation.

use bincode::serde::{decode_from_slice, encode_to_vec};
use mpi::{
    topology::SimpleCommunicator,
    traits::{Communicator, Destination, Root, Source},
};
use serde::{Serialize, de::DeserializeOwned};
use tracing::{debug, info, trace, warn};

/// Message tags for MPI communication protocol.
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
///    to the workers.
///
/// # Panics
///
/// The executor panics on serialization errors or MPI communication failures.
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
    /// use chompi::MpiExecutor;
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
    /// use chompi::MpiExecutor;
    ///
    /// if let Some(executor) = MpiExecutor::from_world() {
    ///     // Use executor for distributed computation
    /// }
    /// ```
    #[must_use]
    pub fn from_world() -> Option<Self> {
        mpi::initialize().map(|universe| Self::new(&universe.world()))
    }

    /// Returns the rank of this process in the stored communicator.
    #[must_use]
    pub fn rank(&self) -> i32 {
        self.comm.rank()
    }

    /// Returns the total number of processes in the stored communicator.
    #[must_use]
    pub fn size(&self) -> i32 {
        self.comm.size()
    }

    /// Returns whether this process is the root (rank 0).
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.comm.rank() == 0
    }

    /// Execute a stateless computation distributed across MPI processes.
    ///
    /// Distributes work items from the iterator across all MPI processes.
    /// Each work item is serialized and sent to a worker, which deserializes
    /// it, applies the compute function, and sends results back.
    ///
    /// For computations that benefit from persistent per-worker state, use
    /// [`execute_with_state`](Self::execute_with_state) instead.
    ///
    /// # Type Parameters
    ///
    /// - `I`: Iterator yielding work items (must be `ExactSizeIterator` for
    ///   progress tracking)
    /// - `C`: Collection type returned by compute function
    /// - `R`: Result element type
    ///
    /// # Returns
    ///
    /// On the root process: vector of all results from all workers.
    /// On worker processes: empty vector.
    pub fn execute<I, C, R>(
        &mut self,
        label: &str,
        work_items: I,
        compute: impl Fn(I::Item) -> C,
    ) -> Vec<R>
    where
        I: ExactSizeIterator,
        I::Item: Serialize + DeserializeOwned,
        C: IntoIterator<Item = R> + Serialize + DeserializeOwned,
    {
        self.execute_with_state(label, work_items, || (), |(), item| compute(item))
    }

    /// Execute a computation with per-worker persistent state.
    ///
    /// Similar to [`execute`](Self::execute), but allows each worker to
    /// maintain state across work items. The `init` function is called once
    /// per process to create the initial state, which is then passed mutably
    /// to each `compute` invocation.
    ///
    /// This is useful when workers need expensive setup (e.g., creating
    /// caches or auxiliary data structures) that should be reused across
    /// multiple work items.
    ///
    /// # Type Parameters
    ///
    /// - `I`: Iterator yielding work items
    /// - `C`: Collection type returned by compute function
    /// - `R`: Result element type
    /// - `S`: Worker state type
    ///
    /// # Returns
    ///
    /// On the root process: vector of all results from all workers.
    /// On worker processes: empty vector.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use chompi::MpiExecutor;
    ///
    /// let mut executor = MpiExecutor::from_world().unwrap();
    ///
    /// // Each worker creates an expensive cache once
    /// let results = executor.execute_with_state(
    ///     "Processing",
    ///     work_items.into_iter(),
    ///     || ExpensiveCache::new(),
    ///     |cache, item| process_with_cache(cache, item),
    /// );
    /// ```
    pub fn execute_with_state<I, C, R, S>(
        &mut self,
        label: &str,
        work_items: I,
        init: impl FnOnce() -> S,
        compute: impl Fn(&mut S, I::Item) -> C,
    ) -> Vec<R>
    where
        I: ExactSizeIterator,
        I::Item: Serialize + DeserializeOwned,
        C: IntoIterator<Item = R> + Serialize + DeserializeOwned,
    {
        let mut state = init();

        if self.is_root() {
            self.run_as_root(label, work_items, &mut state, compute)
        } else {
            self.run_as_worker(&mut state, compute);
            Vec::new()
        }
    }

    /// Root process: distribute work items to workers and collect results.
    fn run_as_root<I, C, R, S>(
        &mut self,
        label: &str,
        work_items: I,
        state: &mut S,
        compute: impl Fn(&mut S, I::Item) -> C,
    ) -> Vec<R>
    where
        I: ExactSizeIterator,
        I::Item: Serialize,
        C: IntoIterator<Item = R> + DeserializeOwned,
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
            return work_items.flat_map(|item| compute(state, item)).collect();
        }

        let mut work_iter = work_items.peekable();
        let mut results = Vec::new();

        info!("{label} (MPI Distributed): 0/{total_work_items}");
        let mut completed = 0usize;

        // Main work-stealing loop
        while active_workers > 0 {
            let (msg_bytes, status): (Vec<u8>, _) = self.comm.any_process().receive_vec();
            let source_rank = status.source_rank();
            let tag = status.tag();

            if tag == MpiTag::WorkRequest as i32 {
                trace!("Received work request from MPI rank {source_rank}");
                self.assign_work_or_shutdown(&mut work_iter, source_rank, &mut active_workers);
            } else if tag == MpiTag::ResultSubmission as i32 {
                trace!("Received results from MPI rank {source_rank}");

                let worker_results: C = decode_from_slice(&msg_bytes, bincode::config::standard())
                    .expect("failed to decode results from MPI worker process")
                    .0;
                results.extend(worker_results);

                completed += 1;
                if completed.is_multiple_of(10) || completed == total_work_items {
                    info!("{label} (MPI Distributed): {completed}/{total_work_items}");
                }

                self.assign_work_or_shutdown(&mut work_iter, source_rank, &mut active_workers);
            } else {
                panic!("root received unexpected message tag {tag} from MPI rank {source_rank}");
            }
        }

        info!("{label} (MPI Distributed): completed");
        results
    }

    /// Helper to assign work to a worker or send shutdown signal.
    fn assign_work_or_shutdown<I>(
        &self,
        work_iter: &mut I,
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
            trace!("Assigned work to MPI rank {worker_rank}");
        } else {
            self.comm
                .process_at_rank(worker_rank)
                .send_with_tag(&Vec::<u8>::new(), MpiTag::Shutdown as i32);
            *active_workers -= 1;
            trace!(
                "Sent shutdown to MPI rank {worker_rank}, {active_workers} worker processes \
                 remaining",
            );
        }
    }

    /// Worker process: request work, compute, and submit results.
    fn run_as_worker<T, C, R, S>(&self, state: &mut S, compute: impl Fn(&mut S, T) -> C)
    where
        T: DeserializeOwned,
        C: IntoIterator<Item = R> + Serialize,
    {
        let rank = self.comm.rank();
        debug!("MPI rank {rank} worker process starting");

        self.comm
            .process_at_rank(0)
            .send_with_tag(&Vec::<u8>::new(), MpiTag::WorkRequest as i32);

        loop {
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

            let work_item: T = decode_from_slice(&msg_bytes, bincode::config::standard())
                .expect("failed to decode work item")
                .0;

            let results = compute(state, work_item);

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
    /// Non-root processes receive the data, ignoring their input value.
    ///
    /// # Panics
    ///
    /// If serialization or deserialization fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use chompi::MpiExecutor;
    ///
    /// let executor = MpiExecutor::from_world().unwrap();
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
        let encoded = if self.is_root() {
            encode_to_vec(data, bincode::config::standard())
                .expect("failed to encode for broadcast")
        } else {
            Vec::new()
        };

        let mut len = encoded.len() as i32;
        self.comm.process_at_rank(0).broadcast_into(&mut len);

        // Sent `len` is originally a usize, so cast signed -> unsigned is fine
        #[allow(clippy::cast_sign_loss)]
        let mut buffer = if self.is_root() {
            encoded
        } else {
            vec![0u8; len as usize]
        };
        self.comm.process_at_rank(0).broadcast_into(&mut buffer);

        decode_from_slice(&buffer, bincode::config::standard())
            .expect("failed to decode broadcast data")
            .0
    }
}
