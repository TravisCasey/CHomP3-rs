// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! MPI-based distributed execution for work-stealing parallelism.
//!
//! This module provides [`MpiExecutor`], which distributes work items across
//! MPI processes using a work-stealing pattern. The root process (rank 0)
//! lazily assigns work items to workers as they request them.
//!
//! # Example
//!
//! ```ignore
//! use chomp3rs::mpi::MpiExecutor;
//!
//! let results: Vec<i32> = MpiExecutor::new(&world)
//!     .label("Compute squares")
//!     .batch_size(100)
//!     .run(0..100, |x| vec![x * x]);
//! ```

use bincode::serde::{decode_from_slice, encode_to_vec};
use mpi::traits::{Destination, Root, Source};
pub use mpi::{
    environment::{Universe, initialize, processor_name},
    topology::SimpleCommunicator,
    traits::Communicator,
};
use serde::{Serialize, de::DeserializeOwned};
use tracing::{Level, debug, trace, warn};

use crate::logging::ProgressTracker;

/// Message tags for the MPI work-stealing protocol.
///
/// These integer values are part of the wire protocol between root and
/// workers and must remain stable.
#[derive(Clone, Copy)]
#[repr(i32)]
pub(crate) enum MpiTag {
    /// Worker to root: ready for work.
    WorkRequest = 1,
    /// Root to worker: batch of items to process.
    WorkAssignment = 2,
    /// Worker to root: computed results.
    ResultSubmission = 3,
    /// Root to worker: no more work, shut down.
    Shutdown = 4,
}

/// MPI-based executor that distributes work across processes using a
/// work-stealing pattern.
///
/// The root process (rank 0) lazily assigns batches from the work iterator
/// to workers as they become available. Workers process batches and return
/// results to the root.
///
/// # Result ordering
///
/// Results are returned in **completion order**, not input order. When
/// multiple workers process batches concurrently, results arrive as workers
/// finish. If input-order results are needed, callers must sort or re-index
/// after collection.
///
/// # Root vs worker return values
///
/// - **Root (rank 0)**: Returns `Vec<R>` containing all collected results.
/// - **Workers**: Return an empty `Vec<R>`, unless
///   [`broadcast_results`](Self::broadcast_results) is enabled, in which case
///   all processes receive the full result vector.
/// - **Single process**: When no workers are available, the root executes all
///   work locally and returns results directly.
///
/// # Panics
///
/// Methods on this type panic on protocol violations or serialization
/// failures. These are deliberate: in a distributed system, partial failure
/// recovery is not meaningful, and fast failure with a clear message is the
/// safest behavior.
///
/// - [`batch_size`](Self::batch_size) panics if `size` is zero.
/// - Serialization/deserialization failures panic with descriptive messages.
/// - Unexpected MPI message tags (protocol violations) panic immediately.
pub struct MpiExecutor {
    comm: SimpleCommunicator,
    label: Option<String>,
    log_level: Option<Level>,
    batch_size: usize,
    broadcast_results: bool,
}

impl MpiExecutor {
    /// Create a new MPI executor from a communicator.
    #[must_use]
    pub fn new(comm: &impl Communicator) -> Self {
        Self {
            comm: comm.duplicate(),
            label: None,
            log_level: None,
            batch_size: 1,
            broadcast_results: false,
        }
    }

    /// Set a label for progress tracking.
    #[must_use]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the log level for progress tracking.
    #[must_use]
    pub fn log_level(mut self, level: Level) -> Self {
        self.log_level = Some(level);
        self
    }

    /// Set the batch size for work distribution.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero.
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        assert!(size > 0, "batch_size must be positive");
        self.batch_size = size;
        self
    }

    /// Broadcast results to all processes after computation.
    ///
    /// When enabled, after the root process collects all results, it broadcasts
    /// them to all worker processes. Workers then return the full result vector
    /// instead of an empty one.
    ///
    /// This is useful when all processes need the complete results for
    /// subsequent computation phases.
    #[must_use]
    pub fn broadcast_results(mut self) -> Self {
        self.broadcast_results = true;
        self
    }

    /// Execute a distributed computation.
    ///
    /// On root: returns all collected results.
    /// On workers: returns empty vector (or full results if `broadcast_results`
    /// is enabled).
    pub fn run<I, C, R>(self, work_items: I, compute: impl Fn(I::Item) -> C) -> Vec<R>
    where
        I: ExactSizeIterator,
        I::Item: Serialize + DeserializeOwned,
        C: IntoIterator<Item = R> + Serialize + DeserializeOwned,
        R: Serialize + DeserializeOwned,
    {
        self.run_with_state(work_items, || (), |(), item| compute(item))
    }

    /// Execute with per-worker persistent state.
    ///
    /// On root: returns all collected results.
    /// On workers: returns empty vector (or full results if `broadcast_results`
    /// is enabled).
    pub fn run_with_state<I, C, R, S>(
        self,
        work_items: I,
        init: impl FnOnce() -> S,
        compute: impl Fn(&mut S, I::Item) -> C,
    ) -> Vec<R>
    where
        I: ExactSizeIterator,
        I::Item: Serialize + DeserializeOwned,
        C: IntoIterator<Item = R> + Serialize + DeserializeOwned,
        R: Serialize + DeserializeOwned,
    {
        let mut state = init();

        let results = if self.comm.rank() == 0 {
            self.run_as_root(work_items, &mut state, &compute)
        } else {
            self.run_as_worker(&mut state, &compute);
            Vec::new()
        };

        if self.broadcast_results {
            broadcast(&self.comm, &results)
        } else {
            results
        }
    }

    fn run_as_root<I, C, R, S>(
        &self,
        work_items: I,
        state: &mut S,
        compute: &impl Fn(&mut S, I::Item) -> C,
    ) -> Vec<R>
    where
        I: ExactSizeIterator,
        I::Item: Serialize,
        C: IntoIterator<Item = R> + DeserializeOwned,
    {
        let total = work_items.len();
        let batch_size = self.batch_size;

        let mut active_workers = (self.comm.size() - 1) as usize;

        // If no workers, execute locally
        if active_workers == 0 {
            warn!("No MPI workers available, executing locally");
            let mut progress = self.make_progress_tracker(total);
            return work_items
                .flat_map(|item| {
                    let result = compute(state, item);
                    progress.increment();
                    result
                })
                .collect();
        }

        let mut work_iter = work_items.peekable();
        let mut results = Vec::new();
        let mut progress = self.make_progress_tracker(total);

        while active_workers > 0 {
            let (msg_bytes, status): (Vec<u8>, _) = self.comm.any_process().receive_vec();
            let source_rank = status.source_rank();
            let tag = status.tag();

            if tag == MpiTag::WorkRequest as i32 {
                trace!("Work request from rank {source_rank}");
                self.assign_batch_or_shutdown(
                    &mut work_iter,
                    batch_size,
                    source_rank,
                    &mut active_workers,
                );
            } else if tag == MpiTag::ResultSubmission as i32 {
                trace!("Results from rank {source_rank}");

                let worker_results: Vec<C> =
                    decode_from_slice(&msg_bytes, bincode::config::standard())
                        .expect("failed to decode results")
                        .0;
                let items_completed = worker_results.len();
                for batch_result in worker_results {
                    results.extend(batch_result);
                }
                progress.increment_by(items_completed);

                self.assign_batch_or_shutdown(
                    &mut work_iter,
                    batch_size,
                    source_rank,
                    &mut active_workers,
                );
            } else {
                panic!("root received unexpected tag {tag} from rank {source_rank}");
            }
        }

        progress.finish();
        results
    }

    fn make_progress_tracker(&self, total: usize) -> ProgressTracker {
        let label = self.label.as_deref().unwrap_or("MPI");
        let mut tracker = ProgressTracker::new(label, total);
        if let Some(level) = self.log_level {
            tracker = tracker.with_level(level);
        }
        tracker
    }

    fn assign_batch_or_shutdown<I>(
        &self,
        work_iter: &mut std::iter::Peekable<I>,
        batch_size: usize,
        worker_rank: i32,
        active_workers: &mut usize,
    ) where
        I: Iterator,
        I::Item: Serialize,
    {
        let batch: Vec<I::Item> = work_iter.take(batch_size).collect();

        if batch.is_empty() {
            self.comm
                .process_at_rank(worker_rank)
                .send_with_tag(&Vec::<u8>::new(), MpiTag::Shutdown as i32);
            *active_workers -= 1;
            trace!("Shutdown rank {worker_rank}, {active_workers} remaining");
        } else {
            let encoded =
                encode_to_vec(&batch, bincode::config::standard()).expect("failed to encode batch");
            self.comm
                .process_at_rank(worker_rank)
                .send_with_tag(&encoded, MpiTag::WorkAssignment as i32);
            trace!("Assigned {} items to rank {worker_rank}", batch.len());
        }
    }

    fn run_as_worker<T, C, R, S>(&self, state: &mut S, compute: &impl Fn(&mut S, T) -> C)
    where
        T: DeserializeOwned,
        C: IntoIterator<Item = R> + Serialize,
    {
        let rank = self.comm.rank();
        debug!("Worker rank {rank} starting");

        self.comm
            .process_at_rank(0)
            .send_with_tag(&Vec::<u8>::new(), MpiTag::WorkRequest as i32);

        loop {
            let (msg_bytes, status): (Vec<u8>, _) = self.comm.process_at_rank(0).receive_vec();
            let tag = status.tag();

            if tag == MpiTag::Shutdown as i32 {
                debug!("Worker rank {rank} shutdown");
                break;
            }

            assert!(
                tag == MpiTag::WorkAssignment as i32,
                "worker rank {rank} received unexpected tag {tag}"
            );

            let batch: Vec<T> = decode_from_slice(&msg_bytes, bincode::config::standard())
                .expect("failed to decode batch")
                .0;

            // Each input item produces exactly one C, preserving the 1:1 count.
            // The root relies on this for item-level progress tracking.
            let results: Vec<C> = batch.into_iter().map(|item| compute(state, item)).collect();

            let encoded = encode_to_vec(&results, bincode::config::standard())
                .expect("failed to encode results");
            self.comm
                .process_at_rank(0)
                .send_with_tag(&encoded, MpiTag::ResultSubmission as i32);
        }
    }
}

/// Broadcast a value from root (rank 0) to all processes.
///
/// All processes must call this function. The root process provides the data,
/// and all processes (including root) receive the broadcast value.
///
/// Works with any serializable type, including vectors:
/// - Single values: `broadcast(comm, &value)`
/// - Vectors: `broadcast(comm, &vec)`
///
/// # Panics
///
/// - If serialization or deserialization fails.
/// - If the serialized payload exceeds `i32::MAX` bytes.
///
/// # Example
///
/// ```ignore
/// use chomp3rs::mpi::{broadcast, initialize};
///
/// let universe = initialize().unwrap();
/// let world = universe.world();
///
/// // Broadcast a single value
/// let data = if world.rank() == 0 { 42 } else { 0 };
/// let result: i32 = broadcast(&world, &data);
/// assert_eq!(result, 42);
///
/// // Broadcast a vector
/// let vec_data = if world.rank() == 0 { vec![1, 2, 3] } else { vec![] };
/// let vec_result: Vec<i32> = broadcast(&world, &vec_data);
/// assert_eq!(vec_result, vec![1, 2, 3]);
/// ```
pub fn broadcast<T>(comm: &SimpleCommunicator, data: &T) -> T
where
    T: Serialize + DeserializeOwned,
{
    let encoded = if comm.rank() == 0 {
        encode_to_vec(data, bincode::config::standard()).expect("failed to encode for broadcast")
    } else {
        Vec::new()
    };

    let mut len: i32 = encoded
        .len()
        .try_into()
        .expect("broadcast payload exceeds i32::MAX bytes");
    comm.process_at_rank(0).broadcast_into(&mut len);

    let mut buffer = if comm.rank() == 0 {
        encoded
    } else {
        vec![0u8; len as usize]
    };
    comm.process_at_rank(0).broadcast_into(&mut buffer);

    decode_from_slice(&buffer, bincode::config::standard())
        .expect("failed to decode broadcast data")
        .0
}
