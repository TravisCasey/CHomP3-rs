// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Backend-agnostic batch parallel map.

#[cfg(feature = "mpi")]
use mpi::{
    topology::SimpleCommunicator,
    traits::{Communicator, Destination, Source},
};
#[cfg(feature = "mpi")]
use postcard::{from_bytes, to_allocvec};
#[cfg(feature = "rayon")]
use rayon::iter::ParallelBridge;
#[cfg(all(feature = "mpi", feature = "rayon"))]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tracing::Level;
#[cfg(feature = "mpi")]
use tracing::{debug, trace};

#[cfg(feature = "mpi")]
use super::mpi_utils::{self, MPITag};
use super::{ExecutionBackend, MapItem, MapResult};
use crate::logging::ProgressTracker;

/// Backend-agnostic batch parallel map.
///
/// Distributes work items across the configured [`ExecutionBackend`] and
/// collects results. Results are returned in **arbitrary order** that may vary
/// between backends and between runs.
pub struct ParallelMap<'a> {
    backend: &'a ExecutionBackend,
    label: Option<String>,
    log_level: Option<Level>,
    batch_size: usize,
    #[cfg(feature = "mpi")]
    broadcast_results: bool,
}

impl<'a> ParallelMap<'a> {
    /// Create a new parallel map using the given backend.
    #[must_use]
    pub fn new(backend: &'a ExecutionBackend) -> Self {
        Self {
            backend,
            label: None,
            log_level: None,
            batch_size: 1,
            #[cfg(feature = "mpi")]
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

    /// Set the batch size for MPI work distribution.
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

    /// Broadcast results from root to all processes after computation.
    ///
    /// When enabled, all processes receive the full result vector instead
    /// of only root. Useful when all processes need results for subsequent
    /// computation phases.
    #[cfg(feature = "mpi")]
    #[must_use]
    pub fn broadcast_results(mut self) -> Self {
        self.broadcast_results = true;
        self
    }

    /// Execute a parallel map over items.
    pub fn run<I, F, C, R>(self, items: I, compute: F) -> Vec<R>
    where
        I: ExactSizeIterator + Send,
        I::Item: MapItem,
        F: Fn(I::Item) -> C + Send + Sync,
        C: IntoIterator<Item = R> + MapResult,
        R: MapItem,
    {
        self.run_with_state(items, || (), |(), item| compute(item))
    }

    /// Execute with per-worker persistent state.
    ///
    /// `init` is called once per worker (once total for Sequential, once
    /// per Rayon thread, once per MPI process). The state is passed to
    /// `compute` for each item processed by that worker.
    pub fn run_with_state<I, S, F, C, R>(
        self,
        items: I,
        init: impl Fn() -> S + Send + Sync,
        compute: F,
    ) -> Vec<R>
    where
        I: ExactSizeIterator + Send,
        I::Item: MapItem,
        S: Send + 'static,
        F: Fn(&mut S, I::Item) -> C + Send + Sync,
        C: IntoIterator<Item = R> + MapResult,
        R: MapItem,
    {
        match self.backend {
            ExecutionBackend::Sequential => self.run_sequential(items, &init, &compute),

            #[cfg(feature = "rayon")]
            ExecutionBackend::Rayon => Self::run_rayon(items, init, compute),

            #[cfg(feature = "mpi")]
            ExecutionBackend::MPI(comm) => {
                let results = if comm.rank() == 0 {
                    self.run_mpi_root(comm, items, &init, &compute)
                } else {
                    let mut state = init();
                    Self::run_mpi_worker(comm, &mut state, &compute);
                    Vec::new()
                };
                self.maybe_broadcast(comm, results)
            },

            #[cfg(all(feature = "mpi", feature = "rayon"))]
            ExecutionBackend::Hybrid(comm) => {
                let results = if comm.rank() == 0 {
                    self.run_mpi_root(comm, items, &init, &compute)
                } else {
                    Self::run_hybrid_worker(comm, &init, &compute);
                    Vec::new()
                };
                self.maybe_broadcast(comm, results)
            },
        }
    }

    fn make_progress_tracker(&self, total: usize) -> ProgressTracker {
        let label = self.label.as_deref().unwrap_or("ParallelMap");
        let mut tracker = ProgressTracker::new(label, total);
        if let Some(level) = self.log_level {
            tracker = tracker.with_level(level);
        }
        tracker
    }

    fn run_sequential<I, S, F, C, R>(&self, items: I, init: &impl Fn() -> S, compute: &F) -> Vec<R>
    where
        I: ExactSizeIterator,
        F: Fn(&mut S, I::Item) -> C,
        C: IntoIterator<Item = R>,
    {
        let total = items.len();
        let mut progress = self.make_progress_tracker(total);
        let mut state = init();
        let results = items
            .flat_map(|item| {
                let result = compute(&mut state, item);
                progress.increment();
                result
            })
            .collect();
        progress.finish();
        results
    }

    #[cfg(feature = "rayon")]
    fn run_rayon<I, S, F, C, R>(items: I, init: impl Fn() -> S + Send + Sync, compute: F) -> Vec<R>
    where
        I: ExactSizeIterator + Send,
        I::Item: Send,
        S: Send,
        F: Fn(&mut S, I::Item) -> C + Send + Sync,
        C: IntoIterator<Item = R> + Send,
        R: Send,
    {
        use rayon::iter::ParallelIterator as _;

        items
            .par_bridge()
            .map_init(init, |state, item| compute(state, item))
            .flat_map_iter(|c| c)
            .collect()
    }

    #[cfg(feature = "mpi")]
    fn run_mpi_root<I, S, F, C, R>(
        &self,
        comm: &SimpleCommunicator,
        items: I,
        init: &impl Fn() -> S,
        compute: &F,
    ) -> Vec<R>
    where
        I: ExactSizeIterator,
        I::Item: MapItem,
        F: Fn(&mut S, I::Item) -> C,
        C: IntoIterator<Item = R> + MapResult,
    {
        let mut active_workers = usize::try_from(comm.size() - 1).expect("invalid worker count");

        if active_workers == 0 {
            tracing::warn!("No MPI workers available, executing locally");
            return self.run_sequential(items, init, compute);
        }

        let total = items.len();
        let batch_size = self.batch_size;
        let mut work_iter = items.peekable();
        let mut results = Vec::new();
        let mut progress = self.make_progress_tracker(total);

        while active_workers > 0 {
            let (msg_bytes, status): (Vec<u8>, _) = comm.any_process().receive_vec();
            let source_rank = status.source_rank();
            let tag = status.tag();

            if tag == MPITag::WorkRequest as i32 {
                trace!("Work request from rank {source_rank}");
                Self::assign_batch_or_shutdown(
                    comm,
                    &mut work_iter,
                    batch_size,
                    source_rank,
                    &mut active_workers,
                );
            } else if tag == MPITag::ResultSubmission as i32 {
                trace!("Results from rank {source_rank}");
                let worker_results: Vec<C> =
                    from_bytes(&msg_bytes).expect("failed to decode results");
                let items_completed = worker_results.len();
                for batch_result in worker_results {
                    results.extend(batch_result);
                }
                progress.increment_by(items_completed);

                Self::assign_batch_or_shutdown(
                    comm,
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

    #[cfg(feature = "mpi")]
    fn run_mpi_worker<T, S, F, C, R>(comm: &SimpleCommunicator, state: &mut S, compute: &F)
    where
        T: MapItem,
        F: Fn(&mut S, T) -> C,
        C: IntoIterator<Item = R> + MapResult,
    {
        let rank = comm.rank();
        debug!("Worker rank {rank} starting");

        comm.process_at_rank(0)
            .send_with_tag(&Vec::<u8>::new(), MPITag::WorkRequest as i32);

        loop {
            let (msg_bytes, status): (Vec<u8>, _) = comm.process_at_rank(0).receive_vec();
            let tag = status.tag();

            if tag == MPITag::Shutdown as i32 {
                debug!("Worker rank {rank} shutdown");
                break;
            }

            assert!(
                tag == MPITag::WorkAssignment as i32,
                "worker rank {rank} received unexpected tag {tag}"
            );

            let batch: Vec<T> = from_bytes(&msg_bytes).expect("failed to decode batch");
            let results: Vec<C> = batch.into_iter().map(|item| compute(state, item)).collect();

            let encoded = to_allocvec(&results).expect("failed to encode results");
            comm.process_at_rank(0)
                .send_with_tag(&encoded, MPITag::ResultSubmission as i32);
        }
    }

    #[cfg(feature = "mpi")]
    fn assign_batch_or_shutdown<T: MapItem>(
        comm: &SimpleCommunicator,
        work_iter: &mut std::iter::Peekable<impl Iterator<Item = T>>,
        batch_size: usize,
        worker_rank: i32,
        active_workers: &mut usize,
    ) {
        let batch: Vec<T> = work_iter.take(batch_size).collect();

        if batch.is_empty() {
            comm.process_at_rank(worker_rank)
                .send_with_tag(&Vec::<u8>::new(), MPITag::Shutdown as i32);
            *active_workers -= 1;
            trace!("Shutdown rank {worker_rank}, {active_workers} remaining");
        } else {
            let count = batch.len();
            let encoded = to_allocvec(&batch).expect("failed to encode batch");
            comm.process_at_rank(worker_rank)
                .send_with_tag(&encoded, MPITag::WorkAssignment as i32);
            trace!("Assigned {count} items to rank {worker_rank}");
        }
    }

    #[cfg(feature = "mpi")]
    fn maybe_broadcast<R: MapItem>(&self, comm: &SimpleCommunicator, results: Vec<R>) -> Vec<R> {
        if self.broadcast_results {
            mpi_utils::broadcast(comm, &results)
        } else {
            results
        }
    }

    #[cfg(all(feature = "mpi", feature = "rayon"))]
    fn run_hybrid_worker<T, S, F, C, R>(
        comm: &SimpleCommunicator,
        init: &(impl Fn() -> S + Send + Sync),
        compute: &F,
    ) where
        T: MapItem,
        S: Send,
        F: Fn(&mut S, T) -> C + Send + Sync,
        C: IntoIterator<Item = R> + MapResult,
        R: Send,
    {
        let rank = comm.rank();
        debug!("Hybrid worker rank {rank} starting");

        comm.process_at_rank(0)
            .send_with_tag(&Vec::<u8>::new(), MPITag::WorkRequest as i32);

        loop {
            let (msg_bytes, status): (Vec<u8>, _) = comm.process_at_rank(0).receive_vec();
            let tag = status.tag();

            if tag == MPITag::Shutdown as i32 {
                debug!("Hybrid worker rank {rank} shutdown");
                break;
            }

            assert!(
                tag == MPITag::WorkAssignment as i32,
                "hybrid worker rank {rank} received unexpected tag {tag}"
            );

            let batch: Vec<T> = from_bytes(&msg_bytes).expect("failed to decode batch");
            let results: Vec<C> = batch
                .into_par_iter()
                .map_init(init, |state, item| compute(state, item))
                .collect();

            let encoded = to_allocvec(&results).expect("failed to encode results");
            comm.process_at_rank(0)
                .send_with_tag(&encoded, MPITag::ResultSubmission as i32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_and_flat_map() {
        let backend = ExecutionBackend::Sequential;

        let mut results: Vec<i32> = ParallelMap::new(&backend).run(0..5, |x| vec![x * 2]);
        results.sort_unstable();
        assert_eq!(results, vec![0, 2, 4, 6, 8]);

        // Compute closure can return multiple items per input
        let mut results: Vec<i32> = ParallelMap::new(&backend).run(0..3, |x| vec![x, x * 10]);
        results.sort_unstable();
        assert_eq!(results, vec![0, 0, 1, 2, 10, 20]);
    }

    #[test]
    fn persistent_state() {
        let backend = ExecutionBackend::Sequential;
        let results: Vec<i32> = ParallelMap::new(&backend).run_with_state(
            0..5,
            || 10i32,
            |state, x| {
                let result = x + *state;
                *state += 1;
                vec![result]
            },
        );
        // Sequential processes in order with persistent state:
        // state=10: 0+10=10, state=11: 1+11=12, ..., state=14: 4+14=18
        assert_eq!(results, vec![10, 12, 14, 16, 18]);
    }

    #[test]
    fn empty_and_single() {
        let backend = ExecutionBackend::Sequential;
        let empty: Vec<i32> =
            ParallelMap::new(&backend).run(std::iter::empty::<i32>(), |x| vec![x]);
        assert!(empty.is_empty());

        let single: Vec<i32> = ParallelMap::new(&backend).run(std::iter::once(42), |x| vec![x * 2]);
        assert_eq!(single, vec![84]);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn rayon_correctness() {
        let backend = ExecutionBackend::Rayon;

        // Basic map
        let mut results: Vec<i32> = ParallelMap::new(&backend).run(0..100, |x| vec![x * 2]);
        results.sort_unstable();
        let expected: Vec<i32> = (0..100).map(|x| x * 2).collect();
        assert_eq!(results, expected);

        // Flat map
        let mut results: Vec<i32> = ParallelMap::new(&backend).run(0..50, |x| vec![x, x + 100]);
        results.sort_unstable();
        let mut expected: Vec<i32> = (0..50).flat_map(|x| vec![x, x + 100]).collect();
        expected.sort_unstable();
        assert_eq!(results, expected);

        // Empty input
        let empty: Vec<i32> =
            ParallelMap::new(&backend).run(std::iter::empty::<i32>(), |x| vec![x]);
        assert!(empty.is_empty());
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn rayon_per_thread_state() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let backend = ExecutionBackend::Rayon;
        let init_count = AtomicUsize::new(0);
        let results: Vec<usize> = ParallelMap::new(&backend).run_with_state(
            0..100usize,
            || {
                init_count.fetch_add(1, Ordering::Relaxed);
                0usize
            },
            |counter, _x| {
                *counter += 1;
                vec![*counter]
            },
        );

        assert!(results.iter().all(|&v| v >= 1));
        assert_eq!(results.len(), 100);
        assert!(init_count.load(Ordering::Relaxed) >= 1);
    }
}
