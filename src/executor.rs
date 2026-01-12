// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Work executor abstraction layer for parallel and distributed computation.
//!
//! This module provides traits and implementations for executing computations
//! over collections of work items. The abstraction is intended to separate
//! algorithm code from the execution backend, which may be sequential or
//! distributed/parallel.

use crate::logging::ProgressTracker;

/// Base executor trait for running computations over work items.
///
/// This trait defines the core execution interface. Implementations may
/// execute work items sequentially, in parallel via shared memory, or
/// distributed across multiple processes.
///
/// Work items are provided as an iterator, which permits lazy generation of
/// potentially large work sets without upfront memory allocation.
///
/// Each work item may produce zero, one, or many results. The compute function
/// returns a type implementing `IntoIterator`, and all results are flattened
/// into a single output vector, in arrival order.
///
/// For distributed executors, only the root process returns the full results;
/// worker processes should send their results to the root process then return
/// an empty vector.
pub trait Executor {
    /// Execute a computation over work items from an iterator.
    ///
    /// Consumes the iterator and applies the `compute` function to each item.
    /// Results from each work item are flattened into a single output vector.
    ///
    /// # Type Parameters
    ///
    /// - `I`: Iterator yielding work items.
    /// - `O`: Output type from compute function, convertible to an iterator of
    ///   `R`.
    /// - `R`: Result element type.
    /// - `F`: Compute function that transforms work items into result
    ///   collections.
    ///
    /// # Examples
    ///
    /// ```
    /// use chomp3rs::executor::{Executor, SequentialExecutor};
    ///
    /// let mut executor = SequentialExecutor;
    ///
    /// // One-to-one: each input produces exactly one output
    /// let doubled = executor.execute("Doubling", 1..4, |x| Some(x * 2));
    /// assert_eq!(doubled, vec![2, 4, 6]);
    ///
    /// // One-to-many: each input produces multiple outputs
    /// let expanded = executor.execute("Expanding", 1..4, |x| vec![x, x * 10]);
    /// assert_eq!(expanded, vec![1, 10, 2, 20, 3, 30]);
    ///
    /// // Filtering: some inputs produce no output
    /// let evens = executor.execute("Filtering", 1..6, |x| {
    ///     if x % 2 == 0 { Some(x) } else { None }
    /// });
    /// assert_eq!(evens, vec![2, 4]);
    /// ```
    fn execute<I, O, R, F>(&mut self, label: &'static str, work_items: I, compute: F) -> Vec<R>
    where
        I: ExactSizeIterator,
        O: IntoIterator<Item = R>,
        F: Fn(I::Item) -> O;
}

/// Simple single-threaded executor.
///
/// Executes work items sequentially with zero overhead. This is the default
/// executor when parallel features are disabled.
///
/// # Example
///
/// ```
/// use chomp3rs::executor::{Executor, SequentialExecutor};
///
/// let mut executor = SequentialExecutor;
///
/// // One-to-one mapping
/// let doubled = executor.execute("Doubling", 1..4, |x| Some(x * 2));
/// assert_eq!(doubled, vec![2, 4, 6]);
///
/// // One-to-many expansion
/// let pairs = executor.execute("Expanding", 1..3, |x| vec![x, x * 10]);
/// assert_eq!(pairs, vec![1, 10, 2, 20]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SequentialExecutor;

impl Executor for SequentialExecutor {
    fn execute<I, O, R, F>(&mut self, label: &'static str, work_items: I, compute: F) -> Vec<R>
    where
        I: ExactSizeIterator,
        O: IntoIterator<Item = R>,
        F: Fn(I::Item) -> O,
    {
        let mut results = Vec::new();
        let mut progress =
            ProgressTracker::new(format!("{label} (SequentialExecutor)"), work_items.len());

        for item in work_items {
            results.extend(compute(item));
            progress.increment();
        }

        progress.finish();
        results
    }
}

/// Executor trait for distributed computation requiring serialization.
///
/// This trait extends [`Executor`] with serialization bounds needed for
/// inter-process communication. Distributed executors (like [`MpiExecutor`])
/// implement this trait to enforce that work items and results can be
/// serialized.
///
/// # When to Use
///
/// Use `DistributedExecutor` when:
/// - You need to distribute work across processes (MPI, network, etc.)
/// - Your work items and results must be serializable
///
/// Use [`Executor`] when:
/// - You're executing locally (sequential, threads, etc.)
/// - You want to be generic over execution strategy without requiring serde
#[cfg(feature = "mpi")]
pub trait DistributedExecutor: Executor {
    /// Execute a computation with serializable work items and results.
    ///
    /// This method has the same semantics as [`Executor::execute`], but
    /// requires that work items and output collections are serializable
    /// for inter-process communication.
    fn execute_distributed<I, O, R, F>(
        &mut self,
        label: &'static str,
        work_items: I,
        compute: F,
    ) -> Vec<R>
    where
        I: ExactSizeIterator,
        I::Item: serde::Serialize + serde::de::DeserializeOwned,
        O: IntoIterator<Item = R> + serde::Serialize + serde::de::DeserializeOwned,
        F: Fn(I::Item) -> O;
}

#[cfg(feature = "mpi")]
mod mpi;

#[cfg(feature = "mpi")]
pub use mpi::MpiExecutor;

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn sequential_one_to_one() {
        let mut executor = SequentialExecutor;
        let results = executor.execute("test", 1i16..=5, |x| Some(x * 2));
        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn sequential_one_to_many() {
        let mut executor = SequentialExecutor;
        let results = executor.execute("test", 1i16..=3, |x| vec![x, x * 10, x * 100]);
        assert_eq!(results, vec![1, 10, 100, 2, 20, 200, 3, 30, 300]);
    }

    #[test]
    fn sequential_filtering() {
        let mut executor = SequentialExecutor;
        let results = executor.execute(
            "test",
            1i16..=6,
            |x| if x % 2 == 0 { Some(x) } else { None },
        );
        assert_eq!(results, vec![2, 4, 6]);
    }

    #[test]
    fn sequential_empty_input() {
        let mut executor = SequentialExecutor;
        let results = executor.execute("test", std::iter::empty::<i32>(), |x| Some(x * 2));
        assert!(results.is_empty());
    }

    #[test]
    fn sequential_all_empty_outputs() {
        let mut executor = SequentialExecutor;
        let results: Vec<i32> = executor.execute("test", 1i16..=5, |_| Vec::new());
        assert!(results.is_empty());
    }

    #[test]
    fn sequential_lazy_evaluation() {
        let mut executor = SequentialExecutor;
        let counter = Cell::new(0);

        // Create an iterator that tracks how many items have been consumed
        let work = (0..5).inspect(|_| {
            counter.set(counter.get() + 1);
        });

        // Before execution, nothing consumed
        assert_eq!(counter.get(), 0);

        let results = executor.execute("test", work, |x| Some(x * 2));

        // After execution, all items consumed
        assert_eq!(counter.get(), 5);
        assert_eq!(results, vec![0, 2, 4, 6, 8]);
    }
}
