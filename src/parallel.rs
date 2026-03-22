// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Parallel execution backend and utilities.
//!
//! This module provides [`ExecutionBackend`] for selecting a parallelism
//! strategy and [`ParallelMap`](map::ParallelMap) for backend-agnostic batch
//! parallel computation.

use std::fmt::{Debug, Formatter, Result as FmtResult};

#[cfg(feature = "mpi")]
use mpi::{topology::SimpleCommunicator, traits::Communicator};
#[cfg(feature = "mpi")]
use serde::{Serialize, de::DeserializeOwned};

pub mod map;
#[cfg(feature = "mpi")]
pub(crate) mod mpi_utils;

// The parallel traits use a macro to conditionally include serde supertraits
// when the `mpi` feature is enabled.

macro_rules! define_parallel_traits {
    ($($bound:path),* $(,)?) => {
        /// Alias for items processed by [`ParallelMap`](map::ParallelMap).
        pub trait MapItem: Send + 'static $(+ $bound)* {}
        impl<T: Send + 'static $(+ $bound)*> MapItem for T {}

        /// Alias for result collections from [`ParallelMap`](map::ParallelMap).
        pub trait MapResult: Send + 'static $(+ $bound)* {}
        impl<T: Send + 'static $(+ $bound)*> MapResult for T {}

        /// Alias for values that can be synchronized across processes.
        pub trait Syncable: 'static $(+ $bound)* {}
        impl<T: 'static $(+ $bound)*> Syncable for T {}
    };
}

#[cfg(not(feature = "mpi"))]
define_parallel_traits!();

#[cfg(feature = "mpi")]
define_parallel_traits!(Serialize, DeserializeOwned);

/// Execution backend for parallel computation.
///
/// Selects how [`ParallelMap`](map::ParallelMap) distributes work. Also
/// provides coordination primitives ([`is_done`](Self::is_done),
/// [`sync`](Self::sync)) for iterative parallel algorithms that need
/// termination agreement across processes.
#[derive(Default)]
pub enum ExecutionBackend {
    /// Single-threaded sequential execution.
    #[default]
    Sequential,

    /// Shared-memory parallelism via Rayon.
    #[cfg(feature = "rayon")]
    Rayon,

    /// MPI distributed execution across processes.
    #[cfg(feature = "mpi")]
    MPI(SimpleCommunicator),

    /// MPI between nodes, Rayon within each node.
    ///
    /// Behaves identically to [`MPI`](Self::MPI), except that workers process
    /// received batches using Rayon's `par_iter` instead of sequential
    /// iteration. The communicator is the same as for `MPI`.
    #[cfg(all(feature = "mpi", feature = "rayon"))]
    Hybrid(SimpleCommunicator),
}

impl Clone for ExecutionBackend {
    fn clone(&self) -> Self {
        match self {
            Self::Sequential => Self::Sequential,
            #[cfg(feature = "rayon")]
            Self::Rayon => Self::Rayon,
            #[cfg(feature = "mpi")]
            Self::MPI(comm) => Self::MPI(comm.duplicate()),
            #[cfg(all(feature = "mpi", feature = "rayon"))]
            Self::Hybrid(comm) => Self::Hybrid(comm.duplicate()),
        }
    }
}

impl Debug for ExecutionBackend {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Sequential => write!(f, "Sequential"),
            #[cfg(feature = "rayon")]
            Self::Rayon => write!(f, "Rayon"),
            #[cfg(feature = "mpi")]
            Self::MPI(_) => write!(f, "MPI"),
            #[cfg(all(feature = "mpi", feature = "rayon"))]
            Self::Hybrid(_) => write!(f, "Hybrid"),
        }
    }
}

impl ExecutionBackend {
    /// True if this process manages work dispatch.
    ///
    /// Always true for `Sequential` and `Rayon`. True for rank 0 in
    /// `MPI` and `Hybrid`.
    #[must_use]
    pub fn is_root(&self) -> bool {
        match self {
            Self::Sequential => true,
            #[cfg(feature = "rayon")]
            Self::Rayon => true,
            #[cfg(feature = "mpi")]
            Self::MPI(comm) => comm.rank() == 0,
            #[cfg(all(feature = "mpi", feature = "rayon"))]
            Self::Hybrid(comm) => comm.rank() == 0,
        }
    }

    /// Coordinate termination across all processes.
    ///
    /// - **Sequential / Rayon**: returns `local_done` unchanged.
    /// - **MPI / Hybrid**: broadcasts root's value to all processes. Workers'
    ///   `local_done` is ignored; they receive root's value.
    #[must_use]
    pub fn is_done(&self, local_done: bool) -> bool {
        match self {
            Self::Sequential => local_done,
            #[cfg(feature = "rayon")]
            Self::Rayon => local_done,
            #[cfg(feature = "mpi")]
            Self::MPI(comm) => mpi_utils::broadcast(comm, &local_done),
            #[cfg(all(feature = "mpi", feature = "rayon"))]
            Self::Hybrid(comm) => mpi_utils::broadcast(comm, &local_done),
        }
    }

    /// Synchronize a value from root to all processes.
    ///
    /// - **Sequential / Rayon**: returns `value` unchanged.
    /// - **MPI / Hybrid**: broadcasts root's value to all processes.
    #[must_use]
    pub fn sync<T: Syncable>(&self, value: T) -> T {
        match self {
            Self::Sequential => value,
            #[cfg(feature = "rayon")]
            Self::Rayon => value,
            #[cfg(feature = "mpi")]
            Self::MPI(comm) => mpi_utils::broadcast(comm, &value),
            #[cfg(all(feature = "mpi", feature = "rayon"))]
            Self::Hybrid(comm) => mpi_utils::broadcast(comm, &value),
        }
    }

    /// Returns a reference to the MPI communicator, if any.
    #[cfg(feature = "mpi")]
    #[must_use]
    pub fn communicator(&self) -> Option<&SimpleCommunicator> {
        match self {
            Self::MPI(comm) => Some(comm),
            #[cfg(feature = "rayon")]
            Self::Hybrid(comm) => Some(comm),
            _ => None,
        }
    }

    /// Returns the number of MPI processes, if applicable.
    ///
    /// # Panics
    ///
    /// Panics if the process count cannot be represented as a `usize`.
    #[cfg(feature = "mpi")]
    #[must_use]
    pub fn process_count(&self) -> Option<usize> {
        self.communicator()
            .map(|c| usize::try_from(c.size()).expect("invalid process count"))
    }
}
