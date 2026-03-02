// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Configuration types for top cubical Morse matching.

use std::fmt::{Debug, Formatter, Result as FmtResult};

use crate::Orthant;
#[cfg(feature = "mpi")]
use crate::mpi::{Communicator, SimpleCommunicator};

/// Execution strategy for computing the top cubical Morse matching.
///
/// Controls how the matching computation is parallelized or distributed, if at
/// all.
#[derive(Default)]
pub enum ExecutionMode {
    /// Single-threaded sequential execution (default).
    #[default]
    Sequential,
    /// MPI-based distributed execution across multiple processes.
    ///
    /// Requires the `mpi` feature. If `comm` is `None`, the matching will
    /// attempt to initialize the MPI universe and use the world communicator.
    #[cfg(feature = "mpi")]
    Mpi {
        /// Communicator for MPI processes, or `None` to use the world
        /// communicator.
        comm: Option<SimpleCommunicator>,
    },
}

impl Clone for ExecutionMode {
    fn clone(&self) -> Self {
        match self {
            Self::Sequential => Self::Sequential,
            #[cfg(feature = "mpi")]
            Self::Mpi { comm } => Self::Mpi {
                comm: comm.as_ref().map(Communicator::duplicate),
            },
        }
    }
}

impl Debug for ExecutionMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Sequential => write!(f, "Sequential"),
            #[cfg(feature = "mpi")]
            Self::Mpi { comm } => {
                let comm_str = comm
                    .as_ref()
                    .map_or("None".to_string(), |c| format!("Some({})", c.get_name()));
                write!(f, "Mpi {{ comm: {comm_str} }}")
            },
        }
    }
}

impl ExecutionMode {
    /// Returns a reference to the MPI communicator if in MPI mode with a
    /// communicator, `None` otherwise.
    #[cfg(feature = "mpi")]
    #[must_use]
    pub fn communicator(&self) -> Option<&SimpleCommunicator> {
        match self {
            Self::Mpi { comm } => comm.as_ref(),
            Self::Sequential => None,
        }
    }

    /// Returns the number of MPI processes if in MPI mode with a communicator,
    /// `None` otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the MPI communicator returns a process count that cannot be
    /// represented as a `usize`.
    #[cfg(feature = "mpi")]
    #[must_use]
    pub fn process_count(&self) -> Option<usize> {
        self.communicator()
            .map(|c| usize::try_from(c.size()).expect("invalid process count"))
    }
}

/// Configuration for [`TopCubicalMatching`](super::TopCubicalMatching).
///
/// This type controls which critical cells are identified during matching.
///
/// # Fields
///
/// - `maximum_critical_grade`: Only cells with grade at most this value are
///   marked as critical. Set to `u32::MAX` for no constraint.
/// - `maximum_critical_dimension`: Only cells with dimension at most this value
///   are marked as critical. Set to `u32::MAX` for no constraint. Note that
///   setting this to `d` only guarantees accuracy of homology up to dimension
///   `d - 1`.
/// - `subgrid_shape`: Optional shape for subdividing the grid of orthants. Each
///   entry specifies the size along one axis. Larger subgrids improve caching
///   but may exceed memory limits or contain too many empty orthants.
/// - `execution`: Execution strategy (sequential or MPI distributed).
#[derive(Clone, Debug)]
pub struct TopCubicalMatchingConfig {
    /// Maximum grade of critical cells to identify.
    pub maximum_critical_grade: u32,
    /// Maximum dimension of critical cells to identify.
    pub maximum_critical_dimension: u32,
    /// Optional shape for subdividing the grid of orthants for processing.
    pub subgrid_shape: Option<Vec<i16>>,
    /// Base orthants of top-dimensional cubes for subgrid filtering. When
    /// `Some`, only subgrids containing cells of these top-cubes will be
    /// processed. Since a top-cube's faces extend into neighboring orthants,
    /// the filtering accounts for this by including subgrids adjacent to each
    /// provided orthant.
    pub filter_orthants: Option<Vec<Orthant>>,
    /// Execution strategy for the matching computation.
    pub execution: ExecutionMode,
    /// Batch size for MPI work distribution.
    #[cfg(feature = "mpi")]
    pub mpi_batch_size: usize,
}

impl Default for TopCubicalMatchingConfig {
    fn default() -> Self {
        Self {
            maximum_critical_grade: u32::MAX,
            maximum_critical_dimension: u32::MAX,
            subgrid_shape: None,
            filter_orthants: None,
            execution: ExecutionMode::default(),
            #[cfg(feature = "mpi")]
            mpi_batch_size: 10,
        }
    }
}
