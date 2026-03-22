// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Configuration types for top cubical Morse matching.

use crate::{ExecutionBackend, Orthant};

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
/// - `backend`: Execution backend (sequential, Rayon, MPI, or hybrid).
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
    /// Execution backend for matching computation.
    pub backend: ExecutionBackend,
    /// Batch size for MPI work distribution in `ParallelMap`. Default 10.
    /// Meaningful only for MPI backends; ignored by others.
    pub batch_size: usize,
    /// Maximum orthants per sub-batch in level-parallel flow. Default 10,000.
    /// Controls parallelism width vs memory pressure.
    pub sub_batch_size: usize,
}

impl Default for TopCubicalMatchingConfig {
    fn default() -> Self {
        Self {
            maximum_critical_grade: u32::MAX,
            maximum_critical_dimension: u32::MAX,
            subgrid_shape: None,
            filter_orthants: None,
            backend: ExecutionBackend::Sequential,
            batch_size: 10,
            sub_batch_size: 10_000,
        }
    }
}
