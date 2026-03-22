// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Builder pattern for [`TopCubicalMatching`] configuration.

use std::marker::PhantomData;

use super::{TopCubicalMatching, TopCubicalMatchingConfig};
#[cfg(feature = "mpi")]
use crate::mpi::Communicator;
use crate::{CubicalComplex, ExecutionBackend, Grader, Orthant, Ring, TopCubeGrader};

/// Builder pattern helper to build a [`TopCubicalMatching`] object with
/// various options.
pub struct TopCubicalMatchingBuilder<R, G>
where
    R: Ring,
    G: Grader<Orthant>,
{
    config: TopCubicalMatchingConfig,
    _phantom: PhantomData<(R, G)>,
}

impl<R, G> Default for TopCubicalMatchingBuilder<R, G>
where
    R: Ring,
    G: Grader<Orthant>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<R, G> TopCubicalMatchingBuilder<R, G>
where
    R: Ring,
    G: Grader<Orthant>,
{
    /// Create a builder with default configuration.
    ///
    /// Prefer [`TopCubicalMatching::builder`] which infers the generic types
    /// from the return type annotation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TopCubicalMatchingConfig::default(),
            _phantom: PhantomData,
        }
    }

    /// Consume the builder and build a [`TopCubicalMatching`] for the given
    /// complex.
    #[must_use]
    pub fn build(self, complex: CubicalComplex<R, TopCubeGrader<G>>) -> TopCubicalMatching<R, G>
    where
        G: Clone,
    {
        TopCubicalMatching::from_config(self.config, complex)
    }

    /// Set the maximum grade of critical cells found by the matching.
    #[must_use]
    pub fn max_grade(mut self, max_grade: u32) -> Self {
        self.config.maximum_critical_grade = max_grade;
        self
    }

    /// Set the maximum dimension of critical cells found by the matching.
    #[must_use]
    pub fn max_dimension(mut self, max_dimension: u32) -> Self {
        self.config.maximum_critical_dimension = max_dimension;
        self
    }

    /// Configure the shape of the subgrids of orthants matched together.
    ///
    /// Larger subgrids cache results more efficiently, but too large of
    /// subgrids may exceed contiguous cache capabilities or contain too many
    /// empty orthants to be efficient.
    #[must_use]
    pub fn subgrid_shape(mut self, subgrid_shape: Vec<i16>) -> Self {
        self.config.subgrid_shape = Some(subgrid_shape);
        self
    }

    /// Use sequential execution (default).
    #[must_use]
    pub fn sequential(mut self) -> Self {
        self.config.backend = ExecutionBackend::Sequential;
        self
    }

    /// Use Rayon shared-memory parallelism.
    #[cfg(feature = "rayon")]
    #[must_use]
    pub fn rayon(mut self) -> Self {
        self.config.backend = ExecutionBackend::Rayon;
        self
    }

    /// Use MPI with a specific communicator for distributed execution.
    ///
    /// The communicator is duplicated, so the original can be used elsewhere.
    #[cfg(feature = "mpi")]
    #[must_use]
    pub fn mpi_with_comm(mut self, comm: &dyn Communicator) -> Self {
        self.config.backend = ExecutionBackend::MPI(comm.duplicate());
        self
    }

    /// Use hybrid MPI and Rayon with a specific communicator.
    ///
    /// MPI distributes work across nodes; Rayon parallelizes within each
    /// node. The communicator is duplicated.
    #[cfg(all(feature = "mpi", feature = "rayon"))]
    #[must_use]
    pub fn hybrid_with_comm(mut self, comm: &dyn Communicator) -> Self {
        self.config.backend = ExecutionBackend::Hybrid(comm.duplicate());
        self
    }

    /// Set the execution backend directly.
    #[must_use]
    pub fn backend(mut self, backend: ExecutionBackend) -> Self {
        self.config.backend = backend;
        self
    }

    /// Set batch size for MPI work distribution.
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Set sub-batch size for level-parallel flow.
    #[must_use]
    pub fn sub_batch_size(mut self, size: usize) -> Self {
        self.config.sub_batch_size = size;
        self
    }

    /// Enable subgrid filtering with the provided top-cube orthants.
    ///
    /// Only subgrids containing cells of the specified top-dimensional cubes
    /// will be processed. Since each top-cube's faces extend into neighboring
    /// orthants, the filtering includes subgrids adjacent to each provided
    /// orthant.
    ///
    /// This improves performance for sparse complexes embedded in
    /// high-dimensional bounding boxes.
    #[must_use]
    pub fn filter_orthants(mut self, orthants: Vec<Orthant>) -> Self {
        self.config.filter_orthants = Some(orthants);
        self
    }
}
