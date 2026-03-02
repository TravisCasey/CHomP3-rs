// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Builder pattern for [`TopCubicalMatching`] configuration.

use std::marker::PhantomData;

#[cfg(feature = "mpi")]
use super::ExecutionMode;
use super::{TopCubicalMatching, TopCubicalMatchingConfig};
#[cfg(feature = "mpi")]
use crate::mpi::Communicator;
use crate::{CubicalComplex, Grader, Orthant, Ring, TopCubeGrader};

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

    /// Use MPI to distribute the computation across multiple processes.
    ///
    /// The matching will attempt to initialize the MPI universe and use the
    /// world communicator. For a custom communicator, use
    /// [`mpi_with_comm`](Self::mpi_with_comm) instead.
    #[cfg(feature = "mpi")]
    #[must_use]
    pub fn mpi(mut self) -> Self {
        self.config.execution = ExecutionMode::Mpi { comm: None };
        self
    }

    /// Use MPI with a specific communicator for distributed execution.
    ///
    /// The communicator is duplicated, so the original can be used elsewhere.
    #[cfg(feature = "mpi")]
    #[must_use]
    pub fn mpi_with_comm(mut self, comm: &dyn Communicator) -> Self {
        self.config.execution = ExecutionMode::Mpi {
            comm: Some(comm.duplicate()),
        };
        self
    }

    /// Set the batch size for MPI work distribution.
    #[cfg(feature = "mpi")]
    #[must_use]
    pub fn mpi_batch_size(mut self, size: usize) -> Self {
        self.config.mpi_batch_size = size;
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
