// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Efficient discrete Morse matching for cubical complexes.
//!
//! This module provides a specialized algorithm for computing discrete Morse
//! matchings on [`CubicalComplex`] instances using top-dimensional cube grading
//! via [`TopCubeGrader`]. The primary type is [`TopCubicalMatching`], which
//! implements the [`MorseMatching`] trait.
//!
//! # Algorithm Overview
//!
//! The matching algorithm operates on orthants rather than individual cells,
//! processing entire suborthants (subsets of cells with orthant structure)
//! simultaneously. This approach dramatically improves efficiency for large
//! high-dimensional complexes by:
//!
//! - Minimizing top cube grade queries through caching
//! - Processing nearby orthants together for locality
//! - Matching cells in bulk rather than individually
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```
//! use chomp3rs::{
//!     ComplexLike, Cube, CubicalComplex, Cyclic, Grader, HashMapGrader,
//!     HashMapModule, MorseMatching, Orthant, TopCubeGrader,
//!     TopCubicalMatching,
//! };
//!
//! // Create a cubical complex
//! let min = Orthant::from([0, 0]);
//! let max = Orthant::from([3, 3]);
//! let grader = TopCubeGrader::new(HashMapGrader::new(0), None);
//! let complex: CubicalComplex<HashMapModule<Cube, Cyclic<7>>, _> =
//!     CubicalComplex::new(min, max, grader);
//!
//! // Compute matching with default configuration
//! let mut matching = TopCubicalMatching::new();
//! matching.compute_matching(complex);
//!
//! // Access critical cells
//! let critical_cells = matching.critical_cells();
//! ```
//!
//! With configuration:
//!
//! ```
//! use chomp3rs::{
//!     Cube, Cyclic, HashMapGrader, HashMapModule, Orthant, TopCubicalMatching,
//! };
//!
//! // Using builder pattern
//! let matching = TopCubicalMatching::<
//!     HashMapModule<Cube, Cyclic<2>>,
//!     HashMapGrader<Orthant>,
//! >::builder()
//! .max_grade(0)
//! .max_dimension(1)
//! .build::<HashMapModule<Cube, Cyclic<2>>, HashMapGrader<Orthant>>();
//!
//! // Or using convenience constructors
//! let matching = TopCubicalMatching::<
//!     HashMapModule<Cube, Cyclic<2>>,
//!     HashMapGrader<Orthant>,
//! >::with_max_grade(0);
//! let matching = TopCubicalMatching::<
//!     HashMapModule<Cube, Cyclic<2>>,
//!     HashMapGrader<Orthant>,
//! >::with_constraints(0, 1);
//! ```

use std::{collections::HashMap, fmt::Debug, iter::zip};

// Re-export from submodules
pub use gradient::TopCubicalGradientPropagator;
#[cfg(feature = "mpi")]
use mpi::{topology::SimpleCommunicator, traits::Communicator};
#[cfg(feature = "serde")]
use serde::Serialize;
#[cfg(feature = "mpi")]
use serde::de::DeserializeOwned;
pub use subgrid::{GridSubdivision, OrthantMatching, Subgrid};
#[cfg(feature = "mpi")]
use tracing::{info, warn};

#[cfg(feature = "mpi")]
use crate::executor::{DistributedExecutor, MpiExecutor};
use crate::{
    CellMatch, Cube, CubicalComplex, Grader, HashMapModule, ModuleLike, MorseMatching, Orthant,
    RingLike, TopCubeGrader,
    executor::{Executor, SequentialExecutor},
};

// Public submodules for advanced users
pub mod gradient;
pub mod subgrid;

/// Discrete Morse matching for cubical complexes with top-cube grading.
///
/// This type computes an acyclic partial matching (satisfying
/// [`MorseMatching`]) on a [`CubicalComplex`] with grading determined by
/// top-dimensional cubes via [`TopCubeGrader`]. The matching identifies
/// critical cells and computes their boundaries in the resulting Morse complex,
/// enabling efficient homology computation.
///
/// # Overview
///
/// In discrete Morse theory, cells are paired (matched) to reduce complex size
/// while preserving homology. Unpaired cells are called *critical cells* or
/// *Aces*. This implementation uses a specialized algorithm for cubical
/// complexes that processes entire orthants at once rather than individual
/// cells.
///
/// # Algorithm
///
/// Within each orthant, cubes that differ along a single axis and have the same
/// grade are candidates for matching. For example, these two cubes in the same
/// orthant differ only in x-extent and may be matched if they have the same
/// grade:
///
/// ```
/// use chomp3rs::{Cube, Orthant};
///
/// let cube1 =
///     Cube::from_extent(Orthant::from([0, 0, 0]), &[true, false, true]);
/// let cube2 =
///     Cube::from_extent(Orthant::from([0, 0, 0]), &[false, false, true]);
/// ```
///
/// However, the algorithm may choose a different matching if it finds a more
/// efficient way to match cells in the orthant.
///
/// The naive approach of matching cells individually is infeasible for large
/// high-dimensional complexes. Instead, this implementation:
///
/// 1. **Processes orthants in bulk**: Entire suborthants (lower-dimensional
///    orthant-like structures within an orthant) are matched simultaneously
/// 2. **Caches grade queries**: Top-cube grades are queried frequently; caching
///    them for nearby orthants dramatically improves performance
/// 3. **Minimizes recomputation**: Each cube requires at most one top-cube
///    grading query (naive approaches require an exponential (in the ambient
///    dimension) queries.
///
/// This approach is detailed in Harker, Mischaikow, Spendlove, *Morse Theoretic
/// Templates for High Dimensional Homology Computation*.
///
/// # Two-Phase Computation
///
/// The matching is computed in two phases by [`compute_matching`]:
///
/// 1. **Critical cell identification**: All critical cells (Aces) in the
///    complex are identified
/// 2. **Gradient computation**: The boundary of each critical cell is computed
///    *in the Morse complex* (not the original cubical complex) by flowing
///    boundaries through the matching
///
/// After computation, critical cells and their boundaries can be accessed via
/// the [`MorseMatching`] trait methods and used to construct a [`CellComplex`]
/// representing the Morse complex.
///
/// # Configuration
///
/// Matching behavior can be configured to:
/// - Limit critical cells by grade or dimension (optimization for sublevel
///   sets)
/// - Adjust subgrid size for cache efficiency
/// - Enable MPI parallelization (requires `mpi` feature)
///
/// See [`TopCubicalMatchingConfig`] and builder methods for details.
///
/// [`compute_matching`]: MorseMatching::compute_matching
/// [`CellComplex`]: crate::CellComplex
#[derive(Debug, Clone)]
pub struct TopCubicalMatching<UM, G, LM = HashMapModule<u32, <UM as ModuleLike>::Ring>>
where
    UM: ModuleLike<Cell = Cube>,
    G: Grader<Orthant>,
{
    complex: Option<CubicalComplex<UM, TopCubeGrader<G>>>,
    critical_cells: Vec<Cube>,
    boundaries: Vec<LM>,
    projection: HashMap<Cube, u32>,
    config: TopCubicalMatchingConfig,
}

impl<UM, G> TopCubicalMatching<UM, G>
where
    UM: ModuleLike<Cell = Cube>,
    G: Grader<Orthant>,
{
    /// Creates a matching with default configuration.
    ///
    /// All cells are included in the matching regardless of grade or dimension.
    /// Use the builder pattern or convenience constructors for configuration.
    ///
    /// No matching is performed until [`TopCubicalMatching::compute_matching`]
    /// has been called and thus most methods from the [`MorseMatching`] trait
    /// implementation will panic if called before.
    #[must_use]
    pub fn new() -> Self {
        Self::from_config(TopCubicalMatchingConfig::default())
    }

    /// Creates a matching that only finds critical cells up to the given grade.
    ///
    /// This can be a key optimization when you only need the (co)homology of
    /// a complex embedded in a larger one (of higher grade). As matchings
    /// respect grading, there is no accuracy issue with this constraint.
    #[must_use]
    pub fn with_max_grade(max_grade: u32) -> Self {
        Self::builder().max_grade(max_grade).build()
    }

    /// Creates a matching that only finds critical cells up to the given
    /// dimension.
    ///
    /// Note that setting the maximum dimension to `d` will only guarantee
    /// accuracy of homology up to dimensions `d - 1`.
    #[must_use]
    pub fn with_max_dimension(max_dimension: u32) -> Self {
        Self::builder().max_dimension(max_dimension).build()
    }

    /// Creates a matching with both grade and dimension constraints.
    ///
    /// This is equivalent to using the [builder pattern](Self::builder) with
    /// both constraints, but provides a convenient shorthand for the common
    /// case of setting both.
    #[must_use]
    pub fn with_constraints(max_grade: u32, max_dimension: u32) -> Self {
        Self::builder()
            .max_grade(max_grade)
            .max_dimension(max_dimension)
            .build()
    }

    /// Returns a builder for detailed configuration.
    #[must_use]
    pub fn builder() -> TopCubicalMatchingBuilder {
        TopCubicalMatchingBuilder::new()
    }

    /// Creates a matching with MPI-based parallel computation enabled.
    ///
    /// This constructor requires the `mpi` feature flag. It will attempt to
    /// initialize the MPI universe and use the world communicator to control
    /// processes. If MPI initialization fails, the matching will fall back to
    /// sequential computation with a warning.
    ///
    /// If the `mpi` flag is enabled, this configuration can also be specified
    /// using the provided [builder pattern](Self::builder).
    #[cfg(feature = "mpi")]
    #[must_use]
    pub fn new_parallel() -> Self {
        Self::builder().parallel().build()
    }

    fn from_config(config: TopCubicalMatchingConfig) -> Self {
        Self {
            complex: None,
            critical_cells: Vec::new(),
            boundaries: Vec::new(),
            projection: HashMap::new(),
            config,
        }
    }

    /// Returns a reference to the matching configuration.
    ///
    /// This allows introspection of the constraints and settings used for this
    /// matching.
    #[must_use]
    pub fn config(&self) -> &TopCubicalMatchingConfig {
        &self.config
    }
}

impl<UM, G> Default for TopCubicalMatching<UM, G>
where
    UM: ModuleLike<Cell = Cube>,
    G: Grader<Orthant>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<UM, G> From<TopCubicalMatchingConfig> for TopCubicalMatching<UM, G>
where
    UM: ModuleLike<Cell = Cube>,
    G: Grader<Orthant>,
{
    fn from(config: TopCubicalMatchingConfig) -> Self {
        Self::from_config(config)
    }
}

impl<UM, G, LM> TopCubicalMatching<UM, G, LM>
where
    UM: ModuleLike<Cell = Cube>,
    LM: ModuleLike<Cell = u32, Ring = UM::Ring>,
    G: Grader<Orthant> + Clone,
{
    /// Recursive method used to iterate through the nested `orthant_matching`
    /// to find the match of `cube`.
    ///
    /// This is only called when using the `match_cell` method, and not during
    /// the initial critical cell and boundary computation.
    fn match_helper(
        cube: Cube,
        extent: u32,
        orthant_matching: &OrthantMatching,
    ) -> CellMatch<Cube, UM::Ring, Orthant> {
        match orthant_matching {
            OrthantMatching::Critical { .. } => CellMatch::Ace { cell: cube },
            OrthantMatching::Leaf { match_axis, .. } => {
                // Cubical complex incidence (within same orthant):
                // - 111 <-> 110: incidence -1
                // - 111 <-> 101: incidence 1
                // - 111 <-> 011: incidence -1
                let incidence = if ((extent % (1 << match_axis)).count_ones()).is_multiple_of(2) {
                    -UM::Ring::one()
                } else {
                    UM::Ring::one()
                };

                // If the cube has extent along the match axis, it is a king;
                // Else, it is a queen.
                let base_orthant = cube.base().clone();
                let mut matched_cell = cube.clone();
                if cube.base_coord(*match_axis as usize) == cube.dual_coord(*match_axis as usize) {
                    matched_cell.dual_mut()[*match_axis as usize] -= 1;
                    CellMatch::King {
                        cell: cube,
                        queen: matched_cell,
                        incidence,
                        priority: base_orthant,
                    }
                } else {
                    matched_cell.dual_mut()[*match_axis as usize] += 1;
                    CellMatch::Queen {
                        cell: cube,
                        king: matched_cell,
                        incidence,
                        priority: base_orthant,
                    }
                }
            },
            OrthantMatching::Branch {
                upper_extent,
                prime_extent,
                suborthant_matchings,
            } => {
                // The current suborthant index to recurse into is the position
                // of the highest bit set in `extent`, considering only those
                // in upper_extent & !prime_extent (note that upper_extent has
                // at least the bits of prime_extent set).
                //
                // It must be that upper_extent also has at least the bits of
                // extent set.
                //
                // Example:
                // upper_extent: 110101
                // prime_extent: 010100
                // upper_extent & !prime_extent: 100001
                //
                // extent: 000000 -> suborthant index 0
                // extent: 010000 -> suborthant index 0
                // extent: 000101 -> suborthant index 1
                // extent: 100000 -> suborthant index 2
                // extent: 100001 -> suborthant index 2
                // etc.

                debug_assert_eq!(extent & !upper_extent, 0);

                let suborthant_index = {
                    // upper_extent: 01100101011100000111011111110101
                    // prime_extent: 00100101000000000101011100010100
                    // extent:       00100101010000000101011110010101
                    // differing:    00000000010000000000000010000001
                    // to_shift: 31 - 22 = 9
                    // shifted:               11100000010000011100001000000000
                    // popcount: 8

                    let differing = extent & !prime_extent;
                    if differing == 0 {
                        0
                    } else {
                        let to_shift = 31 - differing.ilog2();
                        let shifted = (upper_extent & !prime_extent) << to_shift;
                        shifted.count_ones()
                    }
                } as usize;

                Self::match_helper(cube, extent, &suborthant_matchings[suborthant_index])
            },
        }
    }

    /// Compute the matching using the provided executor.
    ///
    /// This method is executor-agnostic and works with any `Executor`
    /// implementation.
    fn compute_with_executor<E: Executor>(&mut self, mut executor: E) {
        let complex = self.complex.as_ref().unwrap();
        let subdivision = GridSubdivision::new(
            complex.minimum().clone(),
            complex.maximum().clone(),
            self.config.subgrid_shape.clone(),
            self.config.filter_orthants.as_deref(),
        );

        // Phase 1: Compute critical cells from subgrids
        let max_grade = self.config.maximum_critical_grade;
        let max_dim = self.config.maximum_critical_dimension;

        let critical_cells: Vec<Cube> = executor.execute(
            "Critical cells",
            subdivision.into_iter(),
            |(minimum_orthant, maximum_orthant)| {
                let mut subgrid = Subgrid::new(complex, max_grade, max_dim);
                let base_critical_orthant = subgrid.match_subgrid(minimum_orthant, maximum_orthant);

                let mut cells = Vec::new();
                for (_, critical, _) in base_critical_orthant {
                    cells.extend(critical);
                }
                cells
            },
        );

        // Build projection map
        for (index, cube) in critical_cells.iter().enumerate() {
            self.projection.insert(cube.clone(), index as u32);
        }
        self.critical_cells = critical_cells;

        // Phase 2: Compute gradients (boundaries) for each critical cell
        let projection = &self.projection;
        let boundaries: Vec<LM> =
            executor.execute("Gradients", self.critical_cells.iter(), |critical_cell| {
                let mut gradient_computer = TopCubicalGradientPropagator::new(complex);
                let upper_gradient_chain = gradient_computer.compute_gradient(critical_cell);

                let boundary: LM = upper_gradient_chain
                    .into_iter()
                    .map(|(cube, coef)| (projection[&cube], coef))
                    .collect();
                Some(boundary)
            });

        self.boundaries = boundaries;
    }
}

#[cfg(feature = "mpi")]
impl<UM, G, LM> TopCubicalMatching<UM, G, LM>
where
    UM: ModuleLike<Cell = Cube>,
    UM::Ring: Serialize + DeserializeOwned,
    LM: ModuleLike<Cell = u32, Ring = UM::Ring>,
    G: Grader<Orthant> + Clone,
{
    /// Compute the matching using the MPI distributed executor.
    ///
    /// This method distributes work across MPI processes. The root process
    /// coordinates work distribution and collects results. Worker processes
    /// compute assigned work items and return empty results (data is gathered
    /// at root).
    #[allow(clippy::type_complexity)]
    fn compute_with_distributed_executor(&mut self, mut executor: MpiExecutor) {
        info!("Beginning distributed matching with MPI executor...");
        let complex = self.complex.as_ref().unwrap();
        let subdivision = GridSubdivision::new(
            complex.minimum().clone(),
            complex.maximum().clone(),
            self.config.subgrid_shape.clone(),
            self.config.filter_orthants.as_deref(),
        );

        // Phase 1: Compute critical cells from subgrids (distributed)
        let max_grade = self.config.maximum_critical_grade;
        let max_dim = self.config.maximum_critical_dimension;

        let critical_cells: Vec<Cube> = executor.execute_distributed(
            "Critical cells",
            subdivision.into_iter(),
            |(minimum_orthant, maximum_orthant)| {
                let mut subgrid = Subgrid::new(complex, max_grade, max_dim);
                let base_critical_orthant = subgrid.match_subgrid(minimum_orthant, maximum_orthant);

                let mut cells = Vec::new();
                for (_, critical, _) in base_critical_orthant {
                    cells.extend(critical);
                }
                cells
            },
        );

        // Broadcast critical cells and projection to all processes
        let critical_cells: Vec<Cube> = executor.broadcast(&critical_cells);

        // Build projection map on all processes
        let mut projection = HashMap::new();
        for (index, cube) in critical_cells.iter().enumerate() {
            projection.insert(cube.clone(), index as u32);
        }

        // Phase 2: Compute gradients (distributed)
        let boundaries_vec: Vec<(u32, Vec<(u32, UM::Ring)>)> = executor.execute_distributed(
            "Gradients",
            0..critical_cells.len(),
            |cell_index: usize| {
                let critical_cell = &critical_cells[cell_index];
                let mut gradient_computer = TopCubicalGradientPropagator::new(complex);
                let upper_gradient_chain = gradient_computer.compute_gradient(critical_cell);

                let boundary_pairs: Vec<(u32, UM::Ring)> = upper_gradient_chain
                    .into_iter()
                    .map(|(cube, coef)| (projection[&cube], coef))
                    .collect();

                Some((cell_index as u32, boundary_pairs))
            },
        );

        // Reconstruct boundaries in order (only on root, but we handle empty on
        // workers)
        let mut boundaries: Vec<LM> = vec![LM::new(); critical_cells.len()];
        for (cell_index, boundary_pairs) in boundaries_vec {
            boundaries[cell_index as usize] = LM::from_iter(boundary_pairs);
        }

        self.critical_cells = critical_cells;
        self.projection = projection;
        self.boundaries = boundaries;
    }
}

impl<UM, G, LM> TopCubicalMatching<UM, G, LM>
where
    UM: ModuleLike<Cell = Cube>,
    LM: ModuleLike<Cell = u32, Ring = UM::Ring>,
    G: Grader<Orthant> + Clone,
{
    fn match_cell_impl(&self, cube: &Cube) -> CellMatch<Cube, UM::Ring, Orthant> {
        assert!(
            self.complex.is_some(),
            "matching not yet computed; call compute_matching first"
        );

        let mut subgrid = Subgrid::new(self.complex.as_ref().unwrap(), u32::MAX, u32::MAX);
        let orthant_matching =
            &subgrid.match_subgrid(cube.base().clone(), cube.base().clone())[0].2;

        // TODO: this should probably just be a function in Cube: using
        // Vec<bool> is kinda awkward
        let mut extent = 0u32;
        for (axis, (base_coord, dual_coord)) in
            zip(cube.base().iter(), cube.dual().iter()).enumerate()
        {
            if base_coord == dual_coord {
                extent += 1 << axis;
            }
        }

        Self::match_helper(cube.clone(), extent, orthant_matching)
    }

    fn get_upper_complex_impl(&self) -> Option<&CubicalComplex<UM, TopCubeGrader<G>>> {
        self.complex.as_ref()
    }

    fn boundary_and_coboundary_impl(&self) -> (Vec<LM>, Vec<LM>) {
        let mut coboundaries = vec![LM::new(); self.boundaries.len()];
        for (cell, boundary) in self.boundaries.iter().enumerate() {
            for (boundary_cell, coef) in boundary.iter() {
                coboundaries[*boundary_cell as usize].insert_or_add(cell as u32, coef.clone());
            }
        }

        (self.boundaries.clone(), coboundaries)
    }
}

#[cfg(feature = "mpi")]
impl<UM, G, LM> MorseMatching for TopCubicalMatching<UM, G, LM>
where
    UM: ModuleLike<Cell = Cube>,
    UM::Ring: Serialize + DeserializeOwned,
    LM: ModuleLike<Cell = u32, Ring = UM::Ring>,
    G: Grader<Orthant> + Clone,
{
    type LowerModule = LM;
    type Priority = Orthant;
    type Ring = UM::Ring;
    type UpperCell = Cube;
    type UpperComplex = CubicalComplex<UM, TopCubeGrader<G>>;
    type UpperModule = UM;

    fn compute_matching(&mut self, complex: Self::UpperComplex) {
        self.complex = Some(complex);
        self.critical_cells.clear();
        self.boundaries.clear();
        self.projection.clear();

        if self.config.parallel {
            if let Some(executor) = if let Some(comm) = self.config.comm.as_ref() {
                Some(MpiExecutor::new(comm))
            } else {
                MpiExecutor::from_world()
            } {
                let comm_size = executor.size();
                info!("Initialized MPI executor with {comm_size} processes");
                self.compute_with_distributed_executor(executor);
                return;
            }
            warn!("MPI universe cannot be initialized. Falling back to sequential computation.");
        }

        self.compute_with_executor(SequentialExecutor);
    }

    fn match_cell(&self, cube: &Cube) -> CellMatch<Cube, Self::Ring, Self::Priority> {
        self.match_cell_impl(cube)
    }

    fn get_upper_complex(&self) -> Option<&Self::UpperComplex> {
        self.get_upper_complex_impl()
    }

    fn take_upper_complex(self) -> Option<Self::UpperComplex> {
        self.complex
    }

    fn critical_cells(&self) -> Vec<Self::UpperCell> {
        self.critical_cells.clone()
    }

    fn project_cell(&self, cell: Self::UpperCell) -> Option<u32> {
        self.projection.get(&cell).copied()
    }

    fn include_cell(&self, cell: u32) -> Self::UpperCell {
        self.critical_cells[cell as usize].clone()
    }

    fn boundary_and_coboundary(&self) -> (Vec<Self::LowerModule>, Vec<Self::LowerModule>) {
        self.boundary_and_coboundary_impl()
    }
}

#[cfg(not(feature = "mpi"))]
impl<UM, G, LM> MorseMatching for TopCubicalMatching<UM, G, LM>
where
    UM: ModuleLike<Cell = Cube>,
    LM: ModuleLike<Cell = u32, Ring = UM::Ring>,
    G: Grader<Orthant> + Clone,
{
    type LowerModule = LM;
    type Priority = Orthant;
    type Ring = UM::Ring;
    type UpperCell = Cube;
    type UpperComplex = CubicalComplex<UM, TopCubeGrader<G>>;
    type UpperModule = UM;

    fn compute_matching(&mut self, complex: Self::UpperComplex) {
        self.complex = Some(complex);
        self.critical_cells.clear();
        self.boundaries.clear();
        self.projection.clear();

        self.compute_with_executor(SequentialExecutor);
    }

    fn match_cell(&self, cube: &Cube) -> CellMatch<Cube, Self::Ring, Self::Priority> {
        self.match_cell_impl(cube)
    }

    fn get_upper_complex(&self) -> Option<&Self::UpperComplex> {
        self.get_upper_complex_impl()
    }

    fn take_upper_complex(self) -> Option<Self::UpperComplex> {
        self.complex
    }

    fn critical_cells(&self) -> Vec<Self::UpperCell> {
        self.critical_cells.clone()
    }

    fn project_cell(&self, cell: Self::UpperCell) -> Option<u32> {
        self.projection.get(&cell).copied()
    }

    fn include_cell(&self, cell: u32) -> Self::UpperCell {
        self.critical_cells[cell as usize].clone()
    }

    fn boundary_and_coboundary(&self) -> (Vec<Self::LowerModule>, Vec<Self::LowerModule>) {
        self.boundary_and_coboundary_impl()
    }
}

/// Configuration for [`TopCubicalMatching`].
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
/// - `parallel`: Whether to use MPI-based parallel computation (requires `mpi`
///   feature).
#[cfg_attr(not(feature = "mpi"), derive(Clone, Debug))]
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
    /// Whether to use MPI-based parallel computation.
    #[cfg(feature = "mpi")]
    pub parallel: bool,
    /// The communicator to other MPI processes. If unspecified, the matching
    /// will attempt to initialize the MPI universe and use the world
    /// communicator instead.
    #[cfg(feature = "mpi")]
    pub comm: Option<SimpleCommunicator>,
}

#[cfg(not(feature = "mpi"))]
impl Default for TopCubicalMatchingConfig {
    fn default() -> Self {
        Self {
            maximum_critical_grade: u32::MAX,
            maximum_critical_dimension: u32::MAX,
            subgrid_shape: None,
            filter_orthants: None,
        }
    }
}

#[cfg(feature = "mpi")]
impl Default for TopCubicalMatchingConfig {
    fn default() -> Self {
        Self {
            maximum_critical_grade: u32::MAX,
            maximum_critical_dimension: u32::MAX,
            subgrid_shape: None,
            filter_orthants: None,
            parallel: false,
            comm: None,
        }
    }
}

#[cfg(feature = "mpi")]
impl Clone for TopCubicalMatchingConfig {
    fn clone(&self) -> Self {
        Self {
            maximum_critical_grade: self.maximum_critical_grade,
            maximum_critical_dimension: self.maximum_critical_dimension,
            subgrid_shape: self.subgrid_shape.clone(),
            parallel: self.parallel,
            filter_orthants: self.filter_orthants.clone(),
            comm: self.comm.as_ref().map(Communicator::duplicate),
        }
    }
}

#[cfg(feature = "mpi")]
impl Debug for TopCubicalMatchingConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TopCubicalMatchingConfig")
            .field("maximum_critical_grade", &self.maximum_critical_grade)
            .field(
                "maximum_critical_dimension",
                &self.maximum_critical_dimension,
            )
            .field("subgrid_shape", &self.subgrid_shape)
            .field("filter_orthants", &self.filter_orthants)
            .field("parallel", &self.parallel)
            .field(
                "communicator",
                &if self.comm.is_some() {
                    format!("Some({})", self.comm.as_ref().unwrap().get_name())
                } else {
                    "None".into()
                },
            )
            .finish()
    }
}

/// Builder pattern helper to build a [`TopCubicalMatching`] object with
/// various options.
#[derive(Default)]
pub struct TopCubicalMatchingBuilder {
    config: TopCubicalMatchingConfig,
}

impl TopCubicalMatchingBuilder {
    /// Create a builder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TopCubicalMatchingConfig::default(),
        }
    }

    /// Consume the builder and return the configured [`TopCubicalMatching`]
    /// object.
    #[must_use]
    pub fn build<UM, G>(self) -> TopCubicalMatching<UM, G>
    where
        UM: ModuleLike<Cell = Cube>,
        G: Grader<Orthant>,
    {
        TopCubicalMatching::from_config(self.config)
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

    /// Denote that the matching should use MPI to split computations between a
    /// root and the remaining child worker processes.
    #[cfg(feature = "mpi")]
    #[must_use]
    pub fn parallel(mut self) -> Self {
        self.config.parallel = true;
        self
    }

    /// Supply the communicator that should be used for distributed MPI
    /// execution.
    ///
    /// Will not be used unless the `parallel` flag is also set (via
    /// [`TopCubicalMatchingConfig::parallel`]). If unspecified, the matching
    /// will attempt to initialize the MPI universe and use the world
    /// communicator.
    #[cfg(feature = "mpi")]
    #[must_use]
    pub fn with_communicator(mut self, comm: &dyn Communicator) -> Self {
        self.config.comm = Some(comm.duplicate());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ComplexLike, CoreductionMatching, Cube, CubicalComplex, Cyclic, Grader, HashMapGrader,
        HashMapModule, ModuleLike, Orthant, TopCubeGrader,
    };

    fn generate_top_cube_torus_orthants() -> Vec<Orthant> {
        let mut top_cubes = Vec::new();
        for x in 0..7 {
            for y in 0..7 {
                // Create hole in the middle
                if x == 3 && y == 3 {
                    continue;
                }
                for z in 0..3 {
                    if z == 1 {
                        // Make the torus hollow
                        if (x == 1 || x == 5) && (1..=5).contains(&y) {
                            continue;
                        }
                        if (y == 1 || y == 5) && (1..=5).contains(&x) {
                            continue;
                        }
                    }
                    top_cubes.push(Orthant::from([x, y, z]));
                }
            }
        }

        top_cubes
    }

    pub fn top_cube_torus_hashmap()
    -> CubicalComplex<HashMapModule<Cube, Cyclic<2>>, TopCubeGrader<HashMapGrader<Orthant>>> {
        let minimum = Orthant::from([0, 0, 0]);
        let maximum = Orthant::from([7, 7, 3]);

        let top_cubes = generate_top_cube_torus_orthants();

        let grader = TopCubeGrader::new(HashMapGrader::uniform(top_cubes, 0, 1), Some(0));

        CubicalComplex::new(minimum, maximum, grader)
    }

    #[test]
    fn full_reduce_cube_torus_complex() {
        let complex = top_cube_torus_hashmap();
        let mut matching = TopCubicalMatching::default();
        let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

        let mut cells_by_dimension = [Vec::new(), Vec::new(), Vec::new()];
        for cell in morse_complex.iter() {
            if morse_complex.grade(&cell) == 0 {
                cells_by_dimension[morse_complex.cell_dimension(&cell) as usize].push(cell);
                assert_eq!(morse_complex.cell_boundary(&cell), HashMapModule::new());
            }
        }
        assert_eq!(cells_by_dimension[0].len(), 1, "0-dimensional cells");
        assert_eq!(cells_by_dimension[1].len(), 2, "1-dimensional cells");
        assert_eq!(cells_by_dimension[2].len(), 1, "2-dimensional cells");
    }

    #[test]
    fn builder_pattern_works() {
        let matching: TopCubicalMatching<HashMapModule<Cube, Cyclic<2>>, HashMapGrader<Orthant>> =
            TopCubicalMatchingBuilder::new()
                .max_grade(10)
                .max_dimension(5)
                .build();

        assert_eq!(matching.config().maximum_critical_grade, 10);
        assert_eq!(matching.config().maximum_critical_dimension, 5);
    }

    #[test]
    fn convenience_constructors_work() {
        let matching: TopCubicalMatching<HashMapModule<Cube, Cyclic<2>>, HashMapGrader<Orthant>> =
            TopCubicalMatching::with_max_grade(10);
        assert_eq!(matching.config().maximum_critical_grade, 10);

        let matching: TopCubicalMatching<HashMapModule<Cube, Cyclic<2>>, HashMapGrader<Orthant>> =
            TopCubicalMatching::with_max_dimension(5);
        assert_eq!(matching.config().maximum_critical_dimension, 5);

        let matching: TopCubicalMatching<HashMapModule<Cube, Cyclic<2>>, HashMapGrader<Orthant>> =
            TopCubicalMatching::with_constraints(10, 5);
        assert_eq!(matching.config().maximum_critical_grade, 10);
        assert_eq!(matching.config().maximum_critical_dimension, 5);
    }

    /// Test that filtered matching finds all critical cells that are reachable
    /// via gradient flow. This reproduces a bug where subgrid filtering was
    /// too aggressive and excluded orthants containing critical cells that
    /// appear in the boundary of other critical cells.
    #[test]
    fn filtered_matching_includes_all_boundary_cells() {
        // Create a simple complex where filtering could miss critical cells.
        // Use orthants at subgrid boundaries to trigger the bug.
        let top_cubes = vec![Orthant::from([1, 1]), Orthant::from([2, 2])];
        let minimum = Orthant::from([0, 0]);
        let maximum = Orthant::from([3, 3]);
        let grader = TopCubeGrader::new(HashMapGrader::uniform(top_cubes.clone(), 0, 1), Some(0));
        let complex: CubicalComplex<HashMapModule<Cube, Cyclic<2>>, _> =
            CubicalComplex::new(minimum, maximum, grader);

        // Use filter_orthants with the same orthants used for grading
        let mut matching: TopCubicalMatching<
            HashMapModule<Cube, Cyclic<2>>,
            HashMapGrader<Orthant>,
        > = TopCubicalMatching::<HashMapModule<Cube, Cyclic<2>>, HashMapGrader<Orthant>>::builder()
            .filter_orthants(top_cubes)
            .max_grade(0)
            .build();

        // This should not panic - if it does, gradient found a critical cell
        // not in the projection map
        matching.compute_matching(complex);

        // Verify we found critical cells
        assert!(!matching.critical_cells().is_empty());
    }

    /// Test with the torus complex using filtered matching to ensure
    /// all critical boundary cells are included.
    #[test]
    fn filtered_torus_matching_works() {
        let top_cubes = generate_top_cube_torus_orthants();
        let complex = top_cube_torus_hashmap();

        let mut matching: TopCubicalMatching<
            HashMapModule<Cube, Cyclic<2>>,
            HashMapGrader<Orthant>,
        > = TopCubicalMatching::<HashMapModule<Cube, Cyclic<2>>, HashMapGrader<Orthant>>::builder()
            .filter_orthants(top_cubes)
            .max_grade(0)
            .build();

        // Should not panic
        matching.compute_matching(complex);

        // Should find the same critical cells as unfiltered matching
        let critical = matching.critical_cells();
        assert!(!critical.is_empty());
    }
}
