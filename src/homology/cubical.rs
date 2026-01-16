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

#[cfg(feature = "chompi")]
use chompi::{Communicator, MpiExecutor, SimpleCommunicator};
pub use subgrid::{GridSubdivision, OrthantMatching, Subgrid};
#[cfg(feature = "chompi")]
use tracing::{info, warn};
pub use wavefront::{FlowCell, Wavefront, WavefrontConfig};

use crate::{
    CellMatch, ComplexLike, Cube, CubicalComplex, Grader, HashMapModule, ModuleLike, MorseMatching,
    Orthant, RingLike, TopCubeGrader, dispatch, logging::ProgressTracker,
};

pub mod subgrid;
pub mod wavefront;

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
/// - Enable MPI distributed computation (requires `chompi` feature)
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

    /// Creates a matching with MPI-based distributed computation enabled.
    ///
    /// This constructor requires the `chompi` feature flag. It will attempt to
    /// initialize the MPI universe and use the world communicator to control
    /// processes. If MPI initialization fails, the matching will fall back to
    /// sequential computation with a warning.
    ///
    /// This configuration can also be specified using the provided [builder
    /// pattern](Self::builder).
    #[cfg(feature = "chompi")]
    #[must_use]
    pub fn new_mpi() -> Self {
        Self::builder().mpi().build()
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

    /// Compute the orthant matching for a single orthant.
    fn match_orthant(&self, orthant: &Orthant) -> OrthantMatching {
        let mut subgrid = Subgrid::new(self.complex.as_ref().unwrap(), u32::MAX, u32::MAX);
        subgrid.match_subgrid(orthant.clone(), orthant.clone())[0]
            .2
            .clone()
    }

    /// Compute critical cells sequentially with progress tracking.
    fn compute_critical_cells_sequential(&mut self) {
        let complex = self.complex.as_ref().unwrap();
        let subdivision = GridSubdivision::new(
            complex.minimum().clone(),
            complex.maximum().clone(),
            self.config.subgrid_shape.clone(),
            self.config.filter_orthants.as_deref(),
        );

        let max_grade = self.config.maximum_critical_grade;
        let max_dim = self.config.maximum_critical_dimension;
        let total = subdivision.len();

        let mut progress = ProgressTracker::new("Critical cells", total);

        let critical_cells: Vec<Cube> = subdivision
            .into_iter()
            .flat_map(|(minimum_orthant, maximum_orthant)| {
                let mut subgrid = Subgrid::new(complex, max_grade, max_dim);
                let base_critical_orthant = subgrid.match_subgrid(minimum_orthant, maximum_orthant);

                let cells: Vec<Cube> = base_critical_orthant
                    .into_iter()
                    .flat_map(|(_, critical, _)| critical)
                    .collect();

                progress.increment();
                cells
            })
            .collect();

        progress.finish();

        // Build projection map
        for (index, cube) in critical_cells.iter().enumerate() {
            self.projection.insert(cube.clone(), index as u32);
        }
        self.critical_cells = critical_cells;
    }

    /// Compute boundaries for all critical cells.
    fn compute_boundaries(&mut self) {
        let complex = self.complex.as_ref().unwrap();
        self.boundaries = self
            .critical_cells
            .iter()
            .map(|critical_cell| {
                let boundary_chain = complex.cell_boundary(critical_cell);
                self.lower(boundary_chain)
            })
            .collect();
    }
}

#[cfg(feature = "chompi")]
impl<UM, G, LM> TopCubicalMatching<UM, G, LM>
where
    UM: ModuleLike<Cell = Cube>,
    LM: ModuleLike<Cell = u32, Ring = UM::Ring>,
    G: Grader<Orthant> + Clone,
{
    /// Compute critical cells using MPI distributed executor.
    fn compute_critical_cells_distributed(&mut self, executor: &mut MpiExecutor) {
        let complex = self.complex.as_ref().unwrap();
        let subdivision = GridSubdivision::new(
            complex.minimum().clone(),
            complex.maximum().clone(),
            self.config.subgrid_shape.clone(),
            self.config.filter_orthants.as_deref(),
        );

        let max_grade = self.config.maximum_critical_grade;
        let max_dim = self.config.maximum_critical_dimension;

        let critical_cells: Vec<Cube> = executor.execute_with_state(
            "Critical cells",
            subdivision.into_iter(),
            || Subgrid::new(complex, max_grade, max_dim),
            |subgrid, (minimum_orthant, maximum_orthant)| {
                subgrid
                    .match_subgrid(minimum_orthant, maximum_orthant)
                    .into_iter()
                    .flat_map(|(_, critical, _)| critical)
                    .collect::<Vec<Cube>>()
            },
        );

        // Build projection map
        for (index, cube) in critical_cells.iter().enumerate() {
            self.projection.insert(cube.clone(), index as u32);
        }
        self.critical_cells = critical_cells;
    }
}

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

        dispatch!(
            chompi => {
                if let ExecutionMode::Mpi { ref comm } = self.config.execution {
                    let executor = if let Some(c) = comm.as_ref() {
                        Some(MpiExecutor::new(c))
                    } else {
                        MpiExecutor::from_world()
                    };

                    if let Some(mut executor) = executor {
                        info!("Initialized MPI executor with {} processes", executor.size());
                        self.compute_critical_cells_distributed(&mut executor);
                        self.compute_boundaries();
                        return;
                    }
                    warn!("MPI unavailable, falling back to sequential");
                }
                self.compute_critical_cells_sequential();
            },
            _ => {
                self.compute_critical_cells_sequential();
            }
        );

        self.compute_boundaries();
    }

    fn match_cell(&self, cube: &Cube) -> CellMatch<Cube, Self::Ring, Self::Priority> {
        assert!(
            self.complex.is_some(),
            "matching not yet computed; call compute_matching first"
        );
        let matching = self.match_orthant(cube.base());

        let mut extent = 0u32;
        for (axis, (base_coord, dual_coord)) in
            zip(cube.base().iter(), cube.dual().iter()).enumerate()
        {
            if base_coord == dual_coord {
                extent += 1 << axis;
            }
        }

        Self::match_helper(cube.clone(), extent, &matching)
    }

    fn get_upper_complex(&self) -> Option<&Self::UpperComplex> {
        self.complex.as_ref()
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
        let mut coboundaries = vec![LM::new(); self.boundaries.len()];
        for (cell, boundary) in self.boundaries.iter().enumerate() {
            for (boundary_cell, coef) in boundary.iter() {
                coboundaries[*boundary_cell as usize].insert_or_add(cell as u32, coef.clone());
            }
        }
        (self.boundaries.clone(), coboundaries)
    }

    fn lower(&self, chain: Self::UpperModule) -> Self::LowerModule {
        let complex = self.complex.as_ref().unwrap();
        let mut wavefront = Wavefront::new(
            WavefrontConfig::Boundary,
            complex.minimum().clone(),
            complex.maximum().clone(),
        );
        let mut result = LM::new();

        wavefront.seed(chain);

        while let Some((orthant, orthant_chain)) = wavefront.pop_next() {
            let matching = self.match_orthant(&orthant);

            wavefront.flow_orthant(&orthant, orthant_chain, &matching, |cell| {
                if let FlowCell::Ace { extent, coef } = cell {
                    let lower_cell = self.projection[&wavefront::extent_to_cube(&orthant, extent)];
                    result.insert_or_add(lower_cell, coef);
                }
            });
        }

        result
    }

    fn colower(&self, chain: Self::UpperModule) -> Self::LowerModule {
        let complex = self.complex.as_ref().unwrap();
        let mut wavefront = Wavefront::new(
            WavefrontConfig::Coboundary,
            complex.minimum().clone(),
            complex.maximum().clone(),
        );
        let mut result = LM::new();

        wavefront.seed(chain);

        while let Some((orthant, orthant_chain)) = wavefront.pop_next() {
            let matching = self.match_orthant(&orthant);

            wavefront.flow_orthant(&orthant, orthant_chain, &matching, |cell| {
                if let FlowCell::Ace { extent, coef } = cell {
                    let lower_cell = self.projection[&wavefront::extent_to_cube(&orthant, extent)];
                    result.insert_or_add(lower_cell, coef);
                }
            });
        }

        result
    }

    fn lift(&self, chain: Self::LowerModule) -> Self::UpperModule {
        let complex = self.complex.as_ref().unwrap();
        let mut wavefront = Wavefront::new(
            WavefrontConfig::Boundary,
            complex.minimum().clone(),
            complex.maximum().clone(),
        );

        let mut result = UM::new();
        for (lower_cell, coef) in chain {
            result.insert_or_add(self.critical_cells[lower_cell as usize].clone(), coef);
        }
        wavefront.seed(complex.boundary(&result));

        while let Some((orthant, orthant_chain)) = wavefront.pop_next() {
            let matching = self.match_orthant(&orthant);

            wavefront.flow_orthant(&orthant, orthant_chain, &matching, |cell| {
                if let FlowCell::Queen { king_extent, coef } = cell {
                    let king_cube = wavefront::extent_to_cube(&orthant, king_extent);
                    result.insert_or_add(king_cube, coef);
                }
            });
        }

        result
    }

    fn colift(&self, chain: Self::LowerModule) -> Self::UpperModule {
        let complex = self.complex.as_ref().unwrap();
        let mut wavefront = Wavefront::new(
            WavefrontConfig::Coboundary,
            complex.minimum().clone(),
            complex.maximum().clone(),
        );

        let mut result = UM::new();
        for (lower_cell, coef) in chain {
            result.insert_or_add(self.critical_cells[lower_cell as usize].clone(), coef);
        }
        wavefront.seed(complex.coboundary(&result));

        while let Some((orthant, orthant_chain)) = wavefront.pop_next() {
            let matching = self.match_orthant(&orthant);

            wavefront.flow_orthant(&orthant, orthant_chain, &matching, |cell| {
                if let FlowCell::King { queen_extent, coef } = cell {
                    let queen_cube = wavefront::extent_to_cube(&orthant, queen_extent);
                    result.insert_or_add(queen_cube, coef);
                }
            });
        }

        result
    }
}

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
    /// Requires the `chompi` feature. If `comm` is `None`, the matching will
    /// attempt to initialize the MPI universe and use the world communicator.
    #[cfg(feature = "chompi")]
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
            #[cfg(feature = "chompi")]
            Self::Mpi { comm } => Self::Mpi {
                comm: comm.as_ref().map(Communicator::duplicate),
            },
        }
    }
}

impl Debug for ExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sequential => write!(f, "Sequential"),
            #[cfg(feature = "chompi")]
            Self::Mpi { comm } => {
                let comm_str = comm
                    .as_ref()
                    .map_or("None".to_string(), |c| format!("Some({})", c.get_name()));
                write!(f, "Mpi {{ comm: {comm_str} }}")
            },
        }
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
}

impl Default for TopCubicalMatchingConfig {
    fn default() -> Self {
        Self {
            maximum_critical_grade: u32::MAX,
            maximum_critical_dimension: u32::MAX,
            subgrid_shape: None,
            filter_orthants: None,
            execution: ExecutionMode::default(),
        }
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

    /// Use MPI to distribute the computation across multiple processes.
    ///
    /// The matching will attempt to initialize the MPI universe and use the
    /// world communicator. For a custom communicator, use
    /// [`mpi_with_comm`](Self::mpi_with_comm) instead.
    #[cfg(feature = "chompi")]
    #[must_use]
    pub fn mpi(mut self) -> Self {
        self.config.execution = ExecutionMode::Mpi { comm: None };
        self
    }

    /// Use MPI with a specific communicator for distributed execution.
    ///
    /// The communicator is duplicated, so the original can be used elsewhere.
    #[cfg(feature = "chompi")]
    #[must_use]
    pub fn mpi_with_comm(mut self, comm: &dyn Communicator) -> Self {
        self.config.execution = ExecutionMode::Mpi {
            comm: Some(comm.duplicate()),
        };
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

    #[test]
    fn colift_roundtrips_correctly() {
        let complex = top_cube_torus_hashmap();
        let mut matching: TopCubicalMatching<
            HashMapModule<Cube, Cyclic<2>>,
            HashMapGrader<Orthant>,
        > = TopCubicalMatching::default();
        matching.compute_matching(complex);

        for lower_cell in matching
            .critical_cells()
            .into_iter()
            .map(|cube| matching.project_cell(cube).unwrap())
        {
            let chain = HashMapModule::singleton(lower_cell, Cyclic::one());
            let colifted = matching.colift(chain.clone());
            let colowered = matching.colower(colifted);
            assert_eq!(chain, colowered,);
        }
    }

    #[test]
    fn lower_lift_roundtrip() {
        let complex = top_cube_torus_hashmap();
        let mut matching: TopCubicalMatching<
            HashMapModule<Cube, Cyclic<2>>,
            HashMapGrader<Orthant>,
        > = TopCubicalMatching::default();
        matching.compute_matching(complex);

        for lower_cell in matching
            .critical_cells()
            .into_iter()
            .map(|cube| matching.project_cell(cube).unwrap())
        {
            let chain = HashMapModule::singleton(lower_cell, Cyclic::one());
            let lifted = matching.lift(chain.clone());
            let lowered = matching.lower(lifted);
            assert_eq!(chain, lowered,);
        }
    }
}
