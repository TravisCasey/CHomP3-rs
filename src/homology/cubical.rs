// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

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
//!     Complex, CubicalComplex, Cyclic, HashGrader, MorseMatching, Orthant,
//!     TopCubeGrader, TopCubicalMatching,
//! };
//!
//! // Create a cubical complex
//! let min = Orthant::from([0, 0]);
//! let max = Orthant::from([3, 3]);
//! let grader = TopCubeGrader::new(HashGrader::new(0), None);
//! let complex: CubicalComplex<Cyclic<7>, _> =
//!     CubicalComplex::new(min, max, grader);
//!
//! // Compute matching (happens at construction)
//! let matching: TopCubicalMatching<_, _> = TopCubicalMatching::new(complex);
//!
//! // Access critical cells
//! let critical_cells = matching.critical_cells();
//! ```
//!
//! With configuration:
//!
//! ```
//! use chomp3rs::{
//!     CubicalComplex, F2, HashGrader, Orthant, TopCubeGrader,
//!     TopCubicalMatching,
//! };
//!
//! // Using builder pattern
//! let min = Orthant::from([0, 0]);
//! let max = Orthant::from([3, 3]);
//! let grader = TopCubeGrader::new(HashGrader::new(0), None);
//! let complex: CubicalComplex<F2, _> = CubicalComplex::new(min, max, grader);
//!
//! let matching: TopCubicalMatching<F2, HashGrader<Orthant>> =
//!     TopCubicalMatching::builder()
//!         .max_grade(0)
//!         .max_dimension(1)
//!         .build(complex);
//! ```

use std::{collections::HashMap, fmt::Debug, iter::zip};

pub use subgrid::{GridSubdivision, OrthantMatching, Subgrid};
#[cfg(feature = "mpi")]
use tracing::{info, warn};

#[cfg(feature = "mpi")]
use crate::mpi::{Communicator, MpiExecutor, broadcast};
#[cfg(feature = "mpi")]
use crate::{CellComplex, homology::full_reduce_sequential};
use crate::{
    CellMatch, Chain, Complex, Cube, CubicalComplex, Grader, MorseMatching, Orthant, Ring,
    TopCubeGrader, dispatch, logging::ProgressTracker,
};

mod builder;
mod config;

pub use builder::TopCubicalMatchingBuilder;
pub use config::{ExecutionMode, TopCubicalMatchingConfig};

pub mod flow;
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
/// The matching is computed in two phases internally:
///
/// 1. **Critical cell identification**: All critical cells (Aces) in the
///    complex are identified
/// 2. **Gradient computation**: The boundary of each critical cell is computed
///    *in the Morse complex* (not the original cubical complex) by flowing
///    boundaries through the matching
///
/// After construction, critical cells and their boundaries can be accessed via
/// the [`MorseMatching`] trait methods and used to construct a [`CellComplex`]
/// representing the Morse complex.
///
/// # Configuration
///
/// Matching behavior can be configured to:
/// - Limit critical cells by grade or dimension (optimization for sublevel
///   sets)
/// - Adjust subgrid size for cache efficiency
/// - Enable MPI distributed computation (requires `mpi` feature)
///
/// See [`TopCubicalMatchingConfig`] and the [builder](Self::builder) for
/// details.
///
/// [`CellComplex`]: crate::CellComplex
#[derive(Debug, Clone)]
pub struct TopCubicalMatching<R, G>
where
    R: Ring,
    G: Grader<Orthant>,
{
    complex: CubicalComplex<R, TopCubeGrader<G>>,
    critical_cells: Vec<Cube>,
    boundaries: Vec<Chain<u32, R>>,
    projection: HashMap<Cube, u32>,
    config: TopCubicalMatchingConfig,
}

impl<R, G> TopCubicalMatching<R, G>
where
    R: Ring,
    G: Grader<Orthant>,
{
    /// Returns a builder for detailed configuration.
    #[must_use]
    pub fn builder() -> TopCubicalMatchingBuilder<R, G> {
        TopCubicalMatchingBuilder::new()
    }

    /// Returns a reference to the matching configuration.
    #[must_use]
    pub fn config(&self) -> &TopCubicalMatchingConfig {
        &self.config
    }
}

impl<R, G> TopCubicalMatching<R, G>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    /// Compute the matching on `complex` with default configuration.
    ///
    /// All cells are included in the matching regardless of grade or dimension.
    /// Use the [builder pattern](Self::builder) for configuration.
    #[must_use]
    pub fn new(complex: CubicalComplex<R, TopCubeGrader<G>>) -> Self {
        Self::from_config(TopCubicalMatchingConfig::default(), complex)
    }

    /// Compute the matching using MPI-based distributed computation.
    ///
    /// This constructor requires the `mpi` feature flag. It will attempt to
    /// initialize the MPI universe and use the world communicator to control
    /// processes. If MPI initialization fails, the matching will fall back to
    /// sequential computation with a warning.
    ///
    /// This configuration can also be specified using the provided [builder
    /// pattern](Self::builder).
    #[cfg(feature = "mpi")]
    #[must_use]
    pub fn new_mpi(complex: CubicalComplex<R, TopCubeGrader<G>>) -> Self {
        Self::builder().mpi().build(complex)
    }

    fn from_config(
        config: TopCubicalMatchingConfig,
        complex: CubicalComplex<R, TopCubeGrader<G>>,
    ) -> Self {
        let mut matching = Self {
            complex,
            critical_cells: Vec::new(),
            boundaries: Vec::new(),
            projection: HashMap::new(),
            config,
        };
        matching.compute();
        matching
    }

    fn compute(&mut self) {
        dispatch!(
            mpi => {
                if let Some(count) = self.config.execution.process_count() {
                    if count > 1 {
                        info!("MPI execution with {count} processes");
                        self.compute_critical_cells_distributed();
                        self.compute_boundaries();
                    } else {
                        warn!("MPI enabled with single process, using sequential");
                        self.compute_critical_cells_sequential();
                        self.compute_boundaries();
                    }
                } else {
                    if matches!(self.config.execution, ExecutionMode::Mpi { comm: None }) {
                        warn!(
                            "MPI execution mode enabled but no communicator provided; \
                             falling back to sequential. Use .mpi_with_comm(comm) to provide one."
                        );
                    }
                    self.compute_critical_cells_sequential();
                    self.compute_boundaries();
                }
            },
            _ => {
                self.compute_critical_cells_sequential();
                self.compute_boundaries();
            }
        );
    }

    /// Recursive method used to iterate through the nested `orthant_matching`
    /// to find the match of `cube`.
    ///
    /// This is only called when using the `match_cell` method, and not during
    /// the initial critical cell and boundary computation.
    fn match_helper(
        cube: Cube,
        extent: u32,
        orthant_matching: &OrthantMatching,
    ) -> CellMatch<Cube, R, Orthant> {
        match orthant_matching {
            OrthantMatching::Critical { .. } => CellMatch::Ace { cell: cube },
            OrthantMatching::Leaf { match_axis, .. } => {
                // Cellular incidence [king:queen] is the negation of the flow
                // coefficient computed by flow::compute_incidence. See the
                // sign_convention_consistency test and the sign analysis in
                // the plan documentation.
                let incidence = -flow::compute_incidence::<R>(extent, *match_axis);

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

    /// Compute critical cells sequentially with progress tracking.
    fn compute_critical_cells_sequential(&mut self) {
        let complex = &self.complex;
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
        self.store_critical_cells(critical_cells);
    }

    /// Compute boundaries for all critical cells.
    fn compute_boundaries(&mut self) {
        let complex = &self.complex;
        self.boundaries = self
            .critical_cells
            .iter()
            .map(|critical_cell| {
                let boundary_chain = complex.cell_boundary(critical_cell);
                self.lower(boundary_chain)
            })
            .collect();
    }

    /// Store critical cells and build the projection map.
    fn store_critical_cells(&mut self, critical_cells: Vec<Cube>) {
        for (index, cube) in critical_cells.iter().enumerate() {
            self.projection.insert(cube.clone(), index as u32);
        }
        self.critical_cells = critical_cells;
    }
}

#[cfg(feature = "mpi")]
impl<R, G> TopCubicalMatching<R, G>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    /// Compute critical cells using MPI distributed executor.
    ///
    /// After computation, all processes have the sorted critical cells and
    /// projection map populated.
    ///
    /// # Panics
    ///
    /// Panics if the execution mode is not MPI with a communicator.
    fn compute_critical_cells_distributed(&mut self) {
        let comm = self
            .config
            .execution
            .communicator()
            .expect("compute_critical_cells_distributed requires MPI communicator");

        let complex = &self.complex;
        let subdivision = GridSubdivision::new(
            complex.minimum().clone(),
            complex.maximum().clone(),
            self.config.subgrid_shape.clone(),
            self.config.filter_orthants.as_deref(),
        );

        let max_grade = self.config.maximum_critical_grade;
        let max_dim = self.config.maximum_critical_dimension;

        let critical_cells: Vec<Cube> = MpiExecutor::new(comm)
            .label("Critical cells")
            .log_level(tracing::Level::INFO)
            .batch_size(self.config.mpi_batch_size)
            .broadcast_results()
            .run_with_state(
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

        self.store_critical_cells(critical_cells);
    }
}

impl<R, G> MorseMatching for TopCubicalMatching<R, G>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    type Priority = Orthant;
    type Ring = R;
    type UpperCell = Cube;
    type UpperComplex = CubicalComplex<R, TopCubeGrader<G>>;

    fn upper_complex(&self) -> &Self::UpperComplex {
        &self.complex
    }

    fn into_upper_complex(self) -> Self::UpperComplex {
        self.complex
    }

    fn match_cell(&self, cube: &Cube) -> CellMatch<Cube, Self::Ring, Self::Priority> {
        let matching = {
            let mut subgrid = Subgrid::new(&self.complex, u32::MAX, u32::MAX);
            subgrid.match_subgrid(cube.base().clone(), cube.base().clone())[0]
                .2
                .clone()
        };

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

    fn critical_cells(&self) -> &[Self::UpperCell] {
        &self.critical_cells
    }

    fn project_cell(&self, cell: &Self::UpperCell) -> Option<u32> {
        self.projection.get(cell).copied()
    }

    fn include_cell(&self, cell: u32) -> Self::UpperCell {
        self.critical_cells[cell as usize].clone()
    }

    #[cfg(feature = "mpi")]
    fn full_reduce<PM>(
        &self,
        factory: impl Fn(CellComplex<Self::Ring>) -> PM,
    ) -> (Vec<PM>, CellComplex<Self::Ring>)
    where
        PM: MorseMatching<
                UpperCell = u32,
                Ring = Self::Ring,
                UpperComplex = CellComplex<Self::Ring>,
            > + serde::Serialize
            + serde::de::DeserializeOwned,
    {
        let morse_complex = self.construct_morse_complex();

        match &self.config.execution {
            ExecutionMode::Mpi { comm: Some(c) } if c.size() > 1 => {
                // Root performs the coreduction loop
                let (further_matchings, final_complex) = if c.rank() == 0 {
                    full_reduce_sequential(factory, morse_complex)
                } else {
                    (Vec::new(), CellComplex::new(vec![], vec![], vec![], vec![]))
                };

                // Broadcast results to all processes
                let further_matchings = broadcast(c, &further_matchings);
                let final_complex = broadcast(c, &final_complex);

                (further_matchings, final_complex)
            },
            ExecutionMode::Mpi { comm: Some(_) } => {
                warn!("MPI enabled with single process, using sequential full_reduce");
                full_reduce_sequential(factory, morse_complex)
            },
            ExecutionMode::Mpi { comm: None } => {
                warn!("MPI execution mode without communicator, using sequential full_reduce");
                full_reduce_sequential(factory, morse_complex)
            },
            ExecutionMode::Sequential => full_reduce_sequential(factory, morse_complex),
        }
    }

    fn lower(&self, chain: impl IntoIterator<Item = (Cube, R)>) -> Chain<u32, R> {
        flow::lower(self, chain, 0)
    }

    fn lower_capped(
        &self,
        chain: impl IntoIterator<Item = (Cube, R)>,
        min_grade: u32,
    ) -> Chain<u32, R> {
        flow::lower(self, chain, min_grade)
    }

    fn colower(&self, cochain: impl IntoIterator<Item = (Cube, R)>) -> Chain<u32, R> {
        flow::colower(self, cochain, u32::MAX)
    }

    fn colower_capped(
        &self,
        cochain: impl IntoIterator<Item = (Cube, R)>,
        max_grade: u32,
    ) -> Chain<u32, R> {
        flow::colower(self, cochain, max_grade)
    }

    fn lift(&self, chain: impl IntoIterator<Item = (u32, R)>) -> Chain<Cube, R> {
        flow::lift(self, chain, 0)
    }

    fn lift_capped(
        &self,
        chain: impl IntoIterator<Item = (u32, R)>,
        min_grade: u32,
    ) -> Chain<Cube, R> {
        flow::lift(self, chain, min_grade)
    }

    fn colift(&self, cochain: impl IntoIterator<Item = (u32, R)>) -> Chain<Cube, R> {
        flow::colift(self, cochain, u32::MAX)
    }

    fn colift_capped(
        &self,
        cochain: impl IntoIterator<Item = (u32, R)>,
        max_grade: u32,
    ) -> Chain<Cube, R> {
        flow::colift(self, cochain, max_grade)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Chain, Complex, CoreductionMatching, Cube, F2, Grader, HashGrader, Orthant, OrthantTrie,
        TopCubeGrader,
        test_complexes::{
            UniformGrader, assert_all_grade_zero, assert_betti_numbers, complex_from_orthants,
            sn_orthants, torus_orthants,
        },
    };

    #[test]
    fn full_reduce_cube_torus_complex() {
        fn run<G: UniformGrader>() {
            let complex = complex_from_orthants::<G>(&torus_orthants());
            let matching = TopCubicalMatching::new(complex);
            let morse_complex = matching.full_reduce(CoreductionMatching::new).1;
            assert_betti_numbers(&morse_complex, &[1, 2, 1]);
        }
        run::<HashGrader<Orthant>>();
        run::<OrthantTrie>();
    }

    #[test]
    fn full_reduce_sn() {
        fn run<G: UniformGrader>() {
            for n in 1..=4u32 {
                let complex = complex_from_orthants::<G>(&sn_orthants(n as usize));
                let matching = TopCubicalMatching::new(complex);
                let morse_complex = matching.full_reduce(CoreductionMatching::new).1;
                let mut expected = vec![0u32; n as usize + 1];
                expected[0] = 1;
                expected[n as usize] = 1;
                assert_betti_numbers(&morse_complex, &expected);
            }
        }
        run::<HashGrader<Orthant>>();
        run::<OrthantTrie>();
    }

    #[test]
    fn full_reduce_sn_grade_truncated() {
        fn run<G: UniformGrader>() {
            for n in 1..=4u32 {
                let complex = complex_from_orthants::<G>(&sn_orthants(n as usize));
                let matching = TopCubicalMatching::builder().max_grade(0).build(complex);
                let morse_complex = matching.full_reduce(CoreductionMatching::new).1;
                assert_all_grade_zero(&morse_complex);
                let mut expected = vec![0u32; n as usize + 1];
                expected[0] = 1;
                expected[n as usize] = 1;
                assert_betti_numbers(&morse_complex, &expected);
            }
        }
        run::<HashGrader<Orthant>>();
        run::<OrthantTrie>();
    }

    #[test]
    fn full_reduce_torus_grade_truncated() {
        fn run<G: UniformGrader>() {
            let complex = complex_from_orthants::<G>(&torus_orthants());
            let matching = TopCubicalMatching::builder().max_grade(0).build(complex);
            let morse_complex = matching.full_reduce(CoreductionMatching::new).1;
            assert_all_grade_zero(&morse_complex);
            assert_betti_numbers(&morse_complex, &[1, 2, 1]);
        }
        run::<HashGrader<Orthant>>();
        run::<OrthantTrie>();
    }

    #[test]
    fn builder_config_is_preserved() {
        // Build with a tiny complex to verify config values survive into the matching.
        let min = Orthant::from([0, 0]);
        let max = Orthant::from([1, 1]);
        let grader: HashGrader<Orthant> = HashGrader::new(0);
        let complex: CubicalComplex<F2, _> =
            CubicalComplex::new(min, max, TopCubeGrader::new(grader, None));

        let matching: TopCubicalMatching<F2, HashGrader<Orthant>> =
            TopCubicalMatchingBuilder::new()
                .max_grade(10)
                .max_dimension(5)
                .build(complex);

        assert_eq!(matching.config().maximum_critical_grade, 10);
        assert_eq!(matching.config().maximum_critical_dimension, 5);
    }

    /// Test that filtered matching finds all critical cells that are reachable
    /// via gradient flow. This reproduces a bug where subgrid filtering was
    /// too aggressive and excluded orthants containing critical cells that
    /// appear in the boundary of other critical cells.
    #[test]
    fn filtered_matching_includes_all_boundary_cells() {
        fn run<G: UniformGrader>() {
            let top_cubes = vec![Orthant::from([1, 1]), Orthant::from([2, 2])];
            let complex = complex_from_orthants::<G>(&top_cubes);
            let matching = TopCubicalMatching::<F2, G>::builder()
                .filter_orthants(top_cubes)
                .max_grade(0)
                .build(complex);
            assert!(!matching.critical_cells().is_empty());
        }
        run::<HashGrader<Orthant>>();
        run::<OrthantTrie>();
    }

    /// Test with the torus complex using filtered matching to ensure
    /// all critical boundary cells are included.
    #[test]
    fn filtered_torus_matching_works() {
        fn run<G: UniformGrader>() {
            let top_cubes = torus_orthants();
            let complex = complex_from_orthants::<G>(&top_cubes);
            let matching = TopCubicalMatching::<F2, G>::builder()
                .filter_orthants(top_cubes)
                .max_grade(0)
                .build(complex);
            assert!(!matching.critical_cells().is_empty());
        }
        run::<HashGrader<Orthant>>();
        run::<OrthantTrie>();
    }

    #[test]
    fn colift_roundtrips_correctly() {
        fn run<G: UniformGrader>() {
            let complex = complex_from_orthants::<G>(&torus_orthants());
            let matching = TopCubicalMatching::new(complex);

            for lower_cell in matching
                .critical_cells()
                .iter()
                .map(|cube| matching.project_cell(cube).unwrap())
            {
                let chain = Chain::from(lower_cell);
                let colifted = matching.colift(chain.clone());
                let colowered = matching.colower(colifted);
                assert_eq!(chain, colowered);
            }
        }
        run::<HashGrader<Orthant>>();
        run::<OrthantTrie>();
    }

    #[test]
    fn lower_lift_roundtrip() {
        fn run<G: UniformGrader>() {
            let complex = complex_from_orthants::<G>(&torus_orthants());
            let matching = TopCubicalMatching::new(complex);

            for lower_cell in matching
                .critical_cells()
                .iter()
                .map(|cube| matching.project_cell(cube).unwrap())
            {
                let chain = Chain::from(lower_cell);
                let lifted = matching.lift(chain.clone());
                let lowered = matching.lower(lifted);
                assert_eq!(chain, lowered);
            }
        }
        run::<HashGrader<Orthant>>();
        run::<OrthantTrie>();
    }

    #[test]
    fn capped_methods_with_no_filter_match_uncapped() {
        fn run<G: UniformGrader>() {
            let complex = complex_from_orthants::<G>(&torus_orthants());
            let matching = TopCubicalMatching::new(complex);

            for lower_cell in matching
                .critical_cells()
                .iter()
                .map(|cube| matching.project_cell(cube).unwrap())
            {
                let chain: Chain<u32, F2> = Chain::from(lower_cell);

                let colifted = matching.colift(chain.clone());
                let colifted_capped = matching.colift_capped(chain.clone(), u32::MAX);
                assert_eq!(colifted, colifted_capped);

                let colowered = matching.colower(colifted.clone());
                let colowered_capped = matching.colower_capped(colifted, u32::MAX);
                assert_eq!(colowered, colowered_capped);

                let lifted = matching.lift(chain.clone());
                let lifted_capped = matching.lift_capped(chain.clone(), 0);
                assert_eq!(lifted, lifted_capped);

                let lowered = matching.lower(lifted.clone());
                let lowered_capped = matching.lower_capped(lifted, 0);
                assert_eq!(lowered, lowered_capped);
            }
        }
        run::<HashGrader<Orthant>>();
        run::<OrthantTrie>();
    }

    #[test]
    #[allow(clippy::absurd_extreme_comparisons)]
    fn colift_capped_respects_grade_bound() {
        fn run<G: UniformGrader>() {
            let complex = complex_from_orthants::<G>(&torus_orthants());
            let matching = TopCubicalMatching::new(complex.clone());

            let grade_0_cell = matching
                .critical_cells()
                .iter()
                .find(|cube| complex.grade(*cube) == 0)
                .expect("should have grade 0 critical cells");

            let lower_cell = matching.project_cell(grade_0_cell).unwrap();
            let chain: Chain<u32, F2> = Chain::from(lower_cell);

            let colifted_capped = matching.colift_capped(chain, 0);
            for (cube, _) in colifted_capped {
                assert!(
                    complex.grade(&cube) <= 0,
                    "colift_capped(_, 0) produced grade {} cell",
                    complex.grade(&cube)
                );
            }
        }
        run::<HashGrader<Orthant>>();
        run::<OrthantTrie>();
    }

    fn assert_is_cocycle_in_sublevel<C: Complex>(
        complex: &C,
        cochain: &Chain<C::Cell, C::Ring>,
        max_grade: u32,
        context: &str,
    ) {
        let coboundary: Chain<C::Cell, C::Ring> = complex
            .coboundary(cochain)
            .into_iter()
            .filter(|(cell, _)| complex.grade(cell) <= max_grade)
            .collect();
        assert!(
            coboundary.is_empty(),
            "{}: cochain is not a cocycle in sublevel (grade <= {}), coboundary has {} nonzero \
             terms",
            context,
            max_grade,
            coboundary.len()
        );
    }

    #[test]
    fn colift_capped_preserves_cocycle_in_sublevel() {
        fn run<G: UniformGrader>() {
            let complex = complex_from_orthants::<G>(&torus_orthants());
            let matching = TopCubicalMatching::new(complex.clone());

            let morse_complex = matching.construct_morse_complex();

            for lower_cell in morse_complex.iter() {
                if morse_complex.cell_dimension(&lower_cell) != 1 {
                    continue;
                }
                if morse_complex.grade(&lower_cell) != 0 {
                    continue;
                }

                let cocycle: Chain<u32, F2> = Chain::from(lower_cell);
                let colifted_capped = matching.colift_capped(cocycle, 0);

                assert_is_cocycle_in_sublevel(
                    &complex,
                    &colifted_capped,
                    0,
                    &format!("colift_capped of critical 1-cell {lower_cell}"),
                );
            }
        }
        run::<HashGrader<Orthant>>();
        run::<OrthantTrie>();
    }

    #[test]
    fn flow_operations_are_chain_maps() {
        fn run<G: UniformGrader>() {
            let complex = complex_from_orthants::<G>(&torus_orthants());
            let matching = TopCubicalMatching::new(complex.clone());

            let morse_complex = matching.construct_morse_complex();

            // Test lower: parent -> Morse (boundary direction)
            for cube in complex.iter().take(50) {
                let chain: Chain<Cube, F2> = Chain::from(cube.clone());
                let left = matching.lower(complex.boundary(&chain));
                let right = morse_complex.boundary(&matching.lower(chain));
                assert_eq!(left, right, "lower is not a chain map for {cube:?}");
            }

            // Test lift: Morse -> parent (boundary direction)
            for lower_cell in morse_complex.iter() {
                let chain: Chain<u32, F2> = Chain::from(lower_cell);
                let left = complex.boundary(&matching.lift(chain.clone()));
                let right = matching.lift(morse_complex.boundary(&chain));
                assert_eq!(
                    left, right,
                    "lift is not a chain map for Morse cell {lower_cell}"
                );
            }

            // Test colower: parent -> Morse (coboundary direction)
            for cube in complex.iter().take(50) {
                let cochain: Chain<Cube, F2> = Chain::from(cube.clone());
                let left = matching.colower(complex.coboundary(&cochain));
                let right = morse_complex.coboundary(&matching.colower(cochain));
                assert_eq!(left, right, "colower is not a cochain map for {cube:?}");
            }

            // Test colift: Morse -> parent (coboundary direction)
            for lower_cell in morse_complex.iter() {
                let cochain: Chain<u32, F2> = Chain::from(lower_cell);
                let left = complex.coboundary(&matching.colift(cochain.clone()));
                let right = matching.colift(morse_complex.coboundary(&cochain));
                assert_eq!(
                    left, right,
                    "colift is not a cochain map for Morse cell {lower_cell}"
                );
            }
        }
        run::<HashGrader<Orthant>>();
        run::<OrthantTrie>();
    }

    /// Verify that `match_helper`'s incidence equals the negation of
    /// `flow::compute_incidence` for leaf matchings across representative
    /// (extent, `match_axis`) pairs. Uses `Cyclic<3>` where sign matters.
    ///
    /// This also serves as a non-F2 sign correctness test: it verifies
    /// `match_helper`'s incidence against the boundary formula in
    /// `CubicalComplex::cell_boundary`, since `compute_incidence` was
    /// independently verified against that formula.
    #[test]
    fn sign_convention_consistency() {
        type R = crate::Cyclic<3>;

        for match_axis in 0u32..5 {
            for extent in 0u32..32 {
                let upper = extent | (1 << match_axis);
                let lower = extent & !(1 << match_axis);
                if upper == lower {
                    continue;
                }

                let leaf = OrthantMatching::construct_leaf(upper, lower, 0);

                let base = Orthant::from([1i16; 5]);
                let dual: Orthant = (0..5)
                    .map(|axis| i16::from(extent & (1 << axis) != 0))
                    .collect();
                let cube = Cube::new(base, dual);

                let cell_match =
                    TopCubicalMatching::<R, HashGrader<Orthant>>::match_helper(cube, extent, &leaf);
                let match_incidence = match &cell_match {
                    CellMatch::King { incidence, .. } | CellMatch::Queen { incidence, .. } => {
                        *incidence
                    },
                    CellMatch::Ace { .. } => panic!("leaf should not produce Ace"),
                };

                let flow_incidence: R = super::flow::compute_incidence(extent, match_axis);

                assert_eq!(
                    match_incidence, -flow_incidence,
                    "sign mismatch: extent={extent:#07b}, match_axis={match_axis}"
                );
            }
        }
    }
}
