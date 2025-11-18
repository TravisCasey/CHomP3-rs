// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::collections::HashMap;
use std::iter::zip;
use std::marker::PhantomData;
#[cfg(feature = "parallel")]
use std::rc::Rc;

#[cfg(feature = "parallel")]
use mpi::traits::Communicator;
#[cfg(feature = "parallel")]
use serde::de::DeserializeOwned;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "parallel")]
use tracing::{error, info, warn};

#[cfg(feature = "parallel")]
use super::parallel::{TopCubicalParallelChild, TopCubicalParallelHandler, TopCubicalParallelRoot};
#[cfg(feature = "parallel")]
use crate::CHomPMultiprocessingError;
use crate::homology::cubical::gradient::TopCubicalGradientPropagator;
use crate::homology::cubical::orthant_matching::OrthantMatching;
use crate::homology::cubical::subgrid::{GridSubdivision, Subgrid};
use crate::{
    Cube, CubicalComplex, Grader, HashMapModule, MatchResult, ModuleLike, MorseMatching, Orthant,
    RingLike, TopCubeGrader,
};

/// A type implementing an acyclic partial matching (that is, satisfying
/// [`MorseMatching`]) on a [`CubicalComplex`] with grading determined by
/// top-dimensional cubes (implemented via the [`TopCubeGrader`] type.)
///
/// Each cube in the complex attempts to match with other cubes along each axis
/// within its orthant. If the cube extends along the axis, it can attempt to
/// match to the equivalent cube which does not extend along the axis, and vice
/// versa. For instance, the following cubes may match together as they belong
/// to the same base orthant but have different extent along one axis.
/// ```
/// use chomp3rs::{Cube, Orthant};
///
/// Cube::from_extent(Orthant::from([0, 0, 0]), &[true, false, true]);
/// Cube::from_extent(Orthant::from([0, 0, 0]), &[true, false, false]);
/// ```
///
/// This approach, applied to each cell individually with an additional step for
/// checking that the proposed match is available, is detailed in Harker,
/// Mischaikow, Spendlove, *Morse Theoretic Templates for High Dimensional
/// Homology Computation*. However, very large cubical complexes (which can be
/// reasonably constructed and graded via [`TopCubeGrader`]) contain far too
/// many cells to process each individually and to either recur on or store
/// previous matches to check availability.
///
/// The algorithm implemented by this type instead operates on orthants and
/// matches entire suborthants (subsets of cells in an orthant with the
/// structure of orthants of lesser ambient dimension) at a time. Suborthants
/// (and thus, cubes) are processed efficiently so that each cube requires at
/// most one top cube grading query (whereas naive approaches may require up to
/// 2^d queries for a single cube, as the grade of the cube is the minimum of
/// the grades of its surrounding top cubes).
///
/// The grades of top-dimensional cubes are queried and re-queried very commonly
/// by cubes in surrounding orthants. While an efficient grading scheme may be
/// able to rapidly determine the grading of top-cube, the sheer number of
/// queries causes grading (typically with a hash map or tree) to consume most
/// of the running time. By processing nearby orthants together and caching
/// grading results (when there is a reasonable number of surrounding top-cubes,
/// whose grades can be stored contiguously) we further improve the efficiency.
#[derive(Clone)]
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
    #[cfg(feature = "parallel")]
    comm: Option<Rc<dyn Communicator>>,
}

impl<UM, G> TopCubicalMatching<UM, G>
where
    UM: ModuleLike<Cell = Cube>,
    G: Grader<Orthant>,
{
    /// Initialize a matching with options to set the maximum grade and maximum
    /// dimension of cells considered in the matching.
    ///
    /// This behavior can be disabled by setting the corresponding option to
    /// `None`; the `Default` implementation for this type uses values of `None`
    /// for both.
    ///
    /// However, this can be a key optimization when computing (co)homology only
    /// up to a certain dimension or when the complex is embedded in a larger
    /// complex of greater grade, but you only need the (co)homology of the
    /// embedded complex. Note, however, that setting the maximum dimension to
    /// `d` will only guarantee accuracy of homology up to dimensions `d - 1`.
    /// As matchings are computed independently of grades, there is no
    /// corresponding issue for the maximum grade option.
    ///
    /// No matching is performed until [`TopCubicalMatching::compute_matching`]
    /// has been called and thus most methods from the [`MorseMatching`] trait
    /// implementation will panic if called before.
    pub fn new(
        maximum_critical_grade: Option<u32>,
        maximum_critical_dimension: Option<u32>,
        subgrid_sizes: Option<Vec<i16>>,
    ) -> Self {
        Self::from_config(TopCubicalMatchingConfig::new(
            maximum_critical_grade,
            maximum_critical_dimension,
            subgrid_sizes,
            false,
        ))
    }

    /// Builder pattern for setting various options for construction of this
    /// type. See [`TopCubicalMatchingBuilder`] for details.
    pub fn builder() -> TopCubicalMatchingBuilder<UM, G> {
        TopCubicalMatchingBuilder::new()
    }

    /// Identical to [`TopCubicalMatching::new`] using MPI to divide work among
    /// child processes. It is presumed that the MPI Universe has not been
    /// initialized yet; this will attempt to initialize it and use the world
    /// communicator to control processes.
    #[cfg(feature = "parallel")]
    pub fn new_parallel(
        maximum_critical_grade: Option<u32>,
        maximum_critical_dimension: Option<u32>,
        subgrid_sizes: Option<Vec<i16>>,
    ) -> Self {
        Self::from_config(TopCubicalMatchingConfig::new(
            maximum_critical_grade,
            maximum_critical_dimension,
            subgrid_sizes,
            true,
        ))
    }

    #[cfg(not(feature = "parallel"))]
    fn from_config(config: TopCubicalMatchingConfig) -> Self {
        Self {
            complex: None,
            critical_cells: Vec::new(),
            boundaries: Vec::new(),
            projection: HashMap::new(),
            config,
        }
    }

    #[cfg(feature = "parallel")]
    fn from_config(config: TopCubicalMatchingConfig) -> Self {
        Self {
            complex: None,
            critical_cells: Vec::new(),
            boundaries: Vec::new(),
            projection: HashMap::new(),
            config,
            comm: None,
        }
    }

    /// Identical to [`TopCubicalMatching::new_parallel`] except the
    /// communicator of processes is provided. This is useful if the MPI
    /// Universe has been initialized prior to construction of this type.
    #[cfg(feature = "parallel")]
    pub fn new_parallel_with_communicator<C: Communicator>(
        maximum_critical_grade: Option<u32>,
        maximum_critical_dimension: Option<u32>,
        subgrid_sizes: Option<Vec<i16>>,
        communicator: &C,
    ) -> Self {
        Self {
            complex: None,
            critical_cells: Vec::new(),
            boundaries: Vec::new(),
            projection: HashMap::new(),
            config: TopCubicalMatchingConfig::new(
                maximum_critical_grade,
                maximum_critical_dimension,
                subgrid_sizes,
                true,
            ),
            comm: Some(Rc::new(communicator.duplicate())),
        }
    }
}

impl<UM, G> Default for TopCubicalMatching<UM, G>
where
    UM: ModuleLike<Cell = Cube>,
    G: Grader<Orthant>,
{
    fn default() -> Self {
        Self::from_config(TopCubicalMatchingConfig::default())
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
    ) -> MatchResult<Cube, UM::Ring, Orthant> {
        match orthant_matching {
            OrthantMatching::Critical { .. } => MatchResult::Ace { cell: cube },
            OrthantMatching::Leaf { match_axis, .. } => {
                let incidence = if ((extent % (1 << match_axis)).count_ones()).is_multiple_of(2) {
                    -UM::Ring::one()
                } else {
                    UM::Ring::one()
                };
                let base_orthant = cube.base().clone();
                let mut matched_cell = cube.clone();
                if cube.base_coord(*match_axis as usize) == cube.dual_coord(*match_axis as usize) {
                    matched_cell.dual_mut()[*match_axis as usize] -= 1;
                    MatchResult::King {
                        cell: cube,
                        queen: matched_cell,
                        incidence,
                        priority: base_orthant,
                    }
                } else {
                    matched_cell.dual_mut()[*match_axis as usize] += 1;
                    MatchResult::Queen {
                        cell: cube,
                        king: matched_cell,
                        incidence,
                        priority: base_orthant,
                    }
                }
            }
            OrthantMatching::Branch {
                upper_extent,
                prime_extent,
                suborthant_matchings,
            } => {
                let mut differing_axes = extent & !prime_extent;
                let mut suborthant_index = 0usize;
                let mut axis = 0;
                while axis < cube.base().ambient_dimension() {
                    if (upper_extent & (1 << axis)) != 0 {
                        if differing_axes == 0 {
                            break;
                        }
                        if prime_extent & (1 << axis) == 0 {
                            suborthant_index += 1;
                        }
                    }
                    axis += 1;
                    differing_axes >>= 1;
                }
                Self::match_helper(cube, extent, &suborthant_matchings[suborthant_index])
            }
        }
    }

    fn compute_matching_sequential(&mut self) {
        let subdivision = GridSubdivision::new(
            self.complex.as_ref().unwrap().minimum().clone(),
            self.complex.as_ref().unwrap().maximum().clone(),
            self.config.subgrid_shape.clone(),
            None, // TODO
        );

        let mut subgrid = Subgrid::new(
            self.complex.as_ref().unwrap(),
            self.config.maximum_critical_grade,
            self.config.maximum_critical_dimension,
        );

        for (minimum_orthant, maximum_orthant) in subdivision.nonempty_subgrids() {
            let base_critical_orthant =
                subgrid.match_subgrid(minimum_orthant.clone(), maximum_orthant.clone());
            for (_, critical_cells, _) in base_critical_orthant.iter() {
                for cube in critical_cells {
                    self.projection
                        .insert(cube.clone(), self.critical_cells.len() as u32);
                    self.critical_cells.push(cube.clone());
                }
            }
        }

        let mut gradient_computer =
            TopCubicalGradientPropagator::new(self.complex.as_ref().unwrap());
        let mut boundaries: Vec<LM> = Vec::new();
        for critical_cell in self.critical_cells.iter() {
            let upper_gradient_chain = gradient_computer.compute_gradient(critical_cell);
            boundaries.push(
                upper_gradient_chain
                    .into_iter()
                    .map(|(cube, coef)| (self.projection[&cube], coef))
                    .collect(),
            );
        }
        self.boundaries = boundaries;
    }
}

#[cfg(feature = "parallel")]
impl<UM, G, LM, R> TopCubicalMatching<UM, G, LM>
where
    UM: ModuleLike<Cell = Cube, Ring = R>,
    LM: ModuleLike<Cell = u32, Ring = UM::Ring>,
    G: Grader<Orthant> + Clone,
    R: Serialize + DeserializeOwned + RingLike,
{
    fn compute_matching_parallel(&mut self) -> Result<(), CHomPMultiprocessingError> {
        info!("Beginning parallel matching...");
        let mut handler: Box<dyn TopCubicalParallelHandler<LM>> =
            match self.comm.as_ref().unwrap().rank() {
                0 => Box::new(TopCubicalParallelRoot::new(
                    self.comm.clone().unwrap(),
                    self.complex.as_ref().unwrap(),
                    &self.config,
                )),
                _ => Box::new(TopCubicalParallelChild::new(
                    self.comm.clone().unwrap(),
                    self.complex.as_ref().unwrap(),
                    &self.config,
                )),
            };

        handler.verify_configuration(&self.config)?;
        handler.compute_critical_cells();
        handler.compute_gradient()?;
        (self.critical_cells, self.projection, self.boundaries) = handler.finalize();

        Ok(())
    }
}

#[cfg(feature = "parallel")]
impl<UM, G, LM, R> MorseMatching for TopCubicalMatching<UM, G, LM>
where
    UM: ModuleLike<Cell = Cube, Ring = R>,
    LM: ModuleLike<Cell = u32, Ring = R>,
    G: Grader<Orthant> + Clone,
    R: DeserializeOwned + Serialize + RingLike,
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

        if self.config.parallel && self.comm.is_none() {
            match mpi::initialize() {
                Some(universe) => {
                    self.comm = Some(Rc::new(universe.world()));
                    let comm_size = self.comm.as_ref().unwrap().size();
                    info!("Initialized a new MPI universe with {comm_size} processes");
                }
                None => {
                    warn!(
                        "MPI universe cannot be initialized. If it has been initialized before \
                        attempting to compute matching, the TopCubicalMatching instance must \
                        be passed an MPI Communicator at construction. Switching to \
                        non-parallel computation."
                    );
                    self.config.parallel = false;
                }
            }
        }
        if self.config.parallel {
            if let Err(e) = self.compute_matching_parallel() {
                error!(
                    "The following error occurred during multiprocessing in TopCubicalMatching \
                    computation:\n{e}"
                );
            };
        } else {
            self.compute_matching_sequential();
        }
    }

    fn match_cell(&self, cube: &Cube) -> MatchResult<Cube, Self::Ring, Self::Priority> {
        if self.complex.is_none() {
            panic!("MorseMatching method called prior to compute_matching");
        }

        let mut subgrid = Subgrid::new(self.get_upper_complex().unwrap(), u32::MAX, u32::MAX);
        let orthant_matching =
            &subgrid.match_subgrid(cube.base().clone(), cube.base().clone())[0].2;

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
        let mut coboundaries = vec![Self::LowerModule::new(); self.boundaries.len()];
        for (cell, boundary) in self.boundaries.iter().enumerate() {
            for (boundary_cell, coef) in boundary.iter() {
                coboundaries[*boundary_cell as usize].insert_or_add(cell as u32, coef.clone());
            }
        }

        (self.boundaries.clone(), coboundaries)
    }
}

#[cfg(not(feature = "parallel"))]
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

        self.compute_matching_sequential();
    }

    fn match_cell(&self, cube: &Cube) -> MatchResult<Cube, Self::Ring, Self::Priority> {
        if self.complex.is_none() {
            panic!("MorseMatching method called prior to compute_matching");
        }

        let mut subgrid = Subgrid::new(self.get_upper_complex().unwrap(), u32::MAX, u32::MAX);
        let orthant_matching =
            &subgrid.match_subgrid(cube.base().clone(), cube.base().clone())[0].2;

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
        let mut coboundaries = vec![Self::LowerModule::new(); self.boundaries.len()];
        for (cell, boundary) in self.boundaries.iter().enumerate() {
            for (boundary_cell, coef) in boundary.iter() {
                coboundaries[*boundary_cell as usize].insert_or_add(cell as u32, coef.clone());
            }
        }

        (self.boundaries.clone(), coboundaries)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub(super) struct TopCubicalMatchingConfig {
    pub(super) maximum_critical_grade: u32,
    pub(super) maximum_critical_dimension: u32,
    pub(super) subgrid_shape: Option<Vec<i16>>,
    pub(super) parallel: bool,
}

impl TopCubicalMatchingConfig {
    fn new(
        maximum_critical_grade: Option<u32>,
        maximum_critical_dimension: Option<u32>,
        subgrid_shape: Option<Vec<i16>>,
        parallel: bool,
    ) -> Self {
        Self {
            maximum_critical_grade: maximum_critical_grade.unwrap_or(u32::MAX),
            maximum_critical_dimension: maximum_critical_dimension.unwrap_or(u32::MAX),
            subgrid_shape,
            parallel,
        }
    }
}

impl Default for TopCubicalMatchingConfig {
    fn default() -> Self {
        Self::new(None, None, None, false)
    }
}

/// Builder pattern helper to build a [`TopCubicalMatching`] object with
/// various options.
#[derive(Default)]
pub struct TopCubicalMatchingBuilder<UM, G>
where
    UM: ModuleLike<Cell = Cube>,
    G: Grader<Orthant>,
{
    config: TopCubicalMatchingConfig,
    module_phantom: PhantomData<UM>,
    grader_phantom: PhantomData<G>,
}

impl<UM, G> TopCubicalMatchingBuilder<UM, G>
where
    UM: ModuleLike<Cell = Cube>,
    G: Grader<Orthant>,
{
    /// Create a builder, currently with all default options.
    pub fn new() -> Self {
        Self {
            config: TopCubicalMatchingConfig::default(),
            module_phantom: PhantomData,
            grader_phantom: PhantomData,
        }
    }

    /// Consume the builder and return the configured [`TopCubicalMatching`]
    /// object.
    pub fn build(self) -> TopCubicalMatching<UM, G> {
        TopCubicalMatching::from_config(self.config)
    }

    /// Set the maximum grade of critical cells found by the matching.
    pub fn max_grade(mut self, max_grade: u32) -> Self {
        self.config.maximum_critical_grade = max_grade;
        self
    }

    /// Set the maximum dimension of critical cells found by the matching.
    pub fn max_dimension(mut self, max_dimension: u32) -> Self {
        self.config.maximum_critical_dimension = max_dimension;
        self
    }

    /// Configure the shape of the subgrids of orthants matched together.
    /// Larger subgrids cache results more efficiently, but too large of
    /// subgrids may exceed contiguous cache capabilities or contain too many
    /// empty orthants to be efficient.
    pub fn subgrid_shape(mut self, subgrid_shape: Vec<i16>) -> Self {
        self.config.subgrid_shape = Some(subgrid_shape);
        self
    }

    /// Denote that the matching should use MPI to split computations between a
    /// root and the remaining child worker processes.
    #[cfg(feature = "parallel")]
    pub fn parallel(mut self) -> Self {
        self.config.parallel = true;
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
        for cell in morse_complex.cell_iter() {
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
    fn individual_matches_cube_torus_complex() {
        let complex = top_cube_torus_hashmap();
        let mut matching = TopCubicalMatching::default();
        matching.compute_matching(complex);

        let cube = Cube::vertex(Orthant::from([0, 0, 0]));
        let result = MatchResult::Queen {
            cell: cube.clone(),
            king: Cube::from_extent(Orthant::from([0, 0, 0]), &[true, false, false]),
            incidence: Cyclic::one(),
            priority: Orthant::from([0, 0, 0]),
        };
        assert_eq!(matching.match_cell(&cube), result);

        let cube = Cube::from_extent(Orthant::from([1, 1, 1]), &[false, true, true]);
        let result = MatchResult::King {
            cell: cube.clone(),
            queen: Cube::from_extent(Orthant::from([1, 1, 1]), &[false, false, true]),
            incidence: Cyclic::one(),
            priority: Orthant::from([1, 1, 1]),
        };
        assert_eq!(matching.match_cell(&cube), result);

        let cube = Cube::from_extent(Orthant::from([1, 1, 1]), &[true, true, false]);
        let result = MatchResult::Ace { cell: cube.clone() };
        assert_eq!(matching.match_cell(&cube), result);
    }
}
