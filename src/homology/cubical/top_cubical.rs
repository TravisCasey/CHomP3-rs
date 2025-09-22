// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::collections::HashMap;
use std::iter::zip;

use crate::homology::cubical::{OrthantMatching, Subgrid};
use crate::{
    ComplexLike, Cube, CubicalComplex, Grader, HashMapModule, MatchResult, ModuleLike,
    MorseMatching, Orthant, OrthantIterator, RingLike, TopCubeGrader,
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
pub struct TopCubicalMatching<UM, G, LM = HashMapModule<u32, <UM as ModuleLike>::Ring>>
where
    UM: ModuleLike<Cell = Cube>,
    G: Grader<Orthant>,
{
    complex: CubicalComplex<UM, TopCubeGrader<G>>,
    critical_cells: Vec<Cube>,
    boundaries: Vec<LM>,
    projection: HashMap<Cube, u32>,
    gradient: HashMap<Orthant, HashMap<u32, HashMapModule<u32, UM::Ring>>>,
}

impl<UM, G> TopCubicalMatching<UM, G>
where
    UM: ModuleLike<Cell = Cube>,
    G: Grader<Orthant>,
{
    /// Convenience function to call the identically named method from the
    /// [`MorseMatching`] implementation with the default `LM` type parameter.
    pub fn compute_matching(complex: CubicalComplex<UM, TopCubeGrader<G>>) -> Self {
        <Self as MorseMatching>::compute_matching(complex)
    }
}

impl<UM, G, LM> TopCubicalMatching<UM, G, LM>
where
    UM: ModuleLike<Cell = Cube>,
    LM: ModuleLike<Cell = u32, Ring = UM::Ring>,
    G: Grader<Orthant>,
{
    /// Propagate the gradient `chain` of the critical cell (index `ace_index`
    /// in `critical_cells` and `boundaries`) within the orthant
    /// `base_orthant`; its matching is given by `orthant_matching`.
    fn gradient_flow(
        &mut self,
        ace_index: u32,
        base_orthant: &Orthant,
        orthant_matching: &OrthantMatching,
        chain: &mut HashMapModule<u32, UM::Ring>,
    ) {
        match orthant_matching {
            // When the gradient reaches another critical cell, it is saved as
            // the boundary.
            OrthantMatching::Critical {
                ace_dual_orthant,
                ace_extent,
            } => {
                let coef = chain.remove(ace_extent);
                if coef != UM::Ring::zero() {
                    self.boundaries[ace_index as usize].insert_or_add(
                        self.projection[&Cube::new(base_orthant.clone(), ace_dual_orthant.clone())],
                        coef,
                    );
                }
            }
            // Each queen in the chain is transformed to its king, then its
            // boundary is computed and added to `chain` or saved for later
            // base orthants as needed.
            OrthantMatching::Leaf {
                lower_extent,
                match_axis,
            } => {
                let mut suborthant_cells = Vec::new();
                for (cell_extent, coef) in chain.iter() {
                    debug_assert_ne!(*coef, UM::Ring::zero());
                    if (lower_extent & !cell_extent) == 0 {
                        suborthant_cells.push(*cell_extent);
                    }
                }

                let mut boundary_base_orthant = base_orthant.clone();
                for cell_extent in suborthant_cells {
                    if cell_extent & (1 << match_axis) != 0 {
                        // king
                        chain.remove(&cell_extent);
                        continue;
                    }

                    let king_extent = cell_extent + (1 << match_axis);
                    let mut incidence = if ((cell_extent % (1 << match_axis)).count_ones()) % 2 == 0
                    {
                        UM::Ring::one()
                    } else {
                        -UM::Ring::one()
                    } * chain.remove(&cell_extent);

                    for axis in 0..base_orthant.ambient_dimension() as usize {
                        if king_extent & (1 << axis) != 0 {
                            let boundary_extent = king_extent - (1 << axis);
                            if base_orthant[axis] != self.complex.maximum()[axis] {
                                boundary_base_orthant[axis] += 1;
                                self.update_gradient(
                                    ace_index,
                                    boundary_base_orthant.clone(),
                                    boundary_extent,
                                    incidence.clone(),
                                );
                                boundary_base_orthant[axis] -= 1;
                            }
                            if lower_extent & (1 << axis) != 0 {
                                chain.insert_or_add(boundary_extent, -incidence.clone());
                            }
                            incidence = -incidence;
                        }
                    }
                }
            }
            // Split into the suborthants.
            OrthantMatching::Branch {
                suborthant_matchings,
                ..
            } => {
                for suborthant_matching in suborthant_matchings {
                    self.gradient_flow(ace_index, base_orthant, suborthant_matching, chain);
                }
            }
        }
    }

    /// The `gradient` data structure is quite complex; this helper function
    /// updates it.
    fn update_gradient(
        &mut self,
        ace_index: u32,
        base_orthant: Orthant,
        extent: u32,
        coef: UM::Ring,
    ) {
        self.gradient
            .entry(base_orthant)
            .or_default()
            .entry(ace_index)
            .or_insert(HashMapModule::new())
            .insert_or_add(extent, coef);
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
        skip_axes: u32,
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
                prime_extent,
                suborthant_matchings,
            } => {
                let mut differing_axes = extent & !prime_extent;
                let mut suborthant_index = 0usize;
                let mut axis = 0;
                while axis < cube.base().ambient_dimension() {
                    if (skip_axes & (1 << axis)) == 0 {
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
                let next_skip_axes = !(prime_extent | ((1 << axis) - 1));
                Self::match_helper(
                    cube,
                    extent,
                    &suborthant_matchings[suborthant_index],
                    next_skip_axes,
                )
            }
        }
    }
}

impl<UM, G, LM> MorseMatching for TopCubicalMatching<UM, G, LM>
where
    UM: ModuleLike<Cell = Cube>,
    LM: ModuleLike<Cell = u32, Ring = UM::Ring>,
    G: Grader<Orthant>,
{
    type LowerModule = LM;
    type Priority = Orthant;
    type Ring = UM::Ring;
    type UpperCell = Cube;
    type UpperComplex = CubicalComplex<UM, TopCubeGrader<G>>;
    type UpperModule = UM;

    fn compute_matching(complex: Self::UpperComplex) -> Self {
        // When subgrids are fully implemented, this will need to be replaced.
        // Minimum and maximum of each subgrid
        let nonempty_subgrids =
            OrthantIterator::new(complex.minimum().clone(), complex.maximum().clone())
                .map(|orth| (orth.clone(), orth))
                .collect::<Vec<(Orthant, Orthant)>>();

        let mut matching = Self {
            complex,
            critical_cells: Vec::new(),
            boundaries: Vec::new(),
            projection: HashMap::new(),
            gradient: HashMap::new(),
        };

        for (minimum_orthant, maximum_orthant) in nonempty_subgrids {
            let mut subgrid = Subgrid::new(
                matching.get_upper_complex(),
                minimum_orthant,
                maximum_orthant,
                u32::MAX,
                u32::MAX,
            );

            let base_critical_orthant = subgrid.match_subgrid();
            for (_, critical_cells, _) in base_critical_orthant.iter() {
                for cube in critical_cells {
                    matching
                        .projection
                        .insert(cube.clone(), matching.critical_cells.len() as u32);
                    matching.critical_cells.push(cube.clone());
                    matching.boundaries.push(LM::new());
                }
            }

            for (base_orthant, critical_cells, _) in base_critical_orthant.iter() {
                for cube in critical_cells {
                    let boundary_chain = matching.get_upper_complex().cell_boundary(cube);
                    for (boundary_cube, coef) in boundary_chain.into_iter() {
                        if *boundary_cube.base() == *base_orthant
                            && let Some(ace_index) = matching.projection.get(&boundary_cube)
                        {
                            matching.boundaries[matching.projection[cube] as usize]
                                .insert_or_add(*ace_index, coef);
                            continue;
                        }

                        let mut boundary_extent = 0u32;
                        for (axis, (base_coord, dual_coord)) in
                            zip(boundary_cube.base().iter(), boundary_cube.dual().iter())
                                .enumerate()
                        {
                            if base_coord == dual_coord {
                                boundary_extent += 1 << axis;
                            }
                        }
                        matching.update_gradient(
                            matching.projection[cube],
                            boundary_cube.base().clone(),
                            boundary_extent,
                            coef,
                        );
                    }
                }
            }

            for (base_orthant, _, orthant_matching) in base_critical_orthant {
                if let Some(ace_map) = matching.gradient.remove(&base_orthant) {
                    for (ace_index, mut chain) in ace_map {
                        matching.gradient_flow(
                            ace_index,
                            &base_orthant,
                            &orthant_matching,
                            &mut chain,
                        );
                    }
                }
            }
        }

        matching
    }

    fn match_cell(&self, cube: &Cube) -> MatchResult<Cube, Self::Ring, Self::Priority> {
        let mut subgrid = Subgrid::new(
            self.get_upper_complex(),
            cube.base().clone(),
            cube.base().clone(),
            u32::MAX,
            u32::MAX,
        );
        let orthant_matching = &subgrid.match_subgrid()[0].2;

        let mut extent = 0u32;
        for (axis, (base_coord, dual_coord)) in
            zip(cube.base().iter(), cube.dual().iter()).enumerate()
        {
            if base_coord == dual_coord {
                extent += 1 << axis;
            }
        }

        Self::match_helper(cube.clone(), extent, orthant_matching, 0)
    }

    fn get_upper_complex(&self) -> &Self::UpperComplex {
        &self.complex
    }

    fn extract_upper_complex(self) -> Self::UpperComplex {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CellComplex, CoreductionMatching, Cube, CubicalComplex, Cyclic, Grader, HashMapGrader,
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
        let (_top_matching, _further_matchings, morse_complex) = TopCubicalMatching::full_reduce::<
            CoreductionMatching<CellComplex<HashMapModule<u32, Cyclic<2>>>>,
        >(complex);

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
        let matching = TopCubicalMatching::compute_matching(complex);

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
