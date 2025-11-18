// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::collections::BTreeMap;
use std::mem::replace;

use crate::homology::cubical::orthant_matching::OrthantMatching;
use crate::homology::cubical::subgrid::Subgrid;
use crate::{
    Cube, CubicalComplex, Grader, HashMapModule, ModuleLike, Orthant, RingLike, TopCubeGrader,
};

/// Helper struct to encapsulate computation of the gradient of a CubicalComplex
/// under a `TopCubicalMatching`. Immutably borrows the `CubicalComplex` it
/// operates on for the duration of its lifetime.
pub struct TopCubicalGradientPropagator<'a, UM, G>
where
    UM: ModuleLike<Cell = Cube>,
    G: Grader<Orthant>,
{
    complex: &'a CubicalComplex<UM, TopCubeGrader<G>>,
    gradient: BTreeMap<Orthant, HashMapModule<u32, UM::Ring>>,
    boundary: UM,
    subgrid: Subgrid<G>,
}

impl<'a, UM, G> TopCubicalGradientPropagator<'a, UM, G>
where
    UM: ModuleLike<Cell = Cube>,
    G: Grader<Orthant> + Clone,
{
    /// Prepare the propagator to compute the gradient (boundary operator) of
    /// `complex` under the `TopCubicalMatching`. It is expected that the
    /// relevant critical cells are pre-computed and the boundary of each are
    /// computed by calls to [`CubicalGradientPropagator::compute_gradient`].
    pub fn new(complex: &'a CubicalComplex<UM, TopCubeGrader<G>>) -> Self {
        Self {
            complex,
            gradient: BTreeMap::new(),
            boundary: UM::new(),
            subgrid: Subgrid::new(complex, u32::MAX, u32::MAX),
        }
    }

    /// Computes the boundary of the cell represented by `base_orthant` and
    /// `king_extent` and adds boundary elements in `base_orthant` to
    /// `orthant_chain`, and boundary elements in other orthants to the
    /// `gradient` field.
    fn boundary_of_king(
        &mut self,
        base_orthant: &Orthant,
        orthant_chain: &mut HashMapModule<u32, UM::Ring>,
        king_extent: u32,
        suborthant_lower_extent: u32,
        mut incidence: UM::Ring,
    ) {
        let mut boundary_base_orthant = base_orthant.clone();
        for axis in 0..base_orthant.ambient_dimension() as usize {
            let axis_flag = 1 << axis;
            if king_extent & axis_flag != 0 {
                let boundary_extent = king_extent ^ axis_flag;

                // Boundary element in neighboring orthant
                if base_orthant[axis] != self.complex.maximum()[axis] {
                    boundary_base_orthant[axis] += 1;
                    self.gradient
                        .entry(boundary_base_orthant.clone())
                        .or_insert(HashMapModule::new())
                        .insert_or_add(boundary_extent, incidence.clone());
                    boundary_base_orthant[axis] -= 1;
                }

                // Boundary element in the same orthant.
                // Do not store those in the same suborthant (bounded below by
                // `suborthant_lower_extent`) as these elements must be kings or
                // the queen of this king, and will be discarded in the next
                // round of gradient computation
                if suborthant_lower_extent & axis_flag != 0 {
                    orthant_chain.insert_or_add(boundary_extent, -incidence.clone());
                }

                incidence = -incidence;
            }
        }
    }

    /// Compute the gradient in the current suborthant which has matching
    /// described by `orthant_matching`, for the chain `orthant_chain`.
    fn gradient_flow(
        &mut self,
        base_orthant: &Orthant,
        orthant_matching: &OrthantMatching,
        orthant_chain: &mut HashMapModule<u32, UM::Ring>,
    ) {
        match orthant_matching {
            // When the gradient reaches another critical cell, it is saved as
            // the boundary.
            OrthantMatching::Critical {
                ace_dual_orthant,
                ace_extent,
            } => {
                let coef = orthant_chain.remove(ace_extent);
                if coef != UM::Ring::zero() {
                    self.boundary.insert_or_add(
                        Cube::new(base_orthant.clone(), ace_dual_orthant.clone()),
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
                for (cell_extent, coef) in orthant_chain.iter() {
                    debug_assert_ne!(*coef, UM::Ring::zero());
                    if (lower_extent & !cell_extent) == 0 {
                        suborthant_cells.push(*cell_extent);
                    }
                }

                for cell_extent in suborthant_cells {
                    if cell_extent & (1 << match_axis) != 0 {
                        // king
                        orthant_chain.remove(&cell_extent);
                        continue;
                    }

                    let king_extent = cell_extent + (1 << match_axis);
                    let incidence = if ((cell_extent % (1 << match_axis)).count_ones()) % 2 == 0 {
                        UM::Ring::one()
                    } else {
                        -UM::Ring::one()
                    } * orthant_chain.remove(&cell_extent);

                    self.boundary_of_king(
                        base_orthant,
                        orthant_chain,
                        king_extent,
                        *lower_extent,
                        incidence,
                    );
                }
            }
            // Split into the suborthants.
            OrthantMatching::Branch {
                suborthant_matchings,
                ..
            } => {
                for suborthant_matching in suborthant_matchings.iter().rev() {
                    self.gradient_flow(base_orthant, suborthant_matching, orthant_chain);
                }
            }
        }
    }

    /// Compute and return the gradient of `cell` (which is intended to be a
    /// critical cell) under a `TopCubicalMatching`.
    pub fn compute_gradient(&mut self, cell: &Cube) -> UM {
        let cell_dimension = cell.dimension();
        if cell_dimension == 0 {
            return UM::new();
        }
        self.subgrid
            .set_maximum_kept_grade(self.complex.grade(cell));
        self.subgrid.set_maximum_kept_dimension(cell_dimension - 1);

        let base_orthant = cell.base().clone();
        let extent = cell
            .extent()
            .into_iter()
            .enumerate()
            .map(
                |(axis, has_extent)| {
                    if has_extent { 1 << axis } else { 0u32 }
                },
            )
            .sum::<u32>();
        let mut boundary_chain = HashMapModule::new();

        // Not a king, but the operation is the same.
        self.boundary_of_king(
            &base_orthant,
            &mut boundary_chain,
            extent,
            u32::MAX,
            UM::Ring::one(),
        );
        self.gradient.insert(base_orthant, boundary_chain);

        // Always terminates due to sorting of orthants; each orthant can only
        // add gradient progress into later orthants.
        while let Some((base_orthant, mut orthant_chain)) = self.gradient.pop_first() {
            let orthant_matching = &self
                .subgrid
                .match_subgrid(base_orthant.clone(), base_orthant.clone())[0]
                .2;
            self.gradient_flow(&base_orthant, orthant_matching, &mut orthant_chain);
        }

        // Can run further computations without recreating this struct.
        replace(&mut self.boundary, UM::new())
    }
}
