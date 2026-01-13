// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Wavefront iteration for efficient Morse matching operations.
//!
//! This module provides [`Wavefront`], an abstraction for efficiently computing
//! the `lower`, `colower`, `lift`, and `colift` operations on top-dimensional
//! cubical complexes. The key improvement orthant-local processing: each
//! orthant is matched at most once, and all local work is completed before
//! moving to neighboring orthants.
//!
//! # Algorithm
//!
//! Cells flow through the matching in a wave pattern:
//! - **Boundary operations** (`lower`, `lift`): cells flow to same-or-greater
//!   orthant coordinates, so orthants are processed in ascending order.
//! - **Coboundary operations** (`colower`, `colift`): cells flow to
//!   same-or-lesser orthant coordinates, so orthants are processed in
//!   descending order.
//!
//! The [`Wavefront`] type maintains a frontier of pending cells organized by
//! orthant. The [`flow_orthant`] method processes all cells in an orthant
//! through the matching, automatically recording (co)boundary cells arising
//! from adjacent orthants.
//!
//! [`flow_orthant`]: Wavefront::flow_orthant

use std::collections::BTreeMap;

use crate::{
    Cube, HashMapModule, ModuleLike, Orthant, RingLike, homology::cubical::OrthantMatching,
};

/// Configuration for wavefront traversal direction and cell processing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WavefrontConfig {
    /// Ascending orthant order and transforms queens to kings.
    /// Used by `lower` and `lift` (boundary operations).
    Boundary,

    /// Descending orthant order and transforms kings to queens.
    /// Used by `colower` and `colift` (coboundary operations).
    Coboundary,
}

/// A cell yielded from flowing a chain through the matching.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlowCell<R: RingLike> {
    /// Critical cell with its coefficient in the chain.
    Ace {
        /// The extent bitstring of the critical cell.
        extent: u32,
        /// The coefficient of the cell in the chain.
        coef: R,
    },

    /// Queen cell (yielded for [`WavefrontConfig::Boundary`]).
    Queen {
        /// The extent bitstring of the matched king cell.
        king_extent: u32,
        /// Coefficient of the matched king cell.
        coef: R,
    },

    /// King cell (yielded for [`WavefrontConfig::Coboundary`]).
    King {
        /// The extent bitstring of the matched queen cell.
        queen_extent: u32,
        /// Coefficient of the matched queen cell.
        coef: R,
    },
}

/// Manages the frontier of pending cells organized by orthant.
///
/// Handles orthant-by-orthant traversal and automatically collects (co)boundary
/// cells from adjacent orthants during gradient flow.
pub struct Wavefront<R: RingLike> {
    frontier: BTreeMap<Orthant, HashMapModule<u32, R>>,
    config: WavefrontConfig,
    min_orthant: Orthant,
    max_orthant: Orthant,
}

impl<R: RingLike> Wavefront<R> {
    /// Create a new wavefront with the given configuration and complex bounds.
    #[must_use]
    pub fn new(config: WavefrontConfig, min_orthant: Orthant, max_orthant: Orthant) -> Self {
        Self {
            frontier: BTreeMap::new(),
            config,
            min_orthant,
            max_orthant,
        }
    }

    /// Add a cell to the frontier.
    fn push(&mut self, orthant: Orthant, extent: u32, coef: R) {
        if coef == R::zero() {
            return;
        }
        self.frontier
            .entry(orthant)
            .or_insert_with(HashMapModule::new)
            .insert_or_add(extent, coef);
    }

    /// Seed the frontier from a chain of cubes.
    pub fn seed<I>(&mut self, chain: I)
    where
        I: IntoIterator<Item = (Cube, R)>,
    {
        for (cube, coef) in chain {
            let extent = cube_to_extent(&cube);
            self.push(cube.base().clone(), extent, coef);
        }
    }

    /// Pop the next orthant to process.
    #[must_use]
    pub fn pop_next(&mut self) -> Option<(Orthant, HashMapModule<u32, R>)> {
        match self.config {
            WavefrontConfig::Boundary => self.frontier.pop_first(),
            WavefrontConfig::Coboundary => self.frontier.pop_last(),
        }
    }

    /// Check if the frontier is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frontier.is_empty()
    }

    /// Flow a chain through an orthant's matching, calling `callback` for each
    /// designated cell (see below).
    ///
    /// For each cell in `orthant_chain`:
    /// - Traverses the `OrthantMatching` tree to classify the cell
    /// - Yields `Ace` cells via callback with their coefficient
    /// - For matched cells (Queen or King, depending on the
    ///   [`WavefrontConfig`]):
    ///   - Computes (co)boundary of the matched target
    ///   - Pushes neighbor-orthant faces into the frontier automatically
    ///   - Adds same-orthant faces back to the chain for continued processing
    ///   - Yields the matched cell via callback with its coefficient.
    ///
    /// Only yields cell types according to the [`WavefrontConfig`]:
    /// - `Boundary`: yields Aces and Queens (discards Kings)
    /// - `Coboundary`: yields Aces and Kings (discards Queens)
    pub fn flow_orthant<F>(
        &mut self,
        orthant: &Orthant,
        mut orthant_chain: HashMapModule<u32, R>,
        matching: &OrthantMatching,
        mut callback: F,
    ) where
        F: FnMut(FlowCell<R>),
    {
        self.flow_recursive(orthant, &mut orthant_chain, matching, &mut callback);
    }

    fn flow_recursive<F>(
        &mut self,
        orthant: &Orthant,
        orthant_chain: &mut HashMapModule<u32, R>,
        matching: &OrthantMatching,
        callback: &mut F,
    ) where
        F: FnMut(FlowCell<R>),
    {
        match matching {
            OrthantMatching::Critical { ace_extent, .. } => {
                let coef = orthant_chain.remove(ace_extent);
                if coef != R::zero() {
                    callback(FlowCell::Ace {
                        extent: *ace_extent,
                        coef,
                    });
                }
            },

            OrthantMatching::Leaf {
                lower_extent,
                match_axis,
            } => {
                self.flow_leaf(orthant, orthant_chain, *lower_extent, *match_axis, callback);
            },

            OrthantMatching::Branch {
                suborthant_matchings,
                ..
            } => {
                // Processing suborthants in reverse order ensures we can
                // just use one pass.
                for sub_matching in suborthant_matchings.iter().rev() {
                    self.flow_recursive(orthant, orthant_chain, sub_matching, callback);
                }
            },
        }
    }

    fn flow_leaf<F>(
        &mut self,
        orthant: &Orthant,
        orthant_chain: &mut HashMapModule<u32, R>,
        lower_extent: u32,
        match_axis: u32,
        callback: &mut F,
    ) where
        F: FnMut(FlowCell<R>),
    {
        let axis_flag = 1 << match_axis;

        // Collect cells in this suborthant
        let cells: Vec<(u32, R)> = orthant_chain
            .iter()
            .filter(|(extent, _)| (lower_extent & !**extent) == 0)
            .map(|(e, c)| (*e, c.clone()))
            .collect();

        for (extent, coef) in cells {
            orthant_chain.remove(&extent);
            if coef == R::zero() {
                continue;
            }

            let is_queen = extent & axis_flag == 0;
            let incidence = compute_incidence::<R>(extent, match_axis);

            match self.config {
                WavefrontConfig::Boundary if is_queen => {
                    let king_extent = extent | axis_flag;
                    let king_coef = coef * incidence;

                    // Compute boundary of king, push neighbors to frontier
                    self.push_boundary_neighbors(orthant, king_extent, king_coef.clone());

                    // Add same-orthant boundary faces back to orthant_chain
                    self.add_boundary_same_orthant(
                        orthant_chain,
                        king_extent,
                        lower_extent,
                        king_coef.clone(),
                    );

                    callback(FlowCell::Queen {
                        king_extent,
                        coef: king_coef,
                    });
                },

                WavefrontConfig::Coboundary if !is_queen => {
                    let queen_extent = extent ^ axis_flag;
                    let queen_coef = coef * incidence;

                    // Compute coboundary of queen, push neighbors to frontier
                    self.push_coboundary_neighbors(orthant, queen_extent, queen_coef.clone());

                    // Add same-orthant coboundary faces back to orthant_chain
                    self.add_coboundary_same_orthant(
                        orthant_chain,
                        queen_extent,
                        lower_extent,
                        queen_coef.clone(),
                    );

                    callback(FlowCell::King {
                        queen_extent,
                        coef: queen_coef,
                    });
                },

                _ => {
                    // Cell type not relevant, discard
                },
            }
        }
    }

    /// Push boundary faces to neighboring orthants.
    fn push_boundary_neighbors(&mut self, orthant: &Orthant, extent: u32, coef: R) {
        let mut incidence = coef;
        let mut neighbor_orthant = orthant.clone();

        for axis in 0..orthant.ambient_dimension() as usize {
            let axis_flag = 1 << axis;
            if extent & axis_flag != 0 {
                let face_extent = extent ^ axis_flag;

                // Face in neighboring orthant (coord + 1)
                if orthant[axis] < self.max_orthant[axis] {
                    neighbor_orthant[axis] += 1;
                    self.push(neighbor_orthant.clone(), face_extent, incidence.clone());
                    neighbor_orthant[axis] -= 1;
                }

                incidence = -incidence;
            }
        }
    }

    /// Add boundary faces that stay in the same orthant to `orthant_chain`.
    fn add_boundary_same_orthant(
        &self,
        orthant_chain: &mut HashMapModule<u32, R>,
        extent: u32,
        lower_extent: u32,
        coef: R,
    ) {
        let mut incidence = coef;

        for axis in 0..self.max_orthant.ambient_dimension() as usize {
            let axis_flag = 1 << axis;
            if extent & axis_flag != 0 {
                let face_extent = extent ^ axis_flag;

                // Only add if face is outside the current suborthant
                // (faces inside will be matched and discarded anyway)
                if lower_extent & axis_flag != 0 {
                    orthant_chain.insert_or_add(face_extent, -incidence.clone());
                }

                incidence = -incidence;
            }
        }
    }

    /// Push coboundary cofaces to neighboring orthants.
    fn push_coboundary_neighbors(&mut self, orthant: &Orthant, extent: u32, coef: R) {
        let mut incidence = coef;
        let mut neighbor_orthant = orthant.clone();

        for axis in 0..orthant.ambient_dimension() as usize {
            let axis_flag = 1 << axis;
            if extent & axis_flag == 0 {
                let coface_extent = extent | axis_flag;

                // Coface in neighboring orthant (coord - 1)
                if orthant[axis] > self.min_orthant[axis] {
                    neighbor_orthant[axis] -= 1;
                    self.push(neighbor_orthant.clone(), coface_extent, -incidence.clone());
                    neighbor_orthant[axis] += 1;
                }

                incidence = -incidence;
            }
        }
    }

    /// Add coboundary cofaces that stay in the same orthant to `orthant_chain`.
    fn add_coboundary_same_orthant(
        &self,
        orthant_chain: &mut HashMapModule<u32, R>,
        extent: u32,
        lower_extent: u32,
        coef: R,
    ) {
        let mut incidence = coef;

        for axis in 0..self.max_orthant.ambient_dimension() as usize {
            let axis_flag = 1 << axis;
            if extent & axis_flag == 0 {
                let coface_extent = extent | axis_flag;

                // Only add if coface is outside the current suborthant
                // (cofaces inside will be matched and discarded anyway).
                // A coface is outside if the suborthant lower bound has this
                // axis bit clear (meaning the suborthant doesn't extend in this
                // direction).
                if lower_extent & axis_flag == 0 {
                    orthant_chain.insert_or_add(coface_extent, incidence.clone());
                }

                incidence = -incidence;
            }
        }
    }
}

/// Compute incidence coefficient for matching along axis.
///
/// For a cell with extent `e` matched along axis `a`, the incidence is
/// determined by the parity of bits set below axis `a`.
#[must_use]
pub fn compute_incidence<R: RingLike>(extent: u32, match_axis: u32) -> R {
    if (extent % (1 << match_axis)).count_ones().is_multiple_of(2) {
        R::one()
    } else {
        -R::one()
    }
}

/// Convert a cube to its extent bitstring representation.
#[must_use]
pub fn cube_to_extent(cube: &Cube) -> u32 {
    cube.extent()
        .into_iter()
        .enumerate()
        .map(|(axis, has_extent)| if has_extent { 1u32 << axis } else { 0 })
        .sum()
}

/// Convert an extent bitstring back to a cube.
#[must_use]
pub fn extent_to_cube(orthant: &Orthant, extent: u32) -> Cube {
    let dual_orthant: Orthant = orthant
        .iter()
        .enumerate()
        .map(|(axis, &coord)| {
            if extent & (1 << axis) != 0 {
                coord
            } else {
                coord - 1
            }
        })
        .collect();
    Cube::new(orthant.clone(), dual_orthant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cyclic;

    #[test]
    fn cube_extent_composition() {
        let orthant = Orthant::from([1, 2, 3]);

        // Test various extents
        for extent in 0..8u32 {
            let cube = extent_to_cube(&orthant, extent);
            let recovered = cube_to_extent(&cube);
            assert_eq!(extent, recovered, "extent {extent:#b} failed roundtrip");
        }
    }

    #[test]
    fn incidence_signs() {
        // For extent 0b000, matching along any axis gives incidence 1
        assert_eq!(compute_incidence::<Cyclic<7>>(0b000, 0), Cyclic::one());
        assert_eq!(compute_incidence::<Cyclic<7>>(0b000, 1), Cyclic::one());
        assert_eq!(compute_incidence::<Cyclic<7>>(0b000, 2), Cyclic::one());

        // For extent 0b001, matching along axis 1 checks count_ones(0b001 % 2) = 1,
        // odd -> -1
        assert_eq!(compute_incidence::<Cyclic<7>>(0b001, 1), -Cyclic::one());

        // For extent 0b011, matching along axis 2 checks count_ones(0b011 % 4) = 2,
        // even -> 1
        assert_eq!(compute_incidence::<Cyclic<7>>(0b011, 2), Cyclic::one());
    }

    #[test]
    fn wavefront_ascending_order() {
        let mut wavefront: Wavefront<Cyclic<7>> = Wavefront::new(
            WavefrontConfig::Boundary,
            Orthant::from([0, 0]),
            Orthant::from([3, 3]),
        );

        wavefront.push(Orthant::from([2, 1]), 0b01, Cyclic::one());
        wavefront.push(Orthant::from([1, 0]), 0b10, Cyclic::one());
        wavefront.push(Orthant::from([1, 2]), 0b11, Cyclic::one());

        // Should pop in ascending order
        let (o1, _) = wavefront.pop_next().unwrap();
        let (o2, _) = wavefront.pop_next().unwrap();
        let (o3, _) = wavefront.pop_next().unwrap();

        assert_eq!(o1, Orthant::from([1, 0]));
        assert_eq!(o2, Orthant::from([1, 2]));
        assert_eq!(o3, Orthant::from([2, 1]));
    }

    #[test]
    fn wavefront_descending_order() {
        let mut wavefront: Wavefront<Cyclic<7>> = Wavefront::new(
            WavefrontConfig::Coboundary,
            Orthant::from([0, 0]),
            Orthant::from([3, 3]),
        );

        wavefront.push(Orthant::from([2, 1]), 0b01, Cyclic::one());
        wavefront.push(Orthant::from([1, 0]), 0b10, Cyclic::one());
        wavefront.push(Orthant::from([1, 2]), 0b11, Cyclic::one());

        // Should pop in descending order
        let (o1, _) = wavefront.pop_next().unwrap();
        let (o2, _) = wavefront.pop_next().unwrap();
        let (o3, _) = wavefront.pop_next().unwrap();

        assert_eq!(o1, Orthant::from([2, 1]));
        assert_eq!(o2, Orthant::from([1, 2]));
        assert_eq!(o3, Orthant::from([1, 0]));
    }
}
