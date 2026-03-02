// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Sequential wavefront implementation for gradient flow operations.

use std::collections::BTreeMap;

use tracing::info;

use crate::{
    Chain, Complex, Cube, Grader, MorseMatching, Orthant, Ring, TopCubicalMatching,
    homology::cubical::{OrthantMatching, Subgrid},
};

/// Configuration for wavefront traversal direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WavefrontConfig {
    /// Ascending orthant order. Used for `lower` and `lift`.
    Boundary,
    /// Descending orthant order. Used for `colower` and `colift`.
    Coboundary,
}

/// A cell yielded from flowing through the matching.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum FlowCell<R: Ring> {
    /// Critical cell (Ace).
    Ace {
        /// The extent bitstring of the critical cell.
        extent: u32,
        /// The coefficient of the cell in the chain.
        coef: R,
    },
    /// Queen cell matched to a King.
    Queen {
        /// The extent bitstring of the matched king cell.
        king_extent: u32,
        /// Coefficient of the matched king cell.
        coef: R,
    },
    /// King cell matched to a Queen.
    King {
        /// The extent bitstring of the matched queen cell.
        queen_extent: u32,
        /// Coefficient of the matched queen cell.
        coef: R,
    },
}

/// Sequential wavefront for gradient flow operations.
///
/// Manages orthant-by-orthant traversal, computing matchings on demand.
struct Wavefront<R: Ring> {
    frontier: BTreeMap<Orthant, Chain<u32, R>>,
    config: WavefrontConfig,
    min_orthant: Orthant,
    max_orthant: Orthant,
    grade_cap: u32,
    label: &'static str,
    orthants_processed: usize,
}

impl<R: Ring> Wavefront<R> {
    /// Create a wavefront for boundary operations (ascending order).
    #[must_use]
    pub fn for_boundary(
        min_orthant: Orthant,
        max_orthant: Orthant,
        grade_cap: u32,
        label: &'static str,
    ) -> Self {
        Self {
            frontier: BTreeMap::new(),
            config: WavefrontConfig::Boundary,
            min_orthant,
            max_orthant,
            grade_cap,
            label,
            orthants_processed: 0,
        }
    }

    /// Create a wavefront for coboundary operations (descending order).
    #[must_use]
    pub fn for_coboundary(
        min_orthant: Orthant,
        max_orthant: Orthant,
        grade_cap: u32,
        label: &'static str,
    ) -> Self {
        Self {
            frontier: BTreeMap::new(),
            config: WavefrontConfig::Coboundary,
            min_orthant,
            max_orthant,
            grade_cap,
            label,
            orthants_processed: 0,
        }
    }

    fn push(&mut self, orthant: Orthant, extent: u32, coef: R) {
        if coef == R::zero() {
            return;
        }
        self.frontier
            .entry(orthant)
            .or_default()
            .insert_or_add(extent, coef);
    }

    /// Seed the frontier from a chain of cubes.
    pub fn seed(&mut self, chain: impl IntoIterator<Item = (Cube, R)>) {
        for (cube, coef) in chain {
            let extent = cube_to_extent(&cube);
            self.push(cube.base().clone(), extent, coef);
        }
    }

    /// Pop the next orthant to process.
    #[must_use]
    pub fn pop_next(&mut self) -> Option<(Orthant, Chain<u32, R>)> {
        let result = match self.config {
            WavefrontConfig::Boundary => self.frontier.pop_first(),
            WavefrontConfig::Coboundary => self.frontier.pop_last(),
        };
        if result.is_some() {
            self.orthants_processed += 1;
            if self.orthants_processed.is_multiple_of(10000) {
                info!(
                    "{}: processed {} orthants (frontier: {})",
                    self.label,
                    self.orthants_processed,
                    self.frontier.len()
                );
            }
        }
        result
    }

    /// Flow a chain through an orthant's matching.
    pub fn flow_orthant<F>(
        &mut self,
        orthant: &Orthant,
        orthant_chain: Chain<u32, R>,
        matching: &OrthantMatching,
        mut callback: F,
    ) where
        F: FnMut(FlowCell<R>),
    {
        let min_orthant = self.min_orthant.clone();
        let max_orthant = self.max_orthant.clone();
        super::flow_orthant_impl(
            orthant,
            orthant_chain,
            matching,
            self.config,
            self.grade_cap,
            &min_orthant,
            &max_orthant,
            &mut |o, e, c| self.push(o, e, c),
            &mut callback,
        );
    }
}

/// Convert a cube to extent bitstring.
pub(super) fn cube_to_extent(cube: &Cube) -> u32 {
    cube.extent()
        .into_iter()
        .enumerate()
        .map(|(axis, has_extent)| if has_extent { 1u32 << axis } else { 0 })
        .sum()
}

/// Convert extent bitstring to cube.
pub(super) fn extent_to_cube(orthant: &Orthant, extent: u32) -> Cube {
    let dual: Orthant = orthant
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
    Cube::new(orthant.clone(), dual)
}

/// Compute the orthant matching for a single orthant.
fn match_orthant<R, G>(matching: &TopCubicalMatching<R, G>, orthant: &Orthant) -> OrthantMatching
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    let complex = matching.upper_complex();
    let mut subgrid = Subgrid::new(complex, u32::MAX, u32::MAX);
    subgrid.match_subgrid(orthant.clone(), orthant.clone())[0]
        .2
        .clone()
}

pub(super) fn lower<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (Cube, R)>,
    min_grade: u32,
) -> Chain<u32, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    let complex = matching.upper_complex();
    let mut wavefront = Wavefront::for_boundary(
        complex.minimum().clone(),
        complex.maximum().clone(),
        min_grade,
        "lower",
    );
    let mut result = Chain::new();

    wavefront.seed(chain);

    while let Some((orthant, orthant_chain)) = wavefront.pop_next() {
        let orthant_matching = match_orthant(matching, &orthant);

        wavefront.flow_orthant(&orthant, orthant_chain, &orthant_matching, |cell| {
            if let FlowCell::Ace { extent, coef } = cell {
                let cube = extent_to_cube(&orthant, extent);
                if let Some(idx) = matching.project_cell(&cube) {
                    result.insert_or_add(idx, coef);
                }
            }
        });
    }

    result
}

pub(super) fn colower<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (Cube, R)>,
    max_grade: u32,
) -> Chain<u32, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    let complex = matching.upper_complex();
    let mut wavefront = Wavefront::for_coboundary(
        complex.minimum().clone(),
        complex.maximum().clone(),
        max_grade,
        "colower",
    );
    let mut result = Chain::new();

    wavefront.seed(chain);

    while let Some((orthant, orthant_chain)) = wavefront.pop_next() {
        let orthant_matching = match_orthant(matching, &orthant);

        wavefront.flow_orthant(&orthant, orthant_chain, &orthant_matching, |cell| {
            if let FlowCell::Ace { extent, coef } = cell {
                let cube = extent_to_cube(&orthant, extent);
                if let Some(idx) = matching.project_cell(&cube) {
                    result.insert_or_add(idx, coef);
                }
            }
        });
    }

    result
}

pub(super) fn lift<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (u32, R)>,
    min_grade: u32,
) -> Chain<Cube, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    let complex = matching.upper_complex();
    let mut wavefront = Wavefront::for_boundary(
        complex.minimum().clone(),
        complex.maximum().clone(),
        min_grade,
        "lift",
    );

    let mut result = Chain::new();
    for (idx, coef) in chain {
        result.insert_or_add(matching.include_cell(idx), coef);
    }
    wavefront.seed(complex.boundary(&result));

    while let Some((orthant, orthant_chain)) = wavefront.pop_next() {
        let orthant_matching = match_orthant(matching, &orthant);

        wavefront.flow_orthant(&orthant, orthant_chain, &orthant_matching, |cell| {
            if let FlowCell::Queen { king_extent, coef } = cell {
                let king_cube = extent_to_cube(&orthant, king_extent);
                result.insert_or_add(king_cube, coef);
            }
        });
    }

    result
}

pub(super) fn colift<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (u32, R)>,
    max_grade: u32,
) -> Chain<Cube, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    let complex = matching.upper_complex();
    let mut wavefront = Wavefront::for_coboundary(
        complex.minimum().clone(),
        complex.maximum().clone(),
        max_grade,
        "colift",
    );

    // Note: grade filtering is done in flow.rs before dispatch, so all cells
    // in chain are already within the grade cap.
    let mut result = Chain::new();
    for (idx, coef) in chain {
        result.insert_or_add(matching.include_cell(idx), coef);
    }
    wavefront.seed(complex.coboundary(&result));

    while let Some((orthant, orthant_chain)) = wavefront.pop_next() {
        let orthant_matching = match_orthant(matching, &orthant);

        wavefront.flow_orthant(&orthant, orthant_chain, &orthant_matching, |cell| {
            if let FlowCell::King { queen_extent, coef } = cell {
                let queen_cube = extent_to_cube(&orthant, queen_extent);
                result.insert_or_add(queen_cube, coef);
            }
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cyclic;

    #[test]
    fn cube_extent_roundtrip() {
        let orthant = Orthant::from([1, 2, 3]);
        for extent in 0..8u32 {
            let cube = extent_to_cube(&orthant, extent);
            let recovered = cube_to_extent(&cube);
            assert_eq!(extent, recovered);
        }
    }

    #[test]
    fn incidence_signs() {
        use super::super::compute_incidence;
        assert_eq!(compute_incidence::<Cyclic<7>>(0b000, 0), Cyclic::one());
        assert_eq!(compute_incidence::<Cyclic<7>>(0b001, 1), -Cyclic::one());
        assert_eq!(compute_incidence::<Cyclic<7>>(0b011, 2), Cyclic::one());
    }

    #[test]
    fn wavefront_ascending_order() {
        let mut wavefront: Wavefront<Cyclic<7>> =
            Wavefront::for_boundary(Orthant::from([0, 0]), Orthant::from([3, 3]), 0, "test");

        wavefront.push(Orthant::from([2, 1]), 0b01, Cyclic::one());
        wavefront.push(Orthant::from([1, 0]), 0b10, Cyclic::one());
        wavefront.push(Orthant::from([1, 2]), 0b11, Cyclic::one());

        let (o1, _) = wavefront.pop_next().unwrap();
        let (o2, _) = wavefront.pop_next().unwrap();
        let (o3, _) = wavefront.pop_next().unwrap();

        assert_eq!(o1, Orthant::from([1, 0]));
        assert_eq!(o2, Orthant::from([1, 2]));
        assert_eq!(o3, Orthant::from([2, 1]));
    }

    #[test]
    fn wavefront_descending_order() {
        let mut wavefront: Wavefront<Cyclic<7>> = Wavefront::for_coboundary(
            Orthant::from([0, 0]),
            Orthant::from([3, 3]),
            u32::MAX,
            "test",
        );

        wavefront.push(Orthant::from([2, 1]), 0b01, Cyclic::one());
        wavefront.push(Orthant::from([1, 0]), 0b10, Cyclic::one());
        wavefront.push(Orthant::from([1, 2]), 0b11, Cyclic::one());

        let (o1, _) = wavefront.pop_next().unwrap();
        let (o2, _) = wavefront.pop_next().unwrap();
        let (o3, _) = wavefront.pop_next().unwrap();

        assert_eq!(o1, Orthant::from([2, 1]));
        assert_eq!(o2, Orthant::from([1, 2]));
        assert_eq!(o3, Orthant::from([1, 0]));
    }
}
