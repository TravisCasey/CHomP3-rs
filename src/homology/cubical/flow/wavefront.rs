// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Level-synchronous wavefront for gradient flow operations.
//!
//! Orthants at the same coordinate-sum level are independent (the anti-chain
//! property: boundary pushes increase exactly one coordinate, so all targets
//! are at the next level) and can be processed in parallel via `ParallelMap`.

use std::collections::BTreeMap;

use crate::{
    Chain, Complex, Cube, Grader, MorseMatching, Orthant, Ring, TopCubicalMatching,
    homology::cubical::Subgrid, logging::ProgressTracker, parallel::map::ParallelMap,
};

/// Configuration for wavefront traversal direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WavefrontConfig {
    /// Ascending level order. Used for `lower` and `lift`.
    Boundary,
    /// Descending level order. Used for `colower` and `colift`.
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

/// Convert a cube to its extent bitstring representation.
///
/// Bit `i` is set if and only if the cube has extent along axis `i` (i.e., the
/// base and dual coordinates differ on that axis).
pub(super) fn cube_to_extent(cube: &Cube) -> u32 {
    cube.extent()
        .into_iter()
        .enumerate()
        .map(|(axis, has_extent)| if has_extent { 1u32 << axis } else { 0 })
        .sum()
}

/// Reconstruct a cube from its orthant and extent bitstring.
///
/// Axes with a set bit in `extent` get `dual[axis] = base[axis]` (extent
/// along that axis); cleared bits get `dual[axis] = base[axis] - 1`.
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

/// Coordinate-sum level of an orthant. Orthants at the same level are
/// independent under boundary/coboundary flow.
fn orthant_level(orthant: &Orthant) -> i64 {
    orthant.iter().map(|&c| i64::from(c)).sum()
}

/// Populate the frontier from an input chain, grouping cells by orthant.
fn seed_frontier<R: Ring>(
    frontier: &mut BTreeMap<Orthant, Chain<u32, R>>,
    chain: impl IntoIterator<Item = (Cube, R)>,
) {
    for (cube, coef) in chain {
        if coef == R::zero() {
            continue;
        }
        let extent = cube_to_extent(&cube);
        frontier
            .entry(cube.base().clone())
            .or_default()
            .insert_or_add(extent, coef);
    }
}

/// Remove up to `max_items` orthants at the extreme level from the frontier.
///
/// Boundary mode takes the minimum level; coboundary mode takes the maximum.
fn extract_sub_batch<R: Ring>(
    frontier: &mut BTreeMap<Orthant, Chain<u32, R>>,
    config: WavefrontConfig,
    max_items: usize,
) -> Vec<(Orthant, Chain<u32, R>)> {
    if frontier.is_empty() {
        return Vec::new();
    }

    let target_level = match config {
        WavefrontConfig::Boundary => frontier.keys().map(orthant_level).min().unwrap(),
        WavefrontConfig::Coboundary => frontier.keys().map(orthant_level).max().unwrap(),
    };

    let keys: Vec<Orthant> = frontier
        .keys()
        .filter(|o| orthant_level(o) == target_level)
        .take(max_items)
        .cloned()
        .collect();

    keys.into_iter()
        .map(|key| {
            let chain = frontier.remove(&key).unwrap();
            (key, chain)
        })
        .collect()
}

/// Core level-parallel loop shared by all four flow operations.
///
/// Processes frontier orthants grouped by coordinate-sum level. Within each
/// level, orthants are independent and processed in parallel via `ParallelMap`.
///
/// Returns collected cells as `(orthant, extent, coef)` triples. The caller
/// decides how to merge these into the result chain.
fn level_loop<R, G, CF>(
    matching: &TopCubicalMatching<R, G>,
    frontier: &mut BTreeMap<Orthant, Chain<u32, R>>,
    wavefront_config: WavefrontConfig,
    grade_cap: u32,
    cell_filter: CF,
    label: &str,
) -> Vec<(Orthant, u32, R)>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
    CF: Fn(FlowCell<R>) -> Option<(u32, R)> + Send + Sync,
{
    let complex = matching.upper_complex();
    let backend = &matching.config().backend;
    let sub_batch_size = matching.config().sub_batch_size;
    let min_orthant = complex.minimum().clone();
    let max_orthant = complex.maximum().clone();

    // Track progress across levels using the complex's coordinate-sum range.
    let min_level = orthant_level(&min_orthant);
    let max_level = orthant_level(&max_orthant);
    let total_levels = (max_level - min_level + 1).max(0) as usize;
    let progress = ProgressTracker::new(label, total_levels).with_interval(10);
    let level_origin = match wavefront_config {
        WavefrontConfig::Boundary => min_level,
        WavefrontConfig::Coboundary => max_level,
    };

    let mut collected_cells = Vec::new();

    loop {
        let sub_batch = extract_sub_batch(frontier, wavefront_config, sub_batch_size);

        if backend.is_done(sub_batch.is_empty() && frontier.is_empty()) {
            break;
        }

        // Update progress based on how far through the level range we are.
        if let Some((first_orthant, _)) = sub_batch.first() {
            let current_level = orthant_level(first_orthant);
            let levels_done = match wavefront_config {
                WavefrontConfig::Boundary => current_level - level_origin + 1,
                WavefrontConfig::Coboundary => level_origin - current_level + 1,
            };
            progress.set(levels_done.max(0) as usize);
        }

        let batch_results = ParallelMap::new(backend).run_with_state(
            sub_batch.into_iter(),
            || Subgrid::new(complex, u32::MAX, u32::MAX),
            |subgrid, (orthant, orthant_chain)| {
                let orthant_matching = subgrid.match_subgrid(orthant.clone(), orthant.clone())[0]
                    .2
                    .clone();

                let mut pushes: Vec<(Orthant, u32, R)> = Vec::new();
                let mut cells: Vec<(Orthant, u32, R)> = Vec::new();

                super::flow_orthant_impl(
                    &orthant,
                    orthant_chain,
                    &orthant_matching,
                    wavefront_config,
                    grade_cap,
                    &min_orthant,
                    &max_orthant,
                    &mut |o, e, c| pushes.push((o, e, c)),
                    &mut |cell| {
                        if let Some((extent, coef)) = cell_filter(cell) {
                            cells.push((orthant.clone(), extent, coef));
                        }
                    },
                );

                vec![(cells, pushes)]
            },
        );

        for (cells, pushes) in batch_results {
            collected_cells.extend(cells);
            for (target, extent, coef) in pushes {
                if coef != R::zero() {
                    frontier
                        .entry(target)
                        .or_default()
                        .insert_or_add(extent, coef);
                }
            }
        }
    }

    progress.finish();
    collected_cells
}

/// Project a chain from the parent complex to the Morse complex via boundary
/// flow. Collects critical cells (aces) encountered during the wavefront.
pub(crate) fn lower<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (Cube, R)>,
    min_grade: u32,
) -> Chain<u32, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    let mut frontier = BTreeMap::new();
    seed_frontier(&mut frontier, chain);

    let cells = level_loop(
        matching,
        &mut frontier,
        WavefrontConfig::Boundary,
        min_grade,
        |cell| match cell {
            FlowCell::Ace { extent, coef } => Some((extent, coef)),
            _ => None,
        },
        "Lower",
    );

    let mut result = Chain::new();
    for (orthant, extent, coef) in cells {
        let cube = extent_to_cube(&orthant, extent);
        if let Some(idx) = matching.project_cell(&cube) {
            result.insert_or_add(idx, coef);
        }
    }

    matching.config().backend.sync(result)
}

/// Project a cochain from the parent complex to the Morse complex via
/// coboundary flow. Collects critical cells (aces) encountered.
pub(crate) fn colower<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (Cube, R)>,
    max_grade: u32,
) -> Chain<u32, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    let mut frontier = BTreeMap::new();
    seed_frontier(&mut frontier, chain);

    let cells = level_loop(
        matching,
        &mut frontier,
        WavefrontConfig::Coboundary,
        max_grade,
        |cell| match cell {
            FlowCell::Ace { extent, coef } => Some((extent, coef)),
            _ => None,
        },
        "Colower",
    );

    let mut result = Chain::new();
    for (orthant, extent, coef) in cells {
        let cube = extent_to_cube(&orthant, extent);
        if let Some(idx) = matching.project_cell(&cube) {
            result.insert_or_add(idx, coef);
        }
    }

    matching.config().backend.sync(result)
}

/// Include a Morse chain into the parent complex via boundary flow. Starts
/// from the critical cells and adds matched king cells as queens are
/// encountered.
pub(crate) fn lift<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (u32, R)>,
    min_grade: u32,
) -> Chain<Cube, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    let complex = matching.upper_complex();

    let mut result = Chain::new();
    for (idx, coef) in chain {
        result.insert_or_add(matching.include_cell(idx), coef);
    }

    let mut frontier = BTreeMap::new();
    seed_frontier(&mut frontier, complex.boundary(&result));

    let cells = level_loop(
        matching,
        &mut frontier,
        WavefrontConfig::Boundary,
        min_grade,
        |cell| match cell {
            FlowCell::Queen { king_extent, coef } => Some((king_extent, coef)),
            _ => None,
        },
        "Lift",
    );

    for (orthant, king_extent, coef) in cells {
        result.insert_or_add(extent_to_cube(&orthant, king_extent), coef);
    }

    matching.config().backend.sync(result)
}

/// Include a Morse cochain into the parent complex via coboundary flow.
/// Starts from the critical cells and adds matched queen cells as kings are
/// encountered.
pub(crate) fn colift<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (u32, R)>,
    max_grade: u32,
) -> Chain<Cube, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    let complex = matching.upper_complex();

    let mut result = Chain::new();
    for (idx, coef) in chain {
        result.insert_or_add(matching.include_cell(idx), coef);
    }

    let mut frontier = BTreeMap::new();
    seed_frontier(&mut frontier, complex.coboundary(&result));

    let cells = level_loop(
        matching,
        &mut frontier,
        WavefrontConfig::Coboundary,
        max_grade,
        |cell| match cell {
            FlowCell::King { queen_extent, coef } => Some((queen_extent, coef)),
            _ => None,
        },
        "Colift",
    );

    for (orthant, queen_extent, coef) in cells {
        result.insert_or_add(extent_to_cube(&orthant, queen_extent), coef);
    }

    matching.config().backend.sync(result)
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
    fn level_computation() {
        assert_eq!(orthant_level(&Orthant::from([0, 0])), 0);
        assert_eq!(orthant_level(&Orthant::from([1, 2, 3])), 6);
        assert_eq!(orthant_level(&Orthant::from([-1, 3])), 2);
    }

    #[test]
    fn sub_batch_extracts_by_level() {
        let mut frontier: BTreeMap<Orthant, Chain<u32, Cyclic<7>>> = BTreeMap::new();

        // Level 1: [0,1] and [1,0]
        frontier.insert(Orthant::from([0, 1]), Chain::from([(0b01, Cyclic::one())]));
        frontier.insert(Orthant::from([1, 0]), Chain::from([(0b10, Cyclic::one())]));
        // Level 3: [1,2]
        frontier.insert(Orthant::from([1, 2]), Chain::from([(0b11, Cyclic::one())]));

        // Boundary: extracts minimum level (1), not lexicographic first
        let batch = extract_sub_batch(&mut frontier, WavefrontConfig::Boundary, 100);
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|(o, _)| orthant_level(o) == 1));
        assert_eq!(frontier.len(), 1); // level 3 remains

        // Coboundary: extracts maximum level
        let mut frontier2: BTreeMap<Orthant, Chain<u32, Cyclic<7>>> = BTreeMap::new();
        frontier2.insert(Orthant::from([0, 1]), Chain::from([(0b01, Cyclic::one())]));
        frontier2.insert(Orthant::from([2, 1]), Chain::from([(0b10, Cyclic::one())]));
        let batch = extract_sub_batch(&mut frontier2, WavefrontConfig::Coboundary, 100);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].0, Orthant::from([2, 1]));
    }

    #[test]
    fn sub_batch_respects_max_items() {
        let mut frontier: BTreeMap<Orthant, Chain<u32, Cyclic<7>>> = BTreeMap::new();
        for i in 0..10i16 {
            frontier.insert(Orthant::from([i, 0]), Chain::from([(0b01, Cyclic::one())]));
        }
        // All at different levels, so boundary extracts level 0 only (one item)
        let batch = extract_sub_batch(&mut frontier, WavefrontConfig::Boundary, 5);
        assert_eq!(batch.len(), 1); // only [0,0] is at level 0

        // Put multiple at the same level
        let mut frontier2: BTreeMap<Orthant, Chain<u32, Cyclic<7>>> = BTreeMap::new();
        for i in 0..10i16 {
            frontier2.insert(
                Orthant::from([i, 10 - i]),
                Chain::from([(0b01, Cyclic::one())]),
            );
        }
        // All at level 10, max_items=3
        let batch = extract_sub_batch(&mut frontier2, WavefrontConfig::Boundary, 3);
        assert_eq!(batch.len(), 3);
        assert_eq!(frontier2.len(), 7);
    }
}
