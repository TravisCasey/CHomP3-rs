// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Gradient flow operations for discrete Morse matching.
//!
//! This module provides `lower`, `colower`, `lift`, and `colift` operations
//! that flow chains through a
//! [`TopCubicalMatching`](crate::TopCubicalMatching).
//!
//! # Architecture
//!
//! The implementation is split into two layers:
//!
//! - **Per-orthant flow** (this file): Given a single orthant's matching and a
//!   chain of cells within that orthant, processes each queen/king pair,
//!   emitting results and propagating boundary/coboundary contributions to
//!   neighboring orthants.
//!
//! - **Level-parallel orchestration** (`wavefront`): Maintains a frontier of
//!   orthants grouped by coordinate-sum level. Within each level, orthants are
//!   independent (the anti-chain property) and processed in parallel via
//!   [`ParallelMap`](crate::parallel::map::ParallelMap).

mod wavefront;

use wavefront::{FlowCell, WavefrontConfig};
pub(crate) use wavefront::{colift, colower, lift, lower};

use crate::{Chain, Orthant, Ring, homology::cubical::OrthantMatching};

/// Flow a chain through a single orthant's matching tree.
///
/// Walks the [`OrthantMatching`] tree and, for each matched pair encountered:
/// - Calls `push` with `(neighbor_orthant, extent, coef)` for cells whose
///   boundary/coboundary extends into an adjacent orthant.
/// - Calls `callback` with a [`FlowCell`] for cells the caller should collect
///   (aces, queens, or kings depending on the operation).
#[allow(clippy::too_many_arguments)]
fn flow_orthant_impl<R, P, F>(
    orthant: &Orthant,
    orthant_chain: Chain<u32, R>,
    matching: &OrthantMatching,
    config: WavefrontConfig,
    grade_cap: u32,
    min_orthant: &Orthant,
    max_orthant: &Orthant,
    push: &mut P,
    callback: &mut F,
) where
    R: Ring,
    P: FnMut(Orthant, u32, R),
    F: FnMut(FlowCell<R>),
{
    let mut orthant_chain = orthant_chain;
    flow_recursive(
        orthant,
        &mut orthant_chain,
        matching,
        config,
        grade_cap,
        min_orthant,
        max_orthant,
        push,
        callback,
    );
}

/// Recursively descend the [`OrthantMatching`] tree.
///
/// - **Critical**: Extract the ace coefficient from the chain and emit it.
/// - **Leaf**: Delegate to [`flow_leaf`] for queen/king pair processing.
/// - **Branch**: Descend into sub-matchings. Boundary mode processes
///   suborthants in reverse (high-to-low) so that queens are processed before
///   the kings they generate; coboundary mode processes in forward order.
#[allow(clippy::too_many_arguments)]
fn flow_recursive<R, P, F>(
    orthant: &Orthant,
    orthant_chain: &mut Chain<u32, R>,
    matching: &OrthantMatching,
    config: WavefrontConfig,
    grade_cap: u32,
    min_orthant: &Orthant,
    max_orthant: &Orthant,
    push: &mut P,
    callback: &mut F,
) where
    R: Ring,
    P: FnMut(Orthant, u32, R),
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
        OrthantMatching::Leaf { .. } => {
            flow_leaf(
                orthant,
                orthant_chain,
                matching,
                config,
                grade_cap,
                min_orthant,
                max_orthant,
                push,
                callback,
            );
        },
        OrthantMatching::Branch {
            suborthant_matchings,
            ..
        } => match config {
            WavefrontConfig::Boundary => {
                for sub_matching in suborthant_matchings.iter().rev() {
                    flow_recursive(
                        orthant,
                        orthant_chain,
                        sub_matching,
                        config,
                        grade_cap,
                        min_orthant,
                        max_orthant,
                        push,
                        callback,
                    );
                }
            },
            WavefrontConfig::Coboundary => {
                for sub_matching in suborthant_matchings {
                    flow_recursive(
                        orthant,
                        orthant_chain,
                        sub_matching,
                        config,
                        grade_cap,
                        min_orthant,
                        max_orthant,
                        push,
                        callback,
                    );
                }
            },
        },
    }
}

/// Process a leaf matching node: handle a single queen/king pair.
///
/// For each cell in the chain that belongs to this leaf's suborthant
/// (extent between `lower_extent` and `upper_extent`):
///
/// - **Boundary mode** (queen cell, grade at or above cap): Compute the matched
///   king, push its boundary faces to neighbors, add same-orthant boundary
///   contributions back to the chain, and emit the queen via callback.
///
/// - **Coboundary mode** (king cell, grade at or below cap): Compute the
///   matched queen, push its coboundary cofaces to neighbors, add same-orthant
///   coboundary contributions back to the chain, and emit the king via
///   callback.
#[allow(clippy::too_many_arguments)]
fn flow_leaf<R, P, F>(
    orthant: &Orthant,
    orthant_chain: &mut Chain<u32, R>,
    leaf: &OrthantMatching,
    config: WavefrontConfig,
    grade_cap: u32,
    min_orthant: &Orthant,
    max_orthant: &Orthant,
    push: &mut P,
    callback: &mut F,
) where
    R: Ring,
    P: FnMut(Orthant, u32, R),
    F: FnMut(FlowCell<R>),
{
    let OrthantMatching::Leaf {
        lower_extent,
        upper_extent,
        match_axis,
        grade,
    } = leaf
    else {
        unreachable!();
    };

    let axis_flag = 1 << match_axis;

    let cells: Vec<(u32, R)> = orthant_chain
        .iter()
        .filter(|(extent, _)| (lower_extent & !**extent) == 0 && (**extent & !upper_extent) == 0)
        .map(|(e, c)| (*e, c.clone()))
        .collect();

    for (extent, coef) in cells {
        orthant_chain.remove(&extent);
        if coef == R::zero() {
            continue;
        }

        let is_queen = extent & axis_flag == 0;
        let incidence = compute_incidence::<R>(extent, *match_axis);

        match config {
            WavefrontConfig::Boundary if is_queen && *grade >= grade_cap => {
                let king_extent = extent | axis_flag;
                let king_coef = coef * incidence;

                push_boundary_neighbors(orthant, king_extent, king_coef.clone(), max_orthant, push);
                add_boundary_same_orthant(
                    orthant_chain,
                    king_extent,
                    *lower_extent,
                    king_coef.clone(),
                    max_orthant.ambient_dimension() as usize,
                );

                callback(FlowCell::Queen {
                    king_extent,
                    coef: king_coef,
                });
            },
            WavefrontConfig::Coboundary if !is_queen && *grade <= grade_cap => {
                let queen_extent = extent ^ axis_flag;
                let queen_coef = coef * incidence;

                push_coboundary_neighbors(
                    orthant,
                    queen_extent,
                    queen_coef.clone(),
                    min_orthant,
                    push,
                );
                add_coboundary_same_orthant(
                    orthant_chain,
                    queen_extent,
                    *upper_extent,
                    queen_coef.clone(),
                    max_orthant.ambient_dimension() as usize,
                );

                callback(FlowCell::King {
                    queen_extent,
                    coef: queen_coef,
                });
            },
            _ => {},
        }
    }
}

/// Push boundary face contributions to neighboring orthants.
///
/// For each set bit in `extent`, the corresponding face (with that bit
/// cleared) may live in the neighboring orthant one step further along that
/// axis. Emits `(neighbor, face_extent, coef)` via `push` for faces within
/// the complex bounds.
fn push_boundary_neighbors<R: Ring>(
    orthant: &Orthant,
    extent: u32,
    coef: R,
    max_orthant: &Orthant,
    push: &mut impl FnMut(Orthant, u32, R),
) {
    let mut incidence = coef;
    let mut neighbor = orthant.clone();

    for axis in 0..orthant.ambient_dimension() as usize {
        let axis_flag = 1u32 << axis;
        if extent & axis_flag != 0 {
            let face_extent = extent ^ axis_flag;

            if orthant[axis] < max_orthant[axis] {
                neighbor[axis] += 1;
                push(neighbor.clone(), face_extent, incidence.clone());
                neighbor[axis] -= 1;
            }

            incidence = -incidence;
        }
    }
}

/// Push coboundary coface contributions to neighboring orthants.
///
/// For each cleared bit in `extent`, the corresponding coface (with that bit
/// set) may live in the neighboring orthant one step earlier along that axis.
/// Emits `(neighbor, coface_extent, coef)` via `push` for cofaces within the
/// complex bounds.
fn push_coboundary_neighbors<R: Ring>(
    orthant: &Orthant,
    extent: u32,
    coef: R,
    min_orthant: &Orthant,
    push: &mut impl FnMut(Orthant, u32, R),
) {
    let mut incidence = coef;
    let mut neighbor = orthant.clone();

    for axis in 0..orthant.ambient_dimension() as usize {
        let axis_flag = 1u32 << axis;
        if extent & axis_flag == 0 {
            let coface_extent = extent | axis_flag;

            if orthant[axis] > min_orthant[axis] {
                neighbor[axis] -= 1;
                push(neighbor.clone(), coface_extent, -incidence.clone());
                neighbor[axis] += 1;
            }

            incidence = -incidence;
        }
    }
}

/// Add boundary contributions that remain within the same orthant.
///
/// When a king cell's boundary face has its extent bit within `lower_extent`
/// (i.e., the face is owned by this orthant rather than a neighbor), the
/// face's coefficient is added back into the chain for further processing.
fn add_boundary_same_orthant<R: Ring>(
    orthant_chain: &mut Chain<u32, R>,
    extent: u32,
    lower_extent: u32,
    coef: R,
    ambient_dim: usize,
) {
    let mut incidence = coef;
    for axis in 0..ambient_dim {
        let axis_flag = 1u32 << axis;
        if extent & axis_flag != 0 {
            let face_extent = extent ^ axis_flag;
            if lower_extent & axis_flag != 0 {
                orthant_chain.insert_or_add(face_extent, -incidence.clone());
            }
            incidence = -incidence;
        }
    }
}

/// Add coboundary contributions that remain within the same orthant.
///
/// When a queen cell's coboundary coface has its extent bit within
/// `upper_extent` (i.e., the coface is owned by this orthant rather than a
/// neighbor), the coface's coefficient is added back into the chain for
/// further processing.
fn add_coboundary_same_orthant<R: Ring>(
    orthant_chain: &mut Chain<u32, R>,
    extent: u32,
    upper_extent: u32,
    coef: R,
    ambient_dim: usize,
) {
    let mut incidence = coef;
    for axis in 0..ambient_dim {
        let axis_flag = 1u32 << axis;
        if extent & axis_flag == 0 {
            let coface_extent = extent | axis_flag;
            if upper_extent & axis_flag == 0 {
                orthant_chain.insert_or_add(coface_extent, incidence.clone());
            }
            incidence = -incidence;
        }
    }
}

/// Compute the flow incidence coefficient for matching along `match_axis`.
///
/// Returns `(-1)^{p+1}` where `p` is the number of set bits in `extent`
/// below bit position `match_axis`. This equals `-[king:queen]` minus the
/// negation of the cellular boundary coefficient, which is the form
/// needed by the gradient flow formula. The cellular incidence `[king:queen]`
/// used in [`TopCubicalMatching::match_helper`](super::TopCubicalMatching)
/// is obtained by negating this value.
///
/// See the `sign_convention_consistency` test for verification, and
/// [`CubicalComplex::cell_boundary`](crate::CubicalComplex) for the
/// ground-truth boundary formula.
pub(super) fn compute_incidence<R: Ring>(extent: u32, match_axis: u32) -> R {
    if (extent % (1 << match_axis)).count_ones().is_multiple_of(2) {
        R::one()
    } else {
        -R::one()
    }
}
