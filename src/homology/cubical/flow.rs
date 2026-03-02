// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Gradient flow operations for discrete Morse matching.
//!
//! This module provides implementations for `lower`, `colower`, `lift`, and
//! `colift` operations that flow chains through a discrete Morse matching.
//!
//! Two execution modes are provided:
//! - **Sequential**: Single-threaded wavefront iteration
//! - **Distributed** (requires `mpi` feature): MPI-based parallel matching
//!   computation with speculative precomputation
//!
//! # Distributed Execution
//!
//! When MPI is enabled with multiple processes, flow operations are computed
//! by the root process (rank 0) with worker processes computing orthant
//! matchings on demand. The result is broadcast to all processes before
//! returning, so all processes receive identical results.

mod sequential;

#[cfg(feature = "mpi")]
mod distributed;
#[cfg(feature = "mpi")]
mod frontier;

#[cfg(feature = "mpi")]
use mpi::traits::Communicator;
use sequential::{FlowCell, WavefrontConfig};

#[cfg(feature = "mpi")]
use crate::homology::cubical::ExecutionMode;
use crate::{
    Chain, Cube, Grader, Orthant, Ring, TopCubicalMatching, dispatch,
    homology::cubical::OrthantMatching,
};

/// Compute `lower`, dispatching to distributed if MPI is active.
///
/// Results may be inaccurate if the input chain contains cells outside the
/// grade range implied by the cap.
pub(crate) fn lower<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (Cube, R)>,
    min_grade: u32,
) -> Chain<u32, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    dispatch!(
        mpi => {
            if let ExecutionMode::Mpi { comm: Some(c) } = &matching.config().execution
                && c.size() > 1
            {
                return distributed::lower(matching, chain, min_grade, c);
            }
            sequential::lower(matching, chain, min_grade)
        },
        _ => {
            sequential::lower(matching, chain, min_grade)
        }
    )
}

/// Compute `colower`, dispatching to distributed if MPI is active.
///
/// Results may be inaccurate if the input chain contains cells outside the
/// grade range implied by the cap.
pub(crate) fn colower<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (Cube, R)>,
    max_grade: u32,
) -> Chain<u32, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    dispatch!(
        mpi => {
            if let ExecutionMode::Mpi { comm: Some(c) } = &matching.config().execution
                && c.size() > 1
            {
                return distributed::colower(matching, chain, max_grade, c);
            }
            sequential::colower(matching, chain, max_grade)
        },
        _ => {
            sequential::colower(matching, chain, max_grade)
        }
    )
}

/// Compute `lift`, dispatching to distributed if MPI is active.
///
/// Results may be inaccurate if the input chain contains cells outside the
/// grade range implied by the cap.
pub(crate) fn lift<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (u32, R)>,
    min_grade: u32,
) -> Chain<Cube, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    dispatch!(
        mpi => {
            if let ExecutionMode::Mpi { comm: Some(c) } = &matching.config().execution
                && c.size() > 1
            {
                return distributed::lift(matching, chain, min_grade, c);
            }
            sequential::lift(matching, chain, min_grade)
        },
        _ => {
            sequential::lift(matching, chain, min_grade)
        }
    )
}

/// Compute `colift`, dispatching to distributed if MPI is active.
///
/// Results may be inaccurate if the input chain contains cells outside the
/// grade range implied by the cap.
pub(crate) fn colift<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (u32, R)>,
    max_grade: u32,
) -> Chain<Cube, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    dispatch!(
        mpi => {
            if let ExecutionMode::Mpi { comm: Some(c) } = &matching.config().execution
                && c.size() > 1
            {
                return distributed::colift(matching, chain, max_grade, c);
            }
            sequential::colift(matching, chain, max_grade)
        },
        _ => {
            sequential::colift(matching, chain, max_grade)
        }
    )
}

/// Shared flow implementation used by both `Wavefront` and
/// `DistributedWavefront`.
///
/// Flows a chain through an orthant's matching, calling `push` for cells that
/// propagate to other orthants and `callback` for cells that should be yielded.
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
