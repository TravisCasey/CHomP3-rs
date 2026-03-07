// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Distributed wavefront implementation using MPI.
//!
//! This module provides MPI-based parallel implementations of the flow
//! operations (`lower`, `colower`, `lift`, `colift`). The root process (rank 0)
//! manages the wavefront traversal while worker processes compute orthant
//! matchings on demand. Speculative precomputation hides latency by computing
//! matchings for upcoming orthants while the root processes the current one.
//!
//! All operations broadcast their result to all processes before returning,
//! ensuring consistent state across the MPI communicator.

use mpi::traits::{Communicator, Destination, Source};
use postcard::{from_bytes, to_allocvec};
use tracing::info;

use super::{
    frontier::Frontier,
    sequential::{FlowCell, extent_to_cube},
};
use crate::{
    Chain, Complex, Cube, Grader, MorseMatching, Orthant, Ring, TopCubicalMatching,
    homology::cubical::{OrthantMatching, Subgrid},
    mpi::{MpiTag, SimpleCommunicator, broadcast},
};

/// Maximum number of precomputed matchings to store in the frontier.
const MAX_PRECOMPUTED_MATCHINGS: usize = 100_000;

/// Distributed wavefront with MPI-based matching precomputation.
pub(crate) struct DistributedWavefront<'a, R: Ring> {
    frontier: Frontier<R>,
    comm: &'a SimpleCommunicator,
    min_orthant: Orthant,
    max_orthant: Orthant,
    grade_cap: u32,
    next_worker: i32,
    num_workers: usize,
    label: &'static str,
    orthants_processed: usize,
}

impl<'a, R: Ring> DistributedWavefront<'a, R> {
    pub fn for_boundary(
        comm: &'a SimpleCommunicator,
        min_orthant: Orthant,
        max_orthant: Orthant,
        grade_cap: u32,
        label: &'static str,
    ) -> Self {
        Self::new(
            Frontier::for_boundary(),
            comm,
            min_orthant,
            max_orthant,
            grade_cap,
            label,
        )
    }

    pub fn for_coboundary(
        comm: &'a SimpleCommunicator,
        min_orthant: Orthant,
        max_orthant: Orthant,
        grade_cap: u32,
        label: &'static str,
    ) -> Self {
        Self::new(
            Frontier::for_coboundary(),
            comm,
            min_orthant,
            max_orthant,
            grade_cap,
            label,
        )
    }

    fn new(
        frontier: Frontier<R>,
        comm: &'a SimpleCommunicator,
        min_orthant: Orthant,
        max_orthant: Orthant,
        grade_cap: u32,
        label: &'static str,
    ) -> Self {
        let num_workers = usize::try_from(comm.size() - 1).expect("invalid worker count");
        Self {
            frontier,
            comm,
            min_orthant,
            max_orthant,
            grade_cap,
            next_worker: 1,
            num_workers,
            label,
            orthants_processed: 0,
        }
    }

    pub fn seed(&mut self, chain: impl IntoIterator<Item = (Cube, R)>) {
        self.frontier.seed(chain);
    }

    /// Pop the next orthant with its chain and matching.
    ///
    /// Handles all MPI coordination internally:
    /// 1. Collect ready matchings (non-blocking)
    /// 2. Speculative dispatch up to `MAX_PRECOMPUTED_MATCHINGS`
    /// 3. Ensure next orthant is ready (dispatch and block if needed)
    /// 4. Pop orthant
    /// 5. Priority dispatch the new next orthant
    pub fn pop_next(&mut self) -> Option<(Orthant, Chain<u32, R>, OrthantMatching)> {
        if self.frontier.is_empty() {
            return None;
        }

        // Collect ready matchings (non-blocking)
        self.collect_ready_matchings();

        // Speculative dispatch
        self.dispatch_speculative();

        // Ensure next orthant is ready
        if !self.frontier.next_is_ready() {
            let next = self.frontier.peek_next().unwrap().clone();

            if self.frontier.state(&next).unwrap().is_pending() {
                self.dispatch_orthant(&next);
            }

            self.block_for_matching(&next);
        }

        let result = self.frontier.pop_next();

        if result.is_some() {
            self.orthants_processed += 1;
            if self.orthants_processed.is_multiple_of(10000) {
                info!(
                    "{}: processed {} orthants (frontier: {}, precomputed: {})",
                    self.label,
                    self.orthants_processed,
                    self.frontier.len(),
                    self.frontier.precomputed_count(),
                );
            }
        }

        // Priority dispatch the new next orthant
        if let Some(next) = self.frontier.peek_next()
            && self.frontier.state(next).unwrap().is_pending()
        {
            let next = next.clone();
            self.dispatch_orthant(&next);
        }

        result
    }

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
            self.frontier.config(),
            self.grade_cap,
            &min_orthant,
            &max_orthant,
            &mut |o, e, c| self.frontier.push(o, e, c),
            &mut callback,
        );
    }

    pub fn finish(self) {
        for rank in 1..=self.num_workers as i32 {
            self.comm
                .process_at_rank(rank)
                .send_with_tag(&Vec::<u8>::new(), MpiTag::Shutdown as i32);
        }
    }

    fn dispatch_orthant(&mut self, orthant: &Orthant) {
        self.frontier.mark_dispatched(orthant);

        let encoded = to_allocvec(orthant).expect("failed to encode orthant");

        self.comm
            .process_at_rank(self.next_worker)
            .send_with_tag(&encoded, MpiTag::WorkAssignment as i32);

        self.next_worker = (self.next_worker % self.num_workers as i32) + 1;
    }

    fn dispatch_speculative(&mut self) {
        while self.frontier.precomputed_count() < MAX_PRECOMPUTED_MATCHINGS {
            if let Some(orthant) = self.frontier.next_pending() {
                let orthant = orthant.clone();
                self.dispatch_orthant(&orthant);
            } else {
                break;
            }
        }
    }

    fn collect_ready_matchings(&mut self) {
        while let Some(status) = self
            .comm
            .any_process()
            .immediate_probe_with_tag(MpiTag::ResultSubmission as i32)
        {
            self.receive_and_attach(status.source_rank());
        }
    }

    fn block_for_matching(&mut self, target: &Orthant) {
        loop {
            let (msg, _status) = self
                .comm
                .any_process()
                .receive_vec_with_tag(MpiTag::ResultSubmission as i32);

            let (orthant, matching): (Orthant, OrthantMatching) =
                from_bytes(&msg).expect("failed to decode matching result");

            self.frontier.attach_matching(&orthant, matching);

            if &orthant == target {
                break;
            }
        }
    }

    fn receive_and_attach(&mut self, source_rank: i32) {
        let (msg, _) = self.comm.process_at_rank(source_rank).receive_vec();

        let (orthant, matching): (Orthant, OrthantMatching) =
            from_bytes(&msg).expect("failed to decode matching result");

        self.frontier.attach_matching(&orthant, matching);
    }
}

/// Run as a matching worker process.
pub(crate) fn run_as_worker<R, G>(matching: &TopCubicalMatching<R, G>, comm: &SimpleCommunicator)
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    let complex = matching.upper_complex();
    let mut subgrid = Subgrid::new(complex, u32::MAX, u32::MAX);

    loop {
        let (msg, status) = comm.process_at_rank(0).receive_vec();
        let tag = status.tag();

        if tag == MpiTag::Shutdown as i32 {
            break;
        }

        assert_eq!(tag, MpiTag::WorkAssignment as i32, "unexpected tag {tag}");

        let orthant: Orthant = from_bytes(&msg).expect("failed to decode orthant");

        let orthant_matching = subgrid.match_subgrid(orthant.clone(), orthant.clone())[0]
            .2
            .clone();

        let result = to_allocvec(&(orthant, orthant_matching)).expect("failed to encode result");

        comm.process_at_rank(0)
            .send_with_tag(&result, MpiTag::ResultSubmission as i32);
    }
}

pub(super) fn lower<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (Cube, R)>,
    min_grade: u32,
    comm: &SimpleCommunicator,
) -> Chain<u32, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    let result = if comm.rank() == 0 {
        let complex = matching.upper_complex();
        let mut wavefront = DistributedWavefront::for_boundary(
            comm,
            complex.minimum().clone(),
            complex.maximum().clone(),
            min_grade,
            "lower",
        );
        let mut result = Chain::new();

        wavefront.seed(chain);

        while let Some((orthant, orthant_chain, orthant_matching)) = wavefront.pop_next() {
            wavefront.flow_orthant(&orthant, orthant_chain, &orthant_matching, |cell| {
                if let FlowCell::Ace { extent, coef } = cell {
                    let cube = extent_to_cube(&orthant, extent);
                    if let Some(idx) = matching.project_cell(&cube) {
                        result.insert_or_add(idx, coef);
                    }
                }
            });
        }

        wavefront.finish();
        result
    } else {
        run_as_worker(matching, comm);
        Chain::new()
    };

    broadcast(comm, &result)
}

pub(super) fn colower<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (Cube, R)>,
    max_grade: u32,
    comm: &SimpleCommunicator,
) -> Chain<u32, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    let result = if comm.rank() == 0 {
        let complex = matching.upper_complex();
        let mut wavefront = DistributedWavefront::for_coboundary(
            comm,
            complex.minimum().clone(),
            complex.maximum().clone(),
            max_grade,
            "colower",
        );
        let mut result = Chain::new();

        wavefront.seed(chain);

        while let Some((orthant, orthant_chain, orthant_matching)) = wavefront.pop_next() {
            wavefront.flow_orthant(&orthant, orthant_chain, &orthant_matching, |cell| {
                if let FlowCell::Ace { extent, coef } = cell {
                    let cube = extent_to_cube(&orthant, extent);
                    if let Some(idx) = matching.project_cell(&cube) {
                        result.insert_or_add(idx, coef);
                    }
                }
            });
        }

        wavefront.finish();
        result
    } else {
        run_as_worker(matching, comm);
        Chain::new()
    };

    broadcast(comm, &result)
}

pub(super) fn lift<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (u32, R)>,
    min_grade: u32,
    comm: &SimpleCommunicator,
) -> Chain<Cube, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    let result = if comm.rank() == 0 {
        let complex = matching.upper_complex();
        let mut wavefront = DistributedWavefront::for_boundary(
            comm,
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

        while let Some((orthant, orthant_chain, orthant_matching)) = wavefront.pop_next() {
            wavefront.flow_orthant(&orthant, orthant_chain, &orthant_matching, |cell| {
                if let FlowCell::Queen { king_extent, coef } = cell {
                    let king_cube = extent_to_cube(&orthant, king_extent);
                    result.insert_or_add(king_cube, coef);
                }
            });
        }

        wavefront.finish();
        result
    } else {
        run_as_worker(matching, comm);
        Chain::new()
    };

    broadcast(comm, &result)
}

pub(super) fn colift<R, G>(
    matching: &TopCubicalMatching<R, G>,
    chain: impl IntoIterator<Item = (u32, R)>,
    max_grade: u32,
    comm: &SimpleCommunicator,
) -> Chain<Cube, R>
where
    R: Ring,
    G: Grader<Orthant> + Clone,
{
    let result = if comm.rank() == 0 {
        let complex = matching.upper_complex();
        let mut wavefront = DistributedWavefront::for_coboundary(
            comm,
            complex.minimum().clone(),
            complex.maximum().clone(),
            max_grade,
            "colift",
        );

        let mut result = Chain::new();
        for (idx, coef) in chain {
            result.insert_or_add(matching.include_cell(idx), coef);
        }
        wavefront.seed(complex.coboundary(&result));

        while let Some((orthant, orthant_chain, orthant_matching)) = wavefront.pop_next() {
            wavefront.flow_orthant(&orthant, orthant_chain, &orthant_matching, |cell| {
                if let FlowCell::King { queen_extent, coef } = cell {
                    let queen_cube = extent_to_cube(&orthant, queen_extent);
                    result.insert_or_add(queen_cube, coef);
                }
            });
        }

        wavefront.finish();
        result
    } else {
        run_as_worker(matching, comm);
        Chain::new()
    };

    broadcast(comm, &result)
}
