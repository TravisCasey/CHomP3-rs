// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Discrete Morse theory for homology computation.
//!
//! This module provides tools for computing homology of cell complexes using
//! discrete Morse theory. The core idea is to reduce a cell complex to a
//! smaller *Morse complex* that has the same homology, making computation
//! tractable for large complexes.
//!
//! # Overview
//!
//! Discrete Morse theory partitions cells into three categories based on an
//! acyclic partial matching:
//!
//! - **Ace (critical) cells**: Unmatched cells that survive the reduction and
//!   form the Morse complex.
//! - **Queen cells**: Cells matched to a king cell of one higher dimension.
//! - **King cells**: Cells matched to a queen cell of one lower dimension.
//!
//! The matched pairs (queen-king) can be "collapsed" without changing the
//! homology, leaving only the critical cells. This is encapsulated in the
//! [`CellMatch`] enum and computed by types implementing [`MorseMatching`].
//!
//! # Example
//!
//! Reducing a simple cell complex:
//!
//! ```
//! use chomp3rs::{
//!     CellComplex, CoreductionMatching, Cyclic, HashMapModule, ModuleLike,
//!     MorseMatching, RingLike,
//! };
//!
//! // Create a line segment: two vertices and one edge
//! type Module = HashMapModule<u32, Cyclic<2>>;
//!
//! let mut boundaries = vec![Module::new(), Module::new(), Module::new()];
//! boundaries[2].insert_or_add(1, Cyclic::one());
//! boundaries[2].insert_or_add(0, -Cyclic::one());
//!
//! let mut coboundaries = vec![Module::new(), Module::new(), Module::new()];
//! coboundaries[0].insert_or_add(2, -Cyclic::one());
//! coboundaries[1].insert_or_add(2, Cyclic::one());
//!
//! let complex = CellComplex::new(
//!     vec![0, 0, 1], // dimensions
//!     vec![0, 0, 0], // grades
//!     boundaries,
//!     coboundaries,
//! );
//!
//! // Use CoreductionMatching to reduce the complex
//! let mut matching = CoreductionMatching::new();
//! matching.compute_matching(complex);
//!
//! // Line segment is contractible: reduces to a single point
//! assert_eq!(matching.critical_cells().len(), 1);
//! ```
//!
//! # References
//!
//! For theoretical background, see:
//! - Forman, R. *A user's guide to discrete Morse theory*.
//! - Harker, Mischaikow, Mrozek, and Nanda, *Discrete Morse Theoretic
//!   Algorithms for Computing Homology of Complexes and Maps*.

use std::{cmp::Reverse, collections::BinaryHeap};

pub use coreduction::CoreductionMatching;
pub use cubical::{TopCubicalMatching, TopCubicalMatchingBuilder};
pub use morse::CellMatch;

use crate::{CellComplex, ComplexLike, Grader, ModuleLike, RingLike};

mod coreduction;
pub mod cubical;
mod linked_list;
mod morse;

/// The interface for computing an acyclic partial matching for discrete Morse
/// theoretic cell complex reductions.
///
/// An acyclic partial matching partitions cells into three categories based on
/// the trichotomy induced by a discrete Morse function:
///
/// - **Queen cells**: Matched "upward" to exactly one king cell (a coface with
///   lesser Morse function value).
/// - **King cells**: Matched "downward" to exactly one queen cell (a face with
///   greater Morse function value).
/// - **Ace (critical) cells**: Unmatched cells that form the reduced Morse
///   complex.
///
/// This trichotomy is encapsulated in [`CellMatch`]. Types implementing this
/// trait compute the matching and provide:
/// - Critical cells via [`MorseMatching::critical_cells`]
/// - Boundary operators via [`MorseMatching::boundary_and_coboundary`]
/// - The reduced Morse complex via [`MorseMatching::construct_morse_complex`]
///
/// # Workflow
///
/// 1. Create a matching instance (e.g., `CoreductionMatching::new()`)
/// 2. Call [`compute_matching`](MorseMatching::compute_matching) with the
///    complex
/// 3. Use [`construct_morse_complex`](MorseMatching::construct_morse_complex)
///    to get the reduced complex
/// 4. Optionally, use [`full_reduce`](MorseMatching::full_reduce) to iterate
///    until no further reduction is possible
///
/// # Panics
///
/// Most methods panic if called before
/// [`compute_matching`](MorseMatching::compute_matching). The panic message
/// will indicate that `compute_matching` must be called first.
///
/// # References
///
/// For an involved treatment of this approach, see Harker, Mischaikow, Mrozek,
/// and Nanda, *Discrete Morse Theoretic Algorithms for Computing Homology of
/// Complexes and Maps*.
pub trait MorseMatching
where
    Self: Sized,
{
    /// Cell type of the parent cell complex.
    ///
    /// Must be equivalent to `<Self::UpperComplex as ComplexLike>::Cell` and
    /// `<Self::UpperModule as ModuleLike>::Cell`.
    type UpperCell: Clone + Eq;

    /// Coefficient ring type of the complex.
    ///
    /// Must be equivalent to `<Self::UpperComplex as ComplexLike>::Ring` and
    /// `<Self::UpperModule as ModuleLike>::Ring`.
    type Ring: RingLike;

    /// Module type for (co)chains in the parent cell complex.
    ///
    /// Must be equivalent to `<Self::UpperComplex as ComplexLike>::Module`.
    type UpperModule: ModuleLike<Cell = Self::UpperCell, Ring = Self::Ring>;

    /// Module type for (co)chains in the reduced Morse complex.
    ///
    /// The cell type of the Morse complex is always `u32`, representing indices
    /// into the critical cells vector. The ring type matches the parent
    /// complex.
    type LowerModule: ModuleLike<Cell = u32, Ring = Self::Ring>;

    /// The parent cell complex type on which the matching is performed.
    type UpperComplex: ComplexLike<Cell = Self::UpperCell, Ring = Self::Ring, Module = Self::UpperModule>;

    /// Priority type for ordering cells during (co)lowering and (co)lifting.
    ///
    /// The priority can dramatically improve efficiency by processing cells in
    /// an intelligent order. Specifically, a queen cell `q` matched to king
    /// cell `k` should have priority less than or equal to the queen cells in
    /// the boundary of `k`.
    ///
    /// See [`CellMatch`] and [`match_cell`](MorseMatching::match_cell) for
    /// details.
    type Priority: Ord;

    /// Compute an acyclic partial matching on the given cell complex.
    ///
    /// This method takes ownership of `complex` and determines the critical
    /// (ace) cells, preparing for construction of the reduced Morse complex.
    ///
    /// After calling this method, other methods like
    /// [`construct_morse_complex`](MorseMatching::construct_morse_complex) and
    /// [`critical_cells`](MorseMatching::critical_cells) become available.
    fn compute_matching(&mut self, complex: Self::UpperComplex);

    /// Construct the Morse complex from this acyclic partial matching.
    ///
    /// The resulting [`CellComplex`] has one cell (of type `u32`) for each
    /// critical cell found in the matching. These cells correspond to indices
    /// into the vector returned by
    /// [`critical_cells`](MorseMatching::critical_cells).
    ///
    /// # Panics
    ///
    /// Panics if [`compute_matching`](MorseMatching::compute_matching) has not
    /// been called previously.
    ///
    /// # References
    ///
    /// For theoretical background, see Forman, *A user's guide to discrete
    /// Morse theory*.
    #[must_use]
    fn construct_morse_complex(&self) -> CellComplex<Self::LowerModule> {
        let upper = self
            .get_upper_complex()
            .unwrap_or_else(|| panic!("matching not yet computed; call compute_matching first"));

        let critical_cells = self.critical_cells();
        let cell_dimensions = critical_cells
            .iter()
            .map(|cell| upper.cell_dimension(cell))
            .collect();
        let grades = critical_cells
            .iter()
            .map(|cell| upper.grade(cell))
            .collect();
        let (boundaries, coboundaries) = self.boundary_and_coboundary();
        CellComplex::new(cell_dimensions, grades, boundaries, coboundaries)
    }

    /// Fully reduce `complex` via discrete Morse theory until no further
    /// reduction is possible.
    ///
    /// This method first applies `self` to `complex`, then repeatedly applies
    /// clones of `generic_matching` to the resulting Morse complexes until the
    /// cell count no longer decreases.
    ///
    /// # Matching Types
    ///
    /// The first matching (`self`) can exploit special structure in `complex`
    /// (e.g., [`TopCubicalMatching`] for cubical complexes). However, the
    /// resulting Morse complex is a general [`CellComplex`], so subsequent
    /// matchings use a general-purpose algorithm like [`CoreductionMatching`].
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - A vector of the matchings performed after the first step
    /// - The final Morse complex (no longer reducible via Morse matchings)
    fn full_reduce<PM>(
        &mut self,
        generic_matching: PM,
        complex: Self::UpperComplex,
    ) -> (Vec<PM>, CellComplex<Self::LowerModule>)
    where
        PM: MorseMatching<
                UpperCell = u32,
                Ring = Self::Ring,
                UpperModule = Self::LowerModule,
                LowerModule = Self::LowerModule,
                UpperComplex = CellComplex<Self::LowerModule>,
            > + Clone,
    {
        self.compute_matching(complex);
        let mut morse_complex = Some(self.construct_morse_complex());
        let mut cell_count = morse_complex.as_ref().unwrap().cell_count();
        let mut further_matchings = Vec::new();

        loop {
            let mut next_matching = generic_matching.clone();
            next_matching.compute_matching(morse_complex.take().unwrap());
            if next_matching.critical_cells().len() as u32 == cell_count {
                return (
                    further_matchings,
                    next_matching.take_upper_complex().unwrap(),
                );
            }
            morse_complex = Some(next_matching.construct_morse_complex());
            cell_count = morse_complex.as_ref().unwrap().cell_count();
            further_matchings.push(next_matching);
        }
    }

    /// Return a reference to the parent cell complex.
    ///
    /// Returns `Some` if [`compute_matching`](MorseMatching::compute_matching)
    /// has been called, `None` otherwise.
    fn get_upper_complex(&self) -> Option<&Self::UpperComplex>;

    /// Consume the matching and return the parent cell complex.
    ///
    /// Returns `Some` if [`compute_matching`](MorseMatching::compute_matching)
    /// has been called, `None` otherwise.
    fn take_upper_complex(self) -> Option<Self::UpperComplex>;

    /// Return the critical (ace) cells found by the matching algorithm.
    ///
    /// These are the cells that form the reduced Morse complex. The order of
    /// cells in the returned vector defines their indices in the Morse complex.
    #[must_use]
    fn critical_cells(&self) -> Vec<Self::UpperCell>;

    /// Project a cell from the parent complex to the Morse complex.
    ///
    /// If `cell` is a critical (ace) cell, returns its index in the Morse
    /// complex. Otherwise, returns `None`.
    ///
    /// The returned index is consistent with
    /// [`critical_cells`](MorseMatching::critical_cells).
    #[must_use]
    fn project_cell(&self, cell: Self::UpperCell) -> Option<u32>;

    /// Project a chain from the parent complex onto its critical cells.
    ///
    /// This is *not* a chain map; it simply discards non-critical cells. For
    /// a chain map that commutes with boundary operators, use
    /// [`lower`](MorseMatching::lower) or [`colower`](MorseMatching::colower).
    ///
    /// This method is primarily for translating between module/cell types.
    fn project(&self, chain: Self::UpperModule) -> Self::LowerModule {
        let mut projected_chain = Self::LowerModule::new();
        for (cell, coefficient) in chain {
            if let Some(projected_cell) = self.project_cell(cell) {
                projected_chain.insert_or_add(projected_cell, coefficient);
            }
        }
        projected_chain
    }

    /// Include a Morse complex cell into the parent complex.
    ///
    /// Returns the critical cell in the parent complex corresponding to the
    /// given Morse complex cell index.
    ///
    /// Equivalent to `self.critical_cells()[cell as usize]`.
    #[must_use]
    fn include_cell(&self, cell: u32) -> Self::UpperCell;

    /// Include a chain from the Morse complex into the parent complex.
    ///
    /// This is **not** a chain map; it simply maps each Morse cell to its
    /// corresponding critical cell. For a chain map that commutes with
    /// boundary operators, use [`lift`](MorseMatching::lift) or
    /// [`colift`](MorseMatching::colift).
    ///
    /// This method is primarily for translating between module/cell types.
    fn include(&self, chain: Self::LowerModule) -> Self::UpperModule {
        let mut included_chain = Self::UpperModule::new();
        for (cell, coefficient) in chain {
            included_chain.insert_or_add(self.include_cell(cell), coefficient);
        }
        included_chain
    }

    /// Compute the boundary and coboundary of each critical cell in the Morse
    /// complex.
    ///
    /// # Returns
    ///
    /// A tuple of two vectors:
    /// - First: boundaries of each critical cell
    /// - Second: coboundaries of each critical cell
    ///
    /// The order matches [`critical_cells`](MorseMatching::critical_cells).
    ///
    /// # Panics
    ///
    /// Panics if [`compute_matching`](MorseMatching::compute_matching) has not
    /// been called.
    fn boundary_and_coboundary(&self) -> (Vec<Self::LowerModule>, Vec<Self::LowerModule>) {
        let upper = self
            .get_upper_complex()
            .unwrap_or_else(|| panic!("matching not yet computed; call compute_matching first"));

        let critical_cells = self.critical_cells();
        let mut boundaries = Vec::with_capacity(critical_cells.len());
        let mut coboundaries = Vec::with_capacity(critical_cells.len());

        for cell in &critical_cells {
            boundaries.push(self.lower(upper.cell_boundary(cell)));
            coboundaries.push(self.lower(upper.cell_coboundary(cell)));
        }

        (boundaries, coboundaries)
    }

    /// Query the match status of a cell in the parent complex.
    ///
    /// Returns a [`CellMatch`] indicating whether `cell` is a king, queen,
    /// or ace (critical), along with match details.
    ///
    /// # Priority
    ///
    /// The priority in the returned [`CellMatch`] affects efficiency of
    /// (co)lowering and (co)lifting. For optimal performance, a queen cell `q`
    /// matched to king `k` should have priority less than or equal to the
    /// queens in the boundary of `k`.
    #[must_use]
    fn match_cell(
        &self,
        cell: &Self::UpperCell,
    ) -> CellMatch<Self::UpperCell, Self::Ring, Self::Priority>;

    /// Lower a chain from the parent complex to the Morse complex.
    ///
    /// This is a **chain map**: it commutes with boundary operators and maps
    /// cycles to cycles. The resulting chain has support only on critical
    /// cells.
    ///
    /// # Panics
    ///
    /// Panics if [`compute_matching`](MorseMatching::compute_matching) has not
    /// been called.
    fn lower(&self, chain: Self::UpperModule) -> Self::LowerModule {
        let upper = self
            .get_upper_complex()
            .unwrap_or_else(|| panic!("matching not yet computed; call compute_matching first"));

        // Remaining queens and coefficients to be eliminated
        let mut queen_chain = Self::UpperModule::new();
        // Result chain with only critical (ace) cells
        let mut lowered_chain = Self::LowerModule::new();
        // Current chain being processed
        let mut boundary_chain = chain;

        // Min heap via Reverse(_)
        let mut queen_queue = BinaryHeap::new();
        loop {
            // Categorize cells as queens or aces
            for (cell, coef) in boundary_chain {
                let match_result = self.match_cell(&cell);
                match match_result {
                    CellMatch::Queen { .. } => {
                        queen_chain.insert_or_add(cell, coef);
                        queen_queue.push(Reverse(match_result));
                    },
                    CellMatch::Ace { .. } => {
                        lowered_chain.insert_or_add(
                            self.project_cell(cell)
                                .expect("project_cell returned None on critical cell"),
                            coef,
                        );
                    },
                    CellMatch::King { .. } => (),
                }
            }

            // Find next queen with nonzero coefficient
            if let Some(Reverse(CellMatch::Queen {
                cell,
                king,
                incidence,
                ..
            })) = pop_until(&mut queen_queue, |item| {
                if let CellMatch::Queen { cell, .. } = &item.0 {
                    return queen_chain.coefficient(cell) != Self::Ring::zero();
                }
                panic!("Queen queue populated with non-queen cell");
            }) {
                let queen_coefficient = queen_chain.coefficient(&cell);
                let cancel_coefficient = -queen_coefficient * incidence.invert();
                boundary_chain = upper.cell_boundary(&king).scalar_mul(cancel_coefficient);
            } else {
                break;
            }
        }

        debug_assert_eq!(queen_chain, Self::UpperModule::new());
        lowered_chain
    }

    /// Lower a single cell from the parent complex to the Morse complex.
    ///
    /// Convenience method equivalent to `self.lower(singleton_chain(cell, 1))`.
    fn lower_cell(&self, cell: Self::UpperCell) -> Self::LowerModule {
        let mut cell_chain = Self::UpperModule::new();
        cell_chain.insert_or_add(cell, Self::Ring::one());
        self.lower(cell_chain)
    }

    /// Lift a chain from the Morse complex to the parent complex.
    ///
    /// This is a **chain map**: it commutes with boundary operators and maps
    /// cycles to cycles.
    ///
    /// # Panics
    ///
    /// Panics if [`compute_matching`](MorseMatching::compute_matching) has not
    /// been called.
    fn lift(&self, chain: Self::LowerModule) -> Self::UpperModule {
        let upper = self
            .get_upper_complex()
            .unwrap_or_else(|| panic!("matching not yet computed; call compute_matching first"));

        // Remaining queens and coefficients to be eliminated
        let mut queen_chain = Self::UpperModule::new();
        // Result chain in the upper complex
        let mut lifted_chain = self.include(chain);
        // Current boundary being processed
        let mut boundary_chain = upper.boundary(&lifted_chain);

        // Min heap via Reverse(_)
        let mut queen_queue = BinaryHeap::new();
        loop {
            // Boundary cells which are queens need further propagation
            for (cell, coef) in boundary_chain {
                let match_result = self.match_cell(&cell);
                if let CellMatch::Queen { .. } = match_result {
                    queen_chain.insert_or_add(cell, coef);
                    queen_queue.push(Reverse(match_result));
                }
            }

            // Find next queen with nonzero coefficient
            if let Some(Reverse(CellMatch::Queen {
                cell,
                king,
                incidence,
                ..
            })) = pop_until(&mut queen_queue, |item| {
                if let CellMatch::Queen { cell, .. } = &item.0 {
                    return queen_chain.coefficient(cell) != Self::Ring::zero();
                }
                panic!("Queen queue populated with non-queen cell");
            }) {
                let queen_coefficient = queen_chain.coefficient(&cell);
                let cancel_coefficient = -queen_coefficient * incidence.invert();
                boundary_chain = upper
                    .cell_boundary(&king)
                    .scalar_mul(cancel_coefficient.clone());
                lifted_chain.insert_or_add(king.clone(), cancel_coefficient);
            } else {
                break;
            }
        }

        debug_assert_eq!(queen_chain, Self::UpperModule::new());
        lifted_chain
    }

    /// Lift a single Morse complex cell to the parent complex.
    ///
    /// Convenience method equivalent to `self.lift(singleton_chain(cell, 1))`.
    fn lift_cell(&self, cell: u32) -> Self::UpperModule {
        let mut cell_chain = Self::LowerModule::new();
        cell_chain.insert_or_add(cell, Self::Ring::one());
        self.lift(cell_chain)
    }

    /// Lower a cochain from the parent complex to the Morse complex.
    ///
    /// This is a **cochain map**: it commutes with coboundary operators and
    /// maps cocycles to cocycles. Queens and kings are reversed compared to
    /// [`lower`](MorseMatching::lower).
    ///
    /// # Panics
    ///
    /// Panics if [`compute_matching`](MorseMatching::compute_matching) has not
    /// been called.
    fn colower(&self, cochain: Self::UpperModule) -> Self::LowerModule {
        let upper = self
            .get_upper_complex()
            .unwrap_or_else(|| panic!("matching not yet computed; call compute_matching first"));

        // Remaining kings to be eliminated
        let mut king_cochain = Self::UpperModule::new();
        // Result cochain with only critical cells
        let mut colowered_cochain = Self::LowerModule::new();
        // Current cochain being processed
        let mut coboundary_cochain = cochain.clone();

        // Max heap
        let mut king_queue = BinaryHeap::new();
        loop {
            // Categorize cells as kings or aces
            for (cell, coef) in coboundary_cochain {
                let match_result = self.match_cell(&cell);
                match match_result {
                    CellMatch::King { .. } => {
                        king_cochain.insert_or_add(cell, coef);
                        king_queue.push(match_result);
                    },
                    CellMatch::Ace { .. } => {
                        colowered_cochain.insert_or_add(
                            self.project_cell(cell)
                                .expect("project_cell returned None on critical cell"),
                            coef,
                        );
                    },
                    CellMatch::Queen { .. } => (),
                }
            }

            // Find next king with nonzero coefficient
            if let Some(CellMatch::King {
                cell,
                queen,
                incidence,
                ..
            }) = pop_until(&mut king_queue, |item| {
                if let CellMatch::King { cell, .. } = &item {
                    return king_cochain.coefficient(cell) != Self::Ring::zero();
                }
                panic!("King queue populated with non-king cell");
            }) {
                let king_coefficient = king_cochain.coefficient(&cell);
                let cancel_coefficient = -king_coefficient * incidence.invert();
                coboundary_cochain = upper.cell_coboundary(&queen).scalar_mul(cancel_coefficient);
            } else {
                break;
            }
        }

        debug_assert_eq!(king_cochain, Self::UpperModule::new());
        colowered_cochain
    }

    /// Lower a single cell as a cochain from the parent complex.
    ///
    /// Convenience method equivalent to `self.colower(singleton_cochain(cell,
    /// 1))`.
    fn colower_cell(&self, cell: Self::UpperCell) -> Self::LowerModule {
        let mut cell_cochain = Self::UpperModule::new();
        cell_cochain.insert_or_add(cell, Self::Ring::one());
        self.colower(cell_cochain)
    }

    /// Lift a cochain from the Morse complex to the parent complex.
    ///
    /// This is a **cochain map**: it commutes with coboundary operators and
    /// maps cocycles to cocycles. Queens and kings are reversed compared to
    /// [`lift`](MorseMatching::lift).
    ///
    /// # Panics
    ///
    /// Panics if [`compute_matching`](MorseMatching::compute_matching) has not
    /// been called.
    fn colift(&self, cochain: Self::LowerModule) -> Self::UpperModule {
        let upper = self
            .get_upper_complex()
            .unwrap_or_else(|| panic!("matching not yet computed; call compute_matching first"));

        // Remaining kings to be eliminated
        let mut king_cochain = Self::UpperModule::new();
        // Result cochain in the upper complex
        let mut colifted_cochain = self.include(cochain);
        // Current coboundary being processed
        let mut coboundary_cochain = upper.coboundary(&colifted_cochain);

        // Max heap
        let mut king_queue = BinaryHeap::new();
        loop {
            // Coboundary cells which are kings need further propagation
            for (cell, coef) in coboundary_cochain {
                let match_result = self.match_cell(&cell);
                if let CellMatch::King { .. } = match_result {
                    king_cochain.insert_or_add(cell, coef);
                    king_queue.push(match_result);
                }
            }

            // Find next king with nonzero coefficient
            if let Some(CellMatch::King {
                cell,
                queen,
                incidence,
                ..
            }) = pop_until(&mut king_queue, |item| {
                if let CellMatch::King { cell, .. } = &item {
                    return king_cochain.coefficient(cell) != Self::Ring::zero();
                }
                panic!("King queue populated with non-king cell");
            }) {
                let king_coefficient = king_cochain.coefficient(&cell);
                let cancel_coefficient = -king_coefficient * incidence.invert();
                coboundary_cochain = upper
                    .cell_coboundary(&queen)
                    .scalar_mul(cancel_coefficient.clone());
                colifted_cochain.insert_or_add(queen.clone(), cancel_coefficient);
            } else {
                break;
            }
        }

        debug_assert_eq!(king_cochain, Self::UpperModule::new());
        colifted_cochain
    }

    /// Lift a single Morse complex cell as a cochain to the parent complex.
    ///
    /// Convenience method equivalent to `self.colift(singleton_cochain(cell,
    /// 1))`.
    fn colift_cell(&self, cell: u32) -> Self::UpperModule {
        let mut cell_cochain = Self::LowerModule::new();
        cell_cochain.insert_or_add(cell, Self::Ring::one());
        self.colift(cell_cochain)
    }
}

/// Pop elements from `heap` until `predicate` is satisfied or heap is empty.
fn pop_until<T: Ord>(heap: &mut BinaryHeap<T>, predicate: impl Fn(&T) -> bool) -> Option<T> {
    while let Some(item) = heap.pop() {
        if predicate(&item) {
            return Some(item);
        }
    }
    None
}
