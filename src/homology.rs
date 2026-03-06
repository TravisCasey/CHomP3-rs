// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

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
//!     CellComplex, Chain, CoreductionMatching, F2, MorseMatching, Ring,
//! };
//!
//! // Create a line segment: two vertices and one edge
//! let mut boundaries: Vec<Chain<u32, F2>> = vec![Chain::new(); 3];
//! boundaries[2].insert_or_add(1, F2::one());
//! boundaries[2].insert_or_add(0, -F2::one());
//!
//! let mut coboundaries: Vec<Chain<u32, F2>> = vec![Chain::new(); 3];
//! coboundaries[0].insert_or_add(2, -F2::one());
//! coboundaries[1].insert_or_add(2, F2::one());
//!
//! let complex = CellComplex::new(
//!     vec![0, 0, 1], // dimensions
//!     vec![0, 0, 0], // grades
//!     boundaries,
//!     coboundaries,
//! );
//!
//! // Use CoreductionMatching to reduce the complex
//! let matching: CoreductionMatching<_> = CoreductionMatching::new(complex);
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

use std::{cmp::Reverse, collections::BinaryHeap, fmt::Debug, hash::Hash};

pub use coreduction::CoreductionMatching;
pub use cubical::{TopCubicalMatching, TopCubicalMatchingBuilder};
pub use morse::CellMatch;

use crate::{CellComplex, Chain, Complex, Grader, Ring};

mod coreduction;
pub mod cubical;
mod linked_list;
mod morse;

// Macro to generate `full_reduce` with cfg-gated serde bounds on `PM`.
// When `serde` is enabled, `PM` must be serializable so that specialized
// overrides can broadcast results across processes.
macro_rules! define_full_reduce {
    ($($bound:path),* $(,)?) => {
        /// Fully reduce `self` via discrete Morse theory until no further
        /// reduction is possible.
        ///
        /// This method builds the Morse complex from `self`, then repeatedly
        /// applies `factory` to the resulting Morse complexes until the cell
        /// count no longer decreases.
        ///
        /// # Matching Types
        ///
        /// The first matching (`self`) can exploit special structure in the
        /// original complex (e.g., [`TopCubicalMatching`] for cubical
        /// complexes). However, the resulting Morse complex is a general
        /// [`CellComplex`], so subsequent matchings use a general-purpose
        /// algorithm. Pass [`CoreductionMatching::new`] (the function itself,
        /// not a call to it) as `factory` for this purpose.
        ///
        /// # Returns
        ///
        /// A tuple of:
        /// - A vector of the matchings performed after the first step
        /// - The final Morse complex (no longer reducible via Morse matchings)
        fn full_reduce<PM>(
            &self,
            factory: impl Fn(CellComplex<Self::Ring>) -> PM,
        ) -> (Vec<PM>, CellComplex<Self::Ring>)
        where
            PM: MorseMatching<
                    UpperCell = u32,
                    Ring = Self::Ring,
                    UpperComplex = CellComplex<Self::Ring>,
                >
                $(+ $bound)*,
        {
            let morse_complex = self.construct_morse_complex();
            full_reduce_sequential(factory, morse_complex)
        }
    };
}

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
/// trait compute the matching at construction time and provide:
/// - Critical cells via [`MorseMatching::critical_cells`]
/// - Boundary operators via [`MorseMatching::boundary_and_coboundary`]
/// - The reduced Morse complex via [`MorseMatching::construct_morse_complex`]
///
/// # Workflow
///
/// 1. Construct the matching with the complex (e.g.,
///    `CoreductionMatching::new(complex)`); computation happens here.
/// 2. Use the resulting matching to query critical cells, build the Morse
///    complex, or further reduce with
///    [`full_reduce`](MorseMatching::full_reduce).
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
    /// Must be equivalent to `<Self::UpperComplex as Complex>::Cell`.
    type UpperCell: Clone + Debug + Eq + Hash;

    /// Coefficient ring type of the complex.
    ///
    /// Must be equivalent to `<Self::UpperComplex as Complex>::Ring`.
    type Ring: Ring;

    /// The parent cell complex type on which the matching is performed.
    type UpperComplex: Complex<Cell = Self::UpperCell, Ring = Self::Ring>;

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

    /// Construct the Morse complex from this acyclic partial matching.
    ///
    /// The resulting [`CellComplex`] has one cell (of type `u32`) for each
    /// critical cell found in the matching. These cells correspond to indices
    /// into the slice returned by
    /// [`critical_cells`](MorseMatching::critical_cells).
    ///
    /// # References
    ///
    /// For theoretical background, see Forman, *A user's guide to discrete
    /// Morse theory*.
    #[must_use]
    fn construct_morse_complex(&self) -> CellComplex<Self::Ring> {
        let upper = self.upper_complex();

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

    // The `full_reduce` default method requires serde bounds on `PM` when
    // the `serde` feature is enabled, because specialized overrides (e.g.,
    // `TopCubicalMatching`) may broadcast `PM` across MPI processes. A macro
    // avoids duplicating the documentation between the two cfg variants.
    #[cfg(not(feature = "serde"))]
    define_full_reduce!();
    #[cfg(feature = "serde")]
    define_full_reduce!(serde::Serialize, serde::de::DeserializeOwned);

    /// Return a reference to the parent cell complex.
    #[must_use]
    fn upper_complex(&self) -> &Self::UpperComplex;

    /// Consume the matching and return the parent cell complex.
    #[must_use]
    fn into_upper_complex(self) -> Self::UpperComplex;

    /// Return the critical (ace) cells found by the matching algorithm.
    ///
    /// These are the cells that form the reduced Morse complex. The order of
    /// cells in the returned slice defines their indices in the Morse complex.
    #[must_use]
    fn critical_cells(&self) -> &[Self::UpperCell];

    /// Project a cell from the parent complex to the Morse complex.
    ///
    /// If `cell` is a critical (ace) cell, returns its index in the Morse
    /// complex. Otherwise, returns `None`.
    ///
    /// The returned index is consistent with
    /// [`critical_cells`](MorseMatching::critical_cells).
    #[must_use]
    fn project_cell(&self, cell: &Self::UpperCell) -> Option<u32>;

    /// Project a chain from the parent complex onto its critical cells.
    ///
    /// This is *not* a chain map; it simply discards non-critical cells. For
    /// a chain map that commutes with boundary operators, use
    /// [`lower`](MorseMatching::lower) or [`colower`](MorseMatching::colower).
    ///
    /// This method is primarily for translating between module/cell types.
    fn project(
        &self,
        chain: impl IntoIterator<Item = (Self::UpperCell, Self::Ring)>,
    ) -> Chain<u32, Self::Ring> {
        let mut projected_chain = Chain::new();
        for (cell, coefficient) in chain {
            if let Some(projected_cell) = self.project_cell(&cell) {
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
    fn include(
        &self,
        chain: impl IntoIterator<Item = (u32, Self::Ring)>,
    ) -> Chain<Self::UpperCell, Self::Ring> {
        let mut included_chain = Chain::new();
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
    #[allow(clippy::type_complexity)]
    fn boundary_and_coboundary(
        &self,
    ) -> (Vec<Chain<u32, Self::Ring>>, Vec<Chain<u32, Self::Ring>>) {
        let upper = self.upper_complex();
        let critical_cells = self.critical_cells();

        let mut boundaries = Vec::with_capacity(critical_cells.len());
        for cell in critical_cells {
            boundaries.push(self.lower(upper.cell_boundary(cell)));
        }

        // Coboundaries are the transpose of the boundary matrix
        let mut coboundaries: Vec<Chain<u32, Self::Ring>> =
            (0..critical_cells.len()).map(|_| Chain::new()).collect();
        for (cell, boundary) in boundaries.iter().enumerate() {
            for (bd_cell, coef) in boundary {
                *coboundaries[*bd_cell as usize].coefficient_mut(&(cell as u32)) = coef.clone();
            }
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
    fn lower(
        &self,
        chain: impl IntoIterator<Item = (Self::UpperCell, Self::Ring)>,
    ) -> Chain<u32, Self::Ring> {
        let upper = self.upper_complex();

        // Remaining queens and coefficients to be eliminated
        let mut queen_chain = Chain::<Self::UpperCell, Self::Ring>::new();
        // Result chain with only critical (ace) cells
        let mut lowered_chain = Chain::<u32, Self::Ring>::new();

        // Min heap via Reverse(_)
        let mut queen_queue = BinaryHeap::new();

        // Categorize initial input
        for (cell, coef) in chain {
            let match_result = self.match_cell(&cell);
            match match_result {
                CellMatch::Queen { .. } => {
                    queen_chain.insert_or_add(cell, coef);
                    queen_queue.push(Reverse(match_result));
                },
                CellMatch::Ace { .. } => {
                    lowered_chain.insert_or_add(
                        self.project_cell(&cell)
                            .expect("project_cell returned None on critical cell"),
                        coef,
                    );
                },
                CellMatch::King { .. } => (),
            }
        }

        // Eliminate queens
        while let Some(Reverse(CellMatch::Queen {
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
            // Incidence between matched cells must be invertible
            let cancel_coefficient = -queen_coefficient * incidence.invert().unwrap();
            for (cell, coef) in upper.cell_boundary(&king).scalar_mul(&cancel_coefficient) {
                let match_result = self.match_cell(&cell);
                match match_result {
                    CellMatch::Queen { .. } => {
                        queen_chain.insert_or_add(cell, coef);
                        queen_queue.push(Reverse(match_result));
                    },
                    CellMatch::Ace { .. } => {
                        lowered_chain.insert_or_add(
                            self.project_cell(&cell)
                                .expect("project_cell returned None on critical cell"),
                            coef,
                        );
                    },
                    CellMatch::King { .. } => (),
                }
            }
        }

        debug_assert_eq!(queen_chain, Chain::new());
        lowered_chain
    }

    /// Lower a single cell from the parent complex to the Morse complex.
    ///
    /// Convenience method equivalent to lowering a singleton chain with unit
    /// coefficient.
    fn lower_cell(&self, cell: Self::UpperCell) -> Chain<u32, Self::Ring> {
        self.lower([(cell, Self::Ring::one())])
    }

    /// Lift a chain from the Morse complex to the parent complex.
    ///
    /// This is a **chain map**: it commutes with boundary operators and maps
    /// cycles to cycles.
    fn lift(
        &self,
        chain: impl IntoIterator<Item = (u32, Self::Ring)>,
    ) -> Chain<Self::UpperCell, Self::Ring> {
        let upper = self.upper_complex();

        // Remaining queens and coefficients to be eliminated
        let mut queen_chain = Chain::<Self::UpperCell, Self::Ring>::new();
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
                // Incidence between matched cells must be invertible
                let cancel_coefficient = -queen_coefficient * incidence.invert().unwrap();
                boundary_chain = upper.cell_boundary(&king).scalar_mul(&cancel_coefficient);
                lifted_chain.insert_or_add(king.clone(), cancel_coefficient);
            } else {
                break;
            }
        }

        debug_assert_eq!(queen_chain, Chain::new());
        lifted_chain
    }

    /// Lift a single Morse complex cell to the parent complex.
    ///
    /// Convenience method equivalent to lifting a singleton chain with unit
    /// coefficient.
    fn lift_cell(&self, cell: u32) -> Chain<Self::UpperCell, Self::Ring> {
        self.lift([(cell, Self::Ring::one())])
    }

    /// Lower a cochain from the parent complex to the Morse complex.
    ///
    /// This is a **cochain map**: it commutes with coboundary operators and
    /// maps cocycles to cocycles. Queens and kings are reversed compared to
    /// [`lower`](MorseMatching::lower).
    fn colower(
        &self,
        cochain: impl IntoIterator<Item = (Self::UpperCell, Self::Ring)>,
    ) -> Chain<u32, Self::Ring> {
        let upper = self.upper_complex();

        // Remaining kings to be eliminated
        let mut king_cochain = Chain::<Self::UpperCell, Self::Ring>::new();
        // Result cochain with only critical cells
        let mut colowered_cochain = Chain::<u32, Self::Ring>::new();

        // Max heap
        let mut king_queue = BinaryHeap::new();

        // Categorize initial input
        for (cell, coef) in cochain {
            let match_result = self.match_cell(&cell);
            match match_result {
                CellMatch::King { .. } => {
                    king_cochain.insert_or_add(cell, coef);
                    king_queue.push(match_result);
                },
                CellMatch::Ace { .. } => {
                    colowered_cochain.insert_or_add(
                        self.project_cell(&cell)
                            .expect("project_cell returned None on critical cell"),
                        coef,
                    );
                },
                CellMatch::Queen { .. } => (),
            }
        }

        // Eliminate kings
        while let Some(CellMatch::King {
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
            // Incidence between matched cells must be invertible
            let cancel_coefficient = -king_coefficient * incidence.invert().unwrap();
            for (cell, coef) in upper
                .cell_coboundary(&queen)
                .scalar_mul(&cancel_coefficient)
            {
                let match_result = self.match_cell(&cell);
                match match_result {
                    CellMatch::King { .. } => {
                        king_cochain.insert_or_add(cell, coef);
                        king_queue.push(match_result);
                    },
                    CellMatch::Ace { .. } => {
                        colowered_cochain.insert_or_add(
                            self.project_cell(&cell)
                                .expect("project_cell returned None on critical cell"),
                            coef,
                        );
                    },
                    CellMatch::Queen { .. } => (),
                }
            }
        }

        debug_assert_eq!(king_cochain, Chain::new());
        colowered_cochain
    }

    /// Lower a single cell as a cochain from the parent complex.
    ///
    /// Convenience method equivalent to colowering a singleton cochain with
    /// unit coefficient.
    fn colower_cell(&self, cell: Self::UpperCell) -> Chain<u32, Self::Ring> {
        self.colower([(cell, Self::Ring::one())])
    }

    /// Lift a cochain from the Morse complex to the parent complex.
    ///
    /// This is a **cochain map**: it commutes with coboundary operators and
    /// maps cocycles to cocycles. Queens and kings are reversed compared to
    /// [`lift`](MorseMatching::lift).
    fn colift(
        &self,
        cochain: impl IntoIterator<Item = (u32, Self::Ring)>,
    ) -> Chain<Self::UpperCell, Self::Ring> {
        let upper = self.upper_complex();

        // Remaining kings to be eliminated
        let mut king_cochain = Chain::<Self::UpperCell, Self::Ring>::new();
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
                // Incidence between matched cells must be invertible
                let cancel_coefficient = -king_coefficient * incidence.invert().unwrap();
                coboundary_cochain = upper
                    .cell_coboundary(&queen)
                    .scalar_mul(&cancel_coefficient);
                colifted_cochain.insert_or_add(queen.clone(), cancel_coefficient);
            } else {
                break;
            }
        }

        debug_assert_eq!(king_cochain, Chain::new());
        colifted_cochain
    }

    /// Lift a single Morse complex cell as a cochain to the parent complex.
    ///
    /// Convenience method equivalent to colifting a singleton cochain with
    /// unit coefficient.
    fn colift_cell(&self, cell: u32) -> Chain<Self::UpperCell, Self::Ring> {
        self.colift([(cell, Self::Ring::one())])
    }

    /// Lower a chain, including only cells at or above `min_grade`.
    ///
    /// This is an optimization for computing chain representatives in
    /// superlevel sets. The default implementation filters the output of
    /// [`lower`]; types with specialized matching structures may override
    /// for efficiency.
    ///
    /// [`lower`]: MorseMatching::lower
    fn lower_capped(
        &self,
        chain: impl IntoIterator<Item = (Self::UpperCell, Self::Ring)>,
        min_grade: u32,
    ) -> Chain<u32, Self::Ring> {
        let upper = self.upper_complex();
        self.lower(chain)
            .into_iter()
            .filter(|(cell, _)| upper.grade(&self.include_cell(*cell)) >= min_grade)
            .collect()
    }

    /// Lift a chain, including only cells at or above `min_grade`.
    ///
    /// This is an optimization for computing chain representatives in
    /// superlevel sets. The default implementation filters the output of
    /// [`lift`]; types with specialized matching structures may override
    /// for efficiency.
    ///
    /// [`lift`]: MorseMatching::lift
    fn lift_capped(
        &self,
        chain: impl IntoIterator<Item = (u32, Self::Ring)>,
        min_grade: u32,
    ) -> Chain<Self::UpperCell, Self::Ring> {
        let upper = self.upper_complex();
        self.lift(chain)
            .into_iter()
            .filter(|(cell, _)| upper.grade(cell) >= min_grade)
            .collect()
    }

    /// Lower a cochain, including only cells at or below `max_grade`.
    ///
    /// This is an optimization for computing cochain representatives in
    /// sublevel sets. The default implementation filters the output of
    /// [`colower`]; types with specialized matching structures may override
    /// for efficiency.
    ///
    /// [`colower`]: MorseMatching::colower
    fn colower_capped(
        &self,
        cochain: impl IntoIterator<Item = (Self::UpperCell, Self::Ring)>,
        max_grade: u32,
    ) -> Chain<u32, Self::Ring> {
        let upper = self.upper_complex();
        self.colower(cochain)
            .into_iter()
            .filter(|(cell, _)| upper.grade(&self.include_cell(*cell)) <= max_grade)
            .collect()
    }

    /// Lift a cochain, including only cells at or below `max_grade`.
    ///
    /// This is an optimization for computing cochain representatives in
    /// sublevel sets. The default implementation filters the output of
    /// [`colift`]; types with specialized matching structures may override
    /// for efficiency.
    ///
    /// [`colift`]: MorseMatching::colift
    fn colift_capped(
        &self,
        cochain: impl IntoIterator<Item = (u32, Self::Ring)>,
        max_grade: u32,
    ) -> Chain<Self::UpperCell, Self::Ring> {
        let upper = self.upper_complex();
        self.colift(cochain)
            .into_iter()
            .filter(|(cell, _)| upper.grade(cell) <= max_grade)
            .collect()
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

/// Sequential implementation of full reduction via discrete Morse theory.
///
/// This is the core logic used by both the default
/// [`MorseMatching::full_reduce`] implementation and specialized overrides that
/// need a sequential fallback.
pub(crate) fn full_reduce_sequential<R, PM>(
    factory: impl Fn(CellComplex<R>) -> PM,
    mut morse_complex: CellComplex<R>,
) -> (Vec<PM>, CellComplex<R>)
where
    R: Ring,
    PM: MorseMatching<UpperCell = u32, Ring = R, UpperComplex = CellComplex<R>>,
{
    let mut cell_count = morse_complex.cell_count();
    let mut further_matchings = Vec::new();

    loop {
        let next_matching = factory(morse_complex);
        if next_matching.critical_cells().len() as u32 == cell_count {
            return (further_matchings, next_matching.into_upper_complex());
        }
        morse_complex = next_matching.construct_morse_complex();
        cell_count = morse_complex.cell_count();
        further_matchings.push(next_matching);
    }
}
