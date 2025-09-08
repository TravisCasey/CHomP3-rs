// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::{ComplexLike, MatchResult, ModuleLike, RingLike};

/// The interface for a type implementing an acyclic partial matching used in
/// discrete Morse theoretic cell complex reductions.
///
/// An acyclic partial matching is a more computable surrogate for the discrete
/// Morse function used to reduce chain complexes. A partial matching uses the
/// trichotomy induced by the Morse function to match `Queen` cells (those cells
/// having exactly one coface of less Morse function value) to `King` cells
/// (those cells having exactly one face of greater Morse function value). Cells
/// that are neither kings nor queens, referred to as `Ace` cells or critical
/// cells, are left unmatched. This trichotomy is encapsulated in the
/// [`MatchResult`] enum.
///
/// This trichotomy is used to define a Morse cell complex only on the ace
/// (critical) cells. Types implementing this trait compute the matching and
/// provide the critical cells (via the [`MorseMatching::critical_cells`]
/// method) as well as their new (co)boundary operators (via the
/// [`MorseMatching::boundary_and_coboundary`] method) to construct the Morse
/// cell complex.
///
/// For an involved treatment of this approach see Harker, Mischaikow, Mrozek,
/// and Nanda, *Discrete Morse Theoretic Algorithms for Computing Homology of
/// Complexes and Maps*.
pub trait MorseMatching {
    /// Cell type of the parent cell complex; must be equivalent to
    /// `<Self::Complex as ComplexLike>::Cell` and `<Self::Module as
    /// ModuleLike>::Cell`.
    type UpperCell: Clone + Eq;
    /// Coefficient ring type of the complex, and must thus be equivalent to
    /// `<Self::Complex as ComplexLike>::Ring` and `<Self::Module as
    /// ModuleLike>::Ring`.
    type Ring: RingLike;
    /// Module type used to represent (co)chains in the parent cell complex;
    /// must be equivalent to `<Self::Complex as ComplexLike>::Module`.
    type UpperModule: ModuleLike<Cell = Self::UpperCell, Ring = Self::Ring>;
    /// Module type used to represent (co)chains in the reduced Morse complex
    /// formed from this partial matching. The cell type of the Morse complex is
    /// `u32`, with the ring type being the same as the parent complex.
    type LowerModule: ModuleLike<Cell = u32, Ring = Self::Ring>;
    /// The parent cell complex type on which the matching is performed.
    type UpperComplex: ComplexLike<Cell = Self::UpperCell, Ring = Self::Ring, Module = Self::UpperModule>;
    /// The priority type used to order cells for efficient (co)lowering and
    /// (co)lifting operations and related operations. See [`MatchResult`]
    /// and [`MorseMatching::match_cell`] for details.
    type Priority: Ord;

    /// Compute an acyclic partial matching on the given cell complex,
    /// determining the critical (ace) cells and preparing for construction of
    /// the reduced Morse complex.
    fn compute_matching(complex: Self::UpperComplex) -> Self;

    /// Return an immutable reference to the owned parent cell complex.
    fn get_complex(&self) -> &Self::UpperComplex;

    /// Return the critical cells found by the matching algorithm. These are
    /// primarily used to construct the reduced Morse complex.
    fn critical_cells(&self) -> Vec<Self::UpperCell>;

    /// If `cell` is a critical cell of the parent complex, return its
    /// represenative as a cell in the Morse complex. Else, return `None`.
    ///
    /// The representative must also be consistent with its index into the
    /// returned vector of [`MorseMatching::critical_cells`].
    fn project_cell(&self, cell: Self::UpperCell) -> Option<u32>;

    /// A provided method to project a chain from the parent cell complex onto
    /// its critical cells in the Morse complex.
    ///
    /// The map implemented by this method is not a chain map; see the
    /// [`MorseMatching::lower`] and [`MorseMatching::colower`] methods for
    /// a chain map used for representing chains of the parent complex in the
    /// Morse complex. The intention of this method is largely to translate from
    /// the cell and module types of the parent cell complex to that of the
    /// Morse complex.
    fn project(&self, chain: Self::UpperModule) -> Self::LowerModule {
        let mut projected_chain = Self::LowerModule::new();
        for (cell, coefficient) in chain.into_iter() {
            if let Some(projected_cell) = self.project_cell(cell) {
                projected_chain.insert_or_add(projected_cell, coefficient);
            }
        }
        projected_chain
    }

    /// For the cell `cell` in the Morse complex, get the corresponding critical
    /// cell in the parent cell complex.
    ///
    /// This must be equivalent to `matching.critical_cell[cell as usize]`,
    /// where `matchings` is the type implementing [`MorseMatching`].
    fn include_cell(&self, cell: u32) -> Self::UpperCell;

    /// A provided method to include `chain` from the Morse complex into the
    /// parent cell complex, by mapping each cell of the Morse complex to
    /// the corresponding critical cell in the parent cell complex.
    ///
    /// The map implemented by this method is not a chain map; see the
    /// [`MorseMatching::lift`] and [`MorseMatching::colift`] methods for a
    /// chain map used for representing chains of the Morse complex in the
    /// parent cell complex. The intention of this method is largely to
    /// translate the cell and module types of the Morse complex to that of
    /// the parent cell complex.
    fn include(&self, chain: Self::LowerModule) -> Self::UpperModule {
        let mut included_chain = Self::UpperModule::new();
        for (cell, coefficient) in chain.into_iter() {
            included_chain.insert_or_add(self.include_cell(cell), coefficient);
        }
        included_chain
    }

    /// Compute and return the boundary and coboundary of each critical cell,
    /// projected to the Morse complex.
    ///
    /// The first vector contains the boundaries while the second contains the
    /// coboundaries. The order of the (co)boundaries must be consistent with
    /// the returned vector of [`MorseMatching::critical_cells`].
    fn boundary_and_coboundary(&self) -> (Vec<Self::LowerModule>, Vec<Self::LowerModule>) {
        let critical_cells = self.critical_cells();
        let mut boundaries = Vec::with_capacity(critical_cells.len());
        let mut coboundaries = Vec::with_capacity(critical_cells.len());

        for cell in critical_cells.iter() {
            boundaries.push(self.lower(self.get_complex().cell_boundary(cell)));
            coboundaries.push(self.lower(self.get_complex().cell_coboundary(cell)));
        }

        (boundaries, coboundaries)
    }

    /// Query the match of a `cell` in the parent cell complex.
    ///
    /// The returned type holds the cell it matches to, and the category of
    /// `cell` (whether it is a queen, king, or critical (ace) cell).
    ///
    /// The priority of the match can dramatically improve the efficiency of the
    /// provided (co)lowering and (co)lifting algorithms and related operations
    /// by processing cells in an intelligent order. Specifically, a queen cell
    /// `q` matched to the king cell `k` should (but is not required to) have
    /// priority less than or equal to (in its implementation of `Ord`) the
    /// queen cells in the boundary of `k`.
    fn match_cell(
        &self,
        cell: &Self::UpperCell,
    ) -> &MatchResult<Self::UpperCell, Self::Ring, Self::Priority>;

    /// Lower `chain` from the parent cell complex to its representative in the
    /// Morse complex having support solely on critical cells.
    ///
    /// This is a chain map between the parent cell complex and the Morse
    /// complex, and thus it commutes with the boundary operators and maps
    /// cycles to cycles.
    fn lower(&self, chain: Self::UpperModule) -> Self::LowerModule {
        // Remaining queens and coefficients to be eliminated. Iteration in this
        // method ceases once this chain is empty.
        let mut queen_chain = Self::UpperModule::new();
        // Exists in the Morse complex, has only critical (ace) cells.
        let mut lowered_chain = Self::LowerModule::new();
        // Each queen maps to its king, and the boundary of the king is stored
        // in this chain until it is split into queen_chain and lowered_chain.
        let mut boundary_chain = chain;

        // Using Reverse(_) for min heap
        let mut queen_queue = BinaryHeap::new();
        loop {
            // Categorize cells as queens or aces; aces are in the final chain,
            // queens need to be propagated further, and kings are ignored.
            for (cell, coef) in boundary_chain {
                let match_result = self.match_cell(&cell);
                match match_result {
                    MatchResult::Queen { .. } => {
                        queen_chain.insert_or_add(cell, coef);
                        queen_queue.push(Reverse(match_result));
                    }
                    MatchResult::Ace { .. } => {
                        lowered_chain.insert_or_add(
                            self.project_cell(cell)
                                .expect("project_cell returned None on critical cell"),
                            coef,
                        );
                    }
                    MatchResult::King { .. } => (),
                };
            }

            // Find next queen cell with nonzero coefficient
            if let Some(Reverse(MatchResult::Queen {
                cell,
                king,
                incidence,
                ..
            })) = pop_until(&mut queen_queue, |item| {
                if let MatchResult::Queen { cell, .. } = &item.0 {
                    return queen_chain.coef(cell) != Self::Ring::zero();
                }
                panic!("Queen queue populated with non-queen cell");
            }) {
                let queen_coef = queen_chain.coef(cell);

                // The coefficient which cancels the original queen when the
                // boundary of its king is added to queen_chain.
                let cancel_coef = -queen_coef * incidence.invert();

                // Compute the scaled boundary of the king
                boundary_chain = self
                    .get_complex()
                    .cell_boundary(king)
                    .scalar_mul(cancel_coef);
            } else {
                break;
            }
        }

        debug_assert_eq!(queen_chain, Self::UpperModule::new());

        lowered_chain
    }

    /// Provided convenience method for finding the representative `chain` of
    /// `cell` from the parent cell complex in the Morse complex.
    ///
    /// `cell` is treated as a singleton chain with a coefficient of one.
    fn lower_cell(&self, cell: Self::UpperCell) -> Self::LowerModule {
        let mut cell_chain = Self::UpperModule::new();
        cell_chain.insert_or_add(cell, Self::Ring::one());
        self.lower(cell_chain)
    }

    /// Lift `chain` from the Morse complex to the parent cell complex.
    ///
    /// This is a chain map between the parent cell complex and the Morse
    /// complex, and thus it commutes with the boundary operators and maps
    /// cycles to cycles.
    fn lift(&self, chain: Self::LowerModule) -> Self::UpperModule {
        // Remaining queens and coefficients to be eliminated. Iteration in this
        // method ceases once this chain is empty.
        let mut queen_chain = Self::UpperModule::new();
        // The representative of `chain` in the upper complex.
        let mut lifted_chain = self.include(chain);
        // Each queen maps to its king, and the boundary of the king is stored
        // in this chain; the queens of this chain are propagated further.
        let mut boundary_chain = self.get_complex().boundary(&lifted_chain);

        // Using Reverse(_) for min heap
        let mut queen_queue = BinaryHeap::new();
        loop {
            // Boundary cells which are queens are added to queen_chain and
            // queen_queue to propagate further.
            for (cell, coef) in boundary_chain {
                let match_result = self.match_cell(&cell);
                if let MatchResult::Queen { .. } = match_result {
                    queen_chain.insert_or_add(cell, coef);
                    queen_queue.push(Reverse(match_result));
                }
            }

            // Find next queen cell with nonzero coefficient
            if let Some(Reverse(MatchResult::Queen {
                cell,
                king,
                incidence,
                ..
            })) = pop_until(&mut queen_queue, |item| {
                if let MatchResult::Queen { cell, .. } = &item.0 {
                    return queen_chain.coef(cell) != Self::Ring::zero();
                }
                panic!("Queen queue populated with non-queen cell");
            }) {
                let queen_coef = queen_chain.coef(cell);

                // The coefficient which cancels the original queen when the
                // boundary of its king is added to queen_chain.
                let cancel_coef = -queen_coef * incidence.invert();

                // Compute the scaled boundary of the king
                boundary_chain = self
                    .get_complex()
                    .cell_boundary(king)
                    .scalar_mul(cancel_coef.clone());
                lifted_chain.insert_or_add(king.clone(), cancel_coef);
            } else {
                break;
            }
        }

        debug_assert_eq!(queen_chain, Self::UpperModule::new());

        lifted_chain
    }

    /// Provided convenience method for finding the representative `chain` of
    /// `cell` from the Morse complex in the parent cell complex.
    ///
    /// `cell` is treated as a singleton chain with a coefficient of one.
    fn lift_cell(&self, cell: u32) -> Self::UpperModule {
        let mut cell_chain = Self::LowerModule::new();
        cell_chain.insert_or_add(cell, Self::Ring::one());
        self.lift(cell_chain)
    }

    /// Lower `cochain` from the parent cell complex to its representative in
    /// the Morse complex having support solely on critical cells.
    ///
    /// This is a cochain map between the parent cell complex and the Morse
    /// complex, and thus it commutes with the coboundary operators and maps
    /// cocycles to cocycles. Queens and kings are reversed compared to the
    /// `lower` method.
    fn colower(&self, cochain: Self::UpperModule) -> Self::LowerModule {
        // Remaining kings and coefficients to be eliminated. Iteration in this
        // method ceases once this cochain is empty.
        let mut king_cochain = Self::UpperModule::new();
        // Exists in the Morse complex, has only critical (ace) cells.
        let mut colowered_cochain = Self::LowerModule::new();
        // Each king maps to its queen, and the coboundary of the queen is stored
        // in this cochain until it is split into king_cochain and colowered_cochain.
        let mut coboundary_cochain = cochain.clone();

        // Using max heap
        let mut king_queue = BinaryHeap::new();
        loop {
            // Categorize cells as kings or aces; aces are in the final cochain,
            // kings need to be propagated further, and queens are ignored.
            for (cell, coef) in coboundary_cochain {
                let match_result = self.match_cell(&cell);
                match match_result {
                    MatchResult::King { .. } => {
                        king_cochain.insert_or_add(cell, coef);
                        king_queue.push(match_result);
                    }
                    MatchResult::Ace { .. } => {
                        colowered_cochain.insert_or_add(
                            self.project_cell(cell)
                                .expect("project_cell returned None on critical cell"),
                            coef,
                        );
                    }
                    MatchResult::Queen { .. } => (),
                };
            }

            // Find next king cell with nonzero coefficient
            if let Some(MatchResult::King {
                cell,
                queen,
                incidence,
                ..
            }) = pop_until(&mut king_queue, |item| {
                if let MatchResult::King { cell, .. } = &item {
                    return king_cochain.coef(cell) != Self::Ring::zero();
                }
                panic!("King queue populated with non-king cell");
            }) {
                let king_coef = king_cochain.coef(cell);

                // The coefficient which cancels the original king when the
                // coboundary of its queen is added to king_cochain.
                let cancel_coef = -king_coef * incidence.invert();

                // Compute the scaled coboundary of the queen
                coboundary_cochain = self
                    .get_complex()
                    .cell_coboundary(queen)
                    .scalar_mul(cancel_coef);
            } else {
                break;
            }
        }

        debug_assert_eq!(king_cochain, Self::UpperModule::new());

        colowered_cochain
    }

    /// Provided convenience method for finding the representative `cochain` of
    /// `cell` from the parent cell complex in the Morse complex.
    ///
    /// `cell` is treated as a singleton cochain with a coefficient of one.
    fn colower_cell(&self, cell: Self::UpperCell) -> Self::LowerModule {
        let mut cell_cochain = Self::UpperModule::new();
        cell_cochain.insert_or_add(cell, Self::Ring::one());
        self.colower(cell_cochain)
    }

    /// Lift `cochain` from the Morse complex to the parent cell complex.
    ///
    /// This is a cochain map between the parent cell complex and the Morse
    /// complex, and thus it commutes with the coboundary operators and maps
    /// cocycles to cocycles. Queens and kings are reversed compared to the
    /// `lift` method.
    fn colift(&self, cochain: Self::LowerModule) -> Self::UpperModule {
        // Remaining kings and coefficients to be eliminated. Iteration in this
        // method ceases once this cochain is empty.
        let mut king_cochain = Self::UpperModule::new();
        // The representative of `cochain` in the upper complex.
        let mut colifted_cochain = self.include(cochain);
        // Each king maps to its queen, and the coboundary of the queen is stored
        // in this cochain; the kings of this cochain are propagated further.
        let mut coboundary_cochain = self.get_complex().coboundary(&colifted_cochain);

        // Using max heap
        let mut king_queue = BinaryHeap::new();
        loop {
            // Coboundary cells which are kings are added to king_cochain and
            // king_queue to propagate further.
            for (cell, coef) in coboundary_cochain {
                let match_result = self.match_cell(&cell);
                if let MatchResult::King { .. } = match_result {
                    king_cochain.insert_or_add(cell, coef);
                    king_queue.push(match_result);
                }
            }

            // Find next king cell with nonzero coefficient
            if let Some(MatchResult::King {
                cell,
                queen,
                incidence,
                ..
            }) = pop_until(&mut king_queue, |item| {
                if let MatchResult::King { cell, .. } = &item {
                    return king_cochain.coef(cell) != Self::Ring::zero();
                }
                panic!("King queue populated with non-king cell");
            }) {
                let king_coef = king_cochain.coef(cell);

                // The coefficient which cancels the original king when the
                // coboundary of its queen is added to king_cochain.
                let cancel_coef = -king_coef * incidence.invert();

                // Compute the scaled coboundary of the queen
                coboundary_cochain = self
                    .get_complex()
                    .cell_coboundary(queen)
                    .scalar_mul(cancel_coef.clone());
                colifted_cochain.insert_or_add(queen.clone(), cancel_coef);
            } else {
                break;
            }
        }

        debug_assert_eq!(king_cochain, Self::UpperModule::new());

        colifted_cochain
    }

    /// Provided convenience method for colifting the representative `cochain`
    /// of `cell` from the Morse complex to the parent cell complex.
    ///
    /// `cell` is treated as a singleton cochain with a coefficient of one.
    fn colift_cell(&self, cell: u32) -> Self::UpperModule {
        let mut cell_cochain = Self::LowerModule::new();
        cell_cochain.insert_or_add(cell, Self::Ring::one());
        self.colift(cell_cochain)
    }
}

/// Helper function to pop elements from `heap` until `predicate` is satisfied.
fn pop_until<T: Ord>(heap: &mut BinaryHeap<T>, predicate: impl Fn(&T) -> bool) -> Option<T> {
    while let Some(item) = heap.pop() {
        if predicate(&item) {
            return Some(item);
        }
    }
    None
}
