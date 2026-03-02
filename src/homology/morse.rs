// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Cell classification types for discrete Morse theory.
//!
//! This module defines [`CellMatch`], which represents the classification of
//! a cell in an acyclic partial matching. Every cell falls into one of three
//! categories:
//!
//! - **King**: Matched "downward" to a queen cell of one lower dimension
//! - **Queen**: Matched "upward" to a king cell of one higher dimension
//! - **Ace**: Unmatched (critical) cell that survives in the Morse complex

use std::cmp::Ordering;

/// The classification of a cell in an acyclic partial matching.
///
/// This enum is returned by
/// [`MorseMatching::match_cell`](super::MorseMatching::match_cell)
/// and indicates whether a cell is matched (as a king or queen) or unmatched
/// (an ace/critical cell).
///
/// # Type Parameters
///
/// - `T`: The cell type
/// - `R`: The coefficient ring type (for incidence values)
/// - `P`: The priority type for ordering (see below)
///
/// # Priority and Ordering
///
/// The priority type `P` enables efficient (co)lowering and (co)lifting
/// operations by allowing cells to be processed in an optimal order. For best
/// performance, a queen cell `q` matched to king `k` should have priority less
/// than or equal to the queen cells in the boundary of `k`.
///
/// The [`Ord`] implementation uses the convention `King > Ace > Queen` when
/// comparing across variants. Within the same variant, comparison uses the
/// priority field. This ordering is designed for use with
/// [`BinaryHeap`](std::collections::BinaryHeap).
///
/// # Examples
///
/// Pattern matching on a `CellMatch`:
///
/// ```
/// use chomp3rs::CellMatch;
///
/// fn describe_cell<T: std::fmt::Debug, R, P>(
///     result: &CellMatch<T, R, P>,
/// ) -> String {
///     match result {
///         CellMatch::King { cell, queen, .. } => {
///             format!("{:?} is a king matched to {:?}", cell, queen)
///         },
///         CellMatch::Queen { cell, king, .. } => {
///             format!("{:?} is a queen matched to {:?}", cell, king)
///         },
///         CellMatch::Ace { cell } => {
///             format!("{:?} is a critical cell", cell)
///         },
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CellMatch<T, R, P> {
    /// A king cell matched to a queen of one lower dimension.
    King {
        /// The king cell itself.
        cell: T,
        /// The queen cell matched to this king.
        queen: T,
        /// The incidence coefficient between king and queen. Must be
        /// invertible in the coefficient ring.
        incidence: R,
        /// Priority for ordering operations. See [`CellMatch`] docs.
        priority: P,
    },

    /// A queen cell matched to a king of one higher dimension.
    Queen {
        /// The queen cell itself.
        cell: T,
        /// The king cell matched to this queen.
        king: T,
        /// The incidence coefficient between queen and king. Must be
        /// invertible in the coefficient ring.
        incidence: R,
        /// Priority for ordering operations. See [`CellMatch`] docs.
        priority: P,
    },

    /// An ace (critical/unmatched) cell.
    ///
    /// Ace cells are not matched and form the reduced Morse complex.
    Ace {
        /// The ace cell itself.
        cell: T,
    },
}

impl<T, R, P> CellMatch<T, R, P> {
    /// Returns a reference to the cell this result describes.
    ///
    /// This works for all variants and returns the primary cell.
    #[must_use]
    pub fn cell(&self) -> &T {
        match self {
            CellMatch::King { cell, .. }
            | CellMatch::Queen { cell, .. }
            | CellMatch::Ace { cell } => cell,
        }
    }

    /// Returns `true` if this is an ace (critical/unmatched) cell.
    #[must_use]
    pub fn is_ace(&self) -> bool {
        matches!(self, CellMatch::Ace { .. })
    }

    /// Returns `true` if this is a king cell.
    #[must_use]
    pub fn is_king(&self) -> bool {
        matches!(self, CellMatch::King { .. })
    }

    /// Returns `true` if this is a queen cell.
    #[must_use]
    pub fn is_queen(&self) -> bool {
        matches!(self, CellMatch::Queen { .. })
    }

    /// Returns a reference to the matched cell, if any.
    ///
    /// - For a king, returns the matched queen.
    /// - For a queen, returns the matched king.
    /// - For an ace, returns `None`.
    #[must_use]
    pub fn matched_cell(&self) -> Option<&T> {
        match self {
            CellMatch::King { queen, .. } => Some(queen),
            CellMatch::Queen { king, .. } => Some(king),
            CellMatch::Ace { .. } => None,
        }
    }

    /// Returns a reference to the incidence coefficient, if any.
    ///
    /// The incidence is the coefficient relating the king and queen in the
    /// boundary/coboundary relationship. It must be invertible for the
    /// matching to be valid.
    ///
    /// Returns `None` for ace cells.
    #[must_use]
    pub fn incidence(&self) -> Option<&R> {
        match self {
            CellMatch::King { incidence, .. } | CellMatch::Queen { incidence, .. } => {
                Some(incidence)
            },
            CellMatch::Ace { .. } => None,
        }
    }
}

impl<T: Eq, R: Eq, P: Ord> PartialOrd for CellMatch<T, R, P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq, R: Eq, P: Ord> Ord for CellMatch<T, R, P> {
    /// Compare two match results.
    ///
    /// The ordering follows the convention `King > Ace > Queen`. Within the
    /// same variant (two kings or two queens), comparison uses the priority
    /// field.
    fn cmp(&self, other: &Self) -> Ordering {
        match self {
            CellMatch::King { priority, .. } => match other {
                CellMatch::King {
                    priority: other_priority,
                    ..
                } => priority.cmp(other_priority),
                _ => Ordering::Greater,
            },
            CellMatch::Queen { priority, .. } => match other {
                CellMatch::Queen {
                    priority: other_priority,
                    ..
                } => priority.cmp(other_priority),
                _ => Ordering::Less,
            },
            CellMatch::Ace { .. } => match other {
                CellMatch::King { .. } => Ordering::Less,
                CellMatch::Queen { .. } => Ordering::Greater,
                CellMatch::Ace { .. } => Ordering::Equal,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessors() {
        let king: CellMatch<i32, i32, u32> = CellMatch::King {
            cell: 1,
            queen: 0,
            incidence: -1,
            priority: 10,
        };

        assert_eq!(king.cell(), &1);
        assert!(king.is_king());
        assert!(!king.is_queen());
        assert!(!king.is_ace());
        assert_eq!(king.matched_cell(), Some(&0));
        assert_eq!(king.incidence(), Some(&-1));

        let queen: CellMatch<i32, i32, u32> = CellMatch::Queen {
            cell: 0,
            king: 1,
            incidence: -1,
            priority: 5,
        };

        assert_eq!(queen.cell(), &0);
        assert!(!queen.is_king());
        assert!(queen.is_queen());
        assert!(!queen.is_ace());
        assert_eq!(queen.matched_cell(), Some(&1));
        assert_eq!(queen.incidence(), Some(&-1));

        let ace: CellMatch<i32, i32, u32> = CellMatch::Ace { cell: 2 };

        assert_eq!(ace.cell(), &2);
        assert!(!ace.is_king());
        assert!(!ace.is_queen());
        assert!(ace.is_ace());
        assert_eq!(ace.matched_cell(), None);
        assert_eq!(ace.incidence(), None);
    }

    #[test]
    fn ordering_within_variant() {
        let king1: CellMatch<i32, i32, u32> = CellMatch::King {
            cell: 1,
            queen: 0,
            incidence: 1,
            priority: 5,
        };
        let king2: CellMatch<i32, i32, u32> = CellMatch::King {
            cell: 2,
            queen: 0,
            incidence: 1,
            priority: 10,
        };

        assert!(king1 < king2); // Lower priority is less

        let queen1: CellMatch<i32, i32, u32> = CellMatch::Queen {
            cell: 0,
            king: 1,
            incidence: 1,
            priority: 5,
        };
        let queen2: CellMatch<i32, i32, u32> = CellMatch::Queen {
            cell: 0,
            king: 2,
            incidence: 1,
            priority: 10,
        };

        assert!(queen1 < queen2); // Lower priority is less
    }

    #[test]
    fn ordering_across_variants() {
        let king: CellMatch<i32, i32, u32> = CellMatch::King {
            cell: 1,
            queen: 0,
            incidence: 1,
            priority: 0,
        };
        let queen: CellMatch<i32, i32, u32> = CellMatch::Queen {
            cell: 0,
            king: 1,
            incidence: 1,
            priority: 100,
        };
        let ace: CellMatch<i32, i32, u32> = CellMatch::Ace { cell: 2 };

        // King > Ace > Queen regardless of priority
        assert!(king > ace);
        assert!(king > queen);
        assert!(ace > queen);
        assert!(ace < king);
        assert!(queen < ace);
        assert!(queen < king);
    }

    #[test]
    fn ace_equality() {
        let ace1: CellMatch<i32, i32, u32> = CellMatch::Ace { cell: 1 };
        let ace2: CellMatch<i32, i32, u32> = CellMatch::Ace { cell: 2 };

        // Different aces compare as equal in ordering (both unmatched)
        assert_eq!(ace1.cmp(&ace2), Ordering::Equal);

        // But they are not equal by PartialEq
        assert_ne!(ace1, ace2);
    }
}
