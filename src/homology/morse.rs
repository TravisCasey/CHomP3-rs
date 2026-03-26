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
///
/// # Examples
///
/// Pattern matching on a `CellMatch`:
///
/// ```
/// use chomp3rs::CellMatch;
///
/// fn describe_cell<T: std::fmt::Debug, R>(
///     result: &CellMatch<T, R>,
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
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CellMatch<T, R> {
    /// A king cell matched to a queen of one lower dimension.
    King {
        /// The king cell itself.
        cell: T,
        /// The queen cell matched to this king.
        queen: T,
        /// The incidence coefficient between king and queen. Must be
        /// invertible in the coefficient ring.
        incidence: R,
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
    },

    /// An ace (critical/unmatched) cell.
    ///
    /// Ace cells are not matched and form the reduced Morse complex.
    Ace {
        /// The ace cell itself.
        cell: T,
    },
}

impl<T, R> CellMatch<T, R> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessors() {
        let king: CellMatch<i32, i32> = CellMatch::King {
            cell: 1,
            queen: 0,
            incidence: -1,
        };

        assert_eq!(king.cell(), &1);
        assert!(king.is_king());
        assert!(!king.is_queen());
        assert!(!king.is_ace());
        assert_eq!(king.matched_cell(), Some(&0));
        assert_eq!(king.incidence(), Some(&-1));

        let queen: CellMatch<i32, i32> = CellMatch::Queen {
            cell: 0,
            king: 1,
            incidence: -1,
        };

        assert_eq!(queen.cell(), &0);
        assert!(!queen.is_king());
        assert!(queen.is_queen());
        assert!(!queen.is_ace());
        assert_eq!(queen.matched_cell(), Some(&1));
        assert_eq!(queen.incidence(), Some(&-1));

        let ace: CellMatch<i32, i32> = CellMatch::Ace { cell: 2 };

        assert_eq!(ace.cell(), &2);
        assert!(!ace.is_king());
        assert!(!ace.is_queen());
        assert!(ace.is_ace());
        assert_eq!(ace.matched_cell(), None);
        assert_eq!(ace.incidence(), None);
    }

    #[test]
    fn ace_equality() {
        let ace1: CellMatch<i32, i32> = CellMatch::Ace { cell: 1 };
        let ace2: CellMatch<i32, i32> = CellMatch::Ace { cell: 2 };

        // Aces with different cells are not equal
        assert_ne!(ace1, ace2);
    }
}
