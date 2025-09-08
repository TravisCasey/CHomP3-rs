// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::cmp::Ordering;

/// Enum representing the result of a match query to a type implementing
/// `MorseMatching`.
///
/// Type `T` is the cell type and type `R` is the coefficient ring type of the
/// matched cell complex. An ordering on matches (note `Ace` cells are
/// unmatched) is imposed by the type `P`. Use of this type is optional and can
/// be ignored by setting all priorities to the same value.
///
/// However, the (co)lowering and (co)lifting operations from the
/// `MorseMatching` trait can be made significantly more efficient by setting
/// the priority values subject to the condition:
/// > A queen cell `q` matched to the king cell `k` has priority less than or
/// > equal to (in its implementation of `Ord`) the queen cells in the boundary
/// > of `k`.
///
/// The `Ord` implementation of this enum is intended to only be used between
/// two king cells or between two queens. The implementation does not panic if
/// this is violated, and instead uses the convention that `King > Ace > Queen`.
/// However, we reiterate that this may be nonsensical. This likely should be
/// implemented as `PartialOrd` instead, but `Ord` currently permits its use in
/// `std::collections::BinaryHeap`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchResult<T, R, P> {
    /// `cell` is a `King` if it is matched with `queen` of exactly one lesser
    /// dimension. The incidence of `cell` and `queen` is given in `incidence`,
    /// and must be a unit (i.e., `incidence.invert()` must be valid). See the
    /// documentation of `MatchResult` for details on `priority`.
    King {
        /// The king cell.
        cell: T,
        /// The queen cell matched to `cell`.
        queen: T,
        /// The incidence of `cell` and `queen`. Must be invertible.
        incidence: R,
        /// Priority of the match, used for ordering `MatchResult` instances.
        priority: P,
    },
    /// `cell` is a `Queen` if it is matched with `king` of exactly one greater
    /// dimension. The incidence of `cell` and `king` is given in `incidence`,
    /// and must be a unit (i.e., `incidence.invert()` must be valid). See the
    /// documentation of `MatchResult` for details on `priority`.
    Queen {
        /// The queen cell.
        cell: T,
        /// The king cell matched to `cell`.
        king: T,
        /// The incidence of `cell` and `king`. Must be invertible.
        incidence: R,
        /// Priority of the match, used for ordering `MatchResult` instances.
        priority: P,
    },
    /// `cell` is an `Ace` if it is matched with no other cell. See the
    /// documentation of `MatchResult` for details on `priority`.
    Ace {
        /// The ace cell.
        cell: T,
    },
}

impl<T: Eq, R: Eq, P: Ord> PartialOrd for MatchResult<T, R, P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq, R: Eq, P: Ord> Ord for MatchResult<T, R, P> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self {
            MatchResult::King { priority, .. } => match other {
                MatchResult::King {
                    priority: other_priority,
                    ..
                } => priority.cmp(other_priority),
                _ => Ordering::Greater,
            },
            MatchResult::Queen { priority, .. } => match other {
                MatchResult::Queen {
                    priority: other_priority,
                    ..
                } => priority.cmp(other_priority),
                _ => Ordering::Less,
            },
            MatchResult::Ace { .. } => match other {
                MatchResult::King { .. } => Ordering::Less,
                MatchResult::Queen { .. } => Ordering::Greater,
                MatchResult::Ace { .. } => Ordering::Equal,
            },
        }
    }
}
