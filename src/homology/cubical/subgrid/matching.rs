// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Matching result type for subgrid orthant matching routine.

use std::fmt::{Display, Formatter, Result};

use crate::Orthant;

/// Represents the matching structure for cells within an orthant.
///
/// This enum describes how cells in a suborthant are matched:
///
/// - `Branch`: The suborthant is subdivided into further suborthants in a
///   regular pattern, each with their own matching.
/// - `Leaf`: All cells in the interval between `lower_extent` and the implicit
///   upper extent are matched along a single axis.
/// - `Critical`: The suborthant contains a single critical cell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OrthantMatching {
    /// A branch node that subdivides the orthant into multiple suborthants.
    Branch {
        /// Extent of the maximal cell in this suborthant.
        upper_extent: u32,
        /// Upper extent of the first suborthant.
        prime_extent: u32,
        /// Matchings on each child suborthant.
        suborthant_matchings: Vec<OrthantMatching>,
    },
    /// A leaf node in which all cells match along a single axis.
    Leaf {
        /// Extent of the minimal cell in the matched interval.
        lower_extent: u32,
        /// The axis along which cells are matched.
        match_axis: u32,
    },
    /// A single critical cell: it is not matched to any other cell.
    Critical {
        /// The dual orthant of the critical cell.
        ace_dual_orthant: Orthant,
        /// Extent of the critical cell.
        ace_extent: u32,
    },
}

impl OrthantMatching {
    /// Computes the match axis of a suborthant with upper extent `upper` and
    /// lower extent `lower`.
    ///
    /// This is the first bit position (from the least bit) that is set in
    /// `upper` but not in `lower`.
    #[must_use]
    pub fn construct_leaf(upper: u32, lower: u32) -> Self {
        debug_assert_eq!(
            upper & lower,
            lower,
            "attempted to construct an OrthantMatching::Leaf from a lower cell which is not a \
             face of the upper cell"
        );
        debug_assert_ne!(
            upper, lower,
            "attempted to construct an OrthantMatching::Leaf from a single cell, which should \
             instead be an OrthantMatching::Critical"
        );

        // bit magic - trust
        Self::Leaf {
            lower_extent: lower,
            match_axis: ((upper ^ lower) & ((u32::MAX - upper) + lower + 1)).ilog2(),
        }
    }

    fn display_with_indent(&self, f: &mut Formatter<'_>, indent: usize) -> Result {
        write!(f, "{}", " ".repeat(indent))?;
        match self {
            Self::Branch {
                upper_extent,
                prime_extent,
                suborthant_matchings,
            } => {
                writeln!(
                    f,
                    "Branch {{ upper_extent: {upper_extent:b}, prime_extent: {prime_extent:b}, \
                     suborthant_matchings: ["
                )?;
                for matching in suborthant_matchings {
                    matching.display_with_indent(f, indent + 4)?;
                }
                writeln!(f, "{}] }}", " ".repeat(indent))
            },
            Self::Leaf {
                lower_extent,
                match_axis,
            } => writeln!(
                f,
                "Leaf {{ lower_extent: {lower_extent:b}, match_axis: {match_axis} }}"
            ),
            Self::Critical {
                ace_dual_orthant,
                ace_extent,
            } => writeln!(
                f,
                "Critical {{ ace_dual_orthant: {ace_dual_orthant}, ace_extent: {ace_extent:b} }}"
            ),
        }
    }
}

impl Display for OrthantMatching {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        self.display_with_indent(f, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_computation() {
        let orthant_matching = OrthantMatching::construct_leaf(0b11111, 0b00000);
        assert!(matches!(
            orthant_matching,
            OrthantMatching::Leaf { match_axis: 0, .. }
        ));

        let orthant_matching = OrthantMatching::construct_leaf(0b100100, 0b000000);
        assert!(matches!(
            orthant_matching,
            OrthantMatching::Leaf { match_axis: 2, .. }
        ));

        let orthant_matching = OrthantMatching::construct_leaf(0b11111, 0b01111);
        assert!(matches!(
            orthant_matching,
            OrthantMatching::Leaf { match_axis: 4, .. }
        ));

        let orthant_matching = OrthantMatching::construct_leaf(0b11011, 0b10011);
        assert!(matches!(
            orthant_matching,
            OrthantMatching::Leaf { match_axis: 3, .. }
        ));
    }

    #[test]
    fn display() {
        let orthant_matching = OrthantMatching::Branch {
            upper_extent: 0b1111,
            prime_extent: 0b1101,
            suborthant_matchings: vec![
                OrthantMatching::Branch {
                    upper_extent: 0b1101,
                    prime_extent: 0b0001,
                    suborthant_matchings: vec![OrthantMatching::Critical {
                        ace_dual_orthant: Orthant::from([0, 1, -1, 2]),
                        ace_extent: 0b0001,
                    }],
                },
                OrthantMatching::construct_leaf(0b1111, 0b0010),
            ],
        };

        assert_eq!(
            orthant_matching.to_string(),
            r"Branch { upper_extent: 1111, prime_extent: 1101, suborthant_matchings: [
    Branch { upper_extent: 1101, prime_extent: 1, suborthant_matchings: [
        Critical { ace_dual_orthant: (0, 1, -1, 2), ace_extent: 1 }
    ] }
    Leaf { lower_extent: 10, match_axis: 0 }
] }
"
        );
    }
}
