// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::fmt::Display;

use crate::Orthant;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OrthantMatching {
    Branch {
        prime_extent: u32,
        suborthant_matchings: Vec<OrthantMatching>,
    },
    Leaf {
        lower_extent: u32,
        match_axis: u32,
    },
    Critical {
        ace_dual_orthant: Orthant,
        ace_extent: u32,
    },
}

impl OrthantMatching {
    /// Computes the match axis of a suborthant with upper extent `upper` and
    /// lower extent `lower`. This is the first bit position (from the least
    /// bit) that is set in `upper` but not in `lower`.
    pub fn construct_leaf(upper: u32, lower: u32) -> Self {
        debug_assert_eq!(
            upper & lower,
            lower,
            "attempted to construct a Leaf OrthantMatching from a lower cell which is not a face of the upper cell"
        );
        debug_assert_ne!(
            upper, lower,
            "attempted to construct a Leaf OrthantMatching from a single cell, which should be a Critical OrthantMatching"
        );

        Self::Leaf {
            lower_extent: lower,
            match_axis: ((upper ^ lower) & ((u32::MAX - upper) + lower + 1)).ilog2(),
        }
    }

    fn display_with_indent(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        indent: usize,
    ) -> std::fmt::Result {
        write!(f, "{}", " ".repeat(indent))?;
        match self {
            Self::Branch {
                prime_extent,
                suborthant_matchings,
            } => {
                writeln!(
                    f,
                    "Branch {{ prime_extent: {prime_extent:b}, suborthant_matchings: ["
                )?;
                for matching in suborthant_matchings {
                    matching.display_with_indent(f, indent + 4)?;
                }
                writeln!(f, "{}] }}", " ".repeat(indent))
            }
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with_indent(f, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_axis_computation() {
        let orthant_matching = OrthantMatching::construct_leaf(0b11111, 0b00000);
        assert!(matches!(
            orthant_matching,
            OrthantMatching::Leaf { match_axis: 0, .. }
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
    fn test_display() {
        let orthant_matching = OrthantMatching::Branch {
            prime_extent: 0b1101,
            suborthant_matchings: vec![
                OrthantMatching::Branch {
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
            "Branch { prime_extent: 1101, suborthant_matchings: [\n    \
                 Branch { prime_extent: 1, suborthant_matchings: [\n        \
                     Critical { ace_dual_orthant: (0, 1, -1, 2), ace_extent: 1 }\n    \
                 ] }\n    \
                 Leaf { lower_extent: 10, match_axis: 0 }\n\
             ] }\n"
        );
    }
}
