// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::Orthant;

#[derive(Debug, Clone, Eq, PartialEq)]
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
}
