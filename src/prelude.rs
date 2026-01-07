// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Convenient re-exports for common types and traits.
//!
//! This module provides a single import for the most commonly used items in
//! `chomp3rs`. Use it with a glob import:
//!
//! ```
//! use chomp3rs::prelude::*;
//! ```

pub use crate::{
    algebra::{Cyclic, HashMapModule, ModuleLike, RingLike},
    complexes::{
        CellComplex, ComplexLike, Cube, CubicalComplex, Grader, HashMapGrader, Orthant,
        OrthantTrie, TopCubeGrader,
    },
    homology::{CellMatch, CoreductionMatching, MorseMatching, TopCubicalMatching},
};
