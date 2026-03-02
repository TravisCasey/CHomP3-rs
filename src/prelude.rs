// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Ergonomic re-exports for common types and traits.
//!
//! This prelude provides a single glob import for the core types and traits
//! needed in a typical `chomp3rs` workflow: constructing complexes, computing
//! Morse matchings, and inspecting results. It mirrors the crate root
//! re-exports.
//!
//! Specialized types like [`CubeIterator`](crate::complexes::CubeIterator)
//! and [`OrthantIterator`](crate::complexes::OrthantIterator) are not
//! included; import them from [`complexes`](crate::complexes) when needed.
//! Feature-gated items (e.g., `mpi`) are also excluded.
//!
//! ```
//! use chomp3rs::prelude::*;
//! ```

pub use crate::{
    algebra::{Chain, Cyclic, F2, OrderedChain, Ring},
    complexes::{
        CellComplex, Complex, Cube, CubicalComplex, Grader, HashGrader, Orthant, OrthantTrie,
        TopCubeGrader,
    },
    homology::{
        CellMatch, CoreductionMatching, MorseMatching, TopCubicalMatching,
        TopCubicalMatchingBuilder,
    },
};
