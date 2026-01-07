// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Efficient algorithms for high-dimensional homology computation using
//! discrete Morse theory.
//!
//! This crate is in an early developmental state; interfaces may change at any
//! point.
//!
//! # Quick Start
//!
//! Use the [`prelude`] for convenient imports:
//!
//! ```
//! use chomp3rs::prelude::*;
//! ```
//!
//! # Feature Flags
//!
//! - **`serde`**: Enables serialization support for core data structures.
//! - **`mpi`**: Enables MPI-based distributed computation for large cubical
//!   complexes. Implies `serde`.

#![warn(missing_docs)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

pub use algebra::{
    Additive, AlgebraicBase, Cyclic, HashMapModule, ModuleLike, Multiplicative, RingLike,
};
pub use complexes::{
    CellComplex, ComplexLike, Cube, CubeIterator, CubicalComplex, Grader, HashMapGrader, Orthant,
    OrthantIterator, OrthantTrie, TopCubeGrader,
};
// temp subgrid, GridSubdivision, CubicalGradeintPropagator
pub use homology::{
    CellMatch, CoreductionMatching, MorseMatching, TopCubicalMatching, TopCubicalMatchingBuilder,
};

pub mod algebra;
pub mod complexes;
pub mod executor;
pub mod homology;
pub mod prelude;
