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
//! - **`chompi`**: Enables MPI-based distributed computation for large cubical
//!   complexes via the `chompi` crate. Implies `serde`.

#![warn(missing_docs)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::unreadable_literal
)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

pub use algebra::{
    Additive, AlgebraicBase, Cyclic, HashMapModule, ModuleLike, Multiplicative, RingLike,
};
#[cfg(feature = "chompi")]
pub use chompi::{MpiExecutor, SimpleCommunicator};
pub use complexes::{
    CellComplex, ComplexLike, Cube, CubeIterator, CubicalComplex, Grader, HashMapGrader, Orthant,
    OrthantIterator, OrthantTrie, TopCubeGrader,
};
pub use homology::{
    CellMatch, CoreductionMatching, MorseMatching, TopCubicalMatching, TopCubicalMatchingBuilder,
};

pub mod algebra;
pub mod complexes;
pub mod homology;
mod logging;
pub mod prelude;

/// Dispatch between feature-gated implementations.
///
/// Executes the first branch when the `chompi` feature is enabled,
/// otherwise executes the fallback branch. This is useful for code that
/// needs different behavior depending on whether MPI support is available.
///
/// # Example
///
/// ```ignore
/// use chomp3rs::dispatch;
///
/// dispatch!(
///     chompi => {
///         // MPI-enabled code path
///         run_distributed();
///     },
///     _ => {
///         // Fallback code path
///         run_sequential();
///     }
/// );
/// ```
#[macro_export]
macro_rules! dispatch {
    (chompi => $chompi:expr,_ => $fallback:expr) => {{
        #[cfg(feature = "chompi")]
        {
            $chompi
        }
        #[cfg(not(feature = "chompi"))]
        {
            $fallback
        }
    }};
}
