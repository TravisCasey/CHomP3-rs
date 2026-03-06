// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

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
//! - **`serde`**: Enables serde `Serialize` and `Deserialize` implementations
//!   on core types. Requires serde bounds on custom ring types.
//! - **`mpi`**: Enables MPI-based distributed computation for large cubical
//!   complexes. Implies `serde`.

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

pub use algebra::{Chain, Cyclic, F2, OrderedChain, Ring};
pub use complexes::{
    CellComplex, Complex, Cube, CubicalComplex, Grader, HashGrader, Orthant, OrthantTrie,
    TopCubeGrader,
};
pub use homology::{
    CellMatch, CoreductionMatching, MorseMatching, TopCubicalMatching, TopCubicalMatchingBuilder,
};

pub mod algebra;
pub mod complexes;
pub mod homology;
pub(crate) mod logging;
#[cfg(feature = "mpi")]
pub mod mpi;
pub mod prelude;
#[cfg(test)]
mod test_complexes;

/// Dispatch between MPI and non-MPI code paths.
///
/// Executes the first branch when the `mpi` feature is enabled,
/// otherwise executes the fallback branch. This is useful for code that
/// needs different behavior depending on whether MPI support is available.
///
/// # Example
///
/// ```ignore
/// use chomp3rs::dispatch;
///
/// dispatch!(
///     mpi => {
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
    (mpi => $mpi:expr,_ => $fallback:expr) => {{
        #[cfg(feature = "mpi")]
        {
            $mpi
        }
        #[cfg(not(feature = "mpi"))]
        {
            $fallback
        }
    }};
}
