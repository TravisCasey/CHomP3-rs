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
//! - **`rayon`**: Enables shared-memory parallelism via Rayon for matching and
//!   flow operations.
//! - **`mpi`**: Enables MPI-based distributed computation for large cubical
//!   complexes. Implies `serde`. Can be combined with `rayon` for hybrid
//!   parallelism (MPI between nodes, Rayon within each node).

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
pub use parallel::ExecutionBackend;

pub mod algebra;
pub mod complexes;
pub mod homology;
pub(crate) mod logging;
#[cfg(feature = "mpi")]
pub mod mpi;
pub mod parallel;
pub mod prelude;
#[cfg(test)]
mod test_complexes;
