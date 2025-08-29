// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The `chomp3rs` crate provides efficient and optimized algorithms for high-dimensional
//! computation of homology. The project is currently in an early developmental state.

#![warn(missing_docs)]

pub use crate::algebra::{
    Additive, AlgebraicBase, Cyclic, FieldLike, HashMapModule, ModuleLike, Multiplicative, RingLike,
};
pub use crate::complexes::{CellComplex, ComplexLike};

mod algebra;
mod complexes;
