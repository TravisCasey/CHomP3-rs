// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Algebraic structures for computational homology.
//!
//! This module provides the core algebraic abstractions and implementations
//! used throughout `CHomP3-rs` for ring and chain arithmetic. The primary
//! components are:
//!
//! - **Trait**: [`Ring`] defines the expected behavior for coefficient rings.
//! - **Cyclic fields**: [`Cyclic<MOD>`] implements finite, cyclic fields for
//!   prime `MOD`.
//! - **Sparse chains**: [`Chain`] provides a hash table-based implementation
//!   for sparse linear combinations, while [`OrderedChain`] provides sorted
//!   storage instead.
//!
//! # Examples
//!
//! Working with cyclic fields:
//!
//! ```
//! use chomp3rs::Cyclic;
//!
//! let a = Cyclic::<7>::from(5);
//! let b = Cyclic::<7>::from(4);
//! assert_eq!(a + b, Cyclic::<7>::from(2)); // 5 + 4 = 9 = 2 (mod 7)
//! ```
//!
//! Constructing and manipulating (co)chains:
//!
//! ```
//! use chomp3rs::{Chain, Cyclic};
//!
//! let mut chain = Chain::<u32, Cyclic<5>>::new();
//! chain.insert_or_add(1, Cyclic::from(3));
//! chain.insert_or_add(2, Cyclic::from(4));
//!
//! assert_eq!(chain.coefficient(&1), Cyclic::from(3));
//! ```

use std::{
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub use chain::{Chain, ChainIter, ChainIterMut};
pub use cyclic::{Cyclic, F2};
pub use ordered::{OrderedChain, OrderedChainIter, OrderedChainIterMut};
#[cfg(feature = "serde")]
use serde::{Serialize, de::DeserializeOwned};

mod chain;
mod cyclic;
mod ordered;

/// Trait for types which support additive operations.
///
/// Automatically implemented for types which fulfill all supertraits. Required
/// by the [`Ring`] algebraic trait.
pub trait Additive:
    Sized + Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign + Neg<Output = Self>
{
}

/// Trait for types which support multiplicative operations.
///
/// Automatically implemented for types which fulfill all supertraits. Required
/// by the [`Ring`] algebraic trait.
pub trait Multiplicative: Sized + Mul<Output = Self> + MulAssign {}

impl<T> Additive for T where
    T: Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign + Neg<Output = Self>
{
}

impl<T> Multiplicative for T where T: Mul<Output = Self> + MulAssign {}

// The Ring trait uses a macro to conditionally include serde supertraits when
// the `serde` feature is enabled. This avoids duplicating serde bounds on every
// function signature throughout the crate. When `serde` is disabled, custom
// ring types need not implement serde.

macro_rules! define_ring_trait {
    ($($bound:path),* $(,)?) => {
        /// Required functionality for coefficient rings in `CHomP3-rs`.
        ///
        /// ### Mathematical Correctness
        ///
        /// These coefficient rings are expected to be
        /// [integral domains](https://en.wikipedia.org/wiki/Integral_domain)
        /// with unity, though this is not enforced by the trait.
        /// Implementations should ensure mathematical correctness of their
        /// algebraic operations themselves.
        ///
        /// ### Serialization
        ///
        /// When the `serde` feature flag is enabled, serde `Serialize` and
        /// `DeserializeOwned` bounds are required.
        pub trait Ring: Sized + Clone + Eq + Debug + Additive + Multiplicative $(+ $bound)* {
            /// Creates a ring element representing the additive identity.
            #[must_use]
            fn zero() -> Self;

            /// Creates a ring element representing the multiplicative identity.
            #[must_use]
            fn one() -> Self;

            /// Check if the element is invertible in the ring.
            #[must_use]
            fn is_invertible(&self) -> bool;

            /// Return the multiplicative inverse of `self`, if it exists.
            #[must_use]
            fn invert(&self) -> Option<Self>;
        }
    };
}

#[cfg(not(feature = "serde"))]
define_ring_trait!();

#[cfg(feature = "serde")]
define_ring_trait!(Serialize, DeserializeOwned);

/// Formats a chain as a sum of terms for [`Display`] output.
///
/// Shared implementation for [`Chain`] and [`OrderedChain`].
fn fmt_chain<'a, B, R>(
    pairs: impl Iterator<Item = (&'a B, &'a R)>,
    f: &mut Formatter<'_>,
) -> FmtResult
where
    B: Display + 'a,
    R: Ring + Display + 'a,
{
    let non_zero: Vec<_> = pairs
        .filter(|(_, coefficient)| **coefficient != R::zero())
        .collect();

    if non_zero.is_empty() {
        return write!(f, "0");
    }

    let mut first = true;
    for (cell, coefficient) in non_zero {
        if !first {
            write!(f, " + ")?;
        }
        first = false;

        if *coefficient == R::one() {
            write!(f, "{cell}")?;
        } else {
            write!(f, "{coefficient}*{cell}")?;
        }
    }
    Ok(())
}
