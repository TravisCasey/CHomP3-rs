// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Algebraic structures for computational homology.
//!
//! This module provides the core algebraic abstractions and implementations
//! used throughout `chomp3rs`, including rings, modules, and fields. The
//! primary components are:
//!
//! - **Traits**: [`RingLike`] and [`ModuleLike`] define the expected behavior
//!   for coefficient rings and modules over those rings.
//! - **Cyclic fields**: [`Cyclic<MOD>`] implements finite, cyclic fields for
//!   prime `MOD`.
//! - **Sparse modules**: [`HashMapModule`] provides a hash table-based
//!   implementation of modules for sparse linear combinations.
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
//! Creating and manipulating modules:
//!
//! ```
//! use chomp3rs::{Cyclic, HashMapModule, ModuleLike};
//!
//! let mut module = HashMapModule::<u32, Cyclic<5>>::new();
//! module.insert_or_add(1, Cyclic::from(3));
//! module.insert_or_add(2, Cyclic::from(4));
//!
//! assert_eq!(module.coefficient(&1), Cyclic::from(3));
//! ```

use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub use cyclic::Cyclic;
pub use module::HashMapModule;

mod cyclic;
mod module;

/// Helper trait for types that support additive operations.
pub trait Additive:
    Sized + Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign + Neg<Output = Self>
{
}

/// Helper trait for types that support multiplicative operations.
pub trait Multiplicative: Sized + Mul<Output = Self> + MulAssign {}

/// Helper trait for basic requirements required of algebraic structures.
pub trait AlgebraicBase: Sized + Clone + Eq + Debug {}

impl<T> Additive for T where
    T: Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign + Neg<Output = Self>
{
}

impl<T> Multiplicative for T where T: Mul<Output = Self> + MulAssign {}

impl<T> AlgebraicBase for T where T: Sized + Clone + Eq + Debug {}

/// Expected functionality for coefficient rings throughout `chomp3rs`.
///
/// These coefficient rings are expected to be
/// [integral domains](https://en.wikipedia.org/wiki/Integral_domain) with
/// unity, though this is not enforced by the trait. Implementations should
/// ensure mathematical correctness themselves.
///
/// # Examples
///
/// Implementing `RingLike` for a custom type:
///
/// ```
/// use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
///
/// use chomp3rs::{Additive, AlgebraicBase, Multiplicative, RingLike};
///
/// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// struct ModTwo(bool);
///
/// impl Add for ModTwo {
///     type Output = Self;
///
///     fn add(self, rhs: Self) -> Self {
///         ModTwo(self.0 ^ rhs.0)
///     }
/// }
///
/// impl AddAssign for ModTwo {
///     fn add_assign(&mut self, rhs: Self) {
///         self.0 ^= rhs.0;
///     }
/// }
///
/// impl Sub for ModTwo {
///     type Output = Self;
///
///     fn sub(self, rhs: Self) -> Self {
///         ModTwo(self.0 ^ rhs.0)
///     }
/// }
///
/// impl SubAssign for ModTwo {
///     fn sub_assign(&mut self, rhs: Self) {
///         self.0 ^= rhs.0;
///     }
/// }
///
/// impl Neg for ModTwo {
///     type Output = Self;
///
///     fn neg(self) -> Self {
///         self
///     }
/// }
///
/// impl Mul for ModTwo {
///     type Output = Self;
///
///     fn mul(self, rhs: Self) -> Self {
///         ModTwo(self.0 && rhs.0)
///     }
/// }
///
/// impl MulAssign for ModTwo {
///     fn mul_assign(&mut self, rhs: Self) {
///         self.0 = self.0 && rhs.0;
///     }
/// }
///
/// impl RingLike for ModTwo {
///     fn zero() -> Self {
///         ModTwo(false)
///     }
///
///     fn one() -> Self {
///         ModTwo(true)
///     }
///
///     fn is_invertible(&self) -> bool {
///         self.0
///     }
///
///     fn invert(&self) -> Self {
///         assert!(self.0, "cannot invert zero");
///         *self
///     }
/// }
/// ```
pub trait RingLike: AlgebraicBase + Additive + Multiplicative {
    /// Creates a new ring element representing the additive identity.
    fn zero() -> Self;

    /// Creates a new ring element representing the multiplicative identity.
    fn one() -> Self;

    /// Check if the element is invertible in the ring.
    fn is_invertible(&self) -> bool;

    /// Return the multiplicative inverse of `self`, for those values at which
    /// it exists. Implementations should panic if the inverse does not
    /// exist.
    #[must_use]
    fn invert(&self) -> Self;
}

/// Expected functionality for
/// [algebraic modules](https://en.wikipedia.org/wiki/Module_(mathematics)) over
/// a coefficient ring.
///
/// Objects implementing `ModuleLike` represent linear combinations of basis
/// elements (of the associated type `Self::Cell`) with coefficients from a ring
/// (of the associated type `Self::Ring`, notably implementing [`RingLike`]).
/// Zero coefficient terns may be stored or omitted depending on implementation;
/// querying the coefficient of a cell which is not present, however, should
/// return zero (in the coefficient ring).
///
/// # Examples
///
/// Basic module operations:
///
/// ```
/// use chomp3rs::{Cyclic, HashMapModule, ModuleLike};
///
/// let module = HashMapModule::<i32, Cyclic<5>>::from([
///     (1, Cyclic::from(2)),
///     (2, Cyclic::from(3)),
/// ]);
///
/// assert_eq!(module.coefficient(&1), Cyclic::from(2));
/// assert_eq!(module.coefficient(&99), Cyclic::from(0)); // Not present
/// ```
pub trait ModuleLike:
    AlgebraicBase
    + Additive
    + FromIterator<(Self::Cell, Self::Ring)>
    + IntoIterator<Item = (Self::Cell, Self::Ring)>
{
    /// The type of the basis elements of the module.
    type Cell;

    /// Coefficient type applied to cells in the module.
    type Ring: RingLike;

    /// Iterator type for iterating over (cell, coefficient) pairs.
    type Iter<'a>: Iterator<Item = (&'a Self::Cell, &'a Self::Ring)>
    where
        Self: 'a;

    /// Create an empty module element.
    fn new() -> Self;

    /// Empty the module element `self`; the implementation details (e.g. memory
    /// management, etc.) are not otherwise prescribed by this trait.
    fn clear(&mut self);

    /// Return the coefficient of `cell` in `self`. If `cell` is not present,
    /// returns `Self::Ring::zero()`.
    fn coefficient(&self, cell: &Self::Cell) -> Self::Ring;

    /// Return a mutable reference to the coefficient of `cell` in `self`.
    /// If `cell` is not present, it is inserted with coefficient zero first.
    fn coefficient_mut(&mut self, cell: &Self::Cell) -> &mut Self::Ring;

    /// Remove `cell` from the module, returning its coefficient. If `cell` is
    /// not in the module, return `Self::Ring::zero()`.
    fn remove(&mut self, cell: &Self::Cell) -> Self::Ring;

    /// Perform scalar multiplication of `self` with `coefficient`. This
    /// effectively multiplies each coefficient in `self` by `coefficient`.
    #[must_use]
    fn scalar_mul(self, coefficient: Self::Ring) -> Self;

    /// Returns an iterator over all (cell, coefficient) pairs in the module
    /// for which coefficient is nonzero.
    fn iter(&self) -> Self::Iter<'_>;

    /// If `cell` is not in `self`, insert it with the given `coefficient`.
    /// Otherwise, add `coefficient` to the existing coefficient of `cell`.
    fn insert_or_add(&mut self, cell: Self::Cell, coefficient: Self::Ring);

    /// Returns `true` if the module contains no non-zero terms.
    ///
    /// Note that this method iterates over all terms to check for non-zero
    /// coefficients, so it may be O(n) in the number of stored terms
    /// depending on the implementation.
    ///
    /// # Examples
    ///
    /// ```
    /// use chomp3rs::{Cyclic, HashMapModule, ModuleLike};
    ///
    /// let empty = HashMapModule::<u32, Cyclic<5>>::new();
    /// assert!(empty.is_empty());
    ///
    /// let non_empty =
    ///     HashMapModule::<u32, Cyclic<5>>::from([(1, Cyclic::from(2))]);
    /// assert!(!non_empty.is_empty());
    /// ```
    fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }

    /// Returns the number of terms with non-zero coefficients.
    ///
    /// Note that this method iterates over all terms to count non-zero
    /// coefficients, so it may be O(n) in the number of stored terms
    /// depending on the implementation.
    ///
    /// # Examples
    ///
    /// ```
    /// use chomp3rs::{Cyclic, HashMapModule, ModuleLike};
    ///
    /// let module = HashMapModule::<u32, Cyclic<5>>::from([
    ///     (1, Cyclic::from(2)),
    ///     (2, Cyclic::from(3)),
    ///     (3, Cyclic::from(0)), // Zero coefficient, not counted
    /// ]);
    /// assert_eq!(module.len(), 2);
    /// ```
    fn len(&self) -> usize {
        self.iter().count()
    }

    /// Creates a module with a single term.
    ///
    /// If `coefficient` is zero, returns an empty module.
    ///
    /// # Examples
    ///
    /// ```
    /// use chomp3rs::{Cyclic, HashMapModule, ModuleLike};
    ///
    /// let module =
    ///     HashMapModule::<u32, Cyclic<5>>::singleton(42, Cyclic::from(3));
    /// assert_eq!(module.coefficient(&42), Cyclic::from(3));
    /// assert_eq!(module.len(), 1);
    /// ```
    fn singleton(cell: Self::Cell, coefficient: Self::Ring) -> Self {
        let mut module = Self::new();
        module.insert_or_add(cell, coefficient);
        module
    }
}
