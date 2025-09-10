// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Helper trait for types that support additive operations.
pub trait Additive:
    Sized + Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign + Neg<Output = Self>
{
}

/// Helper trait for types that support multiplicative operations.
pub trait Multiplicative: Sized + Mul<Output = Self> + MulAssign {}

/// Helper trait for basic algebraic structure requirements.
pub trait AlgebraicBase: Sized + Clone + Eq + Debug {}

impl<T> Additive for T where
    T: Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign + Neg<Output = Self>
{
}

impl<T> Multiplicative for T where T: Mul<Output = Self> + MulAssign {}

impl<T> AlgebraicBase for T where T: Sized + Clone + Eq + Debug {}

/// Expected functionality for coefficient rings throughout `chomp3rs`. These
/// coefficient rings are expected to be integral domains with unity, though
/// this is not checked by this trait.
pub trait RingLike: AlgebraicBase + Additive + Multiplicative {
    /// Creates a new ring element representing the additive identity.
    fn zero() -> Self;
    /// Creates a new ring element representing the multiplicative identity.
    fn one() -> Self;
    /// Check if the element is invertible in the ring.
    fn is_invertible(&self) -> bool;
    /// Return the multiplicative inverse of `self`, for thos values at which it
    /// exists. It is good practice to panic if the inverse does not exist
    /// (i.e. at the zero element).
    fn invert(&self) -> Self;
}

/// The expected functionality for types implementing algebraic modules over the
/// coefficient ring `Self::Ring`. Objects of a type satisfying `ModuleLike` represent
/// linear combinations of objects of the basis type `Self::Cell`.
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
    /// Return the coefficient of `cell` in `self`, if it exists.
    fn coef(&self, cell: &Self::Cell) -> Self::Ring;
    /// Return a mutable reference to the coefficient of `cell` in `self`, if it
    /// exists.
    fn coef_mut(&mut self, cell: &Self::Cell) -> &mut Self::Ring;
    /// Perform scalar multiplication of `self` with `coef`. Effectively, this
    /// multiplies each coefficient in `self` by `coef`.
    fn scalar_mul(self, coef: Self::Ring) -> Self;

    /// Returns an iterator over all (cell, coefficient) pairs in the module
    /// for which coefficient is nonzero.
    fn iter(&self) -> Self::Iter<'_>;

    /// If `cell` is not in `self`, insert it with coefficient `coef`. Else, add
    /// `coef` to the existing coefficient of `cell` in `self`.
    fn insert_or_add(&mut self, cell: Self::Cell, coef: Self::Ring) {
        *self.coef_mut(&cell) += coef
    }
}
