// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Helper trait for types that support additive operations.
pub trait Additive:
    Sized + Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign + Neg<Output = Self>
{
}

/// Helper trait for types that support multiplicative operations.
pub trait Multiplicative: Sized + Mul<Output = Self> + MulAssign {}

/// Helper trait for basic algebraic structure requirements.
pub trait AlgebraicBase: Sized + Clone + Eq {}

impl<T> Additive for T where
    T: Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign + Neg<Output = Self>
{
}

impl<T> Multiplicative for T where T: Mul<Output = Self> + MulAssign {}

impl<T> AlgebraicBase for T where T: Sized + Clone + Eq {}

/// Expected functionality for coefficient rings throughout `chomp3rs`. These
/// coefficient rings are expected to be integral domains with unity, though
/// this is not checked by this trait.
pub trait RingLike: AlgebraicBase + Additive + Multiplicative {
    /// Creates a new ring element representing the additive identity.
    fn zero() -> Self;
    /// Creates a new ring element representing the multiplicative identity.
    fn one() -> Self;
}

/// A type satisfying `FieldLike` is a ring in which every nonzero value is
/// invertible; this inverse is fetched by the prescribed `invert` method.
pub trait FieldLike: RingLike {
    /// Return the multiplicative inverse of `self`. Should not be called on the
    /// zero element of the ring `Self` (i.e. `Self::zero()`) and may panic
    /// if done so. Otherwise, the method is expected to return correctly.
    fn invert(&self) -> Self;
}

/// The expected functionality for types implementing algebraic modules over the
/// coefficient ring `R`. Objects of a type satisfying `ModuleLike` represent
/// `R`-linear combinations of objects of the basis type `C`.
pub trait ModuleLike: AlgebraicBase + Additive + FromIterator<(Self::Cell, Self::Ring)> {
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

    /// Returns an iterator over all (cell, coefficient) pairs in the module.
    ///
    /// This iterator provides immutable access to the terms in the module,
    /// yielding pairs of cells and their corresponding coefficients. It may be
    /// efficient to filter out terms with zero coefficient, but it is not
    /// required.
    fn iter(&self) -> Self::Iter<'_>;

    /// If `cell` is not in `self`, insert it with coefficient `coef`. Else, add
    /// `coef` to the existing coefficient of `cell` in `self`.
    fn insert_or_add(&mut self, cell: &Self::Cell, coef: Self::Ring) {
        *self.coef_mut(cell) += coef
    }
}
