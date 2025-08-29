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

// Blanket implementations for helper traits
impl<T> Additive for T where
    T: Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign + Neg<Output = Self>
{
}

impl<T> Multiplicative for T where T: Mul<Output = Self> + MulAssign {}

impl<T> AlgebraicBase for T where T: Sized + Clone + Eq {}

/// Expected functionality for coefficient rings throughout `chomp3rs`. These coefficient rings are
/// expected to be integral domains with unity, though this is not checked by this trait. The
/// arithmetic operations are expected to represent the ring operations; the `From<u32>` trait is
/// employed as a constructor that yields the ring element that is the prescribed sum of its unity.
///
/// Example
/// ```rust
/// use chomp3rs::Cyclic;
/// assert_eq!(Cyclic::<5>::from(8), Cyclic::<5>::from(3));
/// assert_ne!(Cyclic::<7>::from(8), Cyclic::<7>::from(3));
/// ```
pub trait Ring: AlgebraicBase + Additive + Multiplicative {
    /// Creates a new ring element representing the additive identity.
    fn zero() -> Self;
    /// Creates a new ring element representing the multiplicative identity.
    fn one() -> Self;
}

/// A type satisfying `Field` is a ring in which every nonzero value is invertible; this inverse is
/// fetched by the prescribed `invert` method.
pub trait Field: Ring {
    /// Return the multiplicative inverse of `self`. Should not be called on the zero element of
    /// the ring `Self` (i.e. `Self::from(0)`) and may panic if done so. Otherwise, the method is
    /// expected to return correctly.
    fn invert(&self) -> Self;
}

/// The expected functionality for types implementing algebraic modules over the coefficient ring
/// `R`. Objects of a type satisfying `Module` represent `R`-linear combinations of objects of
/// the basis type `C`.
pub trait Module<C, R: Ring>: AlgebraicBase + Additive {
    /// Create an empty module element.
    fn new() -> Self;
    /// Empty the module element `self`; the implementation details (e.g. memory management, etc.)
    /// are not otherwise prescribed by this trait.
    fn clear(&mut self);
    /// Return the coefficient of `cell` in `self`, if it exists.
    fn coef(&self, cell: &C) -> R;
    /// Return a mutable reference to the coefficient of `cell` in `self`, if it exists.
    fn coef_mut(&mut self, cell: &C) -> &mut R;
    /// Perform scalar multiplication of `self` with `coef`. Effectively, this multiplies each
    /// coefficient in `self` by `coef`. If `coef` is the zero element, the default implementation
    /// clears `self` to avoid storing cells with zero coefficient. Otherwise, as types implementing
    /// [`Ring`] are presumed to not have zero divisors, the number of nonzero elements does not
    /// change.
    fn scalar_mul(&mut self, coef: R);

    /// If `cell` is not in `self`, insert it with coefficient `coef`. Else, add `coef` to the
    /// existing coefficient of `cell` in `self`.
    fn insert_or_add(&mut self, cell: &C, coef: R) {
        *self.coef_mut(cell) += coef
    }
}
