// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The `Cyclic` class implementing the cyclic field of integers with
//! configurable modulus.

use std::convert::From;
use std::fmt::{Display, Error, Formatter};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use flint_sys::nmod_vec::{nmod_add, nmod_init, nmod_inv, nmod_mul, nmod_neg, nmod_sub, nmod_t};

use crate::algebra::traits::RingLike;

/// The field of integers modulo `MOD`, for prime modulus values `MOD`.
///
/// # Important Note
/// `MOD` **must** be a prime number for the `FieldLike` implementation to be
/// mathematically correct. While this is not explicitly checked at compile
/// time, the implementation of `invert` and other dependent methods assume this
/// property. Using a composite modulus may lead to incorrect results or panics.
///
/// Overflow and underflow are handled by the implementation.
///
/// # Examples
/// ## Equality Modulo `MOD`
/// ```rust
/// use chomp3rs::Cyclic;
/// assert_eq!(Cyclic::<5>::from(8), Cyclic::<5>::from(3));
/// assert_ne!(Cyclic::<7>::from(8), Cyclic::<7>::from(3));
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Cyclic<const MOD: u64> {
    remainder: u64,
    modulus: nmod_t,
}

impl<const MOD: u64> Cyclic<MOD> {
    /// Create a new `Cyclic` instance with the given value modulo `MOD`. Panics
    /// if `MOD` is less than 2; `MOD` is expected to be a prime number,
    /// though this is not explicitly checked.
    pub fn new(value: u64) -> Self {
        assert!(
            MOD > 1,
            "modulus values must be a prime number greater than or equal to 2"
        );

        let mut modulus = nmod_t {
            n: 0,
            ninv: 0,
            norm: 0,
        };
        unsafe {
            nmod_init(&mut modulus, MOD);
        }

        Self {
            remainder: value % MOD,
            modulus,
        }
    }
}

impl<const MOD: u64> From<u64> for Cyclic<MOD> {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl<const MOD: u64> RingLike for Cyclic<MOD> {
    fn zero() -> Self {
        Self::new(0)
    }

    fn one() -> Self {
        Self::new(1)
    }

    fn is_invertible(&self) -> bool {
        self.remainder != 0
    }

    fn invert(&self) -> Self {
        assert!(
            self.remainder != 0,
            "attempting to invert equivalency class zero"
        );
        if MOD == 2 {
            return *self;
        }

        Self {
            remainder: unsafe { nmod_inv(self.remainder, self.modulus) },
            modulus: self.modulus,
        }
    }
}

impl<const MOD: u64> Display for Cyclic<MOD> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{} (mod {})", self.remainder, MOD)
    }
}

impl<const MOD: u64> Neg for Cyclic<MOD> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            remainder: unsafe { nmod_neg(self.remainder, self.modulus) },
            modulus: self.modulus,
        }
    }
}

impl<const MOD: u64> AddAssign for Cyclic<MOD> {
    fn add_assign(&mut self, rhs: Self) {
        self.remainder = unsafe { nmod_add(self.remainder, rhs.remainder, self.modulus) };
    }
}

impl<const MOD: u64> Add for Cyclic<MOD> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            remainder: unsafe { nmod_add(self.remainder, rhs.remainder, self.modulus) },
            modulus: self.modulus,
        }
    }
}

impl<const MOD: u64> SubAssign for Cyclic<MOD> {
    fn sub_assign(&mut self, rhs: Self) {
        self.remainder = unsafe { nmod_sub(self.remainder, rhs.remainder, self.modulus) };
    }
}

impl<const MOD: u64> Sub for Cyclic<MOD> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            remainder: unsafe { nmod_sub(self.remainder, rhs.remainder, self.modulus) },
            modulus: self.modulus,
        }
    }
}

impl<const MOD: u64> MulAssign for Cyclic<MOD> {
    fn mul_assign(&mut self, rhs: Self) {
        self.remainder = unsafe { nmod_mul(self.remainder, rhs.remainder, self.modulus) };
    }
}

impl<const MOD: u64> Mul for Cyclic<MOD> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            remainder: unsafe { nmod_mul(self.remainder, rhs.remainder, self.modulus) },
            modulus: self.modulus,
        }
    }
}

impl<const MOD: u64> PartialEq for Cyclic<MOD> {
    fn eq(&self, other: &Self) -> bool {
        self.remainder == other.remainder
    }
}

impl<const MOD: u64> Eq for Cyclic<MOD> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction() {
        let _a = Cyclic::<2>::from(0);
        let _b = Cyclic::<65521>::from(1);
    }

    #[test]
    #[should_panic(expected = "modulus values must be a prime number greater than or equal to 2")]
    fn modulus_too_low() {
        let _a = Cyclic::<1>::from(0);
    }

    #[test]
    fn negation() {
        assert_eq!(-Cyclic::<7>::from(0), Cyclic::<7>::from(0));
        assert_eq!(-Cyclic::<5>::from(3), Cyclic::<5>::from(2));
        assert_eq!(-Cyclic::<2>::from(3), Cyclic::<2>::from(1));
    }

    #[test]
    fn addition() {
        assert_eq!(
            Cyclic::<5>::from(1) + Cyclic::<5>::from(2),
            Cyclic::<5>::from(3)
        );
        assert_eq!(
            Cyclic::<3>::from(4) + Cyclic::<3>::from(2),
            Cyclic::<3>::from(0)
        );

        let mut a = Cyclic::<11>::from(15);
        a += Cyclic::<11>::from(2);
        assert_eq!(a, Cyclic::<11>::from(6));
    }

    #[test]
    fn subtraction() {
        assert_eq!(
            Cyclic::<13>::from(11) - Cyclic::<13>::from(10),
            Cyclic::<13>::from(1)
        );
        assert_eq!(
            Cyclic::<3>::from(0) - Cyclic::<3>::from(7),
            Cyclic::<3>::from(2)
        );

        let mut a = Cyclic::<23>::from(11);
        a -= Cyclic::<23>::from(38);
        assert_eq!(a, Cyclic::<23>::from(19));
    }

    #[test]
    fn multiplication() {
        assert_eq!(
            Cyclic::<17>::from(4) * Cyclic::<17>::from(20),
            Cyclic::<17>::from(12)
        );
        assert_eq!(
            Cyclic::<31>::from(61) * Cyclic::<31>::from(29),
            Cyclic::<31>::from(2)
        );

        let mut a = Cyclic::<11>::from(21);
        a *= Cyclic::<11>::from(2);
        assert_eq!(a, Cyclic::<11>::from(9));
        a *= Cyclic::<11>::from(11);
        assert_eq!(a, Cyclic::<11>::from(0));
    }

    #[test]
    fn inversion() {
        assert_eq!(Cyclic::<2>::from(1).invert(), Cyclic::<2>::from(1));
        assert_eq!(
            Cyclic::<2>::from(1) * Cyclic::<2>::from(1).invert(),
            Cyclic::<2>::from(1)
        );

        assert_eq!(Cyclic::<5>::from(3).invert(), Cyclic::<5>::from(2));
        assert_eq!(
            Cyclic::<5>::from(3) * Cyclic::<5>::from(3).invert(),
            Cyclic::<5>::from(1)
        );

        assert_eq!(Cyclic::<541>::from(327).invert(), Cyclic::<541>::from(316));
        assert_eq!(
            Cyclic::<541>::from(327) * Cyclic::<541>::from(327).invert(),
            Cyclic::<541>::from(1)
        );
    }

    #[test]
    #[should_panic(expected = "attempting to invert equivalency class zero")]
    fn attempt_zero_inversion() {
        Cyclic::<17>::from(0).invert();
    }

    #[test]
    fn handle_overflow_and_underflow() {
        assert_eq!(
            Cyclic::<7>::from(u64::MAX - 3) + Cyclic::<7>::from(5),
            Cyclic::<7>::from(3)
        );
        assert_eq!(
            Cyclic::<5>::from(u64::MAX - 4) * Cyclic::<5>::from(2),
            Cyclic::<5>::from(2)
        );
        assert_eq!(
            Cyclic::<3>::from(1) - Cyclic::<3>::from(18),
            Cyclic::<3>::from(1)
        );
    }

    #[test]
    fn display() {
        assert_eq!(Cyclic::<7>::from(16).to_string(), "2 (mod 7)");
    }
}
