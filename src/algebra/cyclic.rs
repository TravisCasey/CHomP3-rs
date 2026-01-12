// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The `Cyclic` class implementing the cyclic field of integers with
//! configurable modulus.

use std::{
    convert::From,
    fmt::{Debug, Display, Error, Formatter},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use flint_sys::nmod_vec::{nmod_add, nmod_init, nmod_inv, nmod_mul, nmod_neg, nmod_sub, nmod_t};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize, de::Visitor};

use super::RingLike;

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
#[derive(Copy, Clone)]
pub struct Cyclic<const MOD: u64> {
    remainder: u64,
    modulus: nmod_t,
}

impl<const MOD: u64> Cyclic<MOD> {
    /// Create a new `Cyclic` instance with the given value modulo `MOD`.
    ///
    /// # Panics
    ///
    /// If `MOD` is less than 2; `MOD` is also expected to be a prime number,
    /// though this is not explicitly checked.
    #[must_use]
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
            nmod_init(&raw mut modulus, MOD);
        }

        Self {
            remainder: value % MOD,
            modulus,
        }
    }

    /// Return the canonical representative value (remainder) of this element.
    ///
    /// The returned value is always in the range `[0, MOD)`.
    #[must_use]
    pub fn value(&self) -> u64 {
        self.remainder
    }
}

impl<const MOD: u64> Debug for Cyclic<MOD> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{} (mod {})", self.remainder, MOD)
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

#[cfg(feature = "serde")]
impl<const MOD: u64> Serialize for Cyclic<MOD> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u64(self.remainder)
    }
}

#[cfg(feature = "serde")]
impl<'de, const MOD: u64> Deserialize<'de> for Cyclic<MOD> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct U64Visitor;

        impl Visitor<'_> for U64Visitor {
            type Value = u64;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("any 64 bit integer")
            }

            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(v)
            }
        }

        deserializer.deserialize_u64(U64Visitor).map(Cyclic::new)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "serde")]
    use serde_test::{Token, assert_tokens};

    use super::*;

    #[test]
    fn construction_and_modulo() {
        assert_eq!(Cyclic::<2>::from(4).value(), 0);
        assert_eq!(Cyclic::<2>::from(5).value(), 1);
        assert_eq!(Cyclic::<65521>::from(1).value(), 1);
        assert_eq!(Cyclic::<7>::from(16).value(), 2);
    }

    #[test]
    #[should_panic(expected = "modulus values must be a prime number greater than or equal to 2")]
    fn invalid_modulus() {
        let _a = Cyclic::<1>::from(0);
    }

    #[test]
    fn arithmetic_operations() {
        // Addition
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

        // Subtraction
        assert_eq!(
            Cyclic::<13>::from(11) - Cyclic::<13>::from(10),
            Cyclic::<13>::from(1)
        );
        assert_eq!(
            Cyclic::<3>::from(0) - Cyclic::<3>::from(7),
            Cyclic::<3>::from(2)
        );

        let mut b = Cyclic::<23>::from(11);
        b -= Cyclic::<23>::from(38);
        assert_eq!(b, Cyclic::<23>::from(19));

        // Multiplication
        assert_eq!(
            Cyclic::<17>::from(4) * Cyclic::<17>::from(20),
            Cyclic::<17>::from(12)
        );
        assert_eq!(
            Cyclic::<31>::from(61) * Cyclic::<31>::from(29),
            Cyclic::<31>::from(2)
        );

        let mut c = Cyclic::<11>::from(21);
        c *= Cyclic::<11>::from(2);
        assert_eq!(c, Cyclic::<11>::from(9));
        c *= Cyclic::<11>::from(11);
        assert_eq!(c, Cyclic::<11>::from(0));

        // Negation
        assert_eq!(-Cyclic::<7>::from(0), Cyclic::<7>::from(0));
        assert_eq!(-Cyclic::<5>::from(3), Cyclic::<5>::from(2));
        assert_eq!(-Cyclic::<2>::from(3), Cyclic::<2>::from(1));
    }

    #[test]
    fn field_properties() {
        // Zero and one
        assert_eq!(Cyclic::<5>::zero(), Cyclic::<5>::from(0));
        assert_eq!(Cyclic::<5>::one(), Cyclic::<5>::from(1));

        // Invertibility
        assert!(!Cyclic::<7>::from(0).is_invertible());
        assert!(Cyclic::<7>::from(1).is_invertible());
        assert!(Cyclic::<7>::from(5).is_invertible());

        // Inversion for modulus 2
        assert_eq!(Cyclic::<2>::from(1).invert(), Cyclic::<2>::from(1));
        assert_eq!(
            Cyclic::<2>::from(1) * Cyclic::<2>::from(1).invert(),
            Cyclic::<2>::one()
        );

        // General inversion
        let x = Cyclic::<5>::from(3);
        assert_eq!(x * x.invert(), Cyclic::<5>::one());

        let y = Cyclic::<541>::from(327);
        assert_eq!(y.invert(), Cyclic::<541>::from(316));
        assert_eq!(y * y.invert(), Cyclic::<541>::one());
    }

    #[test]
    #[should_panic(expected = "attempting to invert equivalency class zero")]
    fn zero_inversion_panics() {
        let _ = Cyclic::<17>::from(0).invert();
    }

    #[test]
    fn overflow_underflow_handling() {
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
    fn display_formatting() {
        assert_eq!(Cyclic::<7>::from(16).to_string(), "2 (mod 7)");
        assert_eq!(format!("{:?}", Cyclic::<5>::from(3)), "3 (mod 5)");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization() {
        let a = Cyclic::<11>::from(6);
        assert_tokens(&a, &[Token::U64(6)]);
    }
}
