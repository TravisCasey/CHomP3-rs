// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Finite cyclic field with configurable prime modulus.

use std::{
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use flint_sys::nmod_vec::{nmod_add, nmod_init, nmod_inv, nmod_mul, nmod_neg, nmod_sub, nmod_t};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize, de::Visitor};

use super::Ring;

/// The field of integers modulo a prime `MOD`.
///
/// ### Mathematical Correctness
///
/// `MOD` *must* be a prime number to form an
/// [`integral domain`](https://en.wikipedia.org/wiki/Integral_domain), which
/// is required (but not checked) by the [`Ring`] trait. Primality of `MOD` is
/// not checked at compile time, but downstream functionality assumes this
/// property. Using a composite modulus may lead to incorrect results.
///
/// Overflow and underflow are handled by the implementation.
///
/// # Examples
///
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

/// The field with two elements. Alias for [`Cyclic<2>`].
pub type F2 = Cyclic<2>;

// All FLINT `nmod_*` functions are safe when called with a valid `nmod_t`
// initialized by `nmod_init`. The `modulus` field is always initialized in
// `Cyclic::new`, so all `unsafe` calls below satisfy this precondition.

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
        // SAFETY: `nmod_init` only writes to the provided pointer with no other side
        // effects.
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
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{} (mod {})", self.remainder, MOD)
    }
}

impl<const MOD: u64> From<u64> for Cyclic<MOD> {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl<const MOD: u64> Ring for Cyclic<MOD> {
    fn zero() -> Self {
        Self::new(0)
    }

    fn one() -> Self {
        Self::new(1)
    }

    fn is_invertible(&self) -> bool {
        self.remainder != 0
    }

    fn invert(&self) -> Option<Self> {
        match self.remainder {
            0 => None,
            1 => Some(*self),
            _ => Some(Self {
                // SAFETY: `self.modulus` initialized in `new`.
                remainder: unsafe { nmod_inv(self.remainder, self.modulus) },
                modulus: self.modulus,
            }),
        }
    }
}

impl<const MOD: u64> Display for Cyclic<MOD> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{} (mod {})", self.remainder, MOD)
    }
}

impl<const MOD: u64> Neg for Cyclic<MOD> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        // SAFETY: `self.modulus` initialized in `new`.
        Self {
            remainder: unsafe { nmod_neg(self.remainder, self.modulus) },
            modulus: self.modulus,
        }
    }
}

impl<const MOD: u64> AddAssign for Cyclic<MOD> {
    fn add_assign(&mut self, rhs: Self) {
        // SAFETY: `self.modulus` initialized in `new`.
        self.remainder = unsafe { nmod_add(self.remainder, rhs.remainder, self.modulus) };
    }
}

impl<const MOD: u64> Add for Cyclic<MOD> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // SAFETY: `self.modulus` initialized in `new`.
        Self {
            remainder: unsafe { nmod_add(self.remainder, rhs.remainder, self.modulus) },
            modulus: self.modulus,
        }
    }
}

impl<const MOD: u64> SubAssign for Cyclic<MOD> {
    fn sub_assign(&mut self, rhs: Self) {
        // SAFETY: `self.modulus` initialized in `new`.
        self.remainder = unsafe { nmod_sub(self.remainder, rhs.remainder, self.modulus) };
    }
}

impl<const MOD: u64> Sub for Cyclic<MOD> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        // SAFETY: `self.modulus` initialized in `new`.
        Self {
            remainder: unsafe { nmod_sub(self.remainder, rhs.remainder, self.modulus) },
            modulus: self.modulus,
        }
    }
}

impl<const MOD: u64> MulAssign for Cyclic<MOD> {
    fn mul_assign(&mut self, rhs: Self) {
        // SAFETY: `self.modulus` initialized in `new`.
        self.remainder = unsafe { nmod_mul(self.remainder, rhs.remainder, self.modulus) };
    }
}

impl<const MOD: u64> Mul for Cyclic<MOD> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // SAFETY: `self.modulus` initialized in `new`.
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

impl<const MOD: u64> Hash for Cyclic<MOD> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.remainder.hash(state);
    }
}

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

            fn expecting(&self, formatter: &mut Formatter) -> FmtResult {
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
        assert_eq!(F2::from(4).value(), 0);
        assert_eq!(F2::from(5).value(), 1);
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
        assert_eq!(-F2::from(3), F2::from(1));
    }

    #[test]
    fn inversion() {
        // Zero and one
        assert_eq!(Cyclic::<5>::zero(), Cyclic::<5>::from(0));
        assert_eq!(Cyclic::<5>::one(), Cyclic::<5>::from(1));

        // Invertibility
        assert!(!Cyclic::<7>::from(0).is_invertible());
        assert!(Cyclic::<7>::from(1).is_invertible());
        assert!(Cyclic::<7>::from(5).is_invertible());

        // Inversion for modulus 2
        assert_eq!(F2::from(1).invert().unwrap(), F2::from(1));
        assert_eq!(F2::from(1) * F2::from(1).invert().unwrap(), F2::one());

        // General inversion
        let x = Cyclic::<5>::from(3);
        assert_eq!(x * x.invert().unwrap(), Cyclic::<5>::one());

        let y = Cyclic::<541>::from(327);
        assert_eq!(y.invert().unwrap(), Cyclic::<541>::from(316));
        assert_eq!(y * y.invert().unwrap(), Cyclic::<541>::one());
    }

    #[test]
    fn zero_inversion_none() {
        assert!(Cyclic::<17>::from(0).invert().is_none());
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

    #[test]
    #[cfg(feature = "serde")]
    fn serialization() {
        let a = Cyclic::<11>::from(6);
        assert_tokens(&a, &[Token::U64(6)]);
    }
}
