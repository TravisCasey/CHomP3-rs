/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
//! The `Cyclic` class implementing the cyclic field of integers with configurable modulus.

use std::fmt::{Display, Error, Formatter};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The field of integers modulo `MOD`, for prime modulus values between 2 and 65521 (representable
/// in 16 bits). `MOD` is not explicitly checked to be prime, but the implementation of `invert` and
/// other dependent methods may assume this. Overflow and underflow are handled by the
/// implementation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Cyclic<const MOD: u32> {
    remainder: u32,
}

impl<const MOD: u32> Cyclic<MOD> {
    fn new_unchecked(value: u32) -> Self {
        Self { remainder: value }
    }

    pub fn new(value: u32) -> Self {
        assert!(
            MOD > 1,
            "modulus values must be a prime number greater than or equal to 2"
        );
        assert!(
            MOD <= u32::MAX / MOD,
            "modulus values must be a prime number no greater than 65536"
        );

        Self {
            remainder: value % MOD,
        }
    }

    pub fn invert(&self) -> Self {
        assert!(
            self.remainder != 0,
            "attempting to invert equivalency class zero"
        );
        if MOD == 2 {
            return *self;
        }

        // Based on Euler's theorem and the assumption that MOD is prime. In this case, the inverse
        // of self is self to the power of (MOD - 2) (mod MOD).
        // This code is inefficient and may need to be reworked if this method is used in time-
        // critical code.
        let mut inverse = *self;
        for _ in 0..(MOD - 3) {
            inverse *= *self;
        }
        inverse
    }
}

impl<const MOD: u32> Display for Cyclic<MOD> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{} (mod {})", self.remainder, MOD)
    }
}

impl<const MOD: u32> Neg for Cyclic<MOD> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self.remainder {
            0u32 => self,
            _ => Self::new_unchecked(MOD - self.remainder),
        }
    }
}

impl<const MOD: u32> AddAssign for Cyclic<MOD> {
    fn add_assign(&mut self, rhs: Self) {
        self.remainder += rhs.remainder;
        if self.remainder >= MOD {
            self.remainder -= MOD;
        }
    }
}

impl<const MOD: u32> Add for Cyclic<MOD> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut elem = self;
        elem += rhs;
        elem
    }
}

impl<const MOD: u32> SubAssign for Cyclic<MOD> {
    fn sub_assign(&mut self, rhs: Self) {
        if self.remainder >= rhs.remainder {
            self.remainder -= rhs.remainder;
        } else {
            self.remainder += MOD - rhs.remainder;
        }
    }
}

impl<const MOD: u32> Sub for Cyclic<MOD> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut elem = self;
        elem -= rhs;
        elem
    }
}

impl<const MOD: u32> MulAssign for Cyclic<MOD> {
    fn mul_assign(&mut self, rhs: Self) {
        // MOD is required to satisfy MOD * MOD <= u32::MAX by Cyclic::new
        self.remainder *= rhs.remainder;
        self.remainder %= MOD;
    }
}

impl<const MOD: u32> Mul for Cyclic<MOD> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut elem = self;
        elem *= rhs;
        elem
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction() {
        Cyclic::<2>::new(0);
        Cyclic::<65521>::new(1);
    }

    #[test]
    #[should_panic(expected = "modulus values must be a prime number greater than or equal to 2")]
    fn modulus_too_low() {
        Cyclic::<1>::new(0);
    }

    #[test]
    #[should_panic(expected = "modulus values must be a prime number no greater than 65536")]
    fn modulus_too_high() {
        Cyclic::<65537>::new(0);
    }

    #[test]
    fn negation() {
        assert_eq!(-Cyclic::<7>::new(0), Cyclic::<7>::new(0));
        assert_eq!(-Cyclic::<5>::new(3), Cyclic::<5>::new(2));
        assert_eq!(-Cyclic::<2>::new(3), Cyclic::<2>::new(1));
    }

    #[test]
    fn addition() {
        assert_eq!(
            Cyclic::<5>::new(1) + Cyclic::<5>::new(2),
            Cyclic::<5>::new(3)
        );
        assert_eq!(
            Cyclic::<3>::new(4) + Cyclic::<3>::new(2),
            Cyclic::<3>::new(0)
        );

        let mut a = Cyclic::<11>::new(15);
        a += Cyclic::<11>::new(2);
        assert_eq!(a, Cyclic::<11>::new(6));
    }

    #[test]
    fn subtraction() {
        assert_eq!(
            Cyclic::<13>::new(11) - Cyclic::<13>::new(10),
            Cyclic::<13>::new(1)
        );
        assert_eq!(
            Cyclic::<3>::new(0) - Cyclic::<3>::new(7),
            Cyclic::<3>::new(2)
        );

        let mut a = Cyclic::<23>::new(11);
        a -= Cyclic::<23>::new(38);
        assert_eq!(a, Cyclic::<23>::new(19));
    }

    #[test]
    fn multiplication() {
        assert_eq!(
            Cyclic::<17>::new(4) * Cyclic::<17>::new(20),
            Cyclic::<17>::new(12)
        );
        assert_eq!(
            Cyclic::<31>::new(61) * Cyclic::<31>::new(29),
            Cyclic::<31>::new(2)
        );

        let mut a = Cyclic::<11>::new(21);
        a *= Cyclic::<11>::new(2);
        assert_eq!(a, Cyclic::<11>::new(9));
        a *= Cyclic::<11>::new(11);
        assert_eq!(a, Cyclic::<11>::new(0));
    }

    #[test]
    fn inversion() {
        assert_eq!(Cyclic::<2>::new(1).invert(), Cyclic::<2>::new(1));
        assert_eq!(Cyclic::<5>::new(3).invert(), Cyclic::<5>::new(2));
        assert_eq!(Cyclic::<541>::new(327).invert(), Cyclic::<541>::new(316));
    }

    #[test]
    #[should_panic(expected = "attempting to invert equivalency class zero")]
    fn attempt_zero_inversion() {
        Cyclic::<17>::new(0).invert();
    }

    #[test]
    fn display() {
        assert_eq!(Cyclic::<7>::new(16).to_string(), "2 (mod 7)");
    }
}
