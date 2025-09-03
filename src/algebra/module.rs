// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::collections::HashMap;
use std::convert::From;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::hash::{BuildHasher, Hash, RandomState};
use std::iter::FromIterator;
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use crate::{ModuleLike, RingLike};

/// HashMap-based implementation of algebraic modules.
///
/// This module provides [`HashMapModule`], a concrete implementation of the
/// [`ModuleLike`] trait using [`std::collections::HashMap`] as the underlying
/// storage mechanism. This implementation is recommended when the number of
/// basis elements cannot be explicitly stored and only a small fraction of
/// basis elements have non-zero coefficients.
///
/// # Mathematical Background
///
/// An algebraic module over a ring `R` with basis type `C` represents formal
/// linear combinations of the form:
///
/// $ a_1c_1 + a_2c_2 + ... + a_nc_n $
///
/// where:
/// - `a_i` are coefficients in the ring `R`
/// - `c_i` are basis elements of type `C`
///
/// When `R` is a field (satisfies [`crate::FieldLike`]), the module becomes a
/// vector space.
///
/// # Examples
///
/// ## Creating and manipulating modules
///
/// ```rust
/// use chomp3rs::{Cyclic, HashMapModule, ModuleLike};
///
/// // Create a module over the field Z/5Z with basis in u32
/// let mut module = HashMapModule::<u32, Cyclic<5>>::new();
///
/// // Insert some basis elements with coefficients
/// module.insert_or_add(&1, Cyclic::from(3));
/// module.insert_or_add(&2, Cyclic::from(4));
/// module.insert_or_add(&1, Cyclic::from(3)); // Adds to existing coefficient
///
/// assert_eq!(module.coef(&1), Cyclic::from(1)); // 3 + 3 = 1 (mod 5)
/// assert_eq!(module.coef(&2), Cyclic::from(4));
/// ```
///
/// ## Module arithmetic
///
/// ```rust
/// use chomp3rs::{Cyclic, HashMapModule, ModuleLike};
///
/// let module1 = HashMapModule::<i32, Cyclic<7>>::from([
///     (1, Cyclic::from(3)),
///     (2, Cyclic::from(3)),
/// ]);
/// let module2 =
///     HashMapModule::from([(1, Cyclic::from(5)), (3, Cyclic::from(1))]);
///
/// let sum = module1 + module2;
/// assert_eq!(sum.coef(&1), Cyclic::from(1)); // 3 + 5 = 1 (mod 7)
/// assert_eq!(sum.coef(&2), Cyclic::from(3));
/// assert_eq!(sum.coef(&3), Cyclic::from(1));
/// ```
///
/// ## Scalar multiplication
///
/// ```rust
/// use chomp3rs::{Cyclic, HashMapModule, ModuleLike};
///
/// let mut module = HashMapModule::<i32, Cyclic<5>>::from([
///     (1, Cyclic::from(2)),
///     (2, Cyclic::from(3)),
/// ]);
/// module.scalar_mul(Cyclic::from(2));
///
/// assert_eq!(module.coef(&1), Cyclic::from(4)); // 2 * 2 = 4 (mod 4)
/// assert_eq!(module.coef(&2), Cyclic::from(1)); // 3 * 2 = 6 = 1 (mod 5)
/// ```
#[derive(Clone, Debug)]
pub struct HashMapModule<C, R, H = RandomState> {
    map: HashMap<C, R, H>,
}

impl<C, R, H: Default> HashMapModule<C, R, H> {
    /// Create a new `HashMapModule` with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: HashMap::<C, R, H>::with_capacity_and_hasher(capacity, H::default()),
        }
    }
}

impl<C: Clone + Eq + Hash, R: RingLike, H: BuildHasher + Default + Clone> ModuleLike
    for HashMapModule<C, R, H>
{
    type Cell = C;
    type Ring = R;

    fn new() -> Self {
        Self {
            map: HashMap::<C, R, H>::with_hasher(H::default()),
        }
    }

    fn clear(&mut self) {
        self.map.clear();
    }

    fn coef(&self, cell: &C) -> R {
        self.map.get(cell).cloned().unwrap_or(R::zero())
    }

    fn coef_mut(&mut self, cell: &C) -> &mut R {
        self.map.entry(cell.clone()).or_insert_with(|| R::zero())
    }

    fn scalar_mul(&mut self, coef: R) {
        if coef == R::zero() {
            self.clear();
        } else if coef != R::one() {
            for (_, cell_coef) in self.map.iter_mut() {
                *cell_coef *= coef.clone();
            }
        }
    }
}

impl<C, R, H> Display for HashMapModule<C, R, H>
where
    C: Display,
    R: RingLike + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let non_zero: Vec<_> = self
            .map
            .iter()
            .filter(|(_, coef)| **coef != R::zero())
            .collect();

        if non_zero.is_empty() {
            return write!(f, "0");
        }

        let mut first = true;
        for (cell, coef) in non_zero {
            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if *coef == R::one() {
                write!(f, "{}", cell)?;
            } else {
                write!(f, "{}*{}", coef, cell)?;
            }
        }
        Ok(())
    }
}

impl<C, R: RingLike, H> Neg for HashMapModule<C, R, H> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for (_cell, coef) in self.map.iter_mut() {
            *coef = -coef.clone();
        }
        self
    }
}

impl<C: Clone + Eq + Hash, R: RingLike, H: BuildHasher> AddAssign for HashMapModule<C, R, H>
where
    HashMapModule<C, R, H>: ModuleLike<Cell = C, Ring = R>,
{
    fn add_assign(&mut self, rhs: Self) {
        for (cell, coef) in rhs.map.iter() {
            if *coef != R::zero() {
                let new_coef = self.coef(cell) + coef.clone();
                if new_coef != R::zero() {
                    self.map.insert(cell.clone(), new_coef);
                } else {
                    self.map.remove(cell);
                }
            }
        }
    }
}

impl<C, R: RingLike, H> Add for HashMapModule<C, R, H>
where
    HashMapModule<C, R, H>: ModuleLike,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<C: Clone + Eq + Hash, R: RingLike, H: BuildHasher> SubAssign for HashMapModule<C, R, H>
where
    HashMapModule<C, R, H>: ModuleLike<Cell = C, Ring = R>,
{
    fn sub_assign(&mut self, rhs: Self) {
        for (cell, coef) in rhs.map.iter() {
            if *coef != R::zero() {
                let new_coef = self.coef(cell) - coef.clone();
                if new_coef != R::zero() {
                    self.map.insert(cell.clone(), new_coef);
                } else {
                    self.map.remove(cell);
                }
            }
        }
    }
}

impl<C, R: RingLike, H> Sub for HashMapModule<C, R, H>
where
    HashMapModule<C, R, H>: ModuleLike,
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<C, R, H> PartialEq for HashMapModule<C, R, H>
where
    C: Clone + Eq + Hash,
    R: RingLike,
{
    fn eq(&self, other: &Self) -> bool {
        // Filter out zero-coefficient entries and clone to build new Hashmaps
        let self_non_zero: HashMap<_, _> = self
            .map
            .iter()
            .filter(|(_, coef)| **coef != R::zero())
            .map(|(cell, coef)| (cell.clone(), coef.clone()))
            .collect();

        let other_non_zero: HashMap<_, _> = other
            .map
            .iter()
            .filter(|(_, coef)| **coef != R::zero())
            .map(|(cell, coef)| (cell.clone(), coef.clone()))
            .collect();

        self_non_zero == other_non_zero
    }
}

impl<C, R, H> Eq for HashMapModule<C, R, H> where HashMapModule<C, R, H>: PartialEq {}

impl<C, R, H, const N: usize> From<[(C, R); N]> for HashMapModule<C, R, H>
where
    C: Eq + Hash,
    H: BuildHasher + Default,
{
    fn from(items: [(C, R); N]) -> Self {
        Self {
            map: HashMap::<C, R, H>::from_iter(items),
        }
    }
}

impl<C, R, H> FromIterator<(C, R)> for HashMapModule<C, R, H>
where
    C: Eq + Hash,
    H: BuildHasher + Default,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (C, R)>,
    {
        Self {
            map: HashMap::<C, R, H>::from_iter(iter),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::hash::Hasher;

    use super::*;
    use crate::Cyclic;

    #[derive(Clone, Debug, Default)]
    struct SimpleHashBuilder {}

    impl BuildHasher for SimpleHashBuilder {
        type Hasher = SimpleHash;

        fn build_hasher(&self) -> Self::Hasher {
            SimpleHash { state: 0 }
        }
    }

    struct SimpleHash {
        state: u64,
    }

    impl Hasher for SimpleHash {
        fn finish(&self) -> u64 {
            self.state
        }

        fn write(&mut self, bytes: &[u8]) {
            for byte in bytes {
                self.state = self.state.wrapping_add(*byte as u64);
            }
        }
    }

    fn initialize_test_objects() -> (
        HashMapModule<u32, Cyclic<5>>,
        HashMapModule<(u16, u16), Cyclic<31>, SimpleHashBuilder>,
    ) {
        (
            HashMapModule::<u32, Cyclic<5>>::new(),
            HashMapModule::<(u16, u16), Cyclic<31>, SimpleHashBuilder>::with_capacity(20),
        )
    }

    fn populate_test_objects(
        rs_module: &mut HashMapModule<u32, Cyclic<5>>,
        sh_module: &mut HashMapModule<(u16, u16), Cyclic<31>, SimpleHashBuilder>,
    ) {
        rs_module.insert_or_add(&0, Cyclic::from(2));
        rs_module.insert_or_add(&10, Cyclic::from(0));
        rs_module.insert_or_add(&12, Cyclic::from(4));

        sh_module.insert_or_add(&(13, 0), Cyclic::from(21));
        sh_module.insert_or_add(&(2, 25), Cyclic::from(4));
        sh_module.insert_or_add(&(17, 17), Cyclic::from(0));
    }

    #[test]
    fn construction() {
        let (_rs_module, _sh_module) = initialize_test_objects();
    }

    #[test]
    fn insertion_and_access() {
        let (mut rs_module, mut sh_module) = initialize_test_objects();
        populate_test_objects(&mut rs_module, &mut sh_module);

        assert_eq!(
            rs_module,
            HashMapModule::from([
                (0, Cyclic::from(2)),
                (10, Cyclic::from(0)),
                (12, Cyclic::from(4)),
                (20, Cyclic::from(0)), // not present
            ])
        );
        assert_eq!(
            sh_module,
            HashMapModule::from([
                ((13, 0), Cyclic::from(21)),
                ((2, 25), Cyclic::from(4)),
                ((17, 17), Cyclic::from(0)),
                ((11, 0), Cyclic::from(0)), // not present
            ])
        );

        let old_rs_module = rs_module.clone();
        let old_sh_module = sh_module.clone();

        rs_module.insert_or_add(&13, Cyclic::from(1)); // new
        rs_module.insert_or_add(&12, Cyclic::from(2)); // add to existing
        sh_module.insert_or_add(&(8, 90), Cyclic::from(10)); // new
        sh_module.insert_or_add(&(17, 17), Cyclic::from(18)); // add to existing

        assert_ne!(rs_module, old_rs_module);
        assert_ne!(sh_module, old_sh_module);

        assert_eq!(
            rs_module,
            HashMapModule::from([
                (0, Cyclic::from(2)),
                (10, Cyclic::from(0)),
                (12, Cyclic::from(1)),
                (20, Cyclic::from(0)), // not present
                (13, Cyclic::from(1)),
            ])
        );
        assert_eq!(
            sh_module,
            HashMapModule::from([
                ((13, 0), Cyclic::from(21)),
                ((2, 25), Cyclic::from(4)),
                ((17, 17), Cyclic::from(18)),
                ((11, 0), Cyclic::from(0)), // not present
                ((8, 90), Cyclic::from(10)),
            ])
        );
    }

    #[test]
    fn test_addition() {
        let (mut rs_module, mut sh_module) = initialize_test_objects();
        populate_test_objects(&mut rs_module, &mut sh_module);

        let rs_result = rs_module.clone()
            + HashMapModule::<u32, Cyclic<5>>::from([(0, Cyclic::from(1)), (10, Cyclic::from(2))]);
        let sh_result = sh_module.clone()
            + HashMapModule::<(u16, u16), Cyclic<31>, SimpleHashBuilder>::from([
                ((13, 0), Cyclic::from(10)),
                ((2, 25), Cyclic::from(5)),
            ]);

        assert_eq!(
            rs_result,
            HashMapModule::from([
                (0, Cyclic::from(3)),
                (10, Cyclic::from(2)),
                (12, Cyclic::from(4)),
                (20, Cyclic::from(0)),
            ]),
        );
        assert_eq!(
            sh_result,
            HashMapModule::from([
                ((13, 0), Cyclic::from(0)),
                ((2, 25), Cyclic::from(9)),
                ((17, 17), Cyclic::from(0)),
                ((11, 0), Cyclic::from(0)),
            ]),
        );

        rs_module +=
            HashMapModule::<u32, Cyclic<5>>::from([(0, Cyclic::from(1)), (10, Cyclic::from(2))]);
        sh_module += HashMapModule::<(u16, u16), Cyclic<31>, SimpleHashBuilder>::from([
            ((13, 0), Cyclic::from(10)),
            ((2, 25), Cyclic::from(5)),
        ]);

        assert_eq!(
            rs_result,
            HashMapModule::from([
                (0, Cyclic::from(3)),
                (10, Cyclic::from(2)),
                (12, Cyclic::from(4)),
                (20, Cyclic::from(0)),
            ]),
        );
        assert_eq!(
            sh_result,
            HashMapModule::from([
                ((13, 0), Cyclic::from(0)),
                ((2, 25), Cyclic::from(9)),
                ((17, 17), Cyclic::from(0)),
                ((11, 0), Cyclic::from(0)),
            ]),
        );
    }

    #[test]
    fn test_subtraction() {
        let (mut rs_module, mut sh_module) = initialize_test_objects();
        populate_test_objects(&mut rs_module, &mut sh_module);

        let rs_result = rs_module.clone()
            - HashMapModule::<u32, Cyclic<5>>::from([(0, Cyclic::from(1)), (10, Cyclic::from(2))]);
        let sh_result = sh_module.clone()
            - HashMapModule::<(u16, u16), Cyclic<31>, SimpleHashBuilder>::from([
                ((13, 0), Cyclic::from(10)),
                ((2, 25), Cyclic::from(5)),
            ]);

        assert_eq!(
            rs_result,
            HashMapModule::from([
                (0, Cyclic::from(1)),
                (10, Cyclic::from(3)),
                (12, Cyclic::from(4)),
                (20, Cyclic::from(0)),
            ]),
        );
        assert_eq!(
            sh_result,
            HashMapModule::from([
                ((13, 0), Cyclic::from(11)),
                ((2, 25), Cyclic::from(30)),
                ((17, 17), Cyclic::from(0)),
                ((11, 0), Cyclic::from(0)),
            ]),
        );

        rs_module -=
            HashMapModule::<u32, Cyclic<5>>::from([(0, Cyclic::from(1)), (10, Cyclic::from(2))]);
        sh_module -= HashMapModule::<(u16, u16), Cyclic<31>, SimpleHashBuilder>::from([
            ((13, 0), Cyclic::from(10)),
            ((2, 25), Cyclic::from(5)),
        ]);

        assert_eq!(
            rs_result,
            HashMapModule::from([
                (0, Cyclic::from(1)),
                (10, Cyclic::from(3)),
                (12, Cyclic::from(4)),
                (20, Cyclic::from(0)),
            ]),
        );
        assert_eq!(
            sh_result,
            HashMapModule::from([
                ((13, 0), Cyclic::from(11)),
                ((2, 25), Cyclic::from(30)),
                ((17, 17), Cyclic::from(0)),
                ((11, 0), Cyclic::from(0)),
            ]),
        );
    }

    #[test]
    fn test_scalar_multiplication() {
        let mut module =
            HashMapModule::<u32, Cyclic<5>>::from([(1, Cyclic::from(2)), (2, Cyclic::from(3))]);

        // Multiplication by 1 should be no-op
        let original = module.clone();
        module.scalar_mul(Cyclic::from(1));
        assert_eq!(module, original);

        module.scalar_mul(Cyclic::from(2));
        assert_eq!(module.coef(&1), Cyclic::from(4));
        assert_eq!(module.coef(&2), Cyclic::from(1));

        // Multiplication by 0 should clear
        module.scalar_mul(Cyclic::from(0));
        assert_eq!(module, HashMapModule::new());
    }
}
