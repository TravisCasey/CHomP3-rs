// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! HashMap-based module implementation.
//!
//! This module provides [`HashMapModule`], a sparse implementation of
//! [`ModuleLike`] using hash tables for coefficient storage. This is the
//! recommended module implementation when the basis set is large and most
//! coefficients are zero.

use std::{
    collections::{HashMap, hash_map, hash_map::Entry},
    convert::From,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{BuildHasher, Hash, RandomState},
    iter::{Filter, FromIterator},
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

use super::{ModuleLike, RingLike};

/// HashMap-based implementation of algebraic modules.
///5
/// This module provides [`HashMapModule`], a concrete implementation of the
/// [`ModuleLike`] trait using [`std::collections::HashMap`] as the underlying
/// storage mechanism. This implementation is recommended when the number of
/// basis elements cannot be explicitly stored and only a small fraction of
/// basis elements have non-zero coefficients.
///
/// # Mathematical Background
///
/// An algebraic module over a ring `R` with basis type `B` represents formal
/// linear combinations of coefficients in the ring `R` and basis elements of
/// type `B`
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
/// module.insert_or_add(1, Cyclic::from(3));
/// module.insert_or_add(2, Cyclic::from(4));
/// module.insert_or_add(1, Cyclic::from(3)); // Adds to existing coefficient
///
/// assert_eq!(module.coefficient(&1), Cyclic::from(1)); // 3 + 3 = 1 (mod 5)
/// assert_eq!(module.coefficient(&2), Cyclic::from(4));
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
/// assert_eq!(sum.coefficient(&1), Cyclic::from(1)); // 3 + 5 = 1 (mod 7)
/// assert_eq!(sum.coefficient(&2), Cyclic::from(3));
/// assert_eq!(sum.coefficient(&3), Cyclic::from(1));
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
/// module = module.scalar_mul(Cyclic::from(2));
///
/// assert_eq!(module.coefficient(&1), Cyclic::from(4)); // 2 * 2 = 4 (mod 4)
/// assert_eq!(module.coefficient(&2), Cyclic::from(1)); // 3 * 2 = 6 = 1 (mod 5)
/// ```
#[derive(Clone)]
pub struct HashMapModule<B, R, S = RandomState> {
    map: HashMap<B, R, S>,
}

impl<B, R, S: Default> HashMapModule<B, R, S> {
    /// Create a new `HashMapModule` with the given capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: HashMap::<B, R, S>::with_capacity_and_hasher(capacity, S::default()),
        }
    }
}

impl<B: Clone + Debug + Eq + Hash, R: RingLike, H: BuildHasher + Default + Clone> ModuleLike
    for HashMapModule<B, R, H>
{
    type Cell = B;
    type Iter<'a>
        = Filter<hash_map::Iter<'a, B, R>, fn(&(&'a B, &'a R)) -> bool>
    where
        Self: 'a;
    type Ring = R;

    fn new() -> Self {
        Self {
            map: HashMap::<B, R, H>::with_hasher(H::default()),
        }
    }

    fn clear(&mut self) {
        self.map.clear();
    }

    fn coefficient(&self, cell: &B) -> R {
        self.map.get(cell).cloned().unwrap_or(R::zero())
    }

    fn coefficient_mut(&mut self, cell: &B) -> &mut R {
        self.map.entry(cell.clone()).or_insert_with(|| R::zero())
    }

    fn remove(&mut self, cell: &B) -> R {
        self.map.remove(cell).unwrap_or(R::zero())
    }

    fn scalar_mul(mut self, coefficient: R) -> Self {
        if coefficient == R::zero() {
            self.clear();
        } else if coefficient != R::one() {
            for cell_coefficient in self.map.values_mut() {
                *cell_coefficient *= coefficient.clone();
            }
        }
        self
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.map
            .iter()
            .filter(|&(_cell, coefficient)| *coefficient != R::zero())
    }

    fn insert_or_add(&mut self, cell: Self::Cell, coefficient: Self::Ring) {
        if coefficient != Self::Ring::zero() {
            match self.map.entry(cell) {
                Entry::Occupied(mut o) => {
                    if coefficient.clone() + o.get().clone() == Self::Ring::zero() {
                        o.remove();
                    } else {
                        *o.get_mut() += coefficient;
                    }
                },
                Entry::Vacant(v) => {
                    v.insert(coefficient);
                },
            }
        }
    }
}

impl<B, R, H> Display for HashMapModule<B, R, H>
where
    B: Display,
    R: RingLike + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let non_zero: Vec<_> = self
            .map
            .iter()
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
}

impl<B: Debug, R: Debug, H> Debug for HashMapModule<B, R, H> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{:?}", self.map)
    }
}

impl<B, R: RingLike, H> Neg for HashMapModule<B, R, H> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for coefficient in &mut self.map.values_mut() {
            *coefficient = -coefficient.clone();
        }
        self
    }
}

impl<B: Clone + Eq + Hash, R: RingLike, H: BuildHasher> AddAssign for HashMapModule<B, R, H>
where
    HashMapModule<B, R, H>: ModuleLike<Cell = B, Ring = R>,
{
    fn add_assign(&mut self, rhs: Self) {
        for (cell, coefficient) in &rhs.map {
            if *coefficient != R::zero() {
                let new_coefficient = self.coefficient(cell) + coefficient.clone();
                if new_coefficient == R::zero() {
                    self.map.remove(cell);
                } else {
                    self.map.insert(cell.clone(), new_coefficient);
                }
            }
        }
    }
}

impl<B, R: RingLike, H> Add for HashMapModule<B, R, H>
where
    HashMapModule<B, R, H>: ModuleLike,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<B: Clone + Eq + Hash, R: RingLike, H: BuildHasher> SubAssign for HashMapModule<B, R, H>
where
    HashMapModule<B, R, H>: ModuleLike<Cell = B, Ring = R>,
{
    fn sub_assign(&mut self, rhs: Self) {
        for (cell, coefficient) in &rhs.map {
            if *coefficient != R::zero() {
                let new_coefficient = self.coefficient(cell) - coefficient.clone();
                if new_coefficient == R::zero() {
                    self.map.remove(cell);
                } else {
                    self.map.insert(cell.clone(), new_coefficient);
                }
            }
        }
    }
}

impl<B, R: RingLike, H> Sub for HashMapModule<B, R, H>
where
    HashMapModule<B, R, H>: ModuleLike,
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<B, R, H> PartialEq for HashMapModule<B, R, H>
where
    B: Clone + Eq + Hash,
    R: RingLike,
{
    fn eq(&self, other: &Self) -> bool {
        // Filter out zero-coefficient entries and clone to build new Hashmaps
        let self_non_zero: HashMap<_, _> = self
            .map
            .iter()
            .filter(|(_, coefficient)| **coefficient != R::zero())
            .map(|(cell, coefficient)| (cell.clone(), coefficient.clone()))
            .collect();

        let other_non_zero: HashMap<_, _> = other
            .map
            .iter()
            .filter(|(_, coefficient)| **coefficient != R::zero())
            .map(|(cell, coefficient)| (cell.clone(), coefficient.clone()))
            .collect();

        self_non_zero == other_non_zero
    }
}

impl<B, R, H> Eq for HashMapModule<B, R, H> where HashMapModule<B, R, H>: PartialEq {}

impl<B, R, H, const N: usize> From<[(B, R); N]> for HashMapModule<B, R, H>
where
    B: Eq + Hash,
    H: BuildHasher + Default,
{
    fn from(items: [(B, R); N]) -> Self {
        Self {
            map: HashMap::<B, R, H>::from_iter(items),
        }
    }
}

impl<B, R, H> IntoIterator for HashMapModule<B, R, H>
where
    B: Eq + Hash,
    H: BuildHasher + Default,
{
    type IntoIter = std::collections::hash_map::IntoIter<B, R>;
    type Item = (B, R);

    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

impl<B, R, H> FromIterator<(B, R)> for HashMapModule<B, R, H>
where
    B: Eq + Hash,
    H: BuildHasher + Default,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (B, R)>,
    {
        Self {
            map: HashMap::<B, R, H>::from_iter(iter),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cyclic;

    #[test]
    fn construction_and_capacity() {
        let module1 = HashMapModule::<u32, Cyclic<5>>::new();
        assert_eq!(module1.coefficient(&0), Cyclic::from(0));

        let module2 = HashMapModule::<i32, Cyclic<7>>::with_capacity(100);
        assert_eq!(module2.coefficient(&99), Cyclic::from(0));
    }

    #[test]
    fn coefficient_access_and_mutation() {
        let mut module = HashMapModule::<u32, Cyclic<5>>::new();

        // Access non-existent returns zero
        assert_eq!(module.coefficient(&1), Cyclic::from(0));

        // Insert and access
        module.insert_or_add(1, Cyclic::from(3));
        assert_eq!(module.coefficient(&1), Cyclic::from(3));

        // Add to existing
        module.insert_or_add(1, Cyclic::from(2));
        assert_eq!(module.coefficient(&1), Cyclic::from(0)); // 3 + 2 = 0 (mod 5)

        // Mutable access
        *module.coefficient_mut(&2) = Cyclic::from(4);
        assert_eq!(module.coefficient(&2), Cyclic::from(4));

        // Remove
        assert_eq!(module.remove(&2), Cyclic::from(4));
        assert_eq!(module.remove(&2), Cyclic::from(0)); // Already removed

        // Clear
        module.insert_or_add(5, Cyclic::from(1));
        module.clear();
        assert_eq!(module.coefficient(&5), Cyclic::from(0));
    }

    #[test]
    fn module_arithmetic() {
        let module1 =
            HashMapModule::<i32, Cyclic<7>>::from([(1, Cyclic::from(3)), (2, Cyclic::from(5))]);
        let module2 = HashMapModule::from([(1, Cyclic::from(4)), (3, Cyclic::from(2))]);

        // Addition
        let sum = module1.clone() + module2.clone();
        assert_eq!(sum.coefficient(&1), Cyclic::from(0)); // 3 + 4 = 0 (mod 7)
        assert_eq!(sum.coefficient(&2), Cyclic::from(5));
        assert_eq!(sum.coefficient(&3), Cyclic::from(2));

        // Subtraction
        let diff = module1.clone() - module2.clone();
        assert_eq!(diff.coefficient(&1), Cyclic::from(6)); // 3 - 4 = -1 = 6 (mod 7)
        assert_eq!(diff.coefficient(&2), Cyclic::from(5));
        assert_eq!(diff.coefficient(&3), Cyclic::from(5)); // 0 - 2 = -2 = 5 (mod 7)

        // AddAssign
        let mut module3 = module1.clone();
        module3 += module2.clone();
        assert_eq!(module3, sum);

        // SubAssign
        let mut module4 = module1.clone();
        module4 -= module2;
        assert_eq!(module4, diff);

        // Negation
        let neg = -module1.clone();
        assert_eq!(neg.coefficient(&1), Cyclic::from(4)); // -3 = 4 (mod 7)
        assert_eq!(neg.coefficient(&2), Cyclic::from(2)); // -5 = 2 (mod 7)
    }

    #[test]
    fn scalar_multiplication() {
        let module =
            HashMapModule::<u32, Cyclic<5>>::from([(1, Cyclic::from(2)), (2, Cyclic::from(3))]);

        // Identity
        assert_eq!(module.clone().scalar_mul(Cyclic::one()), module);

        // General scalar
        let scaled = module.clone().scalar_mul(Cyclic::from(2));
        assert_eq!(scaled.coefficient(&1), Cyclic::from(4));
        assert_eq!(scaled.coefficient(&2), Cyclic::from(1)); // 3 * 2 = 6 ≡ 1 (mod 5)

        // Zero clears
        let zeroed = module.scalar_mul(Cyclic::zero());
        assert_eq!(zeroed.coefficient(&1), Cyclic::from(0));
        assert_eq!(zeroed.coefficient(&2), Cyclic::from(0));
    }

    #[test]
    fn equality_with_zeros() {
        let module1 = HashMapModule::<u32, Cyclic<5>>::from([
            (1, Cyclic::from(2)),
            (2, Cyclic::from(0)), // Explicit zero
        ]);
        let module2 = HashMapModule::from([(1, Cyclic::from(2))]);

        assert_eq!(module1, module2);
    }

    #[test]
    fn iterator_filters_zeros() {
        let mut module = HashMapModule::<u32, Cyclic<7>>::from([
            (1, Cyclic::from(3)),
            (2, Cyclic::from(0)),
            (3, Cyclic::from(5)),
        ]);
        module.insert_or_add(4, Cyclic::from(0));

        let mut collected: Vec<_> = module
            .iter()
            .map(|(cell, coefficient)| (*cell, *coefficient))
            .collect();
        collected.sort_by_key(|(cell, _)| *cell);

        assert_eq!(collected, vec![(1, Cyclic::from(3)), (3, Cyclic::from(5))]);

        // Empty module
        let empty = HashMapModule::<u32, Cyclic<7>>::new();
        assert_eq!(empty.iter().count(), 0);
    }

    #[test]
    fn from_array_and_into_iter() {
        let arr = [(1, Cyclic::<5>::from(2)), (3, Cyclic::from(4))];
        let module: HashMapModule<i32, Cyclic<5>> = HashMapModule::from(arr);

        assert_eq!(module.coefficient(&1), Cyclic::from(2));
        assert_eq!(module.coefficient(&3), Cyclic::from(4));

        let collected: Vec<_> = module.into_iter().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn display_formatting() {
        let empty = HashMapModule::<u32, Cyclic<5>>::new();
        assert_eq!(empty.to_string(), "0");

        let single: HashMapModule<u32, Cyclic<5>> =
            HashMapModule::from([(2, Cyclic::<5>::from(1))]);
        assert_eq!(single.to_string(), "2");

        let with_coef: HashMapModule<u32, Cyclic<5>> =
            HashMapModule::from([(3, Cyclic::<5>::from(2))]);
        assert_eq!(with_coef.to_string(), "2 (mod 5)*3");
    }
}
