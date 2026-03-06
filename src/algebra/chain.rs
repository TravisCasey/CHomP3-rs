// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Hash table-based sparse chain implementation.
//!
//! [`Chain`] stores basis element/coefficient pairs in a [`HashMap`], providing
//! O(1) average-case access. Iteration order is non-deterministic.
//!
//! For deterministic, sorted iteration order, use
//! [`OrderedChain`] instead.

use std::{
    collections::{
        HashMap,
        hash_map::{self, Entry},
    },
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{BuildHasher, Hash, RandomState},
    iter::FusedIterator,
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

use super::{OrderedChain, Ring};

/// Hash table-based sparse chain.
///
/// Stores basis element/coefficient pairs in a [`HashMap`], providing O(1)
/// average-case access. Iteration order is non-deterministic.
///
/// For deterministic iteration order, prefer [`OrderedChain`] instead. Use
/// [`to_ordered`](Self::to_ordered) to convert chain types.
///
/// # Examples
///
/// ## Construction and Manipulation
///
/// ```rust
/// use chomp3rs::{Chain, Cyclic};
///
/// let chain1 = Chain::<i32, Cyclic<7>>::from([
///     (1, Cyclic::from(3)),
///     (2, Cyclic::from(3)),
/// ]);
/// let chain2 = Chain::from([(1, Cyclic::from(5)), (3, Cyclic::from(1))]);
///
/// let sum = chain1 + chain2;
/// assert_eq!(sum.coefficient(&1), Cyclic::from(1)); // 3 + 5 = 1 (mod 7)
/// assert_eq!(sum.coefficient(&2), Cyclic::from(3));
/// assert_eq!(sum.coefficient(&3), Cyclic::from(1));
/// ```
///
/// ## Scalar Multiplication
///
/// ```rust
/// use chomp3rs::{Chain, Cyclic};
///
/// let chain = Chain::<u32, Cyclic<5>>::from([
///     (1, Cyclic::from(2)),
///     (2, Cyclic::from(3)),
/// ]);
/// let scaled = chain.scalar_mul(&Cyclic::from(3));
/// assert_eq!(
///     scaled.coefficient(&1),
///     Cyclic::from(1)
/// ); // 2 * 3 = 6 = 1 (mod 5)
/// assert_eq!(
///     scaled.coefficient(&2),
///     Cyclic::from(4)
/// ); // 3 * 3 = 9 = 4 (mod 5)
/// ```
///
/// # Zero-Storage Semantics
///
/// Core operations ([`insert_or_add`](Self::insert_or_add), arithmetic
/// operators) eagerly remove zero-coefficient entries. However,
/// constructors ([`From`], [`FromIterator`]) and direct map access
/// ([`inner_mut`](Self::inner_mut)) do not canonicalize, so zero
/// entries may be present. [`PartialEq`] filters zeros, so chains
/// with incidental zeros still compare correctly.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "B: serde::Serialize + Eq + Hash, R: serde::Serialize, S: BuildHasher",
        deserialize = "B: serde::Deserialize<'de> + Eq + Hash, R: serde::Deserialize<'de>, S: \
                       BuildHasher + Default"
    ))
)]
pub struct Chain<B, R, S = RandomState> {
    map: HashMap<B, R, S>,
}

/// Iterator over `(cell, coefficient)` pairs in a [`Chain`].
///
/// Created by [`Chain::iter`]. Does not filter zero-coefficient entries;
/// core operations ([`Chain::insert_or_add`], arithmetic) eagerly remove
/// zeros, so the map should not contain zero entries under normal use.
pub struct ChainIter<'a, B, R>(hash_map::Iter<'a, B, R>);

impl<'a, B, R> Iterator for ChainIter<'a, B, R> {
    type Item = (&'a B, &'a R);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<B, R> ExactSizeIterator for ChainIter<'_, B, R> {}
impl<B, R> FusedIterator for ChainIter<'_, B, R> {}

/// Mutable iterator over `(cell, coefficient)` pairs in a [`Chain`].
///
/// Created by [`Chain::iter_mut`]. Yields `(&B, &mut R)` pairs, allowing
/// in-place modification of coefficients.
pub struct ChainIterMut<'a, B, R>(hash_map::IterMut<'a, B, R>);

impl<'a, B, R> Iterator for ChainIterMut<'a, B, R> {
    type Item = (&'a B, &'a mut R);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<B, R> ExactSizeIterator for ChainIterMut<'_, B, R> {}
impl<B, R> FusedIterator for ChainIterMut<'_, B, R> {}

impl<B, R, S> Chain<B, R, S>
where
    B: Clone + Eq + Hash,
    R: Ring,
    S: BuildHasher + Default,
{
    /// Create an empty chain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: HashMap::with_hasher(S::default()),
        }
    }

    /// Empty the chain.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Return the coefficient of `cell` in this chain.
    ///
    /// If `cell` is not present, returns the ring zero.
    #[must_use]
    pub fn coefficient(&self, cell: &B) -> R {
        self.map.get(cell).cloned().unwrap_or_else(R::zero)
    }

    /// Return a mutable reference to the coefficient of `cell`.
    ///
    /// If `cell` is not present, it is inserted with coefficient zero first.
    pub fn coefficient_mut(&mut self, cell: &B) -> &mut R {
        self.map.entry(cell.clone()).or_insert_with(R::zero)
    }

    /// Remove `cell` from the chain, returning its coefficient.
    ///
    /// If `cell` is not present, returns the ring zero.
    pub fn remove(&mut self, cell: &B) -> R {
        self.map.remove(cell).unwrap_or_else(R::zero)
    }

    /// Scalar multiplication of this chain by `coefficient`.
    ///
    /// Multiplies each coefficient by `coefficient`.
    #[must_use]
    pub fn scalar_mul(mut self, coefficient: &R) -> Self {
        if *coefficient == R::zero() {
            self.clear();
        } else if *coefficient != R::one() {
            for cell_coefficient in self.map.values_mut() {
                *cell_coefficient *= coefficient.clone();
            }
        }
        self
    }

    /// If `cell` is not in this chain, insert it with the given
    /// `coefficient`. Otherwise, add `coefficient` to the existing
    /// coefficient of `cell`.
    ///
    /// # Examples
    ///
    /// ```
    /// use chomp3rs::{Chain, Cyclic, Ring};
    ///
    /// let mut chain = Chain::<u32, Cyclic<7>>::new();
    /// assert!(chain.is_empty());
    ///
    /// chain.insert_or_add(1, Cyclic::one());
    /// assert!(!chain.is_empty());
    /// assert_eq!(chain.coefficient(&1), Cyclic::one());
    ///
    /// chain.insert_or_add(1, -Cyclic::from(1));
    /// assert!(chain.is_empty());
    /// assert_eq!(chain.coefficient(&1), Cyclic::zero());
    /// ```
    pub fn insert_or_add(&mut self, cell: B, coefficient: R) {
        if coefficient != R::zero() {
            match self.map.entry(cell) {
                Entry::Occupied(mut o) => {
                    if coefficient.clone() + o.get().clone() == R::zero() {
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

    /// Iterator over all `(cell, coefficient)` pairs in the chain.
    #[must_use]
    pub fn iter(&self) -> ChainIter<'_, B, R> {
        ChainIter(self.map.iter())
    }

    /// Mutable iterator over all `(cell, coefficient)` pairs in the chain.
    #[must_use]
    pub fn iter_mut(&mut self) -> ChainIterMut<'_, B, R> {
        ChainIterMut(self.map.iter_mut())
    }

    /// Returns `true` if the chain has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the number of entries in the chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Create a chain with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity_and_hasher(capacity, S::default()),
        }
    }

    /// Access the underlying [`HashMap`].
    #[must_use]
    pub fn inner(&self) -> &HashMap<B, R, S> {
        &self.map
    }

    /// Mutably access the underlying [`HashMap`].
    #[must_use]
    pub fn inner_mut(&mut self) -> &mut HashMap<B, R, S> {
        &mut self.map
    }

    /// Convert to an [`OrderedChain`] with deterministic iteration order.
    ///
    /// Requires `B: Ord` for the B-tree ordering.
    #[must_use]
    pub fn to_ordered(&self) -> OrderedChain<B, R>
    where
        B: Ord,
    {
        self.iter()
            .map(|(cell, coef)| (cell.clone(), coef.clone()))
            .collect()
    }
}

/// Formats the chain as a sum of terms.
///
/// Since [`Chain`] uses a [`HashMap`] internally, the order of terms in the
/// output is non-deterministic. For deterministic formatting, use
/// [`OrderedChain`] or [`to_ordered`](Chain::to_ordered) first.
impl<B, R, S> Display for Chain<B, R, S>
where
    B: Display,
    R: Ring + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        super::fmt_chain(self.map.iter(), f)
    }
}

impl<B: Debug, R: Debug, S> Debug for Chain<B, R, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{:?}", self.map)
    }
}

impl<B, R, S> PartialEq for Chain<B, R, S>
where
    B: Eq + Hash,
    R: Ring,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        // Zero-filtering for mathematical correctness: chains that differ only
        // in explicit zero entries are considered equal.
        let self_count = self.map.values().filter(|c| **c != R::zero()).count();
        let other_count = other.map.values().filter(|c| **c != R::zero()).count();
        if self_count != other_count {
            return false;
        }
        self.map
            .iter()
            .filter(|(_, c)| **c != R::zero())
            .all(|(cell, coef)| other.map.get(cell).is_some_and(|c| c == coef))
    }
}

impl<B, R, S> Eq for Chain<B, R, S>
where
    B: Eq + Hash,
    R: Ring,
    S: BuildHasher,
{
}

impl<B, R: Ring, S> Neg for Chain<B, R, S> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for coefficient in self.map.values_mut() {
            *coefficient = -coefficient.clone();
        }
        self
    }
}

impl<B, R, S> AddAssign for Chain<B, R, S>
where
    B: Clone + Eq + Hash,
    R: Ring,
    S: BuildHasher + Default,
{
    fn add_assign(&mut self, rhs: Self) {
        for (cell, coefficient) in rhs.map {
            self.insert_or_add(cell, coefficient);
        }
    }
}

impl<B, R, S> Add for Chain<B, R, S>
where
    B: Clone + Eq + Hash,
    R: Ring,
    S: BuildHasher + Default,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<B, R, S> SubAssign for Chain<B, R, S>
where
    B: Clone + Eq + Hash,
    R: Ring,
    S: BuildHasher + Default,
{
    fn sub_assign(&mut self, rhs: Self) {
        for (cell, coefficient) in rhs.map {
            self.insert_or_add(cell, -coefficient);
        }
    }
}

impl<B, R, S> Sub for Chain<B, R, S>
where
    B: Clone + Eq + Hash,
    R: Ring,
    S: BuildHasher + Default,
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<B, R, S, const N: usize> From<[(B, R); N]> for Chain<B, R, S>
where
    B: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from(items: [(B, R); N]) -> Self {
        Self {
            map: HashMap::from_iter(items),
        }
    }
}

impl<B, R, S> From<B> for Chain<B, R, S>
where
    B: Eq + Hash,
    R: Ring,
    S: BuildHasher + Default,
{
    fn from(cell: B) -> Self {
        Self::from([(cell, R::one())])
    }
}

impl<B, R, S> IntoIterator for Chain<B, R, S>
where
    B: Eq + Hash,
    S: BuildHasher + Default,
{
    type IntoIter = hash_map::IntoIter<B, R>;
    type Item = (B, R);

    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

impl<'a, B, R, S> IntoIterator for &'a Chain<B, R, S>
where
    B: Eq + Hash,
    S: BuildHasher,
{
    type IntoIter = ChainIter<'a, B, R>;
    type Item = (&'a B, &'a R);

    fn into_iter(self) -> Self::IntoIter {
        ChainIter(self.map.iter())
    }
}

impl<'a, B, R, S> IntoIterator for &'a mut Chain<B, R, S>
where
    B: Eq + Hash,
    S: BuildHasher,
{
    type IntoIter = ChainIterMut<'a, B, R>;
    type Item = (&'a B, &'a mut R);

    fn into_iter(self) -> Self::IntoIter {
        ChainIterMut(self.map.iter_mut())
    }
}

impl<B, R, S> FromIterator<(B, R)> for Chain<B, R, S>
where
    B: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (B, R)>,
    {
        Self {
            map: HashMap::from_iter(iter),
        }
    }
}

impl<B, R, S> Default for Chain<B, R, S>
where
    B: Clone + Eq + Hash,
    R: Ring,
    S: BuildHasher + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cyclic;

    #[test]
    fn coefficient_access() {
        let mut chain = Chain::<u32, Cyclic<5>>::new();
        assert_eq!(chain.coefficient(&1), Cyclic::from(0));

        chain.insert_or_add(1, Cyclic::from(3));
        assert_eq!(chain.coefficient(&1), Cyclic::from(3));

        chain.insert_or_add(1, Cyclic::from(2));
        assert_eq!(chain.coefficient(&1), Cyclic::from(0)); // 3 + 2 = 0 (mod 5)

        *chain.coefficient_mut(&2) = Cyclic::from(4);
        assert_eq!(chain.coefficient(&2), Cyclic::from(4));

        assert_eq!(chain.remove(&2), Cyclic::from(4));
        assert_eq!(chain.remove(&2), Cyclic::from(0));

        chain.insert_or_add(5, Cyclic::from(1));
        chain.clear();
        assert_eq!(chain.coefficient(&5), Cyclic::from(0));
    }

    #[test]
    fn arithmetic() {
        let chain1 =
            Chain::<i32, Cyclic<7>>::from_iter([(1, Cyclic::from(3)), (2, Cyclic::from(5))]);
        let chain2 = Chain::from_iter([(1, Cyclic::from(4)), (3, Cyclic::from(2))]);

        let sum = chain1.clone() + chain2.clone();
        assert_eq!(sum.coefficient(&1), Cyclic::from(0));
        assert_eq!(sum.coefficient(&2), Cyclic::from(5));
        assert_eq!(sum.coefficient(&3), Cyclic::from(2));

        let diff = chain1.clone() - chain2.clone();
        assert_eq!(diff.coefficient(&1), Cyclic::from(6));
        assert_eq!(diff.coefficient(&2), Cyclic::from(5));
        assert_eq!(diff.coefficient(&3), Cyclic::from(5));

        let mut add_assign = chain1.clone();
        add_assign += chain2.clone();
        assert_eq!(add_assign, sum);

        let mut sub_assign = chain1.clone();
        sub_assign -= chain2;
        assert_eq!(sub_assign, diff);

        let neg = -chain1.clone();
        assert_eq!(neg.coefficient(&1), Cyclic::from(4));
        assert_eq!(neg.coefficient(&2), Cyclic::from(2));
    }

    #[test]
    fn scalar_mul() {
        let chain =
            Chain::<u32, Cyclic<5>>::from_iter([(1, Cyclic::from(2)), (2, Cyclic::from(3))]);

        assert_eq!(chain.clone().scalar_mul(&Cyclic::one()), chain);

        let scaled = chain.clone().scalar_mul(&Cyclic::from(2));
        assert_eq!(scaled.coefficient(&1), Cyclic::from(4));
        assert_eq!(scaled.coefficient(&2), Cyclic::from(1));

        let zeroed = chain.scalar_mul(&Cyclic::zero());
        assert_eq!(zeroed.coefficient(&1), Cyclic::from(0));
        assert_eq!(zeroed.coefficient(&2), Cyclic::from(0));
    }

    #[test]
    fn equality_with_zeros() {
        let with_zero = Chain::<u32, Cyclic<5>>::from([(1, Cyclic::from(2)), (2, Cyclic::from(0))]);
        let without_zero = Chain::from([(1, Cyclic::from(2))]);
        assert_eq!(with_zero, without_zero);
    }

    #[test]
    fn iterator_does_not_filter_zeros() {
        let mut chain = Chain::<u32, Cyclic<7>>::from([
            (1, Cyclic::from(3)),
            (2, Cyclic::from(0)), // Zero from From - visible in iter
            (3, Cyclic::from(5)),
        ]);
        chain.insert_or_add(4, Cyclic::from(0)); // Zero via insert_or_add - not inserted

        // iter() does not filter; the zero from From<[...]> is present,
        // but the zero via insert_or_add was never inserted.
        let mut collected: Vec<_> = chain
            .iter()
            .map(|(cell, coefficient)| (*cell, *coefficient))
            .collect();
        collected.sort_by_key(|(cell, _)| *cell);

        assert_eq!(
            collected,
            vec![
                (1, Cyclic::from(3)),
                (2, Cyclic::from(0)),
                (3, Cyclic::from(5)),
            ]
        );

        let empty = Chain::<u32, Cyclic<7>>::new();
        assert_eq!(empty.iter().count(), 0);
    }

    #[test]
    fn from_array_and_into_iter() {
        let chain: Chain<i32, Cyclic<5>> =
            Chain::from([(1, Cyclic::<5>::from(2)), (3, Cyclic::from(4))]);

        assert_eq!(chain.coefficient(&1), Cyclic::from(2));
        assert_eq!(chain.coefficient(&3), Cyclic::from(4));

        let collected: Vec<_> = chain.into_iter().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn display_formatting() {
        let empty = Chain::<u32, Cyclic<5>>::new();
        assert_eq!(empty.to_string(), "0");

        let single: Chain<u32, Cyclic<5>> = Chain::from([(2, Cyclic::from(1))]);
        assert_eq!(single.to_string(), "2");

        let with_coef: Chain<u32, Cyclic<5>> = Chain::from([(3, Cyclic::from(2))]);
        assert_eq!(with_coef.to_string(), "2 (mod 5)*3");
    }

    #[test]
    fn from_cell() {
        let chain = Chain::<u32, Cyclic<5>>::from(42);
        assert_eq!(chain.coefficient(&42), Cyclic::from(1));
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn with_capacity_and_accessors() {
        let mut chain = Chain::<u32, Cyclic<5>>::with_capacity(10);
        chain.insert_or_add(1, Cyclic::from(3));
        assert_eq!(chain.inner().len(), 1);

        chain.inner_mut().insert(2, Cyclic::from(4));
        assert_eq!(chain.coefficient(&2), Cyclic::from(4));
    }

    #[test]
    fn to_ordered() {
        let chain = Chain::<u32, Cyclic<5>>::from([(3, Cyclic::from(2)), (1, Cyclic::from(4))]);
        let ordered = chain.to_ordered();
        let cells: Vec<_> = ordered.iter().map(|(c, _)| *c).collect();
        assert_eq!(cells, vec![1, 3]);
    }

    #[test]
    fn into_iterator_ref() {
        let chain = Chain::<u32, Cyclic<5>>::from([(1, Cyclic::from(2)), (2, Cyclic::from(3))]);
        let mut collected: Vec<_> = (&chain).into_iter().map(|(c, r)| (*c, *r)).collect();
        collected.sort_by_key(|(c, _)| *c);
        assert_eq!(collected, vec![(1, Cyclic::from(2)), (2, Cyclic::from(3))]);
    }

    #[test]
    fn iter_mut_modifies_coefficients() {
        let mut chain = Chain::<u32, Cyclic<5>>::from([(1, Cyclic::from(2)), (2, Cyclic::from(3))]);

        for (_cell, coef) in &mut chain {
            *coef *= Cyclic::from(2);
        }

        assert_eq!(chain.coefficient(&1), Cyclic::from(4));
        assert_eq!(chain.coefficient(&2), Cyclic::from(1));
    }

    #[test]
    fn into_iterator_mut_ref() {
        let mut chain = Chain::<u32, Cyclic<7>>::from([(1, Cyclic::from(3)), (2, Cyclic::from(5))]);

        for (_cell, coef) in &mut chain {
            *coef = -*coef;
        }

        assert_eq!(chain.coefficient(&1), Cyclic::from(4));
        assert_eq!(chain.coefficient(&2), Cyclic::from(2));
    }
}
