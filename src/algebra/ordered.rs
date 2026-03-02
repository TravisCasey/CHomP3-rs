// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! B-tree-based sparse chain implementation with deterministic iteration.
//!
//! [`OrderedChain`] stores basis element/coefficient pairs in a [`BTreeMap`],
//! providing O(log n) access with iteration in sorted order by cell.
//!
//! For better access performance when iteration order doesn't matter, use
//! [`Chain`] instead.

use std::{
    collections::{
        BTreeMap,
        btree_map::{self, Entry},
    },
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::Hash,
    iter::FusedIterator,
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

use super::{Chain, Ring};

/// B-tree-based sparse chain with deterministic iteration.
///
/// Stores basis element/coefficient pairs in a [`BTreeMap`], providing
/// O(log n) access with iteration in sorted order by cell. This is useful
/// for algorithms that require deterministic behavior.
///
/// CHomP3-rs methods typically accept `IntoIterator<Item = (&Cell, &Ring)>`
/// types rather than a specific chain type, so both `Chain` and `OrderedChain`
/// (as well as arrays, vectors, or other types which iterate
/// `(cell, coefficient)` pairs) can be used interchangeably. Where a concrete
/// chain type is needed, the API uses the unordered variant [`Chain`], which is
/// faster on average when deterministic iteration is not required. Use
/// [`to_unordered`](Self::to_unordered) to convert when needed.
///
/// # Examples
///
/// ## Construction and Ordered Iteration
///
/// ```rust
/// use chomp3rs::{Cyclic, OrderedChain};
///
/// let mut chain = OrderedChain::<u32, Cyclic<5>>::new();
/// chain.insert_or_add(3, Cyclic::from(2));
/// chain.insert_or_add(1, Cyclic::from(4));
/// chain.insert_or_add(2, Cyclic::from(1));
///
/// // Iteration is always in sorted order by cell
/// let cells: Vec<_> = chain.iter().map(|(c, _)| *c).collect();
/// assert_eq!(cells, vec![1, 2, 3]);
/// ```
///
/// ## Scalar Multiplication
///
/// ```rust
/// use chomp3rs::{Cyclic, OrderedChain};
///
/// let chain = OrderedChain::<u32, Cyclic<7>>::from([
///     (1, Cyclic::from(3)),
///     (2, Cyclic::from(5)),
/// ]);
/// let scaled = chain.scalar_mul(&Cyclic::from(4));
/// assert_eq!(
///     scaled.coefficient(&1),
///     Cyclic::from(5)
/// ); // 3 * 4 = 12 = 5 (mod 7)
/// assert_eq!(
///     scaled.coefficient(&2),
///     Cyclic::from(6)
/// ); // 5 * 4 = 20 = 6 (mod 7)
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
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "B: serde::Serialize + Ord, R: serde::Serialize",
    deserialize = "B: serde::Deserialize<'de> + Ord, R: serde::Deserialize<'de>"
))]
pub struct OrderedChain<B, R> {
    map: BTreeMap<B, R>,
}

/// Iterator over `(cell, coefficient)` pairs in an [`OrderedChain`], in
/// sorted order.
///
/// Created by [`OrderedChain::iter`]. Does not filter zero-coefficient
/// entries; core operations ([`OrderedChain::insert_or_add`], arithmetic)
/// eagerly remove zeros, so the map should not contain zero entries under
/// normal use.
pub struct OrderedChainIter<'a, B, R>(btree_map::Iter<'a, B, R>);

impl<'a, B, R> Iterator for OrderedChainIter<'a, B, R> {
    type Item = (&'a B, &'a R);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<B, R> ExactSizeIterator for OrderedChainIter<'_, B, R> {}
impl<B, R> FusedIterator for OrderedChainIter<'_, B, R> {}

impl<B, R> DoubleEndedIterator for OrderedChainIter<'_, B, R> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

/// Mutable iterator over `(cell, coefficient)` pairs in an
/// [`OrderedChain`], in sorted order.
///
/// Created by [`OrderedChain::iter_mut`]. Yields `(&B, &mut R)` pairs,
/// allowing in-place modification of coefficients.
pub struct OrderedChainIterMut<'a, B, R>(btree_map::IterMut<'a, B, R>);

impl<'a, B, R> Iterator for OrderedChainIterMut<'a, B, R> {
    type Item = (&'a B, &'a mut R);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<B, R> ExactSizeIterator for OrderedChainIterMut<'_, B, R> {}
impl<B, R> FusedIterator for OrderedChainIterMut<'_, B, R> {}

impl<B, R> DoubleEndedIterator for OrderedChainIterMut<'_, B, R> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<B, R> OrderedChain<B, R>
where
    B: Clone + Ord,
    R: Ring,
{
    /// Create an empty ordered chain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
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
    /// use chomp3rs::{Cyclic, OrderedChain, Ring};
    ///
    /// let mut chain = OrderedChain::<u32, Cyclic<7>>::new();
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

    /// Iterator over all `(cell, coefficient)` pairs in sorted order by cell.
    #[must_use]
    pub fn iter(&self) -> OrderedChainIter<'_, B, R> {
        OrderedChainIter(self.map.iter())
    }

    /// Mutable iterator over all `(cell, coefficient)` pairs in sorted order
    /// by cell.
    #[must_use]
    pub fn iter_mut(&mut self) -> OrderedChainIterMut<'_, B, R> {
        OrderedChainIterMut(self.map.iter_mut())
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

    /// Access the underlying [`BTreeMap`].
    #[must_use]
    pub fn inner(&self) -> &BTreeMap<B, R> {
        &self.map
    }

    /// Mutably access the underlying [`BTreeMap`].
    #[must_use]
    pub fn inner_mut(&mut self) -> &mut BTreeMap<B, R> {
        &mut self.map
    }

    /// Convert to a [`Chain`] with hash-based O(1) access.
    #[must_use]
    pub fn to_unordered(&self) -> Chain<B, R>
    where
        B: Eq + Hash,
    {
        self.iter()
            .map(|(cell, coef)| (cell.clone(), coef.clone()))
            .collect()
    }
}

/// Formats the chain as a sum of terms in sorted order by cell.
impl<B, R> Display for OrderedChain<B, R>
where
    B: Display,
    R: Ring + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        super::fmt_chain(self.map.iter(), f)
    }
}

impl<B: Debug, R: Debug> Debug for OrderedChain<B, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{:?}", self.map)
    }
}

impl<B, R> PartialEq for OrderedChain<B, R>
where
    B: Ord,
    R: Ring,
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

impl<B, R> Eq for OrderedChain<B, R>
where
    B: Ord,
    R: Ring,
{
}

impl<B, R: Ring> Neg for OrderedChain<B, R> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for coefficient in self.map.values_mut() {
            *coefficient = -coefficient.clone();
        }
        self
    }
}

impl<B, R> AddAssign for OrderedChain<B, R>
where
    B: Clone + Ord,
    R: Ring,
{
    fn add_assign(&mut self, rhs: Self) {
        for (cell, coefficient) in rhs.map {
            self.insert_or_add(cell, coefficient);
        }
    }
}

impl<B, R> Add for OrderedChain<B, R>
where
    B: Clone + Ord,
    R: Ring,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<B, R> SubAssign for OrderedChain<B, R>
where
    B: Clone + Ord,
    R: Ring,
{
    fn sub_assign(&mut self, rhs: Self) {
        for (cell, coefficient) in rhs.map {
            self.insert_or_add(cell, -coefficient);
        }
    }
}

impl<B, R> Sub for OrderedChain<B, R>
where
    B: Clone + Ord,
    R: Ring,
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<B, R, const N: usize> From<[(B, R); N]> for OrderedChain<B, R>
where
    B: Ord,
{
    fn from(items: [(B, R); N]) -> Self {
        Self {
            map: BTreeMap::from_iter(items),
        }
    }
}

impl<B, R> From<B> for OrderedChain<B, R>
where
    B: Ord,
    R: Ring,
{
    fn from(cell: B) -> Self {
        Self::from([(cell, R::one())])
    }
}

impl<B, R> IntoIterator for OrderedChain<B, R>
where
    B: Ord,
{
    type IntoIter = btree_map::IntoIter<B, R>;
    type Item = (B, R);

    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

impl<'a, B, R> IntoIterator for &'a OrderedChain<B, R> {
    type IntoIter = OrderedChainIter<'a, B, R>;
    type Item = (&'a B, &'a R);

    fn into_iter(self) -> Self::IntoIter {
        OrderedChainIter(self.map.iter())
    }
}

impl<'a, B, R> IntoIterator for &'a mut OrderedChain<B, R> {
    type IntoIter = OrderedChainIterMut<'a, B, R>;
    type Item = (&'a B, &'a mut R);

    fn into_iter(self) -> Self::IntoIter {
        OrderedChainIterMut(self.map.iter_mut())
    }
}

impl<B, R> FromIterator<(B, R)> for OrderedChain<B, R>
where
    B: Ord,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (B, R)>,
    {
        Self {
            map: BTreeMap::from_iter(iter),
        }
    }
}

impl<B, R> Default for OrderedChain<B, R>
where
    B: Clone + Ord,
    R: Ring,
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
        let mut chain = OrderedChain::<u32, Cyclic<5>>::new();
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
            OrderedChain::<i32, Cyclic<7>>::from_iter([(1, Cyclic::from(3)), (2, Cyclic::from(5))]);
        let chain2 = OrderedChain::from_iter([(1, Cyclic::from(4)), (3, Cyclic::from(2))]);

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
            OrderedChain::<u32, Cyclic<5>>::from_iter([(1, Cyclic::from(2)), (2, Cyclic::from(3))]);

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
        let with_zero =
            OrderedChain::<u32, Cyclic<5>>::from([(1, Cyclic::from(2)), (2, Cyclic::from(0))]);
        let without_zero = OrderedChain::from([(1, Cyclic::from(2))]);
        assert_eq!(with_zero, without_zero);
    }

    #[test]
    fn iterator_sorted_order() {
        let mut chain = OrderedChain::<u32, Cyclic<7>>::from([
            (3, Cyclic::from(5)),
            (1, Cyclic::from(3)),
            (2, Cyclic::from(0)), // Zero from From - visible in iter
        ]);
        chain.insert_or_add(4, Cyclic::from(0)); // Zero via insert_or_add - not inserted

        // iter() does not filter; the zero from From<[...]> is present,
        // but the zero via insert_or_add was never inserted. Sorted order.
        let collected: Vec<_> = chain
            .iter()
            .map(|(cell, coefficient)| (*cell, *coefficient))
            .collect();
        assert_eq!(
            collected,
            vec![
                (1, Cyclic::from(3)),
                (2, Cyclic::from(0)),
                (3, Cyclic::from(5)),
            ]
        );

        let empty = OrderedChain::<u32, Cyclic<7>>::new();
        assert_eq!(empty.iter().count(), 0);
    }

    #[test]
    fn from_array_and_into_iter() {
        let chain: OrderedChain<i32, Cyclic<5>> =
            OrderedChain::from([(1, Cyclic::<5>::from(2)), (3, Cyclic::from(4))]);

        assert_eq!(chain.coefficient(&1), Cyclic::from(2));
        assert_eq!(chain.coefficient(&3), Cyclic::from(4));

        let collected: Vec<_> = chain.into_iter().collect();
        assert_eq!(collected, vec![(1, Cyclic::from(2)), (3, Cyclic::from(4))]);
    }

    #[test]
    fn display_formatting() {
        let empty = OrderedChain::<u32, Cyclic<5>>::new();
        assert_eq!(empty.to_string(), "0");

        let single = OrderedChain::from([(2, Cyclic::<5>::from(1))]);
        assert_eq!(single.to_string(), "2");

        let with_coef = OrderedChain::from([(3, Cyclic::<5>::from(2))]);
        assert_eq!(with_coef.to_string(), "2 (mod 5)*3");

        let multi = OrderedChain::from([(3, Cyclic::<5>::from(2)), (1, Cyclic::from(1))]);
        assert_eq!(multi.to_string(), "1 + 2 (mod 5)*3");
    }

    #[test]
    fn from_cell() {
        let chain = OrderedChain::<u32, Cyclic<5>>::from(42);
        assert_eq!(chain.coefficient(&42), Cyclic::from(1));
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn into_iterator_ref() {
        let chain =
            OrderedChain::<u32, Cyclic<5>>::from([(1, Cyclic::from(2)), (2, Cyclic::from(3))]);
        let collected: Vec<_> = (&chain).into_iter().map(|(c, r)| (*c, *r)).collect();
        assert_eq!(collected, vec![(1, Cyclic::from(2)), (2, Cyclic::from(3))]);
    }

    #[test]
    fn double_ended_iterator() {
        let chain = OrderedChain::<u32, Cyclic<5>>::from([
            (1, Cyclic::from(2)),
            (2, Cyclic::from(3)),
            (3, Cyclic::from(4)),
        ]);
        let (cell, coef) = chain.iter().next_back().unwrap();
        assert_eq!(*cell, 3);
        assert_eq!(*coef, Cyclic::from(4));
    }

    #[test]
    fn iter_mut_modifies_coefficients() {
        let mut chain =
            OrderedChain::<u32, Cyclic<5>>::from([(1, Cyclic::from(2)), (2, Cyclic::from(3))]);

        for (_cell, coef) in &mut chain {
            *coef *= Cyclic::from(2);
        }

        assert_eq!(chain.coefficient(&1), Cyclic::from(4));
        assert_eq!(chain.coefficient(&2), Cyclic::from(1));
    }

    #[test]
    fn into_iterator_mut_ref() {
        let mut chain =
            OrderedChain::<u32, Cyclic<7>>::from([(1, Cyclic::from(3)), (2, Cyclic::from(5))]);

        for (_cell, coef) in &mut chain {
            *coef = -*coef;
        }

        assert_eq!(chain.coefficient(&1), Cyclic::from(4));
        assert_eq!(chain.coefficient(&2), Cyclic::from(2));
    }

    #[test]
    fn inner_accessors() {
        let mut chain = OrderedChain::<u32, Cyclic<5>>::from([(1, Cyclic::from(3))]);
        assert_eq!(chain.inner().len(), 1);

        chain.inner_mut().insert(2, Cyclic::from(4));
        assert_eq!(chain.coefficient(&2), Cyclic::from(4));
    }

    #[test]
    fn to_unordered() {
        let ordered =
            OrderedChain::<u32, Cyclic<5>>::from([(3, Cyclic::from(2)), (1, Cyclic::from(4))]);
        let unordered = ordered.to_unordered();
        assert_eq!(unordered.coefficient(&1), Cyclic::from(4));
        assert_eq!(unordered.coefficient(&3), Cyclic::from(2));
        assert_eq!(unordered.len(), 2);
    }

    #[test]
    fn iter_mut_double_ended() {
        let mut chain = OrderedChain::<u32, Cyclic<5>>::from([
            (1, Cyclic::from(1)),
            (2, Cyclic::from(2)),
            (3, Cyclic::from(3)),
        ]);

        // Modify only the last entry via next_back
        let (_cell, coef) = chain.iter_mut().next_back().unwrap();
        *coef = Cyclic::from(4);

        assert_eq!(chain.coefficient(&1), Cyclic::from(1));
        assert_eq!(chain.coefficient(&3), Cyclic::from(4));
    }
}
