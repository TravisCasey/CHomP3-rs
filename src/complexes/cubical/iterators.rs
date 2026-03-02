// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Iterators for traversing cubical complexes.
//!
//! This module provides iterators for systematically traversing orthants and
//! cubes within a cubical complex. The iterators follow lexicographic order
//! and generate all combinations of cells within specified bounds.

use std::iter::zip;

use super::{Cube, Orthant};

/// This iterator traverses all orthants within a specified bounding box between
/// the provided minimum and maximum orthants (inclusive). The iteration follows
/// lexicographic order starting from the minimum orthant.
///
/// # Examples
///
/// ```rust
/// use chomp3rs::{Orthant, complexes::OrthantIterator};
///
/// let min = Orthant::from([0, 0]);
/// let max = Orthant::from([1, 1]);
/// let mut iter = OrthantIterator::new(min, max);
///
/// assert_eq!(iter.next().unwrap(), Orthant::from([0, 0]));
/// assert_eq!(iter.next().unwrap(), Orthant::from([0, 1]));
/// assert_eq!(iter.next().unwrap(), Orthant::from([1, 0]));
/// assert_eq!(iter.next().unwrap(), Orthant::from([1, 1]));
/// assert_eq!(iter.next(), None);
/// ```
pub struct OrthantIterator {
    next: Option<Orthant>,
    minimum: Orthant,
    maximum: Orthant,
}

impl OrthantIterator {
    /// Create a new orthant iterator for the given range.
    ///
    /// # Panics
    /// Panics if `minimum` and `maximum` have different ambient dimensions.
    #[must_use]
    pub fn new(minimum: Orthant, maximum: Orthant) -> Self {
        assert_eq!(
            minimum.ambient_dimension(),
            maximum.ambient_dimension(),
            "Minimum and maximum orthants must have the same ambient dimension: minimum has \
             dimension {}, maximum has dimension {}",
            minimum.ambient_dimension(),
            maximum.ambient_dimension()
        );

        Self {
            next: Some(minimum.clone()),
            minimum,
            maximum,
        }
    }
}

impl Iterator for OrthantIterator {
    type Item = Orthant;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(mut next) = self.next.clone() {
            for axis in (0..next.ambient_dimension() as usize).rev() {
                if next[axis] < self.maximum[axis] {
                    next[axis] += 1;
                    return self.next.replace(next);
                }
                next[axis] = self.minimum[axis];
            }
            return self.next.take();
        }
        None
    }
}

/// Iterator over all cubes in orthants comprising a rectangular region of
/// n-dimensional space.
///
/// The iterator works by:
/// 1. Iterating through all base orthants in the specified range
/// 2. For each base orthant, generating all possible extent patterns
/// 3. Creating cubes using the extent system (base and dual orthants)
///
/// # Examples
///
/// ```rust
/// use chomp3rs::{Orthant, complexes::CubeIterator};
///
/// let min = Orthant::from([0, 0]);
/// let max = Orthant::from([1, 1]);
/// let mut iter = CubeIterator::new(min, max);
///
/// // Generates 1 vertex, 2 edges, and 1 face for each of the 4 orthants.
/// let cubes: Vec<_> = iter.collect();
/// assert_eq!(cubes.len(), 16);
/// ```
pub struct CubeIterator {
    orthant_iter: OrthantIterator,
    current_base: Option<Orthant>,
    current_dual: Option<Orthant>,
}

impl CubeIterator {
    /// Create a new cube iterator for the given range of orthants.
    ///
    /// # Panics
    /// Panics if `minimum` and `maximum` have different ambient dimensions.
    #[must_use]
    pub fn new(minimum: Orthant, maximum: Orthant) -> Self {
        let mut orthant_iter = OrthantIterator::new(minimum, maximum);
        let current_base = orthant_iter.next();

        let mut cube_iter = Self {
            orthant_iter,
            current_base,
            current_dual: None,
        };
        cube_iter.reset_dual();
        cube_iter
    }

    /// Reset the dual orthant to the vertex for the current base orthant.
    fn reset_dual(&mut self) {
        self.current_dual = self
            .current_base
            .as_ref()
            .map(|base| base.iter().map(|coord| *coord - 1).collect());
    }
}

impl Iterator for CubeIterator {
    type Item = Cube;

    fn next(&mut self) -> Option<Self::Item> {
        // Generate the current cube
        if let (Some(base), Some(dual)) = (self.current_base.as_mut(), self.current_dual.as_mut()) {
            let cube = Cube::new(base.clone(), dual.clone());

            // Try to increment the dual coordinates to generate next extent pattern
            for (axis, (base_coord, dual_coord)) in
                zip(base.iter_mut(), dual.iter_mut()).enumerate()
            {
                if dual_coord != base_coord {
                    debug_assert!(
                        *dual_coord == *base_coord - 1,
                        "Dual coordinate at axis {} should either equal base or be on lesser: \
                         base is {}, dual is {}",
                        axis,
                        *base_coord,
                        *dual_coord
                    );
                    *dual_coord += 1;
                    return Some(cube);
                }
                *dual_coord = *base_coord - 1;
            }

            // Move to the next base orthant
            self.current_base = self.orthant_iter.next();
            self.reset_dual();
            return Some(cube);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthant_iterator() {
        let min = Orthant::from([0, 0]);
        let max = Orthant::from([1, 1]);
        let mut iter = OrthantIterator::new(min, max);

        // Should iterate through all orthants in lexicographic order
        assert_eq!(iter.next().unwrap(), Orthant::from([0, 0]));
        assert_eq!(iter.next().unwrap(), Orthant::from([0, 1]));
        assert_eq!(iter.next().unwrap(), Orthant::from([1, 0]));
        assert_eq!(iter.next().unwrap(), Orthant::from([1, 1]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn trivial_orthant_iterator() {
        let orthant = Orthant::from([5, 3]);
        let mut iter = OrthantIterator::new(orthant.clone(), orthant.clone());

        assert_eq!(iter.next().unwrap(), orthant);
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[should_panic(expected = "Minimum and maximum orthants must have the same ambient dimension")]
    fn orthant_iterator_dimension_mismatch() {
        let min = Orthant::from([0, 0]);
        let max = Orthant::from([1, 1, 1]);
        let _iter = OrthantIterator::new(min, max);
    }

    #[test]
    fn orthant_iterator_3d() {
        let min = Orthant::from([0, 0, 0]);
        let max = Orthant::from([1, 1, 2]);
        let orthants: Vec<_> = OrthantIterator::new(min, max).collect();

        assert_eq!(orthants.len(), 12);

        assert_eq!(orthants[0], Orthant::from([0, 0, 0]));
        assert_eq!(orthants[7], Orthant::from([1, 0, 1]));
        assert_eq!(orthants[11], Orthant::from([1, 1, 2]));
    }

    #[test]
    fn orthant_iterator_large_range() {
        let min = Orthant::from([0, 0]);
        let max = Orthant::from([9, 9]);
        let orthants: Vec<_> = OrthantIterator::new(min, max).collect();

        // Should have 10 * 10 = 100 orthants
        assert_eq!(orthants.len(), 100);

        // Verify first and last
        assert_eq!(orthants[0], Orthant::from([0, 0]));
        assert_eq!(orthants[99], Orthant::from([9, 9]));
    }

    #[test]
    fn cube_iterator_high_dimension() {
        let min = Orthant::from([0, 0, 0, 0]);
        let max = Orthant::from([0, 0, 0, 0]);
        let cubes: Vec<_> = CubeIterator::new(min, max).collect();

        // Single orthant, 2^4 = 16 cubes (all extent patterns in 4D)
        assert_eq!(cubes.len(), 16);

        // Count by dimension
        let by_dim: Vec<usize> = (0..=4)
            .map(|d| cubes.iter().filter(|c| c.dimension() == d).count())
            .collect();

        // Binomial coefficients: C(4,0)=1, C(4,1)=4, C(4,2)=6, C(4,3)=4, C(4,4)=1
        assert_eq!(by_dim, vec![1, 4, 6, 4, 1]);
    }

    #[test]
    fn orthant_iterator_lexicographic_order() {
        let min = Orthant::from([0, 0, 0]);
        let max = Orthant::from([1, 1, 1]);
        let mut iter = OrthantIterator::new(min, max);

        // Verify lexicographic ordering (rightmost coordinate varies fastest)
        assert_eq!(iter.next().unwrap(), Orthant::from([0, 0, 0]));
        assert_eq!(iter.next().unwrap(), Orthant::from([0, 0, 1]));
        assert_eq!(iter.next().unwrap(), Orthant::from([0, 1, 0]));
        assert_eq!(iter.next().unwrap(), Orthant::from([0, 1, 1]));
        assert_eq!(iter.next().unwrap(), Orthant::from([1, 0, 0]));
        assert_eq!(iter.next().unwrap(), Orthant::from([1, 0, 1]));
        assert_eq!(iter.next().unwrap(), Orthant::from([1, 1, 0]));
        assert_eq!(iter.next().unwrap(), Orthant::from([1, 1, 1]));
        assert_eq!(iter.next(), None);
    }
}
