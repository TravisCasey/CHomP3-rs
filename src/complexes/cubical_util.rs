// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Utility types and iterators for cubical complexes.

use std::iter::zip;

use super::cubical::{Cube, Orthant};
use super::traits::Grader;

/// This iterator traverses all orthants within a specified bounding box between
/// the provided minimum and maximum orthants (inclusive). The iteration follows
/// lexicographic order starting from the minimum orthant.
///
/// # Examples
///
/// ```rust
/// use chomp3rs::{Orthant, OrthantIterator};
///
/// let min = Orthant::new(vec![0, 0]);
/// let max = Orthant::new(vec![1, 1]);
/// let mut iter = OrthantIterator::new(min, max);
///
/// assert_eq!(iter.next().unwrap(), Orthant::new(vec![0, 0]));
/// assert_eq!(iter.next().unwrap(), Orthant::new(vec![0, 1]));
/// assert_eq!(iter.next().unwrap(), Orthant::new(vec![1, 0]));
/// assert_eq!(iter.next().unwrap(), Orthant::new(vec![1, 1]));
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
            "Minimum and maximum orthants must have the same ambient dimension"
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
/// use chomp3rs::{CubeIterator, Orthant};
///
/// let min = Orthant::new(vec![0, 0]);
/// let max = Orthant::new(vec![1, 1]);
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
            for (base_coord, dual_coord) in zip(base.iter_mut(), dual.iter_mut()) {
                if dual_coord != base_coord {
                    debug_assert!(*dual_coord == *base_coord - 1);
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

/// A grader that grades a `Cube` based on the minimum of the grades of the
/// surrounding top-dimensional cubes that this cube is a face of.
///
/// This grader takes another grader that can grade `Orthant`s (representing
/// the base orthants of top-dimensional cubes) and uses it to grade the
/// surrounding top cubes, then returns the minimum grade found.
///
/// The grader includes short-circuiting optimization: if a minimum grade is
/// specified and found during iteration, the search terminates early and
/// returns that minimum grade.
///
/// # Examples
///
/// ```rust
/// use std::collections::HashMap;
///
/// use chomp3rs::{Cube, Grader, HashMapGrader, Orthant, TopCubeGrader};
///
/// // Create an orthant grader
/// let mut orthant_grades = HashMap::new();
/// orthant_grades.insert(Orthant::new(vec![0, 0]), 1);
/// orthant_grades.insert(Orthant::new(vec![0, 1]), 2);
/// orthant_grades.insert(Orthant::new(vec![1, 0]), 3);
/// orthant_grades.insert(Orthant::new(vec![1, 1]), 4);
/// let orthant_grader = HashMapGrader::from_map(orthant_grades);
///
/// // Create cube grader with minimum grade for short-circuiting
/// let cube_grader = TopCubeGrader::new(orthant_grader, Some(1));
///
/// // Grade a vertex cube - it's a face of all 4 surrounding top cubes
/// let vertex = Cube::vertex(Orthant::new(vec![1, 1]));
/// let grade = cube_grader.grade(&vertex);
/// assert_eq!(grade, 1); // minimum of surrounding grades: 1, 2, 3, 4
/// ```
#[derive(Clone, Debug)]
pub struct TopCubeGrader<G>
where
    G: Grader<Orthant>,
{
    orthant_grader: G,
    min_grade: Option<u32>,
}

impl<G> TopCubeGrader<G>
where
    G: Grader<Orthant>,
{
    /// Create a new `TopCubeGrader`.
    ///
    /// Top-dimensional cubes are graded by their base orthants using
    /// `orthant_grader`. Providing the minimum grade output by
    /// `orthant_grader` enables the short-circuiting optimization.
    #[must_use]
    pub fn new(orthant_grader: G, min_grade: Option<u32>) -> Self {
        Self {
            orthant_grader,
            min_grade,
        }
    }

    /// Get the minimum grade used for short-circuiting.
    #[must_use]
    pub fn min_grade(&self) -> Option<u32> {
        self.min_grade
    }

    /// Set the minimum grade for short-circuiting optimization.
    pub fn set_min_grade(&mut self, min_grade: Option<u32>) {
        self.min_grade = min_grade;
    }
}

impl<G> Grader<Cube> for TopCubeGrader<G>
where
    G: Grader<Orthant>,
{
    fn grade(&self, cube: &Cube) -> u32 {
        let base = cube.base();
        let dual = cube.dual();

        // Count non-extending axes to determine number of surrounding top cubes
        let non_extending_axes: Vec<usize> = zip(base.iter(), dual.iter())
            .enumerate()
            .filter(|(_, (base_coord, dual_coord))| **base_coord != **dual_coord)
            .map(|(i, _)| i)
            .collect();

        let mut min_grade = u32::MAX;
        let num_combinations = 1 << non_extending_axes.len();

        // Iterate through all combinations of coordinates for non-extending axes
        let mut top_cube_base = base.clone();
        for combination in 0..num_combinations {
            // For each non-extending axis, choose coordinate based on bit pattern
            for (bit_pos, &axis) in non_extending_axes.iter().enumerate() {
                if (combination >> bit_pos) & 1 == 0 {
                    top_cube_base[axis] = dual[axis];
                } else {
                    top_cube_base[axis] = base[axis];
                }
            }

            // Grade the base orthant of this top cube
            let grade = self.orthant_grader.grade(&top_cube_base);
            min_grade = min_grade.min(grade);

            // Short-circuit iteration if we found the minimum possible grade
            if let Some(min_possible) = self.min_grade
                && grade <= min_possible
            {
                return min_possible;
            }
        }

        min_grade
    }
}

impl<G> Grader<Orthant> for TopCubeGrader<G>
where
    G: Grader<Orthant>,
{
    fn grade(&self, orthant: &Orthant) -> u32 {
        self.orthant_grader.grade(orthant)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::HashMapGrader;

    #[test]
    fn test_orthant_iterator() {
        let min = Orthant::new(vec![0, 0]);
        let max = Orthant::new(vec![1, 1]);
        let mut iter = OrthantIterator::new(min, max);

        // Should iterate through all orthants in lexicographic order
        assert_eq!(iter.next().unwrap(), Orthant::new(vec![0, 0]));
        assert_eq!(iter.next().unwrap(), Orthant::new(vec![0, 1]));
        assert_eq!(iter.next().unwrap(), Orthant::new(vec![1, 0]));
        assert_eq!(iter.next().unwrap(), Orthant::new(vec![1, 1]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_orthant_iterator_3d() {
        let min = Orthant::new(vec![0, 0, 0]);
        let max = Orthant::new(vec![1, 1, 2]);
        let orthants: Vec<_> = OrthantIterator::new(min, max).collect();

        assert_eq!(orthants.len(), 12);

        assert_eq!(orthants[0], Orthant::new(vec![0, 0, 0]));
        assert_eq!(orthants[7], Orthant::new(vec![1, 0, 1]));
        assert_eq!(orthants[11], Orthant::new(vec![1, 1, 2]));
    }

    #[test]
    fn test_trivial_orthant_iterator() {
        let orthant = Orthant::new(vec![5, 3]);
        let mut iter = OrthantIterator::new(orthant.clone(), orthant.clone());

        assert_eq!(iter.next().unwrap(), orthant);
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[should_panic(expected = "Minimum and maximum orthants must have the same ambient dimension")]
    fn test_orthant_iterator_dimension_mismatch() {
        let min = Orthant::new(vec![0, 0]);
        let max = Orthant::new(vec![1, 1, 1]);
        let _iter = OrthantIterator::new(min, max);
    }

    #[test]
    fn test_cube_iterator_2d() {
        let min = Orthant::new(vec![0, 0]);
        let max = Orthant::new(vec![1, 1]);
        let cubes: Vec<_> = CubeIterator::new(min, max).collect();

        let vertices: Vec<_> = cubes.iter().filter(|c| c.dimension() == 0).collect();
        let edges: Vec<_> = cubes.iter().filter(|c| c.dimension() == 1).collect();
        let squares: Vec<_> = cubes.iter().filter(|c| c.dimension() == 2).collect();

        assert!(vertices.len() == 4);
        assert!(edges.len() == 8);
        assert!(squares.len() == 4);
    }

    #[test]
    fn test_cube_iterator_1d() {
        let min = Orthant::new(vec![0]);
        let max = Orthant::new(vec![2]);
        let cubes: Vec<_> = CubeIterator::new(min, max).collect();

        let vertices: Vec<_> = cubes.iter().filter(|c| c.dimension() == 0).collect();
        let edges: Vec<_> = cubes.iter().filter(|c| c.dimension() == 1).collect();

        assert_eq!(vertices.len(), 3); // Points at 0, 1, 2
        assert_eq!(edges.len(), 3); // Edges [0,1], [1,2], [2, 3)
    }

    #[test]
    fn test_top_cube_grader() {
        // Create orthant grader with known grades
        let mut orthant_grades = HashMap::new();
        orthant_grades.insert(Orthant::new(vec![0, 0]), 1);
        orthant_grades.insert(Orthant::new(vec![0, 1]), 2);
        orthant_grades.insert(Orthant::new(vec![1, 0]), 3);
        orthant_grades.insert(Orthant::new(vec![1, 1]), 4);
        let orthant_grader = HashMapGrader::from_map(orthant_grades);

        let cube_grader = TopCubeGrader::new(orthant_grader, Some(1));

        // Test vertex cube - should be face of all 4 surrounding top cubes
        let vertex = Cube::vertex(Orthant::new(vec![1, 1]));
        assert_eq!(cube_grader.grade(&vertex), 1); // min of 1, 2, 3, 4

        // Test edge cube along first axis
        let edge = Cube::from_extent(Orthant::new(vec![1, 1]), &[true, false]);
        assert_eq!(cube_grader.grade(&edge), 3); // min of 3, 4

        // Test edge cube along second axis
        let edge = Cube::from_extent(Orthant::new(vec![1, 1]), &[false, true]);
        assert_eq!(cube_grader.grade(&edge), 2); // min of 2, 4

        // Test top-dimensional cube
        let top_cube = Cube::top_cube(Orthant::new(vec![1, 1]));
        assert_eq!(cube_grader.grade(&top_cube), 4); // just grade of (1,1)
    }
}
