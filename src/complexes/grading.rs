// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! General grading implementations for cell complexes.
//!
//! This module provides concrete implementations of the `Grader` trait for
//! assigning grades (filtration levels) to cells in complexes. The primary
//! implementation is `HashMapGrader`, which stores grades in a `HashMap` for
//! efficient lookup.

use std::collections::HashMap;
use std::hash::Hash;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::traits::Grader;

/// A grader that stores cell grades in a `HashMap`. Cells (instances of the
/// type parameter `B`) present in the map return their stored grade, while
/// cells not present return a configurable default grade.
///
/// This implementation provides efficient grade lookup and is suitable for most
/// use cases. The grader is typically constructed once and used in a read-only
/// fashion, though mutable access is available through the
/// [`HashMapGrader::grades_mut`] method for advanced operations.
///
/// # Examples
///
/// ```rust
/// use chomp3rs::{Grader, HashMapGrader};
///
/// // Create from (cell, grade) pairs
/// let grader = HashMapGrader::from_iter([(1, 10), (2, 20)]);
/// assert_eq!(grader.grade(&1), 10);
/// assert_eq!(grader.grade(&2), 20);
/// assert_eq!(grader.grade(&3), 0); // default grade
/// ```
#[derive(Clone, Debug, Default)]
pub struct HashMapGrader<B>
where
    B: Hash + Eq + Clone,
{
    grades: HashMap<B, u32>,
    default_grade: u32,
}

impl<B> HashMapGrader<B>
where
    B: Hash + Eq + Clone,
{
    /// Create a new empty grader with a specified default grade.
    ///
    /// Cells not found in the internal map (currently empty) return
    /// `default_grade`.
    #[must_use]
    pub fn new(default_grade: u32) -> Self {
        Self {
            grades: HashMap::new(),
            default_grade,
        }
    }

    /// Create a grader from an existing `HashMap`.
    ///
    /// The default grade for cells not in the map is set to 0. To choose a
    /// different default grade, use the
    /// [`HashMapGrader::from_map_with_default`] method.
    #[must_use]
    pub fn from_map(grades: HashMap<B, u32>) -> Self {
        Self {
            grades,
            default_grade: 0,
        }
    }

    /// Create a grader from an existing `HashMap` with a custom default grade.
    pub fn from_map_with_default(grades: HashMap<B, u32>, default_grade: u32) -> Self {
        Self {
            grades,
            default_grade,
        }
    }

    /// Create a grader by applying a function `grade_fn` to each cell in an
    /// iterator `cells`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use chomp3rs::{Grader, HashMapGrader};
    ///
    /// let cells = vec![10u16, 25u16, 100u16];
    /// let grader = HashMapGrader::from_cells_with_fn(cells, |cell| (*cell / 10) as u32);
    ///
    /// assert_eq!(grader.grade(&10u16), 1);
    /// assert_eq!(grader.grade(&25u16), 2);
    /// assert_eq!(grader.grade(&100u16), 10);
    /// assert_eq!(grader.grade(&1000u16), 0); // default grade as not in cell iterator
    /// ```
    #[must_use]
    pub fn from_cells_with_fn<I, F>(cells: I, grade_fn: F) -> Self
    where
        I: IntoIterator<Item = B>,
        F: Fn(&B) -> u32,
    {
        let grades = cells
            .into_iter()
            .map(|cell| {
                let grade = grade_fn(&cell);
                (cell, grade)
            })
            .collect();

        Self {
            grades,
            default_grade: 0,
        }
    }

    /// Create a grader in which all cells in the iterator `cells` have grade
    /// `uniform_grade`.
    ///
    /// All other cells have grade `default_grade`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use chomp3rs::{Grader, HashMapGrader};
    ///
    /// let cells = vec![(0, 1), (2, 3), (4, 5)];
    /// let grader = HashMapGrader::uniform(cells, 5, 2);
    ///
    /// assert_eq!(grader.grade(&(0, 1)), 5);
    /// assert_eq!(grader.grade(&(2, 3)), 5);
    /// assert_eq!(grader.grade(&(4, 5)), 5);
    /// assert_eq!(grader.grade(&(6, 7)), 2); // default grade
    /// ```
    #[must_use]
    pub fn uniform<I>(cells: I, uniform_grade: u32, default_grade: u32) -> Self
    where
        I: IntoIterator<Item = B>,
    {
        let grades = cells
            .into_iter()
            .map(|cell| (cell, uniform_grade))
            .collect();

        Self {
            grades,
            default_grade,
        }
    }

    /// Get an immutable reference to the underlying `HashMap`.
    pub fn grades(&self) -> &HashMap<B, u32> {
        &self.grades
    }

    /// Get a mutable reference to the underlying `HashMap`.
    ///
    /// # Warning
    /// Use caution when mutating the grades in conjunction with other
    /// algorithms that may rely on or assume the consistency of grades.
    pub fn grades_mut(&mut self) -> &mut HashMap<B, u32> {
        &mut self.grades
    }

    /// Get the default grade returned for cells not in the map.
    #[must_use]
    pub fn default_grade(&self) -> u32 {
        self.default_grade
    }

    /// Set the default grade for cells not in the map.
    pub fn set_default_grade(&mut self, default_grade: u32) {
        self.default_grade = default_grade;
    }
}

impl<B> Grader<B> for HashMapGrader<B>
where
    B: Hash + Eq + Clone,
{
    fn grade(&self, cell: &B) -> u32 {
        self.grades.get(cell).copied().unwrap_or(self.default_grade)
    }
}

impl<B> FromIterator<(B, u32)> for HashMapGrader<B>
where
    B: Hash + Eq + Clone,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (B, u32)>,
    {
        Self::from_map(HashMap::from_iter(iter))
    }
}

impl<B> Serialize for HashMapGrader<B>
where
    B: Clone + Eq + Hash + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let map_as_vec: Vec<(B, u32)> = self
            .grades
            .iter()
            .map(|(key, value)| (key.clone(), *value))
            .collect();
        (self.default_grade, map_as_vec).serialize(serializer)
    }
}

impl<'de, B> Deserialize<'de> for HashMapGrader<B>
where
    B: Clone + Eq + Hash + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (default_grade, map_as_vec) = <(u32, Vec<(B, u32)>)>::deserialize(deserializer)?;
        let mut grader = HashMapGrader::from_iter(map_as_vec);
        grader.set_default_grade(default_grade);
        Ok(grader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_map() {
        let mut map = HashMap::new();
        map.insert([0, 1], 1);
        map.insert([1, 0], 2);

        let grader = HashMapGrader::from_map(map);
        assert_eq!(grader.grade(&[0, 1]), 1);
        assert_eq!(grader.grade(&[1, 0]), 2);
        assert_eq!(grader.grade(&[1, 1]), 0);

        let mut map = HashMap::new();
        map.insert((5, 10), 5);

        let grader = HashMapGrader::from_map_with_default(map, 999);
        assert_eq!(grader.grade(&(5, 10)), 5);
        assert_eq!(grader.grade(&(0, 0)), 999);
        assert_eq!(grader.default_grade(), 999);
    }

    #[test]
    fn test_from_cells_with_fn() {
        let cells = vec![5i16, 15i16, 150i16];
        let grader = HashMapGrader::from_cells_with_fn(cells, |cell| (*cell / 5) as u32);

        assert_eq!(grader.grade(&5i16), 1);
        assert_eq!(grader.grade(&15i16), 3);
        assert_eq!(grader.grade(&150i16), 30);
        assert_eq!(grader.grade(&999i16), 0);
    }

    #[test]
    fn test_from_iter() {
        let grader: HashMapGrader<i8> = vec![(1, 1), (2, 2), (3, 3)].into_iter().collect();

        assert_eq!(grader.grade(&1), 1);
        assert_eq!(grader.grade(&2), 2);
        assert_eq!(grader.grade(&3), 3);
        assert_eq!(grader.grade(&4), 0);
    }

    #[test]
    fn test_mutable_access() {
        let mut grader = HashMapGrader::new(0);

        // Add grades using mutable reference
        {
            let map = grader.grades_mut();
            map.insert((1, 2, 3, 4), 10);
            map.insert((5, 6, 7, 8), 20);
        }

        assert_eq!(grader.grade(&(1, 2, 3, 4)), 10);
        assert_eq!(grader.grade(&(5, 6, 7, 8)), 20);
    }

    #[test]
    fn test_set_default_grade() {
        let mut grader: HashMapGrader<[u16; 3]> = HashMapGrader::new(0);

        assert_eq!(grader.grade(&[100, 200, 300]), 0);

        grader.set_default_grade(999);

        assert_eq!(grader.grade(&[100, 200, 300]), 999);
        assert_eq!(grader.default_grade(), 999);
    }

    #[test]
    fn test_uniform() {
        let cells = vec!["a", "b", "c"];
        let grader = HashMapGrader::uniform(cells, 42, 7);

        assert_eq!(grader.grade(&"a"), 42);
        assert_eq!(grader.grade(&"b"), 42);
        assert_eq!(grader.grade(&"c"), 42);
        assert_eq!(grader.grade(&"d"), 7); // default grade
    }
}
