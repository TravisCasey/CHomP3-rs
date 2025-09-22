// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Utility types and iterators for cubical complexes.

use std::iter::zip;
use std::mem::transmute;
use std::slice::ChunksExact;

use serde::{Deserialize, Serialize};

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
#[derive(Clone, Debug, Serialize, Deserialize)]
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

    /// Return an immutable reference to the wrapped grading function for
    /// orthants (or, top-dimensional cubes).
    pub fn orthant_grader(&self) -> &G {
        &self.orthant_grader
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

/// Convenience wrapper over slices of the vector in `OrthantTrie` to handle
/// interpreting the slice as a node in the trie. The trie does not store
/// instances of this wrapper; instead, slices are wrapped (via the
/// [`TrieNode::wrap`] method) only when searching the tree.
enum TrieNode<'a> {
    Branch {
        /// The least coordinate in this branch. The bit index (into `bitmap`)
        /// is computed relative to this minimum coordinate.
        minimal: i16,

        /// This slice is the same length as `bitmap`. `prior_popcounts[i]` is
        /// the number of bits set in the subslice `bitmap[0..i]`.
        prior_popcount: &'a [u16],

        /// Regarded as a slice of bits (16 per word), a bit is set if the
        /// corresponding coordinate (relative to `minimal`) is present in the
        /// trie. The index of the next node for this coordinate is given by
        /// `child_indices[p]`, where `p` is the number of bits set before the
        /// corresponding bit. Computed using `u16::count_ones` and
        /// `prior_popcount`.
        bitmap: &'a [u16],

        /// The indices (in the vector implementing the trie) to children nodes;
        /// there is 1 entry for each bit set in `bitmap`. If this is a leaf,
        /// these are instead grades.
        child_indices: &'a [u32],
    },
    Compressed {
        /// The length of chunks in `chunks` to compare against.
        comp_len: usize,

        /// Chunks (subslices of multiple 16-bit coordinates) that comprise all
        /// of the entries of the trie from this node; for small number of
        /// chunks, this may be faster than several layers of very sparsely
        /// populated branches. If chunk `i` matches the sub slice of
        /// coordinates, `child_indices[i]` is its next index or grade.
        chunks: ChunksExact<'a, i16>,

        /// The indices (in the vector implementing the trie) to children nodes;
        /// there is 1 entry for each chunk in `chunk`. If this is a leaf, these
        /// are instead grades.
        child_indices: &'a [u32],
    },
}

impl<'a> TrieNode<'a> {
    fn wrap(slice: &'a [u16]) -> Self {
        let discriminant = slice[0];
        if discriminant == 0 {
            // Branch node structure:
            //   [discriminant, minimal, bitmap_len, num_children]
            //   + bitmap_len * [prior_popcount]
            //   + bitmap_len * [bitmap]
            //   + num_children * [child_indices as u32]
            //   + remainder

            let minimal = slice[1] as i16; // u16 -> i16 reinterprets bits
            let bitmap_len = slice[2] as usize;
            let num_children = slice[3] as usize;

            let (prior_popcount, remainder) = slice.split_at(4).1.split_at(bitmap_len);
            let (bitmap, remainder) = remainder.split_at(bitmap_len);
            let halved_child_indices = remainder.split_at(2 * num_children).0;

            // &[u16] reinterpreted as &[u32]
            // align_to fails here, use transmute instead
            let child_indices = unsafe { transmute::<&[u16], &[u32]>(halved_child_indices) };

            debug_assert_eq!(
                prior_popcount[bitmap_len - 1] as usize
                    + bitmap[bitmap_len - 1].count_ones() as usize,
                num_children,
                "number of children does not match number of set bits in branch bitmap"
            );

            Self::Branch {
                minimal,
                prior_popcount,
                bitmap,
                child_indices,
            }
        } else if discriminant == 1 {
            // Compressed node structure:
            //   [discriminant, comp_len, num_children]
            //   + num_children * [comp_len * [comp]]
            //   + num_children * [child_indices as u32]
            //   + remainder

            let comp_len = slice[1] as usize;
            let num_children = slice[2] as usize;

            let (comp_slice, remainder) = slice.split_at(3).1.split_at(comp_len * num_children);
            let halved_child_indices = remainder.split_at(2 * num_children).0;

            // &[u16] reinterpreted as &[u32]
            // align_to fails here, use transmute instead
            let child_indices = unsafe { transmute::<&[u16], &[u32]>(halved_child_indices) };

            unsafe {
                Self::Compressed {
                    comp_len,
                    chunks: comp_slice.align_to().1.chunks_exact(comp_len),
                    child_indices,
                }
            }
        } else {
            panic!("Invalid discriminant {discriminant} detected in trie");
        }
    }
}

/// A grader that stores grades for top-dimensional cubes in a cubical complex
/// in a prefix tree-like structure with alphabet over the possible coordinates
/// in each dimension.
///
/// `OrthantTrie` implements `Grader<Orthant>`, where the [`Orthant`] object
/// represents the singular top-dimensional cube within that orthant. The height
/// of the trie is less than or equal to the ambient dimension of the cubical
/// complex, as each node in the trie is either a branch among possible values
/// of the coordinate along the corresponding axis, or is a compressed version
/// of multiple levels of the same.
///
/// If there are fewer leaves connected to some node of the trie than some
/// threshold (set by either the `compression_threshold` argument at
/// construction or using the default value of 6), then the remaining
/// coordinates of the `Orthant` objects representing those leaves are instead
/// stored and compared against the orthant being graded directly. This can
/// speed up queries in very sparse, high-dimensional spaces, where many
/// top-dimensional cubes are not included and instead given the default grade.
///
/// Notably, the trie itself cannot be mutated after construction. This
/// simplifies trie structure and trackers and more easily enables the
/// compression optimization. The default grade, however, can be modified via
/// the [`OrthantTrie::set_default_grade`] method.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrthantTrie {
    trie: Vec<u16>,
    default_grade: u32,
}

impl OrthantTrie {
    const DEFAULT_COMPRESSION_THRESHOLD: usize = 6;

    /// Construct an `OrthantTrie` holding the grade relationship in
    /// `orthant_grades`. All other [`Orthant`] objects are assigned the
    /// default grade when queried.
    ///
    /// Duplicates in the `orthant_grades` parameter are unexpected and will be
    /// removed. The `compression_threshold` sets the number of leaves connected
    /// to a node to enable the compression optimization; too few will tend to
    /// create a taller tree that may have longer access times in sparsely
    /// populated regions, while too large may create compressed nodes that take
    /// more time comparing the queried [`Orthant`] directly against possible
    /// coordinates than would be achieved using the normal trie structure. A
    /// value of 0 disables the compression entirely.
    pub fn from_vec_with_threshold(
        mut orthant_grades: Vec<(Orthant, u32)>,
        default_grade: u32,
        compression_threshold: usize,
    ) -> Self {
        let mut orthant_trie = Self {
            trie: Vec::new(),
            default_grade,
        };

        if !orthant_grades.is_empty() {
            orthant_grades.sort_unstable();
            orthant_grades.dedup();
            let dimension = orthant_grades[0].0.ambient_dimension() as usize;

            orthant_trie.dfs_construct(&orthant_grades, compression_threshold, 0, dimension);
        }
        orthant_trie
    }

    /// A version of [`OrthantTrie::from_vec_with_threshold`] which uses the
    /// default compression threshold, 6.
    pub fn from_vec(orthant_grades: Vec<(Orthant, u32)>, default_grade: u32) -> Self {
        Self::from_vec_with_threshold(
            orthant_grades,
            default_grade,
            Self::DEFAULT_COMPRESSION_THRESHOLD,
        )
    }

    /// This method is identical to [`OrthantTrie::from_vec_with_threshold`]
    /// after collecting `iter` into a vector.
    pub fn from_iter_with_threshold(
        iter: impl IntoIterator<Item = (Orthant, u32)>,
        default_grade: u32,
        compression_threshold: usize,
    ) -> Self {
        Self::from_vec_with_threshold(
            iter.into_iter().collect(),
            default_grade,
            compression_threshold,
        )
    }

    /// This method is identical to [`OrthantTrie::from_vec`] after collecting
    /// `iter` into a vector.
    pub fn from_iter(iter: impl IntoIterator<Item = (Orthant, u32)>, default_grade: u32) -> Self {
        Self::from_iter_with_threshold(iter, default_grade, Self::DEFAULT_COMPRESSION_THRESHOLD)
    }

    /// A constructor that assigns the grade `uniform_grade` to every
    /// [`Orthant`] object in `iter` and `default_grade` to every other orthant,
    /// implicitly. This method is otherwise equivalent to
    /// [`OrthantTrie::from_vec_with_threshold`].
    pub fn uniform_with_threshold(
        iter: impl IntoIterator<Item = Orthant>,
        uniform_grade: u32,
        default_grade: u32,
        compression_threshold: usize,
    ) -> Self {
        Self::from_vec_with_threshold(
            iter.into_iter()
                .map(|orthant| (orthant, uniform_grade))
                .collect(),
            default_grade,
            compression_threshold,
        )
    }

    /// This method is identical to [`OrthantTrie::uniform_with_threshold`] but
    /// uses the default compression threshold, 6.
    pub fn uniform(
        iter: impl IntoIterator<Item = Orthant>,
        uniform_grade: u32,
        default_grade: u32,
    ) -> Self {
        Self::uniform_with_threshold(
            iter,
            uniform_grade,
            default_grade,
            Self::DEFAULT_COMPRESSION_THRESHOLD,
        )
    }

    /// Get the default grade assigned to all [`Orthant`] objects for which a
    /// grade is not explicitly stored.
    pub fn default_grade(&self) -> u32 {
        self.default_grade
    }

    /// Set a default grade value assign to all [`Orthant`] objects for which a
    /// grade is not explicitly stored.
    pub fn set_default_grade(&mut self, default_grade: u32) {
        self.default_grade = default_grade;
    }

    fn dfs_construct(
        &mut self,
        orthant_grades: &[(Orthant, u32)],
        compression_threshold: usize,
        depth: usize,
        dimension: usize,
    ) {
        debug_assert!(!orthant_grades.is_empty());

        // Compressed (Leaf) node.
        // Non-leaf nodes (with multiple layers, even) could be implemented
        // for compression sometimes, but not currently.
        // Should this condition also depend on the depth?
        if orthant_grades.len() < compression_threshold {
            let comp_len = dimension - depth;
            let num_children = orthant_grades.len();

            self.trie.push(1); // discriminant
            self.trie.push(comp_len as u16);
            self.trie.push(num_children as u16);

            // Unsafe to reinterpret &[i16] -> &[u16] and u32 -> &[u16] bitwise
            unsafe {
                for (orthant, _grade) in orthant_grades {
                    self.trie
                        .extend_from_slice(orthant.as_slice()[depth..].align_to().1);
                }
                for (_orthant, grade) in orthant_grades {
                    self.trie.extend_from_slice([*grade].align_to().1);
                }
            }

            return;
        }

        // Branch nodes
        // Possibly a leaf, in which case the child index is actually the grade
        let mut coord_counts = vec![(orthant_grades[0].0[depth], 1)];
        for (orthant, _grade) in orthant_grades.iter().skip(1) {
            let last_index = coord_counts.len() - 1;
            if coord_counts[last_index].0 == orthant[depth] {
                coord_counts[last_index].1 += 1;
            } else {
                coord_counts.push((orthant[depth], 1));
            }
        }

        let minimal = coord_counts[0].0;
        let maximal = coord_counts[coord_counts.len() - 1].0;
        // 1 bit for each possible coordinate value between (inclusive) minimal
        // and maximal.
        let bitmap_len = ((maximal - minimal + 1) as usize).div_ceil(16);
        let num_children = coord_counts.len();
        self.trie.push(0); // discriminant
        self.trie.push(minimal as u16);
        self.trie.push(bitmap_len as u16);
        self.trie.push(num_children as u16);

        let popcount_begin_index = self.trie.len();
        let bitmap_begin_index = self.trie.len() + bitmap_len;
        self.trie.resize(self.trie.len() + 2 * bitmap_len, 0u16);

        for (coord, _count) in coord_counts.iter() {
            let bit_index = (coord - minimal) as usize;
            let bitmap_word_index = bit_index / 16;
            let bit_flag = 1 << (bit_index % 16);
            self.trie[bitmap_begin_index + bitmap_word_index] |= bit_flag;
        }

        for bitmap_word_index in 1..bitmap_len {
            let popcount_index = popcount_begin_index + bitmap_word_index;
            let bitmap_index = bitmap_begin_index + bitmap_word_index;
            self.trie[popcount_index] =
                self.trie[popcount_index - 1] + self.trie[bitmap_index - 1].count_ones() as u16;
        }

        // Leaf node: child indices are grades instead of indices
        if depth + 1 == dimension {
            unsafe {
                for (_orthant, grade) in orthant_grades.iter() {
                    self.trie.extend_from_slice([*grade].align_to().1);
                }
            }
            return;
        }

        // Else, the child indices point to where the next node starts. This
        // requires calling `dfs_construct` on each child node first, to get
        // the node spacing right.
        let child_begin_index = self.trie.len();
        // Indices are 32 bit, using two 16 bit words. Treat them as slices
        // instead of using bit operations to achieve this so there's no
        // endianness issues (are there any big endian machines nowadays?)
        self.trie
            .resize(self.trie.len() + 2 * coord_counts.len(), u16::MAX);

        let mut next_coord_slice;
        let mut remainder = orthant_grades;
        for (index, (_coord, count)) in coord_counts.into_iter().enumerate() {
            let next_node_index = self.trie.len();
            unsafe {
                self.trie[child_begin_index + 2 * index..child_begin_index + 2 * index + 2]
                    .copy_from_slice([next_node_index as u32].align_to().1)
            };
            (next_coord_slice, remainder) = remainder.split_at(count);
            self.dfs_construct(
                next_coord_slice,
                compression_threshold,
                depth + 1,
                dimension,
            );
        }
    }

    fn search(&self, node_index: usize, coords: &[i16]) -> Option<u32> {
        if coords.is_empty() {
            return Some(node_index as u32);
        }

        if node_index >= self.trie.len() {
            return None;
        }

        match TrieNode::wrap(&self.trie[node_index..]) {
            TrieNode::Branch {
                minimal,
                prior_popcount,
                bitmap,
                child_indices,
            } => {
                if coords[0] < minimal {
                    return None;
                }

                // If the bit at this index in the bitmap is not set, coords is not in the trie
                let bit_index = (coords[0] - minimal) as usize;
                // Index of the word containing the bit at bit_index
                let bitmap_word_index = bit_index / 16;

                // Exceeds range of bitmap -> not in trie
                if bitmap_word_index >= bitmap.len() {
                    return None;
                }

                // The word containing the bit at bit_index
                let bitmap_word = bitmap[bitmap_word_index];
                // 1 bit set at the correct position to check the bit in bit_word
                let bit_flag = 1 << (bit_index % 16);

                if bitmap_word & bit_flag == 0 {
                    return None;
                }

                let prior_in_word = (bitmap_word & (bit_flag - 1)).count_ones() as usize;
                let next_node_index = child_indices
                    [prior_popcount[bitmap_word_index] as usize + prior_in_word]
                    as usize;
                self.search(next_node_index, coords.split_at(1).1)
            }
            TrieNode::Compressed {
                comp_len,
                chunks,
                child_indices,
            } => {
                for (chunk_index, chunk) in chunks.enumerate() {
                    if coords[..comp_len] == *chunk {
                        return self.search(
                            child_indices[chunk_index] as usize,
                            coords.split_at(comp_len).1,
                        );
                    }
                }
                None
            }
        }
    }
}

impl Grader<Orthant> for OrthantTrie {
    fn grade(&self, orthant: &Orthant) -> u32 {
        self.search(0, orthant.as_slice())
            .unwrap_or(self.default_grade)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::HashMapGrader;

    #[test]
    fn test_trie_grader_compressed() {
        let orthants_and_grades = vec![
            (Orthant::new(vec![0, 0]), 1),
            (Orthant::new(vec![0, 1]), 2),
            (Orthant::new(vec![1, 0]), 3),
            (Orthant::new(vec![1, 1]), 4),
        ];

        let grader = OrthantTrie::from_iter(orthants_and_grades, 0);

        // with compression threshold at 6, this is just one compressed node
        // discriminant 1 -> compressed
        // length 2
        // child (grade) count 4
        let mut correct_trie = vec![1, 2, 4, 0, 0, 0, 1, 1, 0, 1, 1];
        unsafe {
            correct_trie.extend_from_slice([1u32].align_to().1);
            correct_trie.extend_from_slice([2u32].align_to().1);
            correct_trie.extend_from_slice([3u32].align_to().1);
            correct_trie.extend_from_slice([4u32].align_to().1);
        }
        assert_eq!(grader.trie, correct_trie);

        assert_eq!(grader.grade(&Orthant::new(vec![0, 0])), 1);
        assert_eq!(grader.grade(&Orthant::new(vec![0, 1])), 2);
        assert_eq!(grader.grade(&Orthant::new(vec![1, 0])), 3);
        assert_eq!(grader.grade(&Orthant::new(vec![1, 1])), 4);

        // Test default grade for missing entries
        assert_eq!(grader.grade(&Orthant::new(vec![2, 2])), 0);
        assert_eq!(grader.grade(&Orthant::new(vec![-1, 0])), 0);
    }

    #[test]
    fn test_trie_grader_uncompressed() {
        let orthants_and_grades = vec![
            (Orthant::new(vec![0, 2]), 1),
            (Orthant::new(vec![0, -1]), 2),
            (Orthant::new(vec![1, 0]), 3),
            (Orthant::new(vec![1, 1]), 4),
            (Orthant::new(vec![17, 4]), 20),
        ];

        // compression_threshold at 0 -> no compression
        let grader = OrthantTrie::from_iter_with_threshold(orthants_and_grades, 99, 0);

        // First branch, possible values 0..=17
        // discriminant 0 -> branch
        // minimal 0
        // bitmap len 2
        // child count 3
        let mut correct_trie = vec![0, 0, 2, 3, 0, 2, 3, 2];
        unsafe {
            correct_trie.extend_from_slice([14u32].align_to().1);
            correct_trie.extend_from_slice([24u32].align_to().1);
            correct_trie.extend_from_slice([34u32].align_to().1);
        }
        // Second branch, first coord 0, possible values -1..=2
        // discriminant 0 -> branch
        // minimal -1
        // bitmap len 1
        // child count 2
        correct_trie.extend_from_slice(&[0, -1i16 as u16, 1, 2, 0, 9]);
        unsafe {
            correct_trie.extend_from_slice([2u32].align_to().1);
            correct_trie.extend_from_slice([1u32].align_to().1);
        }

        // Third branch, first coord 1, possible values 0..=1
        // discriminant 0 -> branch
        // minimal 0
        // bitmap len 1
        // child count 2
        correct_trie.extend_from_slice(&[0, 0, 1, 2, 0, 3]);
        unsafe {
            correct_trie.extend_from_slice([3u32].align_to().1);
            correct_trie.extend_from_slice([4u32].align_to().1);
        }

        // Fourth branch, first coord 17, possible value 4
        // discriminant 0 -> branch
        // minimal 4
        // bitmap len 1
        // child count 1
        correct_trie.extend_from_slice(&[0, 4, 1, 1, 0, 1]);
        unsafe {
            correct_trie.extend_from_slice([20u32].align_to().1);
        }

        assert_eq!(grader.trie, correct_trie);

        assert_eq!(grader.grade(&Orthant::new(vec![0, 2])), 1);
        assert_eq!(grader.grade(&Orthant::new(vec![0, -1])), 2);
        assert_eq!(grader.grade(&Orthant::new(vec![1, 0])), 3);
        assert_eq!(grader.grade(&Orthant::new(vec![1, 1])), 4);
        assert_eq!(grader.grade(&Orthant::new(vec![17, 4])), 20);
        assert_eq!(grader.grade(&Orthant::new(vec![15, 4])), 99);
        assert_eq!(grader.grade(&Orthant::new(vec![-20, 0])), 99);
        assert_eq!(grader.grade(&Orthant::new(vec![300, -1])), 99);
    }

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
