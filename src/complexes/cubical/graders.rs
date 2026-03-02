// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Specialized graders for cubical complexes.
//!
//! This module provides grading functions specifically designed for cubical
//! complexes, including the top cube grader and the orthant trie data structure
//! for efficient grade lookups.

use std::{iter::zip, slice::ChunksExact};

use super::{Cube, Orthant};
use crate::Grader;

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
/// use chomp3rs::{Cube, Grader, HashGrader, Orthant, TopCubeGrader};
///
/// // Create an orthant grader
/// let mut orthant_grades = HashMap::new();
/// orthant_grades.insert(Orthant::from([0, 0]), 1);
/// orthant_grades.insert(Orthant::from([0, 1]), 2);
/// orthant_grades.insert(Orthant::from([1, 0]), 3);
/// orthant_grades.insert(Orthant::from([1, 1]), 4);
/// let orthant_grader = HashGrader::from_map(orthant_grades, 0);
///
/// // Create cube grader with minimum grade for short-circuiting
/// let cube_grader = TopCubeGrader::new(orthant_grader, Some(1));
///
/// // Grade a vertex cube - it's a face of all 4 surrounding top cubes
/// let vertex = Cube::vertex(Orthant::from([1, 1]));
/// let grade = cube_grader.grade(&vertex);
/// assert_eq!(grade, 1); // minimum of surrounding grades: 1, 2, 3, 4
/// ```
#[derive(Clone, Debug)]
pub struct TopCubeGrader<G> {
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

    /// Returns the minimum grade used for short-circuiting.
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
    #[must_use]
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

        /// The indices (in the vector implementing the trie) to children nodes,
        /// stored as pairs of `u16` values in native byte order (2 slots per
        /// `u32` value). There is 1 logical entry for each bit set in
        /// `bitmap`. If this is a leaf, these are instead grades. Use
        /// [`read_u32`] to extract the `u32` at logical index `i`.
        child_indices: &'a [u16],
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

        /// The indices (in the vector implementing the trie) to children nodes,
        /// stored as pairs of `u16` values in native byte order. There is 1
        /// logical entry for each chunk in `chunks`. If this is a leaf, these
        /// are instead grades. Use [`read_u32`] to extract the `u32` at logical
        /// index `i`.
        child_indices: &'a [u16],
    },
}

/// Read a `u32` value from a `&[u16]` slice where `u32` values are stored as
/// pairs of native-endian `u16` values (written via `align_to`).
fn read_u32(slice: &[u16], index: usize) -> u32 {
    let lo = slice[2 * index].to_ne_bytes();
    let hi = slice[2 * index + 1].to_ne_bytes();
    u32::from_ne_bytes([lo[0], lo[1], hi[0], hi[1]])
}

impl<'a> TrieNode<'a> {
    /// Wrap a slice of the trie's backing vector as a node.
    ///
    /// # Format
    ///
    /// The caller must ensure that `slice` is correctly formatted according to
    /// the node layout expected by `OrthantTrie`. Specifically:
    ///
    /// 1. The first element of the slice is the discriminant: if it is 0, then
    ///    this is a branch node; if it is 1, then this is a compressed node.
    /// 2. For branch nodes:
    ///      - `slice[1]` is the minimal orthant coordinate stored at this
    ///        branch. This is irrelevant for the wrapping procedure, and is
    ///        passed back to the trie.
    ///      - `slice[2]` is the number of 16-bit words in the bitmap following
    ///        the metadata.
    ///      - `slice[3]` is the number of children nodes (number of bits set in
    ///        the bitmap) this branch has.
    ///      - This is followed by the bitmap, which tracks both the popcount
    ///        prior to each word, and the bitmap itself.
    ///      - Then there is a slice of 32-bit words (reinterpreting two
    ///        adjacent 16-bit words) representing child indices or leaf nodes.
    /// 3. For compressed nodes:
    ///      - `slice[1]` is the number of coordinates compressed together here.
    ///      - `slice[2]` is the number of child nodes of this node.
    ///      - The next contains a slice of 16-bit words, slice length equal to
    ///        the number of compressed coordinates. There are a number of these
    ///        slices equal to the number of children.
    ///      - Then there is a slice of 32-bit words (reinterpreting two
    ///        adjacent 16-bit words) representing child indices or leaf nodes.
    /// 4. Finally, there is the rest of the trie that is not accessed by this
    ///    method.
    ///
    /// These invariants are maintained by `OrthantTrie::build_trie` during
    /// construction and are never violated during normal operation.
    unsafe fn wrap(slice: &'a [u16]) -> Self {
        let discriminant = slice[0];
        if discriminant == 0 {
            // Branch node structure:
            //   [discriminant, minimal, bitmap_len, num_children]
            //   + bitmap_len * [prior_popcount]
            //   + bitmap_len * [bitmap]
            //   + num_children * [child_indices as u32, stored as 2*u16 each]
            //   + remainder

            let minimal = slice[1].cast_signed(); // u16 -> i16 reinterprets bits
            let bitmap_len = slice[2] as usize;
            let num_children = slice[3] as usize;

            // SAFETY: Slice layout verified by caller's safety contract
            let (prior_popcount, remainder) =
                unsafe { slice.split_at_unchecked(4).1.split_at_unchecked(bitmap_len) };
            let (bitmap, remainder) = unsafe { remainder.split_at_unchecked(bitmap_len) };
            let child_indices = unsafe { remainder.split_at_unchecked(2 * num_children).0 };

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
            //   + num_children * [child_indices as u32, stored as 2*u16 each]
            //   + remainder

            let comp_len = slice[1] as usize;
            let num_children = slice[2] as usize;

            // SAFETY: Slice layout verified by caller's safety contract
            let (comp_slice, remainder) = unsafe {
                slice
                    .split_at_unchecked(3)
                    .1
                    .split_at_unchecked(comp_len * num_children)
            };
            let child_indices = unsafe { remainder.split_at_unchecked(2 * num_children).0 };

            Self::Compressed {
                comp_len,
                // SAFETY: comp_slice comes from a properly constructed trie where
                // coordinates are stored as u16 values with proper alignment
                chunks: unsafe { comp_slice.align_to().1.chunks_exact(comp_len) },
                child_indices,
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
#[derive(Clone, Debug)]
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
    #[must_use]
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
    #[must_use]
    pub fn from_vec(orthant_grades: Vec<(Orthant, u32)>, default_grade: u32) -> Self {
        Self::from_vec_with_threshold(
            orthant_grades,
            default_grade,
            Self::DEFAULT_COMPRESSION_THRESHOLD,
        )
    }

    /// This method is identical to [`OrthantTrie::from_vec_with_threshold`]
    /// after collecting `iter` into a vector.
    #[must_use]
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
    #[must_use]
    pub fn from_iter(iter: impl IntoIterator<Item = (Orthant, u32)>, default_grade: u32) -> Self {
        Self::from_iter_with_threshold(iter, default_grade, Self::DEFAULT_COMPRESSION_THRESHOLD)
    }

    /// A constructor that assigns the grade `uniform_grade` to every
    /// [`Orthant`] object in `iter` and `default_grade` to every other orthant,
    /// implicitly. This method is otherwise equivalent to
    /// [`OrthantTrie::from_vec_with_threshold`].
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
        debug_assert!(
            !orthant_grades.is_empty(),
            "dfs_construct called with empty orthant_grades at depth {depth}"
        );

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

            // SAFETY: Reinterpreting types via align_to is safe because:
            // - align_to().1 returns the properly aligned middle slice
            // - For &[i16] -> &[u16]: both types have same size/alignment, we only read bit
            //   patterns
            // - For &[u32] -> &[u16]: u32 can be safely viewed as two u16 values (no
            //   endianness issues since we'll read them back the same way)
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

        // Branch nodes: group orthants by coordinate value at current depth
        // coord_counts stores (coordinate_value, count_of_orthants_with_that_value)
        // This determines the branching structure of the trie at this level
        let mut coord_counts = vec![(orthant_grades[0].0[depth], 1)];
        for (orthant, _grade) in orthant_grades.iter().skip(1) {
            let last_index = coord_counts.len() - 1;
            if coord_counts[last_index].0 == orthant[depth] {
                // Same coordinate as previous: increment count
                coord_counts[last_index].1 += 1;
            } else {
                // New coordinate: add new entry
                coord_counts.push((orthant[depth], 1));
            }
        }

        let minimal = coord_counts[0].0;
        let maximal = coord_counts[coord_counts.len() - 1].0;
        // 1 bit for each possible coordinate value between (inclusive) minimal
        // and maximal.
        let bitmap_len = ((maximal as isize - minimal as isize + 1).cast_unsigned()).div_ceil(16);
        let num_children = coord_counts.len();
        self.trie.push(0); // discriminant
        self.trie.push(minimal.cast_unsigned());
        self.trie.push(bitmap_len as u16);
        self.trie.push(num_children as u16);

        // Allocate space for popcount array and bitmap
        // Layout: [popcount[0], ..., popcount[bitmap_len-1], bitmap[0], ...,
        // bitmap[bitmap_len-1]]
        let popcount_begin_index = self.trie.len();
        let bitmap_begin_index = self.trie.len() + bitmap_len;
        self.trie.resize(self.trie.len() + 2 * bitmap_len, 0u16);

        // Set bits in bitmap for each coordinate that appears
        for (coord, _count) in &coord_counts {
            let bit_index = (*coord as isize - minimal as isize).cast_unsigned();
            let bitmap_word_index = bit_index / 16;
            let bit_flag = 1 << (bit_index % 16);
            self.trie[bitmap_begin_index + bitmap_word_index] |= bit_flag;
        }

        // Compute cumulative popcount for each bitmap word
        // popcount[i] = total number of set bits in bitmap[0..i]
        // This allows O(1) lookup of child index from coordinate value
        for bitmap_word_index in 1..bitmap_len {
            let popcount_index = popcount_begin_index + bitmap_word_index;
            let bitmap_index = bitmap_begin_index + bitmap_word_index;
            self.trie[popcount_index] =
                self.trie[popcount_index - 1] + self.trie[bitmap_index - 1].count_ones() as u16;
        }

        // Leaf node: child indices are grades instead of indices
        if depth + 1 == dimension {
            // SAFETY: Reinterpreting u32 as &[u16] via align_to is safe because:
            // - align_to().1 returns the properly aligned middle slice
            // - u32 can be safely viewed as two u16 values (same total size, compatible
            //   alignment)
            // - We maintain consistency by reading them back as u32 later via transmute in
            //   wrap()
            unsafe {
                for (_orthant, grade) in orthant_grades {
                    self.trie.extend_from_slice([*grade].align_to().1);
                }
            }
            return;
        }

        // Non-leaf branch: child indices point to the start of child nodes in the trie
        // vector We recursively construct each child subtree, recording where
        // it starts
        let child_begin_index = self.trie.len();
        // Reserve space for child indices (stored as u32, occupying 2 u16 slots each)
        self.trie
            .resize(self.trie.len() + 2 * coord_counts.len(), u16::MAX);

        // Recursively construct child nodes for each unique coordinate value
        let mut next_coord_slice;
        let mut remainder = orthant_grades;
        for (index, (_coord, count)) in coord_counts.into_iter().enumerate() {
            // Record where this child's node begins in the trie
            let next_node_index = self.trie.len();

            // SAFETY: Reinterpreting u32 as &[u16] and copying into the trie is safe
            // because:
            // - align_to().1 gives us exactly 2 u16 values representing the u32
            // - The target slice has length 2 (verified by the range)
            // - We'll read these back as u32 via transmute in wrap(), maintaining
            //   consistency
            unsafe {
                self.trie[child_begin_index + 2 * index..child_begin_index + 2 * index + 2]
                    .copy_from_slice([next_node_index as u32].align_to().1);
            };
            // Split off orthants with this coordinate value and recurse
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
        // Base case: no more coordinates to match, return the grade stored at this
        // position
        if coords.is_empty() {
            return Some(node_index as u32);
        }

        // Invalid node index indicates a bug in trie construction
        debug_assert!(
            node_index < self.trie.len(),
            "Invalid trie node index {} (trie length {}): child index points beyond trie bounds",
            node_index,
            self.trie.len()
        );

        // SAFETY: The slice starting at node_index is guaranteed to be a valid
        // trie node because:
        // - node_index is either 0 (root) or came from child_indices in a previous node
        // - All nodes are constructed by build_trie which maintains the node layout
        //   invariants
        // - The trie vector is never modified after construction
        match unsafe { TrieNode::wrap(&self.trie[node_index..]) } {
            TrieNode::Branch {
                minimal,
                prior_popcount,
                bitmap,
                child_indices,
            } => {
                // Check if coordinate is below the minimum value in this branch
                if coords[0] < minimal {
                    return None;
                }

                // Compute which bit in the bitmap corresponds to this coordinate
                let bit_index = (coords[0] as isize - minimal as isize).cast_unsigned();
                let bitmap_word_index = bit_index / 16;

                // Check if coordinate exceeds bitmap range
                if bitmap_word_index >= bitmap.len() {
                    return None;
                }

                // Check if this coordinate exists in the trie (bitmap bit is set)
                let bitmap_word = bitmap[bitmap_word_index];
                let bit_flag = 1 << (bit_index % 16);

                if bitmap_word & bit_flag == 0 {
                    return None;
                }

                // Compute child index using popcount:
                // Total set bits before this position = prior_popcount[word] + bits set in
                // current word before bit_flag
                let prior_in_word = (bitmap_word & (bit_flag - 1)).count_ones() as usize;
                let child_array_index = prior_popcount[bitmap_word_index] as usize + prior_in_word;
                let next_node_index = read_u32(child_indices, child_array_index) as usize;
                // SAFETY: coords is non-empty (checked at function start), so splitting at 1 is
                // safe
                self.search(next_node_index, unsafe { coords.split_at_unchecked(1).1 })
            },
            TrieNode::Compressed {
                comp_len,
                chunks,
                child_indices,
            } => {
                for (chunk_index, chunk) in chunks.enumerate() {
                    if coords[..comp_len] == *chunk {
                        // SAFETY: We just successfully indexed coords[..comp_len], so coords has
                        // at least comp_len elements, making split_at_unchecked(comp_len) safe
                        return self
                            .search(read_u32(child_indices, chunk_index) as usize, unsafe {
                                coords.split_at_unchecked(comp_len).1
                            });
                    }
                }
                None
            },
        }
    }
}

impl Grader<Orthant> for OrthantTrie {
    fn grade(&self, orthant: &Orthant) -> u32 {
        if self.trie.is_empty() {
            return self.default_grade;
        }
        self.search(0, orthant.as_slice())
            .unwrap_or(self.default_grade)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::HashGrader;

    #[test]
    fn trie_grader_compressed() {
        let orthants_and_grades = vec![
            (Orthant::from([0, 0]), 1),
            (Orthant::from([0, 1]), 2),
            (Orthant::from([1, 0]), 3),
            (Orthant::from([1, 1]), 4),
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

        assert_eq!(grader.grade(&Orthant::from([0, 0])), 1);
        assert_eq!(grader.grade(&Orthant::from([0, 1])), 2);
        assert_eq!(grader.grade(&Orthant::from([1, 0])), 3);
        assert_eq!(grader.grade(&Orthant::from([1, 1])), 4);

        // Test default grade for missing entries
        assert_eq!(grader.grade(&Orthant::from([2, 2])), 0);
        assert_eq!(grader.grade(&Orthant::from([-1, 0])), 0);
    }

    #[test]
    fn trie_grader_uncompressed() {
        let orthants_and_grades = vec![
            (Orthant::from([0, 2]), 1),
            (Orthant::from([0, -1]), 2),
            (Orthant::from([1, 0]), 3),
            (Orthant::from([1, 1]), 4),
            (Orthant::from([17, 4]), 20),
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

        assert_eq!(grader.grade(&Orthant::from([0, 2])), 1);
        assert_eq!(grader.grade(&Orthant::from([0, -1])), 2);
        assert_eq!(grader.grade(&Orthant::from([1, 0])), 3);
        assert_eq!(grader.grade(&Orthant::from([1, 1])), 4);
        assert_eq!(grader.grade(&Orthant::from([17, 4])), 20);
        assert_eq!(grader.grade(&Orthant::from([15, 4])), 99);
        assert_eq!(grader.grade(&Orthant::from([-20, 0])), 99);
        assert_eq!(grader.grade(&Orthant::from([300, -1])), 99);
    }

    #[test]
    fn top_cube_grader() {
        // Create orthant grader with known grades
        let mut orthant_grades = HashMap::new();
        orthant_grades.insert(Orthant::from([0, 0]), 1);
        orthant_grades.insert(Orthant::from([0, 1]), 2);
        orthant_grades.insert(Orthant::from([1, 0]), 3);
        orthant_grades.insert(Orthant::from([1, 1]), 4);
        let cube_grader = TopCubeGrader::new(HashGrader::from_map(orthant_grades, 0), Some(1));

        // Test vertex cube - should be face of all 4 surrounding top cubes
        let vertex = Cube::vertex(Orthant::from([1, 1]));
        assert_eq!(cube_grader.grade(&vertex), 1); // min of 1, 2, 3, 4

        // Test edge cube along first axis
        let edge = Cube::from_extent(Orthant::from([1, 1]), &[true, false]);
        assert_eq!(cube_grader.grade(&edge), 3); // min of 3, 4

        // Test edge cube along second axis
        let edge = Cube::from_extent(Orthant::from([1, 1]), &[false, true]);
        assert_eq!(cube_grader.grade(&edge), 2); // min of 2, 4

        // Test top-dimensional cube
        let top_cube = Cube::top_cube(Orthant::from([1, 1]));
        assert_eq!(cube_grader.grade(&top_cube), 4); // just grade of (1,1)
    }

    #[test]
    fn trie_empty() {
        let grader = OrthantTrie::from_iter(vec![], 42);
        assert_eq!(grader.grade(&Orthant::from([0, 0])), 42);
        assert_eq!(grader.grade(&Orthant::from([100, -100])), 42);
    }

    #[test]
    fn trie_single_entry() {
        let grader = OrthantTrie::from_iter(vec![(Orthant::from([5, -3, 7]), 99)], 0);
        assert_eq!(grader.grade(&Orthant::from([5, -3, 7])), 99);
        assert_eq!(grader.grade(&Orthant::from([5, -3, 8])), 0);
        assert_eq!(grader.grade(&Orthant::from([0, 0, 0])), 0);
    }

    #[test]
    fn trie_3d_compressed() {
        let orthants = vec![
            (Orthant::from([-1, 0, 0]), 1),
            (Orthant::from([0, -2, 1]), 2),
            (Orthant::from([0, 1, 0]), 3),
            (Orthant::from([1, 0, 0]), 4),
        ];
        let grader = OrthantTrie::from_iter(orthants, 99);

        assert_eq!(grader.grade(&Orthant::from([-1, 0, 0])), 1);
        assert_eq!(grader.grade(&Orthant::from([0, -2, 1])), 2);
        assert_eq!(grader.grade(&Orthant::from([0, 1, 0])), 3);
        assert_eq!(grader.grade(&Orthant::from([1, 0, 0])), 4);
        assert_eq!(grader.grade(&Orthant::from([1, 1, 1])), 99);
    }

    #[test]
    fn trie_negative_coordinates() {
        let orthants = vec![
            (Orthant::from([-10, -20]), 1),
            (Orthant::from([-10, 5]), 2),
            (Orthant::from([15, -20]), 3),
            (Orthant::from([15, 5]), 4),
        ];
        let grader = OrthantTrie::from_iter(orthants, 0);

        assert_eq!(grader.grade(&Orthant::from([-10, -20])), 1);
        assert_eq!(grader.grade(&Orthant::from([-10, 5])), 2);
        assert_eq!(grader.grade(&Orthant::from([15, -20])), 3);
        assert_eq!(grader.grade(&Orthant::from([15, 5])), 4);
        assert_eq!(grader.grade(&Orthant::from([0, 0])), 0);
    }

    #[test]
    fn trie_sparse_coordinates() {
        // Test with coordinates that span multiple bitmap words (>16 apart)
        let orthants = vec![
            (Orthant::from([-50, 0]), 1),
            (Orthant::from([0, 100]), 2),
            (Orthant::from([50, -30]), 3),
            (Orthant::from([50, 100]), 4),
        ];
        let grader = OrthantTrie::from_iter_with_threshold(orthants, 88, 0);

        assert_eq!(grader.grade(&Orthant::from([-50, 0])), 1);
        assert_eq!(grader.grade(&Orthant::from([0, 100])), 2);
        assert_eq!(grader.grade(&Orthant::from([50, -30])), 3);
        assert_eq!(grader.grade(&Orthant::from([50, 100])), 4);
        assert_eq!(grader.grade(&Orthant::from([25, 50])), 88);
    }

    #[test]
    fn trie_mixed_compression() {
        // Create a scenario where some branches compress and others don't
        // by using a threshold that's in between the number of children
        let mut orthants = vec![];
        // First group: 2 children (should compress with threshold >= 3)
        orthants.push((Orthant::from([0, 0]), 1));
        orthants.push((Orthant::from([0, 1]), 2));
        // Second group: 8 children (should not compress with threshold < 8)
        for i in 0..8 {
            orthants.push((Orthant::from([1, i]), 10 + i as u32));
        }

        let grader = OrthantTrie::from_iter_with_threshold(orthants.clone(), 77, 3);

        // Verify all grades
        assert_eq!(grader.grade(&Orthant::from([0, 0])), 1);
        assert_eq!(grader.grade(&Orthant::from([0, 1])), 2);
        for i in 0..8 {
            assert_eq!(grader.grade(&Orthant::from([1, i])), 10 + i as u32);
        }
        assert_eq!(grader.grade(&Orthant::from([2, 0])), 77);
    }

    #[test]
    fn trie_high_dimension() {
        // Test with 5D to ensure the recursive structure works correctly
        let orthants = vec![
            (Orthant::from([0, 0, 0, 0, 0]), 1),
            (Orthant::from([1, -2, 3, 4, 5]), 2),
            (Orthant::from([-1, -2, -3, -4, -5]), 3),
        ];
        let grader = OrthantTrie::from_iter(orthants, 99);

        assert_eq!(grader.grade(&Orthant::from([0, 0, 0, 0, 0])), 1);
        assert_eq!(grader.grade(&Orthant::from([1, -2, 3, 4, 5])), 2);
        assert_eq!(grader.grade(&Orthant::from([-1, -2, -3, -4, -5])), 3);
        assert_eq!(grader.grade(&Orthant::from([0, 1, 2, 3, 4])), 99);
    }

    #[test]
    fn trie_large_dataset() {
        // Test with many entries to stress the trie structure
        let mut orthants = vec![];
        for x in 0..10 {
            for y in 0..10 {
                let grade = (x * 10 + y) as u32;
                orthants.push((Orthant::from([x, y]), grade));
            }
        }

        let grader = OrthantTrie::from_iter(orthants.clone(), 999);

        // Verify all inserted grades
        for x in 0..10 {
            for y in 0..10 {
                let expected = (x * 10 + y) as u32;
                assert_eq!(grader.grade(&Orthant::from([x, y])), expected);
            }
        }

        // Test default grade for missing entries
        assert_eq!(grader.grade(&Orthant::from([10, 10])), 999);
        assert_eq!(grader.grade(&Orthant::from([-1, 5])), 999);
    }

    #[test]
    fn trie_from_vec_constructors() {
        let orthants = vec![(Orthant::from([0, 0]), 1), (Orthant::from([1, 1]), 2)];

        // Test from_vec
        let grader1 = OrthantTrie::from_vec(orthants.clone(), 0);
        assert_eq!(grader1.grade(&Orthant::from([0, 0])), 1);
        assert_eq!(grader1.grade(&Orthant::from([1, 1])), 2);
        assert_eq!(grader1.grade(&Orthant::from([2, 2])), 0);

        // Test from_vec_with_threshold
        let grader2 = OrthantTrie::from_vec_with_threshold(orthants, 55, 10);
        assert_eq!(grader2.grade(&Orthant::from([0, 0])), 1);
        assert_eq!(grader2.grade(&Orthant::from([1, 1])), 2);
        assert_eq!(grader2.grade(&Orthant::from([2, 2])), 55);
    }

    #[test]
    fn trie_uniform_constructors() {
        let orthants = vec![
            Orthant::from([0, 0]),
            Orthant::from([1, 1]),
            Orthant::from([2, 2]),
        ];

        // Test uniform
        let grader1 = OrthantTrie::uniform(orthants.clone(), 42, 0);
        assert_eq!(grader1.grade(&Orthant::from([0, 0])), 42);
        assert_eq!(grader1.grade(&Orthant::from([1, 1])), 42);
        assert_eq!(grader1.grade(&Orthant::from([2, 2])), 42);
        assert_eq!(grader1.grade(&Orthant::from([3, 3])), 0);

        // Test uniform_with_threshold
        let grader2 = OrthantTrie::uniform_with_threshold(orthants, 42, 99, 0);
        assert_eq!(grader2.grade(&Orthant::from([0, 0])), 42);
        assert_eq!(grader2.grade(&Orthant::from([1, 1])), 42);
        assert_eq!(grader2.grade(&Orthant::from([2, 2])), 42);
        assert_eq!(grader2.grade(&Orthant::from([3, 3])), 99);
    }

    #[test]
    fn trie_default_grade_getter() {
        let grader = OrthantTrie::from_iter(vec![(Orthant::from([0, 0]), 1)], 77);
        assert_eq!(grader.default_grade(), 77);
    }
}
