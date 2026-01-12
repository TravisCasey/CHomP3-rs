// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Peak-finding utilities for grade boundary detection in orthants.

use std::{collections::BTreeMap, iter::zip};

use crate::{Grader, Orthant};

/// Helper type to find the peaks of an orthant - those cells which are of
/// strictly lesser grade than each of their cofaces. This implies that their
/// dual top cube is of the same lesser grade, but each dual top cube of the
/// cofaces are not.
///
/// # Invariants
///
/// The following invariants enable safe unchecked indexing in hot loops:
///
/// 1. Extent bounds: All extents are derived from the initial extent `(1 <<
///    ambient_dimension) - 1` by clearing bits. Thus all extents are strictly
///    less than `1 << ambient_dimension`, which equals
///    `extent_to_node_index.len()`.
///
/// 2. Node index validity: `cell_nodes` only grows during BFS iteration (never
///    shrinks until `reset()` is called). Any index validated against
///    `cell_nodes.len()` at loop entry remains valid for the loop body.
///
/// 3. Index synchronization: An index is written to `extent_to_node_index[e]`
///    if and only if a node with extent `e` exists at that index in
///    `cell_nodes`. Both are cleared together by calling `reset()`.
pub(super) struct PeakFinder<G>
where
    G: Grader<Orthant>,
{
    /// Wrapper for the grader that contiguously caches queried grades in the
    /// subgrid.
    grade_cache: SubgridGradeCache<G>,

    /// The dual orthant, extent, and cache index of a cell in the orthant, plus
    /// the bitmask of permissible bits to clear from the extent to get child
    /// nodes.
    cell_nodes: Vec<(OrthantWrapper, u32)>,

    /// Maps a cell extent to the corresponding index in `cell_nodes`. Maintains
    /// length of `1 << ambient_dimension` at all times.
    extent_to_node_index: Vec<Option<usize>>,

    /// Ambient spatial dimension the complex is embedded in.
    ambient_dimension: usize,

    /// Absolute minimum grade of the complex, if supplied by the user.
    grader_min_grade: Option<u32>,

    /// `axis_increments[i]` is the increment to a cache index to move along
    /// axis `i` in the current subgrid.
    axis_increments: Vec<usize>,

    /// The minimum orthant of the current subgrid, used to compute cache
    /// indices relative to the subgrid bounds.
    minimum_orthant: Orthant,

    /// Associative container of (grade, vector of peak extent) pairs. Allocated
    /// memory is reused for future calls to `compute_peaks` when possible.
    peaks: BTreeMap<u32, Vec<u32>>,

    /// Minimum grade found in the last `compute_peaks` call.
    last_min_grade: u32,
}

impl<G> PeakFinder<G>
where
    G: Grader<Orthant> + Clone,
{
    /// Create a new `PeakFinder`.
    ///
    /// If `grader_min_grade` is not `None` and a cell of such minimum grade is
    /// found then it is presumed that each of its faces must have that same
    /// grade.
    pub(super) fn new(
        ambient_dimension: u32,
        grader_min_grade: Option<u32>,
        grading_function: G,
    ) -> Self {
        Self {
            cell_nodes: Vec::with_capacity(1 << ambient_dimension),
            extent_to_node_index: vec![None; 1 << ambient_dimension],
            ambient_dimension: ambient_dimension as usize,
            grader_min_grade,
            grade_cache: SubgridGradeCache::new(grading_function, 0),
            axis_increments: vec![1usize; ambient_dimension as usize],
            minimum_orthant: Orthant::zeros(ambient_dimension as usize),
            peaks: BTreeMap::new(),
            last_min_grade: u32::MAX,
        }
    }

    /// Prepare for a new subgrid with the given bounds.
    ///
    /// Computes `axis_increments` and resets the grade cache.
    pub(super) fn prepare(&mut self, minimum_orthant: &Orthant, maximum_orthant: &Orthant) {
        // We only match orthants between `minimum_orthant` and `maximum_orthant`.
        // However, this may require grades of orthants from 1 less than each minimum
        // coordinate. Grades are thus cached for all orthants between
        // `least_graded_orthant` and the maximum.
        let least_graded_orthant: Orthant =
            minimum_orthant.iter().map(|coord| *coord - 1).collect();

        // Reset axis_increments to 1 for all axes
        self.axis_increments.fill(1);

        // To move one orthant along an axis in `grade_cache`, increment current
        // index by `axis_increments[axis]`.
        for (axis, (lg_coord, max_coord)) in
            zip(least_graded_orthant.iter(), maximum_orthant.iter())
                .enumerate()
                .skip(1)
                .rev()
        {
            debug_assert!(
                max_coord > lg_coord,
                "maximum orthant does not exceed minimum orthant"
            );
            self.axis_increments[axis - 1] = self.axis_increments[axis]
                * (*max_coord as isize - *lg_coord as isize + 1).cast_unsigned();
        }

        debug_assert!(
            maximum_orthant[0] > least_graded_orthant[0],
            "maximum orthant does not exceed minimum orthant"
        );
        let total_orthant_count = self.axis_increments[0]
            * (maximum_orthant[0] as isize - least_graded_orthant[0] as isize + 1).cast_unsigned();
        self.grade_cache.reset(total_orthant_count);
        self.minimum_orthant = minimum_orthant.clone();
    }

    /// Compute the cache index for an orthant.
    pub(super) fn cache_index(&self, orthant: &Orthant, minimum_orthant: &Orthant) -> usize {
        let mut cache_index = 0;
        for (axis, inc) in self.axis_increments.iter().enumerate() {
            cache_index += *inc
                * (orthant[axis] as isize + 1 - minimum_orthant[axis] as isize).cast_unsigned();
        }
        cache_index
    }

    /// Returns a reference to the axis increments.
    pub(super) fn axis_increments(&self) -> &[usize] {
        &self.axis_increments
    }

    fn reset(&mut self) {
        self.cell_nodes.clear();
        self.extent_to_node_index.fill(None);
        self.last_min_grade = u32::MAX;

        // Clear peaks map but retain allocated memory
        for extents in self.peaks.values_mut() {
            extents.clear();
        }
    }

    /// Excludes all faces of the last cell in iteration from further iteration.
    /// This is achieved using a bitmask (the `u32` argument in the tuple of the
    /// `cell_nodes` vector).
    fn exclude_last(&mut self) {
        let last_index = self.cell_nodes.len() - 1;

        // SAFETY: last_index < cell_nodes.len() by definition (invariant #2)
        let extent = unsafe { self.cell_nodes.get_unchecked(last_index).0.extent };
        unsafe { self.cell_nodes.get_unchecked_mut(last_index).1 = 0 };

        // Inform all cofaces that flipping the connecting bit to reach the last
        // cell is no longer allowed; this information is propagated to their
        // other faces who update their bitmasks accordingly.
        let mut axis_flag = 1;
        for _ in 0..self.ambient_dimension {
            // SAFETY: extent derived by clearing bits from max_extent (invariant #1)
            if extent & axis_flag == 0
                && let Some(parent_index) = unsafe {
                    *self
                        .extent_to_node_index
                        .get_unchecked((extent ^ axis_flag) as usize)
                }
            {
                debug_assert_ne!(self.cell_nodes[parent_index].1 & axis_flag, 0);

                // SAFETY: parent_index from extent_to_node_index (invariant #3)
                unsafe { self.cell_nodes.get_unchecked_mut(parent_index).1 ^= axis_flag };
            }
            axis_flag <<= 1;
        }
    }

    /// Find the prime peak for a suborthant.
    ///
    /// Searches grades in descending order (below `upper_grade`) for the
    /// highest grade with a peak that is a coface of `lower_extent`.
    /// Returns the intersection of that peak with `upper_extent` as the
    /// `prime_extent`, along with its grade.
    ///
    /// Returns `None` if `upper_grade` equals `last_min_grade` (short circuit
    /// condition) or if no valid peak is found.
    pub(super) fn prime_peak(
        &self,
        upper_extent: u32,
        lower_extent: u32,
        upper_grade: u32,
    ) -> Option<(u32, u32)> {
        // Short-circuit: no peaks below this orthant's minimum grade
        if upper_grade == self.last_min_grade {
            return None;
        }

        for (grade, extents) in self.peaks.range(..upper_grade).rev() {
            let best_in_grade = extents
                .iter()
                .filter(|peak| *peak & lower_extent == lower_extent)
                .map(|peak| peak & upper_extent)
                .max_by_key(|intersection| intersection.count_ones());

            if let Some(prime_extent) = best_in_grade {
                return Some((prime_extent, *grade));
            }
        }
        None
    }

    /// Compute peaks for an orthant.
    ///
    /// Returns the base grade. After this call, use `peaks()` to access the
    /// peaks map which contains grade associated to vectors of extents of the
    /// peaks. The map always contains at least the entry with the uppermost
    /// cell's grade and extent.
    ///
    /// # Algorithm
    ///
    /// Performs breadth-first traversal of the face lattice starting from the
    /// topmost cell. Each cell is represented as a `(OrthantWrapper, u32)` pair
    /// where the `u32` is a bitmask of axes that can still be flipped in the
    /// cell's extent to produce faces.
    ///
    /// When a cell with a different grade is found (a "peak"), it is recorded
    /// and `exclude_last()` is called to prevent visiting any faces of that
    /// cell. The exclusion propagates through the bitmasks of sibling
    /// cells.
    ///
    /// The `extent_to_node_index` map provides O(1) lookup for whether a cell
    /// has been visited, enabling bitmask updates when cells are reached
    /// via multiple parent paths.
    pub(super) fn compute_peaks(&mut self, base_orthant: &Orthant) -> u32 {
        self.reset();

        let orthant = OrthantWrapper {
            orthant: base_orthant.clone(),
            cache_index: self.cache_index(base_orthant, &self.minimum_orthant),
            extent: (1 << self.ambient_dimension) - 1,
        };

        let base_grade = self.grade_cache.grade(&orthant);
        let base_extent = orthant.extent;

        // Always include the top cell under its grade
        self.peaks.entry(base_grade).or_default().push(base_extent);
        self.last_min_grade = base_grade;

        // --- Early return for absolute minimum grade ---
        if self.grader_min_grade == Some(base_grade) {
            return base_grade;
        }

        // --- Initialize BFS state ---
        // SAFETY: base_extent = (1 << ambient_dimension) - 1 (invariant #1)
        unsafe {
            *self
                .extent_to_node_index
                .get_unchecked_mut(base_extent as usize) = Some(0);
        }
        self.cell_nodes.push((orthant, base_extent));

        // --- BFS iteration ---
        let mut node_index = 0usize;
        let mut axis;
        while node_index < self.cell_nodes.len() {
            axis = 0;
            while axis < self.ambient_dimension {
                let axis_flag = 1 << axis;

                // SAFETY: node_index < cell_nodes.len() checked above (invariant #2)
                if unsafe { self.cell_nodes.get_unchecked(node_index).1 } & axis_flag != 0 {
                    debug_assert_ne!(self.cell_nodes[node_index].0.extent & axis_flag, 0);
                    let child_extent =
                        unsafe { self.cell_nodes.get_unchecked(node_index).0.extent } ^ axis_flag;

                    // SAFETY: child_extent derived by clearing bits (invariant #1)
                    if let Some(prev_node_index) = unsafe {
                        *self
                            .extent_to_node_index
                            .get_unchecked(child_extent as usize)
                    } {
                        // Child already visited; merge bitmask information
                        // SAFETY: prev_node_index from extent_to_node_index (invariant #3)
                        unsafe {
                            self.cell_nodes.get_unchecked_mut(prev_node_index).1 &=
                                self.cell_nodes.get_unchecked(node_index).1;
                        }
                    } else {
                        // New child cell
                        let last_index = self.cell_nodes.len();

                        // SAFETY: node_index < cell_nodes.len() (invariant #2)
                        let (mut child_orthant, mut child_mask) =
                            unsafe { self.cell_nodes.get_unchecked(node_index).clone() };

                        child_orthant.orthant[axis] -= 1;
                        child_orthant.cache_index -= self.axis_increments[axis];
                        child_orthant.extent ^= axis_flag;
                        child_mask ^= axis_flag;

                        let child_grade = self.grade_cache.grade(&child_orthant);

                        // SAFETY: child_extent derived by clearing bits (invariant #1)
                        unsafe {
                            *self
                                .extent_to_node_index
                                .get_unchecked_mut(child_extent as usize) = Some(last_index);
                        }
                        self.cell_nodes.push((child_orthant, child_mask));

                        if child_grade != base_grade {
                            self.last_min_grade = self.last_min_grade.min(child_grade);
                            self.peaks
                                .entry(child_grade)
                                .or_default()
                                .push(child_extent);
                            self.exclude_last();

                            // Restart axis loop to update siblings
                            axis = 0;
                            continue;
                        }
                    }
                }
                axis += 1;
            }
            node_index += 1;
        }
        base_grade
    }
}

/// When matching the subgrid, we represent orthants in the subgrid in three
/// ways for three different purposes, each corresponding to one of the fields
/// of this type:
/// 1. `orthant`: The `Orthant` object itself,
/// 2. `cache_index`: The index of the orthant in the associated
///    `SubgridGradeCache` object. This can be efficiently determined using the
///    `Subgrid::axis_increments` field when incrementing/decrementing orthant
///    axes,
/// 3. `extent`: The extent (see perhaps, [`Cube::from_extent`]), but
///    represented as bits set in an integer (`[false, true, true]` is
///    equivalent to `0b110`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OrthantWrapper {
    pub(super) orthant: Orthant,
    pub(super) cache_index: usize,
    pub(super) extent: u32,
}

/// A basic contiguous cache that stores the results of queries to the given
/// `grading_function` in an array of configurable size.
///
/// The size of the cache is the number of orthants in the subgrid (including
/// the extra orthants that are queried but not matched, see
/// `PeakFinder::prepare`).
#[derive(Clone, Debug)]
struct SubgridGradeCache<G>
where
    G: Grader<Orthant>,
{
    grading_function: G,
    cache: Vec<Option<u32>>,
}

impl<G> SubgridGradeCache<G>
where
    G: Grader<Orthant>,
{
    fn new(grading_function: G, cache_size: usize) -> Self {
        Self {
            grading_function,
            cache: vec![None; cache_size],
        }
    }

    fn reset(&mut self, cache_size: usize) {
        self.cache.resize(cache_size, None);
        self.cache.fill(None);
    }

    fn grade(&mut self, orthant: &OrthantWrapper) -> u32 {
        *self.cache[orthant.cache_index]
            .get_or_insert_with(|| self.grading_function.grade(&orthant.orthant))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HashMapGrader, TopCubeGrader};

    fn prepare_two_cube_peak_finder() -> (PeakFinder<HashMapGrader<Orthant>>, Orthant, Orthant) {
        let grader = TopCubeGrader::new(
            HashMapGrader::uniform(
                [Orthant::from([0, 0, 1, 1]), Orthant::from([1, 1, 0, 0])],
                0,
                1,
            ),
            Some(0),
        );

        let minimum_orthant = Orthant::from([1, 1, 1, 1]);
        let maximum_orthant = Orthant::from([1, 1, 1, 1]);

        let mut peak_finder = PeakFinder::new(4, Some(0), grader.orthant_grader().clone());
        peak_finder.prepare(&minimum_orthant, &maximum_orthant);

        (peak_finder, minimum_orthant, maximum_orthant)
    }

    #[test]
    fn two_cube_subgrid() {
        let (mut peak_finder, minimum_orthant, _maximum_orthant) = prepare_two_cube_peak_finder();

        let base_grade = peak_finder.compute_peaks(&minimum_orthant);

        // Base grade is 1 (default grade, not in the graded orthants)
        assert_eq!(base_grade, 1);

        // Should have two grade levels: 1 (base) and 0 (the two peaks)
        assert_eq!(peak_finder.peaks.len(), 2);

        // Base grade entry contains the top cell
        assert_eq!(peak_finder.peaks.get(&1), Some(&vec![0b1111]));

        // Grade 0 contains the two peaks
        let grade_0_peaks = peak_finder.peaks.get(&0).unwrap();
        assert_eq!(grade_0_peaks.len(), 2);
        assert!(grade_0_peaks.contains(&0b1100));
        assert!(grade_0_peaks.contains(&0b0011));
    }

    #[test]
    fn prime_peak() {
        let peaks = BTreeMap::from_iter([
            (4, vec![0b111111]),
            (3, vec![]),
            (2, vec![0b011011, 0b110010, 0b110100]),
            (1, vec![0b000110, 0b100001]),
            (0, vec![0b000000]),
        ]);

        let mut peak_finder = PeakFinder::new(6, None, HashMapGrader::new(0));
        peak_finder.peaks = peaks;
        peak_finder.last_min_grade = 0; // No short circuit except for minimum grade

        // Calling prime_peak over entire orthant will yield the second-highest
        // grade that has an extent and the greatest intersection:
        assert_eq!(
            peak_finder.prime_peak(0b111111, 0b000000, 4),
            Some((0b011011, 2))
        );

        // But, doing a different slice can yield a larger intersection on a
        // later extent with the same grade
        assert_eq!(
            peak_finder.prime_peak(0b110101, 0b000000, 4),
            Some((0b110100, 2))
        );

        // Alternatively, exclude other peaks as they are not a coface of the
        // lower extent
        assert_eq!(
            peak_finder.prime_peak(0b111111, 0b100010, 4),
            Some((0b110010, 2))
        );

        // Later grade (it is peak 0b100001, but the returned extent is the
        // intersection of that peak with the upper extent 0b100010)
        assert_eq!(
            peak_finder.prime_peak(0b100010, 0b000001, 2),
            Some((0b100000, 1))
        );

        // Short circuits here - nothing of lesser grade
        assert_eq!(peak_finder.prime_peak(0b000000, 0b000000, 0), None);
    }
}
