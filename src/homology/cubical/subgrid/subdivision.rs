// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Grid subdivision utilities for partitioning orthant grids.

use std::{collections::HashSet, vec::IntoIter};

use itertools::Itertools;
use tracing::info;

use crate::{Orthant, logging::ProgressTracker};

/// A utility type for subdividing a rectangular grid of `Orthant` objects into
/// smaller aligned subgrids.
///
/// Given a rectangular grid defined by minimum and maximum orthants
/// (inclusive), this type partitions it into smaller rectangular subgrids of a
/// specified size. Subgrids are aligned to the minimum orthant, and those at
/// the boundary may be truncated to fit within the original grid.
///
/// # Nonempty Subgrids
///
/// This type supports tracking a list of "active" orthants provided at
/// construction time. A subgrid is considered "nonempty" if at least one of
/// these orthants either:
/// 1. Is contained within the subgrid bounds, or
/// 2. Is at most one coordinate less than some orthant in the subgrid along
///    each axis.
///
/// The second condition means that for a subgrid with bounds `[min, max]` and
/// an active orthant `L`, the subgrid is nonempty if for all coordinates `i`:
/// `min[i] - 1 <= L[i] <= max[i]`.
///
/// Nonempty subgrids are computed at construction time and stored for efficient
/// iteration.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GridSubdivision {
    minimum_orthant: Orthant,
    maximum_orthant: Orthant,
    subgrid_shape: Vec<i16>,
    nonempty_subgrids: Vec<(Orthant, Orthant)>,
}

impl GridSubdivision {
    /// Creates a new grid subdivision of the rectangular grid between the given
    /// minimum and maximum orthants (inclusive).
    ///
    /// # Panics
    ///
    /// If the dimensions don't match or if any subgrid size is non-positive.
    #[must_use]
    pub fn new(
        minimum_orthant: Orthant,
        maximum_orthant: Orthant,
        subgrid_shape: Option<Vec<i16>>,
        orthants: Option<&[Orthant]>,
    ) -> Self {
        // Default subgrid shape is to process one orthant at a time.
        let subgrid_shape =
            subgrid_shape.unwrap_or_else(|| vec![1; minimum_orthant.ambient_dimension() as usize]);

        // Input validation
        assert_eq!(
            minimum_orthant.ambient_dimension(),
            maximum_orthant.ambient_dimension(),
            "minimum and maximum orthants must have the same dimension"
        );
        assert_eq!(
            minimum_orthant.ambient_dimension() as usize,
            subgrid_shape.len(),
            "subgrid shape must have exactly as many entries as the ambient dimension"
        );
        assert!(
            subgrid_shape.iter().all(|&s| s > 0),
            "subgrid shape must only contain positive entries"
        );
        for (i, (&min_coord, &max_coord)) in minimum_orthant
            .iter()
            .zip(maximum_orthant.iter())
            .enumerate()
        {
            assert!(
                min_coord <= max_coord,
                "minimum orthant coordinate {min_coord} exceeds maximum at axis {i}"
            );
        }

        let mut subdivision = Self {
            minimum_orthant,
            maximum_orthant,
            subgrid_shape,
            nonempty_subgrids: Vec::new(),
        };

        match orthants {
            None => subdivision.populate_all_subgrids(),
            Some(orthants) => subdivision.populate_nonempty_subgrids(orthants),
        }

        subdivision
    }

    /// Populates `nonempty_subgrids` with all possible subgrids in the grid.
    fn populate_all_subgrids(&mut self) {
        self.nonempty_subgrids = SubgridIterator::new(self).collect();
    }

    /// Populates `nonempty_subgrids` by filtering to only those containing
    /// at least one orthant from the given list (with boundary extension).
    fn populate_nonempty_subgrids(&mut self, orthants: &[Orthant]) {
        let nonempty_set = self.compute_nonempty_subgrid_indices(orthants);
        self.nonempty_subgrids = self.indices_to_bounds(nonempty_set);
        self.log_filtering_statistics();
    }

    /// Computes the set of subgrid indices that contain active orthants.
    fn compute_nonempty_subgrid_indices(&self, orthants: &[Orthant]) -> HashSet<Vec<i16>> {
        let dim = self.minimum_orthant.ambient_dimension() as usize;

        // Compute the maximum valid subgrid index along each axis
        let max_subgrid_index: Vec<i16> = (0..dim)
            .map(|axis| {
                let range = self.maximum_orthant[axis] - self.minimum_orthant[axis];
                range / self.subgrid_shape[axis]
            })
            .collect();

        let mut subgrid_index = Vec::with_capacity(dim);
        let mut next_subgrid = vec![false; dim];
        let mut nonempty_subgrid_set: HashSet<Vec<i16>> = HashSet::new();
        let mut progress =
            ProgressTracker::new("Subdividing grid", orthants.len()).with_interval(10);

        for orthant in orthants {
            subgrid_index.clear();
            next_subgrid.fill(false);

            // Compute the subgrid index for this orthant along each axis
            for axis in 0..dim {
                if orthant[axis] < self.minimum_orthant[axis]
                    || orthant[axis] > self.maximum_orthant[axis]
                {
                    break;
                }
                let relative_coord = orthant[axis] - self.minimum_orthant[axis];
                let idx = relative_coord / self.subgrid_shape[axis];
                subgrid_index.push(idx);

                // Include next subgrid if orthant is at the end of current subgrid
                // AND there is a valid next subgrid
                if (relative_coord + 1) % self.subgrid_shape[axis] == 0
                    && idx < max_subgrid_index[axis]
                {
                    next_subgrid[axis] = true;
                }
            }
            progress.increment();

            if subgrid_index.len() < dim {
                continue;
            }

            nonempty_subgrid_set.extend(
                subgrid_index
                    .iter()
                    .zip(next_subgrid.iter())
                    .map(|(index, next)| *index..=(*index + i16::from(*next)))
                    .multi_cartesian_product(),
            );
        }
        progress.finish();

        nonempty_subgrid_set
    }

    /// Converts subgrid indices to (min, max) orthant bounds.
    fn indices_to_bounds(&self, indices: HashSet<Vec<i16>>) -> Vec<(Orthant, Orthant)> {
        let dim = self.minimum_orthant.ambient_dimension() as usize;

        // Sort indices for deterministic iteration order.
        // Note: This is a diagnostic measure - the algorithm should produce
        // correct results regardless of iteration order.
        let mut sorted_indices: Vec<_> = indices.into_iter().collect();
        sorted_indices.sort();

        sorted_indices
            .into_iter()
            .map(|indices| {
                let subgrid_min: Orthant = (0..dim)
                    .map(|axis| {
                        self.minimum_orthant[axis] + indices[axis] * self.subgrid_shape[axis]
                    })
                    .collect();
                let subgrid_max: Orthant = (0..dim)
                    .map(|axis| {
                        (subgrid_min[axis] + self.subgrid_shape[axis] - 1)
                            .min(self.maximum_orthant[axis])
                    })
                    .collect();
                (subgrid_min, subgrid_max)
            })
            .collect()
    }

    /// Returns the number of nonempty subgrids.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nonempty_subgrids.len()
    }

    /// Returns whether there are no nonempty subgrids.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nonempty_subgrids.is_empty()
    }

    /// Logs statistics about how many subgrids were filtered.
    fn log_filtering_statistics(&self) {
        let dim = self.minimum_orthant.ambient_dimension() as usize;

        // Validated in new() that subgrid_shape[axis] > 0 and
        // maximum_orthant[axis] >= minimum_orthant[axis] for all axes
        let total_subgrids: usize = (0..dim)
            .map(|axis| {
                let range = self.maximum_orthant[axis] - self.minimum_orthant[axis] + 1;
                ((range + self.subgrid_shape[axis] - 1) / self.subgrid_shape[axis]) as usize
            })
            .product();

        let kept = self.nonempty_subgrids.len();
        let filtered_percent = 100.0 * (1.0 - kept as f64 / total_subgrids as f64);
        info!(
            "{}/{} subgrids retained ({:.1}% filtered out)",
            kept, total_subgrids, filtered_percent
        );
    }
}

impl IntoIterator for GridSubdivision {
    type IntoIter = IntoIter<(Orthant, Orthant)>;
    type Item = (Orthant, Orthant);

    fn into_iter(self) -> Self::IntoIter {
        self.nonempty_subgrids.into_iter()
    }
}

/// Iterator over all subgrids of configurable size in a larger rectangular
/// region of cubical space.
///
/// See `GridSubdivision` to filter out empty subgrids instead of keeping all
/// of them.
pub(super) struct SubgridIterator {
    minimum_orthant: Orthant,
    maximum_orthant: Orthant,
    subgrid_shape: Vec<i16>,
    current_indices: Vec<i16>,
    num_subgrids_per_axis: Vec<i16>,
    finished: bool,
}

impl SubgridIterator {
    pub(super) fn new(subdivision: &GridSubdivision) -> Self {
        let dim = subdivision.minimum_orthant.ambient_dimension() as usize;
        let num_subgrids_per_axis: Vec<i16> = (0..dim)
            .map(|axis| {
                let range =
                    subdivision.maximum_orthant[axis] - subdivision.minimum_orthant[axis] + 1;
                (range + subdivision.subgrid_shape[axis] - 1) / subdivision.subgrid_shape[axis]
            })
            .collect();

        Self {
            minimum_orthant: subdivision.minimum_orthant.clone(),
            maximum_orthant: subdivision.maximum_orthant.clone(),
            subgrid_shape: subdivision.subgrid_shape.clone(),
            current_indices: vec![0; dim],
            num_subgrids_per_axis,
            finished: false,
        }
    }
}

impl Iterator for SubgridIterator {
    type Item = (Orthant, Orthant);

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let dim = self.current_indices.len();

        let subgrid_min_orthant: Orthant = (0..dim)
            .map(|axis| {
                self.minimum_orthant[axis] + self.current_indices[axis] * self.subgrid_shape[axis]
            })
            .collect();
        let subgrid_max_orthant: Orthant = (0..dim)
            .map(|axis| {
                (subgrid_min_orthant[axis] + self.subgrid_shape[axis] - 1)
                    .min(self.maximum_orthant[axis])
            })
            .collect();

        // Advance to next subgrid
        let mut axis = 0;
        while axis < dim {
            self.current_indices[axis] += 1;
            if self.current_indices[axis] < self.num_subgrids_per_axis[axis] {
                break;
            }
            self.current_indices[axis] = 0;
            axis += 1;
        }

        if axis == dim {
            self.finished = true;
        }

        Some((subgrid_min_orthant, subgrid_max_orthant))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let subdivision = GridSubdivision::new(
            Orthant::from([0, 0]),
            Orthant::from([9, 9]),
            Some(vec![5, 5]),
            None,
        );
        assert_eq!(subdivision.len(), 4);

        let subgrids: Vec<_> = subdivision.into_iter().collect();

        // Check first subgrid
        assert_eq!(subgrids[0].0, Orthant::from([0, 0]));
        assert_eq!(subgrids[0].1, Orthant::from([4, 4]));

        // Check last subgrid
        assert_eq!(subgrids[3].0, Orthant::from([5, 5]));
        assert_eq!(subgrids[3].1, Orthant::from([9, 9]));
    }

    #[test]
    fn nonempty_subgrids_orthant_inside() {
        // Test case 1: Orthant is directly inside a subgrid
        let subdivision = GridSubdivision::new(
            Orthant::from([0, 0, 0]),
            Orthant::from([9, 9, 9]),
            Some(vec![5, 5, 5]),
            Some(&[Orthant::from([2, 3, 4])]),
        );

        let nonempty: Vec<_> = subdivision.into_iter().collect();
        // The orthant [2, 3, 4] is inside [0, 0, 0] to [4, 4, 4]
        // It's also within range of [0, 0, 5], [4, 4, 9]
        assert_eq!(nonempty.len(), 2);
        assert!(nonempty.contains(&(Orthant::from([0, 0, 0]), Orthant::from([4, 4, 4]))));
        assert!(nonempty.contains(&(Orthant::from([0, 0, 5]), Orthant::from([4, 4, 9]))));
    }

    #[test]
    fn nonempty_subgrids_multiple_orthants() {
        // Multiple orthants affecting different subgrids
        let subdivision = GridSubdivision::new(
            Orthant::from([0, 0]),
            Orthant::from([9, 9]),
            Some(vec![5, 5]),
            Some(&[
                Orthant::from([2, 2]), // Inside [0,0] to [4,4]
                Orthant::from([7, 7]), // Inside [5,5] to [9,9]
            ]),
        );

        let nonempty: Vec<_> = subdivision.into_iter().collect();
        assert_eq!(nonempty.len(), 2);
        assert!(nonempty.contains(&(Orthant::from([0, 0]), Orthant::from([4, 4]))));
        assert!(nonempty.contains(&(Orthant::from([5, 5]), Orthant::from([9, 9]))));
    }

    #[test]
    fn nonempty_subgrids_large_orthant_list() {
        // Create a 20x20 grid with 5x5 subgrids (16 total subgrids)
        let mut orthants = Vec::new();

        // Add 100 orthants scattered throughout the grid
        for i in 0..10 {
            for j in 0..10 {
                orthants.push(Orthant::from([i as i16 * 2, j as i16 * 2]));
            }
        }

        let subdivision = GridSubdivision::new(
            Orthant::from([0, 0]),
            Orthant::from([19, 19]),
            Some(vec![5, 5]),
            Some(&orthants),
        );

        let nonempty: Vec<_> = subdivision.into_iter().collect();
        // All 16 subgrids should be nonempty given the orthant distribution
        assert_eq!(nonempty.len(), 16);
    }

    #[test]
    fn nonempty_subgrids_single_orthant_large_grid() {
        // Single top-cube in a large grid - should filter most subgrids.
        // The top-cube at (55,55,55) has faces extending to (56,56,56),
        // so subgrids containing any of these cells should be retained.
        let subdivision = GridSubdivision::new(
            Orthant::from([0, 0, 0]),
            Orthant::from([99, 99, 99]),
            Some(vec![10, 10, 10]),
            Some(&[Orthant::from([55, 55, 55])]),
        );

        let nonempty: Vec<_> = subdivision.into_iter().collect();
        // Top-cube at [55,55,55] is in subgrid [50,59]^3.
        // Its faces extend to [56,56,56], still within the same subgrid.
        // Due to boundary logic (next_subgrid), adjacent subgrids may also be included.
        assert!(nonempty.len() <= 8); // At most 2^3 adjacent subgrids
        assert!(!nonempty.is_empty()); // At least the containing subgrid
    }

    #[test]
    fn nonempty_subgrids_empty_orthant_list() {
        // Empty top-cube list - no subgrids should be nonempty
        let subdivision = GridSubdivision::new(
            Orthant::from([0, 0]),
            Orthant::from([9, 9]),
            Some(vec![5, 5]),
            Some(&[]),
        );

        assert!(subdivision.is_empty());
    }

    #[test]
    fn filtering_includes_neighbor_orthants() {
        // With subgrid_shape [1,1], each orthant is its own subgrid.
        // A top-cube at orthant O should include subgrids for O and O+1 along each
        // axis.
        let subdivision = GridSubdivision::new(
            Orthant::from([0, 0]),
            Orthant::from([5, 5]),
            None, // Default shape [1, 1]
            Some(&[Orthant::from([2, 2])]),
        );

        let nonempty: Vec<_> = subdivision.into_iter().collect();

        // With a single orthant at [2,2] and subgrid_shape [1,1]:
        // We should get 4 subgrids: [2,2], [2,3], [3,2], [3,3]
        assert_eq!(nonempty.len(), 4);
        assert!(nonempty.contains(&(Orthant::from([2, 2]), Orthant::from([2, 2]))));
        assert!(nonempty.contains(&(Orthant::from([2, 3]), Orthant::from([2, 3]))));
        assert!(nonempty.contains(&(Orthant::from([3, 2]), Orthant::from([3, 2]))));
        assert!(nonempty.contains(&(Orthant::from([3, 3]), Orthant::from([3, 3]))));
    }

    #[test]
    fn filtering_boundary_orthant() {
        // Test that an orthant at the grid boundary correctly handles
        // out-of-bounds extensions.
        let subdivision = GridSubdivision::new(
            Orthant::from([0, 0]),
            Orthant::from([5, 5]),
            None,                           // Default shape [1, 1]
            Some(&[Orthant::from([5, 5])]), // At the boundary
        );

        let nonempty: Vec<_> = subdivision.into_iter().collect();

        // At boundary [5,5], next_subgrid would try to include index 6
        // but that exceeds the grid bounds.
        for (min, max) in &nonempty {
            // Verify all subgrids are within bounds
            for axis in 0..2 {
                assert!(
                    min[axis] >= 0 && min[axis] <= 5,
                    "min out of bounds at axis {axis}: {min}"
                );
                assert!(
                    max[axis] >= 0 && max[axis] <= 5,
                    "max out of bounds at axis {axis}: {max}"
                );
            }
        }
    }

    #[test]
    fn filtering_includes_diagonal_neighbor() {
        // Test that filtering includes orthants that differ by +1 on multiple axes.
        // This is critical for finding all faces of a top-cube.
        let subdivision = GridSubdivision::new(
            Orthant::from([0, 0, 0, 0, 0, 0]),
            Orthant::from([10, 20, 10, 10, 10, 10]),
            None, // Default shape [1, 1, 1, 1, 1, 1]
            Some(&[Orthant::from([3, 13, 4, 1, 2, 2])]),
        );

        let nonempty: Vec<_> = subdivision.into_iter().collect();

        // Should include all 2^6 = 64 neighboring orthants
        assert_eq!(nonempty.len(), 64);

        // Specifically, should include (3, 13, 5, 2, 3, 3) which is +1 on axes 2,3,4,5
        let target = Orthant::from([3, 13, 5, 2, 3, 3]);
        assert!(
            nonempty.iter().any(|(min, _)| *min == target),
            "Target orthant (3, 13, 5, 2, 3, 3) not found in filtered subgrids"
        );
    }
}
