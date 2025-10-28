// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::iter::zip;

use crate::homology::cubical::OrthantMatching;
use crate::{Cube, CubicalComplex, Grader, Orthant, OrthantIterator, TopCubeGrader};

/// Helper type to find the "peaks" of an Orthant - those cells which are of
/// lesser grade than all of their cofaces. This implies that their dual orthant
/// is of the same lesser grade, but each dual orthant of the cofaces of the
/// cell are not.
///
/// The basic idea is to iterate down the orthant in a BFS manner until a cell
/// with a dual orthant of lesser grade is found. All cells below that cell are
/// then excluded, and iteration continues. These peaks are recorded and
/// returned to match the subgrid.
///
/// Currently, the implementation only works for two grades - an upper and a
/// lower. It is also highly recommended that the `minimum_grade` option in
/// the `PeakFinder::new` method is set to the lower grade, else this will
/// needlessly iterate over the entire orthant. The algorithm will eventually
/// be expanded to cover more grades.
struct PeakFinder {
    cell_nodes: Vec<(OrthantWrapper, u32)>,
    extent_to_node_index: Vec<Option<usize>>,
    ambient_dimension: usize,
    minimum_grade: Option<u32>,
}

impl PeakFinder {
    /// If `minimum_grade` is not `None` and a cell of such minimum grade is
    /// found, it is presumed that each of its faces must have the same grade.
    fn new(ambient_dimension: u32, minimum_grade: Option<u32>) -> Self {
        Self {
            cell_nodes: Vec::with_capacity(1 << (ambient_dimension as usize)),
            extent_to_node_index: vec![None; 1 << (ambient_dimension as usize)],
            ambient_dimension: ambient_dimension as usize,
            minimum_grade,
        }
    }

    fn reset(&mut self) {
        // This has a noticeable improvement over clear in profiling, due to it
        // not dropping all entries.
        unsafe { self.cell_nodes.set_len(0) };
        self.extent_to_node_index.fill(None);
    }

    /// Excludes all faces of the last cell in iteration from further iteration.
    /// This is achieved using a bitmask (the u32 argument in the tuple of the
    /// `cell_nodes` vector).
    fn exclude_last(&mut self) {
        let last_index = self.cell_nodes.len() - 1;
        let extent = self.cell_nodes[last_index].0.extent;
        self.cell_nodes[last_index].1 = 0;

        // Inform all parents that flipping the bit to reach the latest cell is
        // no longer allowed; this information is propagated to other children
        // and updates their bitmasks accordingly.
        let mut axis_flag = 1;
        for _ in 0..self.ambient_dimension {
            if extent & axis_flag == 0
                && let Some(parent_index) = self.extent_to_node_index[(extent ^ axis_flag) as usize]
            {
                debug_assert_ne!(self.cell_nodes[parent_index].1 & axis_flag, 0);
                self.cell_nodes[parent_index].1 ^= axis_flag;
            }
            axis_flag <<= 1;
        }
    }

    /// Find the peaks of `base_orthant` and return their (extent, grade).
    fn compute_peaks<G: Grader<Orthant> + Clone>(
        &mut self,
        base_orthant: OrthantWrapper,
        axis_increments: &[usize],
        grade_cache: &mut SubgridGradeCache<G>,
    ) -> Vec<(u32, u32)> {
        let base_grade = grade_cache.grade(&base_orthant);
        if self.minimum_grade == Some(base_grade) {
            return vec![(base_orthant.extent, base_grade)];
        }

        // Initialize fields for iteration
        self.reset();
        let mut peaks = Vec::new();
        let base_extent = base_orthant.extent;
        self.extent_to_node_index[base_extent as usize] = Some(0);
        self.cell_nodes.push((base_orthant, base_extent));

        // Iterate in breadth-first manner; cells may be iterated to multiple
        // times by each of their parents. This allows for updating their
        // bitmasks with all information (see `exclude_last` method).
        let mut node_index = 0usize;
        let mut axis;
        while node_index < self.cell_nodes.len() {
            axis = 0;
            while axis < self.ambient_dimension {
                let axis_flag = 1 << axis;

                // Bit can be flipped to produce a child
                if self.cell_nodes[node_index].1 & axis_flag != 0 {
                    debug_assert_ne!(self.cell_nodes[node_index].0.extent & axis_flag, 0);
                    let child_extent = self.cell_nodes[node_index].0.extent ^ axis_flag;

                    // Child has already been iterated to; pass any information from the bitmask
                    // and continue.
                    if let Some(prev_node_index) = self.extent_to_node_index[child_extent as usize]
                    {
                        self.cell_nodes[prev_node_index].1 &= self.cell_nodes[node_index].1;

                    // Otherwise, produce the new cell.
                    } else {
                        let last_index = self.cell_nodes.len();
                        self.extent_to_node_index[child_extent as usize] = Some(last_index);
                        self.cell_nodes.push(self.cell_nodes[node_index].clone());

                        let (child_orthant, child_mask) = &mut self.cell_nodes[last_index];

                        child_orthant.orthant[axis] -= 1;
                        child_orthant.cache_index -= axis_increments[axis];
                        child_orthant.extent ^= axis_flag;
                        *child_mask ^= axis_flag;

                        let child_grade = grade_cache.grade(child_orthant);

                        if child_grade != base_grade {
                            peaks.push((child_extent, child_grade));
                            self.exclude_last();

                            // Update siblings on following iterations
                            axis = 0;
                            continue;
                        }
                    }
                }
                axis += 1;
            }
            node_index += 1;
        }
        peaks
    }
}

/// When matching the subgrid, we represent orthants in the subgrid in three
/// ways for three different purposes, each corresponding to one of the fields
/// of this type:
/// 1. `orthant`: The `Orthant` object itself,
/// 2. `cache_index`: The index of the orthant in the associated
///    `SubgridGradeCache` object. This can be efficiently dertermined using the
///    `Subgrid::axis_increments` field when incrementing/decrementing orthant
///    axes,
/// 3. `extent`: The extent (see perhaps, [`Orthant::from_extent`]), but
///    represented as bits set in an integer (`[false, true, true]` is
///    equivalent to `0b110`).
#[derive(Clone, Debug, Eq, PartialEq)]
struct OrthantWrapper {
    orthant: Orthant,
    cache_index: usize,
    extent: u32,
}

/// A suborthant is an interval (in the face order) of cells within an orthant
/// (the field `base_orthant`).
///
/// The greatest cell has extent `upper_extent`, while the least cell in the
/// interval has dual orthant `lower_extent`. The grade of the greatest cell and
/// the dimension of the least cell are also stored so they do not need to be
/// recomputed when matching a suborthant (via `Subgrid::match_suborthant`).
#[derive(Clone, Debug, Eq, PartialEq)]
struct Suborthant {
    base_orthant: Orthant,
    upper_extent: u32,
    lower_extent: u32,
    upper_grade: u32,
    lower_dimension: u32,
}

impl Suborthant {
    fn compute_dual_orthant(&self, extent: u32) -> Orthant {
        Orthant::from_iter(self.base_orthant.iter().enumerate().map(|(axis, coord)| {
            if extent & (1 << axis) == 0 {
                *coord - 1
            } else {
                *coord
            }
        }))
    }

    fn lower_dual_orthant(&self) -> Orthant {
        self.compute_dual_orthant(self.lower_extent)
    }
}

/// A basic contiguous cache that stores the results of queries to the given
/// `grading_function` in an array of size `cache_size`.
///
/// The size of the cache is the number of orthants in the subgrid (including
/// the extra orthants that are queried but not matched, see `Subgrid::new`).
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

/// A rectangular grid of orthants between (all coordinates between, not
/// lexicographically) `minimum_orthant` and `maximum_orthant`.
///
/// This type matches all its orthants for use in `TopCubicalMatching`. By
/// matching nearby orthants together, we can improve efficiency by caching the
/// commonly re-queried grades of nearby top-dimensional cubes.
pub struct Subgrid<G>
where
    G: Grader<Orthant>,
{
    minimum_orthant: Orthant,
    maximum_orthant: Orthant,
    axis_increments: Vec<usize>,
    grade_cache: SubgridGradeCache<G>,
    minimum_grade: Option<u32>,
    maximum_kept_grade: u32,
    maximum_kept_dimension: u32,
    peak_finder: PeakFinder,
}

impl<G> Subgrid<G>
where
    G: Grader<Orthant> + Clone,
{
    /// Initialize the subgrid, but do not perform matching yet. That is
    /// done using the `match_subgrid` method.
    ///
    /// Any critical cells found with grade or dimension exceeding
    /// `maximum_kept_grade` or `maximum_kept_dimension`, respectively, are not
    /// returned.
    pub fn new<M>(
        complex: &CubicalComplex<M, TopCubeGrader<G>>,
        maximum_kept_grade: u32,
        maximum_kept_dimension: u32,
    ) -> Self {
        Self {
            minimum_orthant: Orthant::zeros(complex.ambient_dimension() as usize),
            maximum_orthant: Orthant::zeros(complex.ambient_dimension() as usize),
            axis_increments: vec![1usize; complex.ambient_dimension() as usize],
            grade_cache: SubgridGradeCache::new(complex.grader().orthant_grader().clone(), 0),
            minimum_grade: complex.grader().min_grade(),
            maximum_kept_grade,
            maximum_kept_dimension,
            peak_finder: PeakFinder::new(complex.ambient_dimension(), complex.grader().min_grade()),
        }
    }

    fn prepare_subgrid(&mut self, minimum_orthant: Orthant, maximum_orthant: Orthant) {
        debug_assert_eq!(
            minimum_orthant.ambient_dimension(),
            maximum_orthant.ambient_dimension(),
            "mismatched minimum and maximum ambient dimension"
        );

        self.minimum_orthant = minimum_orthant;
        self.maximum_orthant = maximum_orthant;

        // We only match orthants between `minimum_orthant` and `maximum_orthant`.
        // However, this may require grades of orthants from 1 less than each minimum
        // coordinate. Grades are thus cached for all orthants between
        // `least_graded_orthant` and the maximum.
        let least_graded_orthant: Orthant = self
            .minimum_orthant
            .iter()
            .map(|coord| *coord - 1)
            .collect();

        // To move one orthant along an axis in `grade_cache`, increment current
        // index by `axis_increments[axis]`.
        for (axis, (lg_coord, max_coord)) in
            zip(least_graded_orthant.iter(), self.maximum_orthant.iter())
                .enumerate()
                .skip(1)
                .rev()
        {
            debug_assert!(
                max_coord > lg_coord,
                "maximum orthant does not exceed minimum orthant"
            );
            self.axis_increments[axis - 1] =
                self.axis_increments[axis] * (max_coord - lg_coord + 1) as usize;
        }
        debug_assert!(
            self.maximum_orthant[0] > least_graded_orthant[0],
            "maximum orthant does not exceed minimum orthant"
        );
        let total_orthant_count = self.axis_increments[0]
            * (self.maximum_orthant[0] - least_graded_orthant[0] + 1) as usize;
        self.grade_cache.reset(total_orthant_count);
    }

    /// Match each orthant in the subgrid, and return a vector of (base_orthant,
    /// vector of critical cells, and match results).
    pub fn match_subgrid(
        &mut self,
        minimum_orthant: Orthant,
        maximum_orthant: Orthant,
    ) -> Vec<(Orthant, Vec<Cube>, OrthantMatching)> {
        self.prepare_subgrid(minimum_orthant, maximum_orthant);

        let mut base_critical_matching = Vec::new();
        for base_orthant in
            OrthantIterator::new(self.minimum_orthant.clone(), self.maximum_orthant.clone())
        {
            let (critical_cells, orthant_matching) = self.match_orthant(base_orthant.clone());
            base_critical_matching.push((base_orthant, critical_cells, orthant_matching));
        }
        base_critical_matching
    }

    fn match_orthant(&mut self, base_orthant: Orthant) -> (Vec<Cube>, OrthantMatching) {
        // Determine cache indices from orthant object; see `OrthantWrapper` and
        // `SubgridGradeCache` for more information.
        let ambient_dimension = base_orthant.ambient_dimension();
        let mut base_cache_index = 0;
        for (axis, inc) in self.axis_increments.iter().enumerate() {
            base_cache_index +=
                *inc * (base_orthant[axis] + 1 - self.minimum_orthant[axis]) as usize;
        }

        let base_orthant_wrapper = OrthantWrapper {
            orthant: base_orthant.clone(),
            cache_index: base_cache_index,
            extent: (1 << ambient_dimension) - 1,
        };
        let base_grade = self.grade_cache.grade(&base_orthant_wrapper);

        let peaks = self.peak_finder.compute_peaks(
            base_orthant_wrapper,
            &self.axis_increments,
            &mut self.grade_cache,
        );

        let mut critical_cells = Vec::new();
        let mut suborthant = Suborthant {
            base_orthant,
            upper_extent: (1 << ambient_dimension) - 1,
            lower_extent: 0,
            upper_grade: base_grade,
            lower_dimension: 0,
        };
        let orthant_matching = self.match_suborthant(&peaks, &mut critical_cells, &mut suborthant);
        (critical_cells, orthant_matching)
    }

    /// Match all cells in `suborthant`; append all critical cells found (that
    /// do not exceed the maximum kept grade or dimension) to `critical_cells`.
    fn match_suborthant(
        &mut self,
        peaks: &[(u32, u32)],
        critical_cells: &mut Vec<Cube>,
        suborthant: &mut Suborthant,
    ) -> OrthantMatching {
        // If the suborthant has just one cell, it must be critical
        if suborthant.lower_extent == suborthant.upper_extent {
            // Do not record critical cells with grade or dimension exceeding
            // the configured values.
            if suborthant.upper_grade <= self.maximum_kept_grade
                && suborthant.lower_dimension <= self.maximum_kept_dimension
            {
                critical_cells.push(Cube::new(
                    suborthant.base_orthant.clone(),
                    suborthant.lower_dual_orthant(),
                ));
            }
            return OrthantMatching::Critical {
                ace_dual_orthant: suborthant.lower_dual_orthant(),
                ace_extent: suborthant.lower_extent,
            };
        }

        // If all cells have the minimum grade or all exceed the maximum dimension,
        // match without further analysis
        if Some(suborthant.upper_grade) == self.minimum_grade
            || suborthant.lower_dimension > self.maximum_kept_dimension
        {
            return OrthantMatching::construct_leaf(
                suborthant.upper_extent,
                suborthant.lower_extent,
            );
        }

        if let Some((peak_index, prime_extent, prime_grade)) =
            peaks
                .iter()
                .enumerate()
                .find_map(|(peak_index, (extent, grade))| {
                    if extent & suborthant.lower_extent == suborthant.lower_extent {
                        return Some((peak_index, extent & suborthant.upper_extent, *grade));
                    }
                    None
                })
        {
            let original_upper_grade = suborthant.upper_grade;
            let differing_axes = prime_extent ^ suborthant.upper_extent;
            let remaining_peaks = peaks.split_at(peak_index + 1).1;

            // Match the prime suborthant (between the prime cell and the lowest cell of the
            // current suborthant)
            suborthant.upper_extent = prime_extent;
            suborthant.upper_grade = prime_grade;
            let mut suborthant_matchings = Vec::with_capacity(self.axis_increments.len());
            suborthant_matchings.push(self.match_suborthant(
                remaining_peaks,
                critical_cells,
                suborthant,
            ));

            // Match the remaining cells in the original suborthant by forming suborthants
            // from the prime suborthant, and extent along each axis that the prime cell
            // differs from the maximal cell (bits set in `differing_axes`).
            suborthant.upper_grade = original_upper_grade;
            suborthant.lower_dimension += 1;
            let mut axis_flag = 1;
            for _ in 0..self.axis_increments.len() {
                if differing_axes & axis_flag != 0 {
                    suborthant.upper_extent ^= axis_flag;
                    suborthant.lower_extent ^= axis_flag;

                    suborthant_matchings.push(self.match_suborthant(
                        remaining_peaks,
                        critical_cells,
                        suborthant,
                    ));

                    suborthant.lower_extent ^= axis_flag;
                }
                axis_flag <<= 1;
            }
            suborthant.lower_dimension -= 1;

            return OrthantMatching::Branch {
                upper_extent: suborthant.upper_extent,
                prime_extent,
                suborthant_matchings,
            };
        }

        // All elements have the same grade, but there is more than one cell
        // (so, not critical). Thus, all match.
        OrthantMatching::construct_leaf(suborthant.upper_extent, suborthant.lower_extent)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{Cyclic, HashMapGrader, HashMapModule};

    #[test]
    fn match_cube_torus_complex_subgrid() {
        let grader = TopCubeGrader::new(
            HashMapGrader::uniform([Orthant::from([1, 1, 1])], 1, 0),
            Some(0),
        );
        let complex = CubicalComplex::<
            HashMapModule<Cube, Cyclic<2>>,
            TopCubeGrader<HashMapGrader<Orthant>>,
        >::new(Orthant::from([0, 0, 0]), Orthant::from([1, 1, 1]), grader);
        let mut subgrid = Subgrid::new(&complex, u32::MAX, u32::MAX);
        let base_critical_matching =
            subgrid.match_subgrid(Orthant::from([0, 0, 0]), Orthant::from([1, 1, 1]));

        assert_eq!(base_critical_matching.len(), 8);
        for (_base_orthant, critical_cells, orthant_matching) in base_critical_matching[0..7].iter()
        {
            assert_eq!(critical_cells.len(), 0);
            assert!(matches!(
                orthant_matching,
                OrthantMatching::Leaf {
                    lower_extent: 0,
                    match_axis: 0
                }
            ));
        }

        let (base_orthant, critical_cells, orthant_matching) = &base_critical_matching[7];
        assert_eq!(*base_orthant, Orthant::from([1, 1, 1]));
        assert_eq!(
            *critical_cells,
            vec![
                Cube::from_extent(base_orthant.clone(), &[true, true, false]),
                Cube::from_extent(base_orthant.clone(), &[true, true, true])
            ]
        );
        let correct_matching = OrthantMatching::Branch {
            upper_extent: 0b111,
            prime_extent: 0b110,
            suborthant_matchings: vec![
                OrthantMatching::Leaf {
                    lower_extent: 0b000,
                    match_axis: 1,
                },
                OrthantMatching::Branch {
                    upper_extent: 0b111,
                    prime_extent: 0b101,
                    suborthant_matchings: vec![
                        OrthantMatching::Leaf {
                            lower_extent: 0b001,
                            match_axis: 2,
                        },
                        OrthantMatching::Branch {
                            upper_extent: 0b111,
                            prime_extent: 0b011,
                            suborthant_matchings: vec![
                                OrthantMatching::Critical {
                                    ace_dual_orthant: Orthant::from([1, 1, 0]),
                                    ace_extent: 0b011,
                                },
                                OrthantMatching::Critical {
                                    ace_dual_orthant: Orthant::from([1, 1, 1]),
                                    ace_extent: 0b111,
                                },
                            ],
                        },
                    ],
                },
            ],
        };
        assert_eq!(*orthant_matching, correct_matching);
    }

    fn prepare_two_cube_subgrid() -> Subgrid<HashMapGrader<Orthant>> {
        let grader = TopCubeGrader::new(
            HashMapGrader::uniform(
                [Orthant::from([0, 0, 1, 1]), Orthant::from([1, 1, 0, 0])],
                0,
                1,
            ),
            Some(0),
        );
        let complex = CubicalComplex::<
            HashMapModule<Cube, Cyclic<2>>,
            TopCubeGrader<HashMapGrader<Orthant>>,
        >::new(
            Orthant::from([0, 0, 0, 0]),
            Orthant::from([1, 1, 1, 1]),
            grader,
        );
        let mut subgrid = Subgrid::new(&complex, u32::MAX, u32::MAX);
        subgrid.prepare_subgrid(Orthant::from([1, 1, 1, 1]), Orthant::from([1, 1, 1, 1]));
        subgrid
    }

    #[test]
    fn compute_peaks_two_cube_subgrid() {
        let correct_peaks = vec![(0b1100, 0), (0b0011, 0)];
        let mut subgrid = prepare_two_cube_subgrid();

        let base_orthant = OrthantWrapper {
            orthant: Orthant::from([1, 1, 1, 1]),
            cache_index: 15,
            extent: 0b1111,
        };
        let peaks = subgrid.peak_finder.compute_peaks(
            base_orthant,
            &subgrid.axis_increments,
            &mut subgrid.grade_cache,
        );

        assert_eq!(peaks, correct_peaks);
    }

    #[test]
    fn match_two_cube_subgrid() {
        let mut subgrid = prepare_two_cube_subgrid();
        let base_critical_matching =
            subgrid.match_subgrid(Orthant::from([1, 1, 1, 1]), Orthant::from([1, 1, 1, 1]));
        let (base_orthant, critical_cells, orthant_matching) = &base_critical_matching[0];
        assert_eq!(*base_orthant, Orthant::from([1, 1, 1, 1]));
        assert_eq!(
            *critical_cells,
            vec![
                Cube::from_extent(base_orthant.clone(), &[true, false, false, false]),
                Cube::from_extent(base_orthant.clone(), &[true, false, true, false]),
            ]
        );
        let correct_matching = OrthantMatching::Branch {
            upper_extent: 0b1111,
            prime_extent: 0b1100,
            suborthant_matchings: vec![
                OrthantMatching::Leaf {
                    lower_extent: 0b0000,
                    match_axis: 2,
                },
                OrthantMatching::Branch {
                    upper_extent: 0b1101,
                    prime_extent: 0b0001,
                    suborthant_matchings: vec![
                        OrthantMatching::Critical {
                            ace_dual_orthant: Orthant::from([1, 0, 0, 0]),
                            ace_extent: 0b0001,
                        },
                        OrthantMatching::Critical {
                            ace_dual_orthant: Orthant::from([1, 0, 1, 0]),
                            ace_extent: 0b0101,
                        },
                        OrthantMatching::Leaf {
                            lower_extent: 0b1001,
                            match_axis: 2,
                        },
                    ],
                },
                OrthantMatching::Branch {
                    upper_extent: 0b1111,
                    prime_extent: 0b0011,
                    suborthant_matchings: vec![
                        OrthantMatching::Leaf {
                            lower_extent: 0b0010,
                            match_axis: 0,
                        },
                        OrthantMatching::Leaf {
                            lower_extent: 0b0110,
                            match_axis: 0,
                        },
                        OrthantMatching::Leaf {
                            lower_extent: 0b1010,
                            match_axis: 0,
                        },
                    ],
                },
            ],
        };
        assert_eq!(*orthant_matching, correct_matching);
    }
}
