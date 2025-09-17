// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::iter::zip;

use crate::homology::cubical::OrthantMatching;
use crate::{Cube, CubicalComplex, Grader, Orthant, OrthantIterator, TopCubeGrader};

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
/// The greatest cell has dual orthant `upper`, while the least cell in the
/// interval has dual orthant `lower`. The grade of the greatest cell and the
/// dimension of the least cell are also stored so they do not need to be
/// recomputed when matching a suborthant (via `Subgrid::match_subgrid`).
#[derive(Clone, Debug, Eq, PartialEq)]
struct Suborthant {
    base_orthant: Orthant,
    upper: OrthantWrapper,
    lower: OrthantWrapper,
    upper_grade: u32,
    lower_dimension: u32,
}

/// A basic contiguous cache that stores the results of queries to the given
/// `grading_function` in an array of size `cache_size`.
///
/// The size of the cache is the number of orthants in the subgrid (including
/// the extra orthants that are queried but not matched, see `Subgrid::new`).
#[derive(Clone, Debug)]
struct SubgridGradeCache<'a, G>
where
    G: Grader<Orthant>,
{
    grading_function: &'a G,
    cache: Vec<Option<u32>>,
}

impl<'a, G> SubgridGradeCache<'a, G>
where
    G: Grader<Orthant>,
{
    fn new(grading_function: &'a G, cache_size: usize) -> Self {
        Self {
            grading_function,
            cache: vec![None; cache_size],
        }
    }

    fn grade(&mut self, orthant: &OrthantWrapper) -> u32 {
        *self.cache[orthant.cache_index]
            .get_or_insert_with(|| self.grading_function.grade(&orthant.orthant))
    }
}

/// A breadth-first (i.e., all cells of same dimension before moving to the
/// next) iterator over all cells in a suborthant. See the documentation of
/// [`Suborthant`] for the structure of a suborthant.
#[derive(Clone, Debug)]
struct SuborthantBFSIterator<'a> {
    axis_increments: &'a [usize],
    axes_with_extent: Vec<usize>,
    axis_index: usize,
    dual_orthants: Vec<OrthantWrapper>,
    dual_index: usize,
}

impl<'a> SuborthantBFSIterator<'a> {
    fn new(axis_increments: &'a [usize], suborthant: &Suborthant) -> Self {
        let axes_with_extent = zip(
            suborthant.upper.orthant.iter(),
            suborthant.lower.orthant.iter(),
        )
        .enumerate()
        .filter_map(|(axis, (upper, lower))| if *upper != *lower { Some(axis) } else { None })
        .collect::<Vec<_>>();

        let mut dual_orthants = Vec::with_capacity(1 << axes_with_extent.len());
        dual_orthants.push(suborthant.upper.clone());

        Self {
            axis_increments,
            axes_with_extent,
            axis_index: 0,
            dual_orthants,
            dual_index: 0,
        }
    }

    fn decrement_axis(&self, orthant: &OrthantWrapper, axis: usize) -> OrthantWrapper {
        let mut new_orthant = orthant.orthant.clone();
        new_orthant[axis] -= 1;

        OrthantWrapper {
            orthant: new_orthant,
            cache_index: orthant.cache_index - self.axis_increments[axis],
            extent: orthant.extent - (1 << axis),
        }
    }
}

impl Iterator for SuborthantBFSIterator<'_> {
    type Item = OrthantWrapper;

    fn next(&mut self) -> Option<Self::Item> {
        while self.dual_index < self.dual_orthants.len() {
            if self.axis_index < self.axes_with_extent.len() {
                let axis = self.axes_with_extent[self.axis_index];
                if self.dual_orthants[self.dual_index].extent & (1 << axis) != 0 {
                    self.dual_orthants
                        .push(self.decrement_axis(&self.dual_orthants[self.dual_index], axis));
                    self.axis_index += 1;
                    return Some(self.dual_orthants[self.dual_orthants.len() - 1].clone());
                }
            }
            self.dual_index += 1;
            self.axis_index = 0;
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl ExactSizeIterator for SuborthantBFSIterator<'_> {
    fn len(&self) -> usize {
        (1 << self.axes_with_extent.len()) - self.dual_orthants.len()
    }
}

/// A rectangular grid of orthants between (all coordinates between, not
/// lexicographically) `minimum_orthant` and `maximum_orthant`.
///
/// This type matches all its orthants for use in `TopCubicalMatching`. By
/// matching nearby orthants together, we can improve efficiency by caching the
/// commonly re-queried grades of nearby top-dimensional cubes.
pub struct Subgrid<'a, G>
where
    G: Grader<Orthant>,
{
    minimum_orthant: Orthant,
    maximum_orthant: Orthant,
    axis_increments: Vec<usize>,
    grade_cache: SubgridGradeCache<'a, G>,
    minimum_grade: Option<u32>,
    maximum_kept_grade: u32,
    maximum_kept_dimension: u32,
}

impl<'a, G> Subgrid<'a, G>
where
    G: Grader<Orthant>,
{
    /// Initialize the subgrid, but do not perform matching yet. That is
    /// completed using the `match_subgrid` method.
    pub fn new<M>(
        complex: &'a CubicalComplex<M, TopCubeGrader<G>>,
        minimum_orthant: Orthant,
        maximum_orthant: Orthant,
        maximum_kept_grade: u32,
        maximum_kept_dimension: u32,
    ) -> Self {
        debug_assert_eq!(
            minimum_orthant.ambient_dimension(),
            maximum_orthant.ambient_dimension(),
            "mismatched minimum and maximum ambient dimension"
        );

        // We only match orthants between `minimum_orthant` and `maximum_orthant`.
        // However, this may require grades of orthants from 1 less than each minimum
        // coordinate. Grades are thus cached for all orthants between
        // `least_graded_orthant` and the maximum.
        let least_graded_orthant: Orthant =
            minimum_orthant.iter().map(|coord| *coord - 1).collect();

        // To move one orthant along an axis in `grade_cache`, increment current
        // index by `axis_increments[axis]`.
        let mut axis_increments = vec![1usize; minimum_orthant.ambient_dimension() as usize];
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
            axis_increments[axis - 1] = axis_increments[axis] * (max_coord - lg_coord + 1) as usize;
        }
        debug_assert!(
            maximum_orthant[0] > least_graded_orthant[0],
            "maximum orthant does not exceed minimum orthant"
        );
        let total_orthant_count =
            axis_increments[0] * (maximum_orthant[0] - least_graded_orthant[0] + 1) as usize;

        Self {
            minimum_orthant,
            maximum_orthant,
            axis_increments,
            grade_cache: SubgridGradeCache::new(
                complex.grader().orthant_grader(),
                total_orthant_count,
            ),
            minimum_grade: complex.grader().min_grade(),
            maximum_kept_grade,
            maximum_kept_dimension,
        }
    }

    /// Match all cells in `suborthant`; append all critical cells found (that
    /// do not exceed the maximum kept grade or dimension) to `critical_cells`.
    fn match_suborthant(
        &mut self,
        critical_cells: &mut Vec<Cube>,
        suborthant: &mut Suborthant,
    ) -> OrthantMatching {
        // If the suborthant has just one cell, it must be critical
        if suborthant.lower.extent == suborthant.upper.extent {
            // Do not record critical cells with grade or dimension exceeding
            // the configured values.
            if suborthant.upper_grade <= self.maximum_kept_grade
                && suborthant.lower_dimension <= self.maximum_kept_dimension
            {
                critical_cells.push(Cube::new(
                    suborthant.base_orthant.clone(),
                    suborthant.lower.orthant.clone(),
                ));
            }
            return OrthantMatching::Critical {
                ace_dual_orthant: suborthant.lower.orthant.clone(),
                ace_extent: suborthant.lower.extent,
            };
        }

        // If all cells have the minimum grade or all exceed the maximum dimension,
        // match without further analysis
        if Some(suborthant.upper_grade) == self.minimum_grade
            || suborthant.lower_dimension > self.maximum_kept_dimension
        {
            return OrthantMatching::construct_leaf(
                suborthant.upper.extent,
                suborthant.lower.extent,
            );
        }

        let bfs_iter = SuborthantBFSIterator::new(&self.axis_increments, suborthant);

        for prime_orthant in bfs_iter {
            let prime_grade = self.grade_cache.grade(&prime_orthant);

            // Found prime suborthant (largest suborthant for which its maximum
            // cell has differing grade than the max cell in this suborthant)
            if prime_grade != suborthant.upper_grade {
                debug_assert_eq!(
                    prime_orthant.extent & suborthant.upper.extent,
                    prime_orthant.extent,
                    "the prime cell is not a face of the maximal cell in the suborthant"
                );
                let original_upper_grade = suborthant.upper_grade;
                let differing_axes = prime_orthant.extent ^ suborthant.upper.extent;
                let prime_extent = prime_orthant.extent;

                // Match the prime suborthant (between the prime cell and the lowest cell of the
                // current suborthant)
                suborthant.upper = prime_orthant;
                suborthant.upper_grade = prime_grade;
                let mut suborthant_matchings = Vec::with_capacity(self.axis_increments.len());
                suborthant_matchings.push(self.match_suborthant(critical_cells, suborthant));

                // Match the remaining cells in the original suborthant by forming suborthants
                // from the prime suborthant, and extent along each axis that the prime cell
                // differs from the maximal cell (bits set in `differing_axes`).
                suborthant.upper_grade = original_upper_grade;
                suborthant.lower_dimension += 1;
                for axis in 0..self.axis_increments.len() {
                    if differing_axes & (1 << axis) != 0 {
                        self.increment_axis(&mut suborthant.upper, axis);
                        self.increment_axis(&mut suborthant.lower, axis);

                        suborthant_matchings
                            .push(self.match_suborthant(critical_cells, suborthant));

                        self.decrement_axis(&mut suborthant.lower, axis);
                    }
                }
                suborthant.lower_dimension -= 1;

                return OrthantMatching::Branch {
                    prime_extent,
                    suborthant_matchings,
                };
            }
        }

        // All elements the same grade, but there is more than one cell (so, not
        // critical). Thus, all match.
        OrthantMatching::construct_leaf(suborthant.upper.extent, suborthant.lower.extent)
    }

    fn increment_axis(&self, orthant: &mut OrthantWrapper, axis: usize) {
        orthant.cache_index += self.axis_increments[axis];
        orthant.orthant[axis] += 1;
        orthant.extent += 1 << axis;
    }

    fn decrement_axis(&self, orthant: &mut OrthantWrapper, axis: usize) {
        orthant.cache_index -= self.axis_increments[axis];
        orthant.orthant[axis] -= 1;
        orthant.extent -= 1 << axis;
    }

    fn match_orthant(&mut self, base_orthant: Orthant) -> (Vec<Cube>, OrthantMatching) {
        // Determine cache indices from orthant object; see `OrthantWrapper` and
        // `SubgridGradeCache` for more information.
        let ambient_dimension = base_orthant.ambient_dimension();
        let mut base_cache_index = 0;
        let mut lower_orthant = base_orthant.clone();
        let mut lower_cache_index = 0;
        for (axis, inc) in self.axis_increments.iter().enumerate() {
            base_cache_index +=
                *inc * (base_orthant[axis] + 1 - self.minimum_orthant[axis]) as usize;
            lower_orthant[axis] -= 1;
            lower_cache_index +=
                *inc * (lower_orthant[axis] + 1 - self.minimum_orthant[axis]) as usize;
        }

        let upper = OrthantWrapper {
            orthant: base_orthant.clone(),
            cache_index: base_cache_index,
            extent: (1 << ambient_dimension) - 1,
        };
        let upper_grade = self.grade_cache.grade(&upper);

        let lower = OrthantWrapper {
            orthant: lower_orthant,
            cache_index: lower_cache_index,
            extent: 0,
        };

        let mut critical_cells = Vec::new();
        let mut suborthant = Suborthant {
            base_orthant,
            upper,
            lower,
            upper_grade,
            lower_dimension: 0,
        };
        let orthant_matching = self.match_suborthant(&mut critical_cells, &mut suborthant);
        (critical_cells, orthant_matching)
    }

    /// Match each orthant in the subgrid, and return a vector of (base_orthant,
    /// vector of critical cells, and match results).
    pub fn match_subgrid(&mut self) -> Vec<(Orthant, Vec<Cube>, OrthantMatching)> {
        let mut base_critical_matching = Vec::new();
        for base_orthant in
            OrthantIterator::new(self.minimum_orthant.clone(), self.maximum_orthant.clone())
        {
            let (critical_cells, orthant_matching) = self.match_orthant(base_orthant.clone());
            base_critical_matching.push((base_orthant, critical_cells, orthant_matching));
        }
        base_critical_matching
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::{Cyclic, HashMapGrader, HashMapModule};

    #[test]
    fn suborthant_iterator() {
        // 2 x 2 x 3 grid, with least graded orthant 3 x 3 x 4
        let axis_increments = vec![12, 4, 1];

        let suborthant = Suborthant {
            base_orthant: Orthant::from([1, 0, 1]),
            upper: OrthantWrapper {
                orthant: Orthant::from([1, 0, 0]),
                cache_index: 29,
                extent: 0b011,
            },
            lower: OrthantWrapper {
                orthant: Orthant::from([0, -1, 0]),
                cache_index: 13,
                extent: 0b000,
            },
            lower_dimension: 0,
            upper_grade: 1,
        };

        let mut bfs_iter = SuborthantBFSIterator::new(&axis_increments, &suborthant);

        assert_eq!(
            bfs_iter.next(),
            Some(OrthantWrapper {
                orthant: Orthant::from([0, 0, 0]),
                cache_index: 17,
                extent: 0b010
            })
        );
        assert_eq!(
            bfs_iter.next(),
            Some(OrthantWrapper {
                orthant: Orthant::from([1, -1, 0]),
                cache_index: 25,
                extent: 0b001
            })
        );
        assert_eq!(
            bfs_iter.next(),
            Some(OrthantWrapper {
                orthant: Orthant::from([0, -1, 0]),
                cache_index: 13,
                extent: 0b000
            })
        );
        assert_eq!(bfs_iter.next(), None);
        assert_eq!(bfs_iter.next(), None);
    }

    #[test]
    fn match_cube_torus_complex_subgrid() {
        let serialized_complex = fs::read_to_string("testing/complexes/cube_torus_complex.json")
            .expect("Testing complex file not found.");
        let complex: CubicalComplex<
            HashMapModule<Cube, Cyclic<2>>,
            TopCubeGrader<HashMapGrader<Orthant>>,
        > = serde_json::from_str(&serialized_complex)
            .expect("Testing complex could not be deserialized.");
        let mut subgrid = Subgrid::new(
            &complex,
            Orthant::from([0, 0, 0]),
            Orthant::from([1, 1, 1]),
            u32::MAX,
            u32::MAX,
        );
        let base_critical_matching = subgrid.match_subgrid();

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
            prime_extent: 0b110,
            suborthant_matchings: vec![
                OrthantMatching::Leaf {
                    lower_extent: 0,
                    match_axis: 1,
                },
                OrthantMatching::Branch {
                    prime_extent: 0b101,
                    suborthant_matchings: vec![
                        OrthantMatching::Leaf {
                            lower_extent: 0b001,
                            match_axis: 2,
                        },
                        OrthantMatching::Branch {
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
}
