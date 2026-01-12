// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Subgrid matching utilities for cubical complexes.
//!
//! This module provides types for subdividing and matching orthant grids in
//! cubical complexes. These types are used internally by [`TopCubicalMatching`]
//! but exposed for advanced users implementing custom matching workflows.
//!
//! [`TopCubicalMatching`]: super::TopCubicalMatching

mod matching;
mod peaks;
mod subdivision;

pub use matching::OrthantMatching;
use peaks::PeakFinder;
pub use subdivision::GridSubdivision;

use crate::{Cube, CubicalComplex, Grader, Orthant, OrthantIterator, TopCubeGrader};

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
        self.base_orthant
            .iter()
            .enumerate()
            .map(|(axis, coord)| {
                if extent & (1 << axis) == 0 {
                    *coord - 1
                } else {
                    *coord
                }
            })
            .collect::<Orthant>()
    }

    fn lower_dual_orthant(&self) -> Orthant {
        self.compute_dual_orthant(self.lower_extent)
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
    maximum_kept_grade: u32,
    maximum_kept_dimension: u32,
    peak_finder: PeakFinder<G>,
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
            maximum_kept_grade,
            maximum_kept_dimension,
            peak_finder: PeakFinder::new(
                complex.ambient_dimension(),
                complex.grader().min_grade(),
                complex.grader().orthant_grader().clone(),
            ),
        }
    }

    /// Set the maximum grade of returned critical cells for any future
    /// `match_subgrid` calls.
    pub fn set_maximum_kept_grade(&mut self, maximum_kept_grade: u32) {
        self.maximum_kept_grade = maximum_kept_grade;
    }

    /// Set the maximum dimension of returned critical cells for any future
    /// `match_subgrid` calls.
    pub fn set_maximum_kept_dimension(&mut self, maximum_kept_dimension: u32) {
        self.maximum_kept_dimension = maximum_kept_dimension;
    }

    /// Match each orthant in the subgrid, and return a vector of:
    /// `(base orthant, vector of critical cells, match results)`.
    pub fn match_subgrid(
        &mut self,
        minimum_orthant: Orthant,
        maximum_orthant: Orthant,
    ) -> Vec<(Orthant, Vec<Cube>, OrthantMatching)> {
        debug_assert_eq!(
            minimum_orthant.ambient_dimension(),
            maximum_orthant.ambient_dimension(),
            "mismatched minimum and maximum ambient dimension"
        );

        self.minimum_orthant = minimum_orthant;
        self.maximum_orthant = maximum_orthant;
        self.peak_finder
            .prepare(&self.minimum_orthant, &self.maximum_orthant);

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
        let ambient_dimension = base_orthant.ambient_dimension();
        let base_grade = self.peak_finder.compute_peaks(&base_orthant);

        let mut critical_cells = Vec::new();
        let mut suborthant = Suborthant {
            base_orthant,
            upper_extent: (1 << ambient_dimension) - 1,
            lower_extent: 0,
            upper_grade: base_grade,
            lower_dimension: 0,
        };
        let orthant_matching = self.match_suborthant(&mut critical_cells, &mut suborthant);

        (critical_cells, orthant_matching)
    }

    /// Match all cells in `suborthant`; append all critical cells found (that
    /// do not exceed the maximum kept grade or dimension) to `critical_cells`.
    fn match_suborthant(
        &self,
        critical_cells: &mut Vec<Cube>,
        suborthant: &mut Suborthant,
    ) -> OrthantMatching {
        // Single cell: must be critical
        if suborthant.lower_extent == suborthant.upper_extent {
            // Discard the critical cell if grade or dimension exceeds the
            // configurable limits
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

        // If all cells exceed the critical cell dimension limit, declare all
        // cells matched irrespective of grade; they do not matter.
        if suborthant.lower_dimension > self.maximum_kept_dimension {
            return OrthantMatching::construct_leaf(
                suborthant.upper_extent,
                suborthant.lower_extent,
            );
        }

        // Find prime peak; if none, all cells must have the same grade and
        // can be trivially matched.
        let Some((prime_extent, prime_grade)) = self.peak_finder.prime_peak(
            suborthant.upper_extent,
            suborthant.lower_extent,
            suborthant.upper_grade,
        ) else {
            return OrthantMatching::construct_leaf(
                suborthant.upper_extent,
                suborthant.lower_extent,
            );
        };

        // Branch: recurse on prime suborthant and siblings
        let original_upper_extent = suborthant.upper_extent;
        let original_upper_grade = suborthant.upper_grade;
        let differing_axes = prime_extent ^ suborthant.upper_extent;
        let axis_count = self.peak_finder.axis_increments().len();

        // Match prime suborthant: [lower_extent, prime_extent]
        suborthant.upper_extent = prime_extent;
        suborthant.upper_grade = prime_grade;
        let mut suborthant_matchings = Vec::with_capacity(axis_count);
        suborthant_matchings.push(self.match_suborthant(critical_cells, suborthant));

        // Match sibling suborthants of the form:
        // [lower_extent + a_i, prime_extent + a_0 + a_1 + ... + a_i],
        // for `a_i` being the set bits (axes) in upper_extent - prime_extent
        suborthant.upper_grade = original_upper_grade;
        suborthant.lower_dimension += 1;
        let mut axis_flag = 1;
        for _ in 0..axis_count {
            if differing_axes & axis_flag != 0 {
                suborthant.upper_extent ^= axis_flag;
                suborthant.lower_extent ^= axis_flag;

                suborthant_matchings.push(self.match_suborthant(critical_cells, suborthant));

                suborthant.lower_extent ^= axis_flag;
            }
            axis_flag <<= 1;
        }
        suborthant.lower_dimension -= 1;

        OrthantMatching::Branch {
            upper_extent: original_upper_extent,
            prime_extent,
            suborthant_matchings,
        }
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
        for (_base_orthant, critical_cells, orthant_matching) in &base_critical_matching[0..7] {
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
                Cube::from_extent(base_orthant.clone(), &[false, true, true]),
                Cube::from_extent(base_orthant.clone(), &[true, true, true])
            ]
        );

        let correct_matching = OrthantMatching::Branch {
            upper_extent: 0b111,
            prime_extent: 0b011,
            suborthant_matchings: vec![
                OrthantMatching::Leaf {
                    lower_extent: 0b000,
                    match_axis: 0,
                },
                OrthantMatching::Branch {
                    upper_extent: 0b111,
                    prime_extent: 0b101,
                    suborthant_matchings: vec![
                        OrthantMatching::Leaf {
                            lower_extent: 0b100,
                            match_axis: 0,
                        },
                        OrthantMatching::Branch {
                            upper_extent: 0b111,
                            prime_extent: 0b110,
                            suborthant_matchings: vec![
                                OrthantMatching::Critical {
                                    ace_dual_orthant: Orthant::from([0, 1, 1]),
                                    ace_extent: 0b110,
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

    #[test]
    fn match_two_cube_subgrid() {
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
        let base_critical_matching =
            subgrid.match_subgrid(Orthant::from([1, 1, 1, 1]), Orthant::from([1, 1, 1, 1]));

        let (base_orthant, critical_cells, orthant_matching) = &base_critical_matching[0];
        assert_eq!(*base_orthant, Orthant::from([1, 1, 1, 1]));
        assert_eq!(
            *critical_cells,
            vec![
                Cube::from_extent(base_orthant.clone(), &[false, false, true, false]),
                Cube::from_extent(base_orthant.clone(), &[true, false, true, false]),
            ]
        );
        let correct_matching = OrthantMatching::Branch {
            upper_extent: 0b1111,
            prime_extent: 0b0011,
            suborthant_matchings: vec![
                OrthantMatching::Leaf {
                    lower_extent: 0b0000,
                    match_axis: 0,
                },
                OrthantMatching::Branch {
                    upper_extent: 0b0111,
                    prime_extent: 0b0100,
                    suborthant_matchings: vec![
                        OrthantMatching::Critical {
                            ace_dual_orthant: Orthant::from([0, 0, 1, 0]),
                            ace_extent: 0b0100,
                        },
                        OrthantMatching::Critical {
                            ace_dual_orthant: Orthant::from([1, 0, 1, 0]),
                            ace_extent: 0b0101,
                        },
                        OrthantMatching::Leaf {
                            lower_extent: 0b0110,
                            match_axis: 0,
                        },
                    ],
                },
                OrthantMatching::Branch {
                    upper_extent: 0b1111,
                    prime_extent: 0b1100,
                    suborthant_matchings: vec![
                        OrthantMatching::Leaf {
                            lower_extent: 0b1000,
                            match_axis: 2,
                        },
                        OrthantMatching::Leaf {
                            lower_extent: 0b1001,
                            match_axis: 2,
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
