// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Test complex generators and validation helpers.

use crate::{
    CellComplex, Chain, Complex, CubicalComplex, F2, Grader, HashGrader, Orthant, OrthantTrie,
    TopCubeGrader, complexes::OrthantIterator,
};

/// Graders supporting uniform construction from orthants.
///
/// This abstracts over [`HashGrader<Orthant>`] and [`OrthantTrie`], allowing
/// test helpers to build complexes generically over the grader backend.
pub(crate) trait UniformGrader: Clone + Grader<Orthant> {
    /// Create a grader that assigns `uniform_grade` to each cell in the
    /// iterator and `default_grade` to all other cells.
    fn uniform_grade(
        cells: impl IntoIterator<Item = Orthant>,
        uniform_grade: u32,
        default_grade: u32,
    ) -> Self;
}

impl UniformGrader for HashGrader<Orthant> {
    fn uniform_grade(
        cells: impl IntoIterator<Item = Orthant>,
        uniform_grade: u32,
        default_grade: u32,
    ) -> Self {
        Self::uniform(cells, uniform_grade, default_grade)
    }
}

impl UniformGrader for OrthantTrie {
    fn uniform_grade(
        cells: impl IntoIterator<Item = Orthant>,
        uniform_grade: u32,
        default_grade: u32,
    ) -> Self {
        Self::uniform(cells, uniform_grade, default_grade)
    }
}

/// Build a [`CubicalComplex`] from a slice of top-cube orthants with
/// uniform grading.
///
/// The bounding box is computed automatically from the orthants. All given
/// orthants receive grade 0; all other cells receive grade 1.
///
/// # Panics
///
/// - If `orthants` is empty.
/// - If any orthant coordinate equals `i16::MAX` (bounding box expansion would
///   overflow).
pub(crate) fn complex_from_orthants<G: UniformGrader>(
    orthants: &[Orthant],
) -> CubicalComplex<F2, TopCubeGrader<G>> {
    assert!(!orthants.is_empty(), "orthants must not be empty");
    let dim = orthants[0].ambient_dimension() as usize;
    let mut min_coords = vec![i16::MAX; dim];
    let mut max_coords = vec![i16::MIN; dim];

    for orth in orthants {
        for i in 0..dim {
            min_coords[i] = min_coords[i].min(orth[i]);
            let upper = orth[i]
                .checked_add(1)
                .expect("orthant coordinate overflow: coordinate is i16::MAX");
            max_coords[i] = max_coords[i].max(upper);
        }
    }

    let minimum = Orthant::new(&min_coords);
    let maximum = Orthant::new(&max_coords);
    let grader = TopCubeGrader::new(G::uniform_grade(orthants.iter().cloned(), 0, 1), Some(0));
    CubicalComplex::new(minimum, maximum, grader)
}

/// Generate the top-dimensional orthant cells of a discretized n-sphere.
///
/// Returns the `3^(n+1) - 1` top cubes in an `(n+1)`-dimensional grid
/// `[0, 2]^(n+1)` with the center cube `[1]^(n+1)` removed, forming a
/// cubical approximation of S^n.
pub(crate) fn sn_orthants(n: usize) -> Vec<Orthant> {
    let minimum = Orthant::new(&vec![0; n + 1]);
    let maximum_included = Orthant::new(&vec![2; n + 1]);
    let hole = Orthant::new(&vec![1; n + 1]);

    OrthantIterator::new(minimum, maximum_included)
        .filter(|orth| *orth != hole)
        .collect()
}

/// Generate the top-dimensional orthant cells of a discretized torus.
///
/// Returns 128 top cubes forming a cubical approximation of the 2-torus
/// (genus-1 surface) embedded in a 7 by 7 by 3 grid.
pub(crate) fn torus_orthants() -> Vec<Orthant> {
    let mut top_cubes = Vec::new();
    for x in 0..7 {
        for y in 0..7 {
            if x == 3 && y == 3 {
                continue;
            }
            for z in 0..3 {
                if z == 1 {
                    if (x == 1 || x == 5) && (1..=5).contains(&y) {
                        continue;
                    }
                    if (y == 1 || y == 5) && (1..=5).contains(&x) {
                        continue;
                    }
                }
                top_cubes.push(Orthant::from([x, y, z]));
            }
        }
    }
    top_cubes
}

/// Assert that the grade-0 Betti numbers of a Morse complex match `expected`.
///
/// Counts grade-0 critical cells by dimension and verifies each has trivial
/// grade-0 boundary and coboundary.
pub(crate) fn assert_betti_numbers(morse_complex: &CellComplex<F2>, expected: &[u32]) {
    let mut betti = vec![0u32; expected.len()];
    for cell in morse_complex.iter() {
        if morse_complex.grade(&cell) != 0 {
            continue;
        }
        let dim = morse_complex.cell_dimension(&cell) as usize;
        assert!(
            dim < expected.len(),
            "unexpected cell of dimension {dim} (expected at most {})",
            expected.len() - 1
        );
        betti[dim] += 1;

        let boundary: Chain<u32, F2> = morse_complex
            .cell_boundary(&cell)
            .into_iter()
            .filter(|(bd_cell, _)| morse_complex.grade(bd_cell) == 0)
            .collect();
        assert_eq!(
            boundary,
            Chain::new(),
            "non-trivial boundary for grade-0 cell"
        );

        let coboundary: Chain<u32, F2> = morse_complex
            .cell_coboundary(&cell)
            .into_iter()
            .filter(|(cbd_cell, _)| morse_complex.grade(cbd_cell) == 0)
            .collect();
        assert_eq!(
            coboundary,
            Chain::new(),
            "non-trivial coboundary for grade-0 cell"
        );
    }
    assert_eq!(betti, expected, "Betti numbers mismatch");
}

/// Assert that every cell in the Morse complex has grade 0.
pub(crate) fn assert_all_grade_zero(morse_complex: &CellComplex<F2>) {
    for cell in morse_complex.iter() {
        assert_eq!(
            morse_complex.grade(&cell),
            0,
            "expected all cells to be grade 0"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CoreductionMatching, MorseMatching, TopCubicalMatching};

    #[test]
    fn sn_orthants_counts() {
        // S^n has 3^{n+1} - 1 orthants (full grid minus center hole).
        assert_eq!(sn_orthants(1).len(), 8); // 3^2 - 1
        assert_eq!(sn_orthants(2).len(), 26); // 3^3 - 1
        assert_eq!(sn_orthants(3).len(), 80); // 3^4 - 1
    }

    #[test]
    fn sn_orthants_excludes_center() {
        for n in 1..=3 {
            let center = Orthant::new(&vec![1i16; n + 1]);
            assert!(
                !sn_orthants(n).contains(&center),
                "S^{n} should not contain center"
            );
        }
    }

    #[test]
    fn torus_orthants_count_and_uniqueness() {
        let orthants = torus_orthants();
        assert_eq!(orthants.len(), 128);
        let mut sorted = orthants.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 128);
    }

    #[test]
    #[should_panic(expected = "orthants must not be empty")]
    fn complex_from_orthants_panics_on_empty() {
        let _ = complex_from_orthants::<HashGrader<Orthant>>(&[]);
    }

    #[test]
    fn complex_from_orthants_bbox() {
        let orthants = vec![Orthant::from([1i16, 2]), Orthant::from([3i16, 0])];
        let complex = complex_from_orthants::<HashGrader<Orthant>>(&orthants);
        assert_eq!(complex.ambient_dimension(), 2);
    }

    #[test]
    fn assert_betti_numbers_accepts_correct_values() {
        let orthants = sn_orthants(1);
        let complex = complex_from_orthants::<HashGrader<Orthant>>(&orthants);
        let matching = TopCubicalMatching::new(complex);
        let morse = matching.full_reduce(CoreductionMatching::new).1;
        // S^1 has Betti numbers [1, 1].
        assert_betti_numbers(&morse, &[1, 1]);
    }

    #[test]
    #[should_panic(expected = "Betti numbers mismatch")]
    fn assert_betti_numbers_rejects_wrong_values() {
        let orthants = sn_orthants(1);
        let complex = complex_from_orthants::<HashGrader<Orthant>>(&orthants);
        let matching = TopCubicalMatching::new(complex);
        let morse = matching.full_reduce(CoreductionMatching::new).1;
        assert_betti_numbers(&morse, &[1, 0]);
    }
}
