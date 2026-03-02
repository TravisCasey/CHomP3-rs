//! Shared benchmark harness: configuration, scenario types, and the matching
//! bench runner.
//!
//! Data generation utilities live in the [`datagen`] submodule.

// Each bench binary only uses a subset of this module's API, so unused items
// and `pub use` re-exports that are not called by a given binary appear dead
// or as unused imports to the compiler.
#![allow(dead_code, unused_imports)]

use std::fmt;

use chomp3rs::prelude::*;
pub use datagen::{generate_sn_orthants, load_orthants};

mod datagen;

/// Benchmark scenario: a named configuration for a single benchmark case.
///
/// The `dim` field is context-dependent: ambient dimension for manifold
/// benchmarks (S^1, T^2), sphere dimension for S^n benchmarks (ambient = dim +
/// 1).
#[derive(Clone, Copy, Debug)]
pub struct Scenario {
    pub label: &'static str,
    pub dim: u32,
    pub config: MatchingConfig,
}

impl fmt::Display for Scenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label)
    }
}

/// Benchmark configuration for `TopCubicalMatching`.
#[derive(Clone, Copy, Debug)]
pub struct MatchingConfig {
    pub filtered: bool,
    pub subgrid_size: i16,
    pub max_grade: Option<u32>,
}

/// Trait for graders that support uniform construction from an iterator of
/// orthants.
pub trait UniformGrader<C>: Clone + Grader<C> {
    fn uniform_grade<I>(cells: I, uniform_grade: u32, default_grade: u32) -> Self
    where
        I: IntoIterator<Item = Orthant>;
}

impl UniformGrader<Orthant> for HashGrader<Orthant> {
    fn uniform_grade<I>(cells: I, uniform_grade: u32, default_grade: u32) -> Self
    where
        I: IntoIterator<Item = Orthant>,
    {
        Self::uniform(cells, uniform_grade, default_grade)
    }
}

impl UniformGrader<Orthant> for OrthantTrie {
    fn uniform_grade<I>(cells: I, uniform_grade: u32, default_grade: u32) -> Self
    where
        I: IntoIterator<Item = Orthant>,
    {
        Self::uniform(cells, uniform_grade, default_grade)
    }
}

/// Create a cubical complex from a list of top-cube orthants.
///
/// # Panics
///
/// Panics if `orthants` is empty.
#[must_use]
pub fn create_complex<G: UniformGrader<Orthant>>(
    orthants: &[Orthant],
) -> CubicalComplex<F2, TopCubeGrader<G>> {
    assert!(
        !orthants.is_empty(),
        "Cannot build complex from empty orthants"
    );
    let dim = orthants[0].ambient_dimension() as usize;
    let mut min_coords = vec![i16::MAX; dim];
    let mut max_coords = vec![i16::MIN; dim];

    for orth in orthants {
        for i in 0..dim {
            min_coords[i] = min_coords[i].min(orth[i]);
            max_coords[i] = max_coords[i].max(orth[i] + 1);
        }
    }

    let minimum = Orthant::new(&min_coords);
    let maximum = Orthant::new(&max_coords);
    let grader = TopCubeGrader::new(G::uniform_grade(orthants.iter().cloned(), 0, 1), Some(0));
    CubicalComplex::new(minimum, maximum, grader)
}

/// Shared benchmark runner for matching + full reduction + Betti validation.
///
/// # Panics
///
/// Panics if the computed grade-0 Betti numbers do not match
/// `expected_betti`.
pub fn run_matching_bench<G: UniformGrader<Orthant>>(
    bencher: divan::Bencher,
    orthants: &[Orthant],
    ambient_dim: usize,
    config: MatchingConfig,
    expected_betti: &[usize],
) {
    bencher
        .with_inputs(|| (create_complex::<G>(orthants), orthants.to_vec()))
        .bench_local_values(|(complex, filter_orthants)| {
            let mut builder = TopCubicalMatching::<F2, G>::builder()
                .subgrid_shape(vec![config.subgrid_size; ambient_dim]);
            if let Some(grade) = config.max_grade {
                builder = builder.max_grade(grade);
            }
            if config.filtered {
                builder = builder.filter_orthants(filter_orthants);
            }
            let matching = builder.build(complex);
            let (_, morse_complex) = matching.full_reduce(CoreductionMatching::new);

            let mut cells_by_dim = vec![0usize; expected_betti.len()];
            for cell in morse_complex.iter() {
                if morse_complex.grade(&cell) == 0 {
                    let d = morse_complex.cell_dimension(&cell) as usize;
                    if let Some(count) = cells_by_dim.get_mut(d) {
                        *count += 1;
                    }
                }
            }

            assert_eq!(cells_by_dim, expected_betti, "Betti numbers mismatch");
        });
}
