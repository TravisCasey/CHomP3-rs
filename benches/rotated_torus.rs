//! Benchmarks for homology on rotated T^2 datasets.
//!
//! Cases are curated to represent the key strategy tradeoffs while avoiding an
//! overly large Cartesian benchmark matrix.

mod common;

use chomp3rs::prelude::*;
use common::{MatchingConfig, Scenario, UniformGrader, load_orthants, run_matching_bench};

fn main() {
    divan::main();
}

/// Expected Betti numbers for T^2: b0=1, b1=2, b2=1.
const TORUS_BETTI: [usize; 3] = [1, 2, 1];

const CASES: &[Scenario] = &[
    Scenario {
        label: "dim4_unfiltered_full",
        dim: 4,
        config: MatchingConfig {
            filtered: false,
            subgrid_size: 1,
            max_grade: None,
        },
    },
    Scenario {
        label: "dim4_filtered_full",
        dim: 4,
        config: MatchingConfig {
            filtered: true,
            subgrid_size: 1,
            max_grade: None,
        },
    },
    Scenario {
        label: "dim5_filtered_subgrid2_full",
        dim: 5,
        config: MatchingConfig {
            filtered: true,
            subgrid_size: 2,
            max_grade: None,
        },
    },
    Scenario {
        label: "dim5_filtered_subgrid2_grade0",
        dim: 5,
        config: MatchingConfig {
            filtered: true,
            subgrid_size: 2,
            max_grade: Some(0),
        },
    },
];

fn bench_torus<G: UniformGrader<Orthant>>(bencher: divan::Bencher, scenario: Scenario) {
    let path = format!("data/torus/dim{}.csv", scenario.dim);
    let orthants = load_orthants(&path);
    run_matching_bench::<G>(
        bencher,
        &orthants,
        scenario.dim as usize,
        scenario.config,
        &TORUS_BETTI,
    );
}

#[divan::bench_group(sample_count = 10)]
mod curated {
    use super::{CASES, HashGrader, Orthant, OrthantTrie, Scenario, bench_torus};

    #[divan::bench(args = CASES)]
    fn hashmap(bencher: divan::Bencher, scenario: Scenario) {
        bench_torus::<HashGrader<Orthant>>(bencher, scenario);
    }

    #[divan::bench(args = CASES)]
    fn trie(bencher: divan::Bencher, scenario: Scenario) {
        bench_torus::<OrthantTrie>(bencher, scenario);
    }
}
