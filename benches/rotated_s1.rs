//! Benchmarks for homology on rotated S^1 datasets.
//!
//! The suite is intentionally compact and scenario-driven. Cases are selected
//! to capture the most relevant tradeoffs without an exhaustive Cartesian
//! product over every dimension/flag.

mod common;

use chomp3rs::prelude::*;
use common::{MatchingConfig, Scenario, UniformGrader, load_orthants, run_matching_bench};

fn main() {
    divan::main();
}

/// Expected Betti numbers for S^1: b0=1, b1=1.
const S1_BETTI: [usize; 2] = [1, 1];

const CASES: &[Scenario] = &[
    Scenario {
        label: "dim3_unfiltered_full",
        dim: 3,
        config: MatchingConfig {
            filtered: false,
            subgrid_size: 1,
            max_grade: None,
        },
    },
    Scenario {
        label: "dim3_filtered_full",
        dim: 3,
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

fn bench_s1<G: UniformGrader<Orthant>>(bencher: divan::Bencher, scenario: Scenario) {
    let path = format!("data/s1/dim{}.csv", scenario.dim);
    let orthants = load_orthants(&path);
    run_matching_bench::<G>(
        bencher,
        &orthants,
        scenario.dim as usize,
        scenario.config,
        &S1_BETTI,
    );
}

#[divan::bench_group(sample_count = 10)]
mod curated {
    use super::{CASES, HashGrader, Orthant, OrthantTrie, Scenario, bench_s1};

    #[divan::bench(args = CASES)]
    fn hashmap(bencher: divan::Bencher, scenario: Scenario) {
        bench_s1::<HashGrader<Orthant>>(bencher, scenario);
    }

    #[divan::bench(args = CASES)]
    fn trie(bencher: divan::Bencher, scenario: Scenario) {
        bench_s1::<OrthantTrie>(bencher, scenario);
    }
}
