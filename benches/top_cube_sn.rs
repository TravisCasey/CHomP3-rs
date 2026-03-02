//! Benchmarks for homology on S^n boundary-cube constructions.
//!
//! This suite uses a compact set of representative scenarios instead of a full
//! Cartesian product of dimensions and configuration flags.

mod common;

use chomp3rs::prelude::*;
use common::{MatchingConfig, Scenario, UniformGrader, generate_sn_orthants, run_matching_bench};

fn main() {
    divan::main();
}

const CASES: &[Scenario] = &[
    Scenario {
        label: "s4_unfiltered_full",
        dim: 4,
        config: MatchingConfig {
            filtered: false,
            subgrid_size: 1,
            max_grade: None,
        },
    },
    Scenario {
        label: "s4_filtered_full",
        dim: 4,
        config: MatchingConfig {
            filtered: true,
            subgrid_size: 1,
            max_grade: None,
        },
    },
    Scenario {
        label: "s7_filtered_subgrid2_full",
        dim: 7,
        config: MatchingConfig {
            filtered: true,
            subgrid_size: 2,
            max_grade: None,
        },
    },
    Scenario {
        label: "s7_filtered_subgrid2_grade0",
        dim: 7,
        config: MatchingConfig {
            filtered: true,
            subgrid_size: 2,
            max_grade: Some(0),
        },
    },
];

fn expected_betti_sn(n: usize) -> Vec<usize> {
    let mut betti = vec![0; n + 1];
    betti[0] = 1;
    betti[n] = 1;
    betti
}

fn bench_sn<G: UniformGrader<Orthant>>(bencher: divan::Bencher, scenario: Scenario) {
    // For S^n benchmarks, `dim` holds the sphere dimension; ambient = dim + 1.
    let orthants = generate_sn_orthants(scenario.dim as usize);
    let ambient_dim = (scenario.dim + 1) as usize;
    let expected = expected_betti_sn(scenario.dim as usize);
    run_matching_bench::<G>(bencher, &orthants, ambient_dim, scenario.config, &expected);
}

#[divan::bench_group(sample_count = 10)]
mod curated {
    use super::{CASES, HashGrader, Orthant, OrthantTrie, Scenario, bench_sn};

    #[divan::bench(args = CASES)]
    fn hashmap(bencher: divan::Bencher, scenario: Scenario) {
        bench_sn::<HashGrader<Orthant>>(bencher, scenario);
    }

    #[divan::bench(args = CASES)]
    fn trie(bencher: divan::Bencher, scenario: Scenario) {
        bench_sn::<OrthantTrie>(bencher, scenario);
    }
}
