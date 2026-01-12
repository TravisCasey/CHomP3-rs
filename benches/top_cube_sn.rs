use chomp3rs::{
    ComplexLike, CoreductionMatching, Cube, CubicalComplex, Cyclic, HashMapGrader, HashMapModule,
    MorseMatching, Orthant, OrthantIterator, TopCubeGrader, TopCubicalMatching,
};
use test_utilities::{top_cube_sn_hashmap, top_cube_sn_trie};

type Module = HashMapModule<Cube, Cyclic<2>>;
type Matching = TopCubicalMatching<Module, HashMapGrader<Orthant>>;

fn main() {
    divan::main();
}

/// Create the S^n complex along with the list of orthants for filtering.
fn top_cube_sn_hashmap_with_orthants(
    n: usize,
) -> (
    CubicalComplex<Module, TopCubeGrader<HashMapGrader<Orthant>>>,
    Vec<Orthant>,
) {
    let minimum = Orthant::new(&vec![0; n + 1]);
    let maximum = Orthant::new(&vec![3; n + 1]);

    let maximum_included = Orthant::new(&vec![2; n + 1]);
    let orthants: Vec<Orthant> = OrthantIterator::new(minimum.clone(), maximum_included)
        .filter(|top_cube| *top_cube != Orthant::new(&vec![1; n + 1]))
        .collect();

    let grader = TopCubeGrader::new(
        HashMapGrader::uniform(orthants.iter().cloned(), 0, 1),
        Some(0),
    );
    (CubicalComplex::new(minimum, maximum, grader), orthants)
}

#[divan::bench(args = [4, 5, 6, 7], sample_count = 10)]
fn top_cube_reduce_sn_hashmap(bencher: divan::Bencher, n: u32) {
    bencher
        .with_inputs(|| top_cube_sn_hashmap(n as usize))
        .bench_local_values(|complex| {
            let mut matching = TopCubicalMatching::default();
            let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

            // Don't optimize away..
            assert_eq!(morse_complex.dimension(), n + 1);
        });
}

#[divan::bench(args = [4, 5, 6, 7], sample_count = 10)]
fn top_cube_reduce_sn_trie(bencher: divan::Bencher, n: u32) {
    bencher
        .with_inputs(|| top_cube_sn_trie(n as usize))
        .bench_local_values(|complex| {
            let mut matching = TopCubicalMatching::default();
            let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

            // Don't optimize away..
            assert_eq!(morse_complex.dimension(), n + 1);
        });
}

#[divan::bench(args = [4, 5, 6, 7], sample_count = 10)]
fn top_cube_reduce_sn_hashmap_grade_truncated(bencher: divan::Bencher, n: u32) {
    bencher
        .with_inputs(|| top_cube_sn_hashmap(n as usize))
        .bench_local_values(|complex| {
            let mut matching = TopCubicalMatching::with_max_grade(0);
            let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

            // Don't optimize away.. (note truncation)
            assert_eq!(morse_complex.dimension(), n);
        });
}

#[divan::bench(args = [4, 5, 6, 7], sample_count = 10)]
fn top_cube_reduce_sn_trie_grade_truncated(bencher: divan::Bencher, n: u32) {
    bencher
        .with_inputs(|| top_cube_sn_trie(n as usize))
        .bench_local_values(|complex| {
            let mut matching = TopCubicalMatching::with_max_grade(0);
            let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

            // Don't optimize away.. (note truncation)
            assert_eq!(morse_complex.dimension(), n);
        });
}

#[divan::bench(args = [4, 5, 6, 7], sample_count = 10)]
fn top_cube_reduce_sn_hashmap_filtered(bencher: divan::Bencher, n: u32) {
    bencher
        .with_inputs(|| top_cube_sn_hashmap_with_orthants(n as usize))
        .bench_local_values(|(complex, orthants)| {
            let mut matching: Matching = Matching::builder().filter_orthants(orthants).build();
            let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

            // Don't optimize away..
            assert_eq!(morse_complex.dimension(), n + 1);
        });
}
