use chomp3rs::{ComplexLike, CoreductionMatching, MorseMatching, TopCubicalMatching};
use test_utilities::{top_cube_sn_hashmap, top_cube_sn_trie};

fn main() {
    divan::main();
}

#[divan::bench(args = [4, 5, 6, 7, 8], sample_count = 10)]
fn top_cube_reduce_sn_hashmap(bencher: divan::Bencher, n: usize) {
    bencher
        .with_inputs(|| top_cube_sn_hashmap(n))
        .bench_local_values(|complex| {
            let mut matching = TopCubicalMatching::default();
            let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

            // Don't optimize away..
            assert_eq!(morse_complex.dimension(), n as u32 + 1);
        });
}

#[divan::bench(args = [4, 5, 6, 7, 8], sample_count = 10)]
fn top_cube_reduce_sn_trie(bencher: divan::Bencher, n: usize) {
    bencher
        .with_inputs(|| top_cube_sn_trie(n))
        .bench_local_values(|complex| {
            let mut matching = TopCubicalMatching::default();
            let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

            // Don't optimize away..
            assert_eq!(morse_complex.dimension(), n as u32 + 1);
        });
}

#[divan::bench(args = [4, 5, 6, 7, 8], sample_count = 10)]
fn top_cube_reduce_sn_hashmap_grade_truncated(bencher: divan::Bencher, n: usize) {
    bencher
        .with_inputs(|| top_cube_sn_hashmap(n))
        .bench_local_values(|complex| {
            let mut matching = TopCubicalMatching::new(Some(0), None, None);
            let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

            // Don't optimize away.. (note truncation)
            assert_eq!(morse_complex.dimension(), n as u32);
        });
}

#[divan::bench(args = [4, 5, 6, 7, 8], sample_count = 10)]
fn top_cube_reduce_sn_trie_grade_truncated(bencher: divan::Bencher, n: usize) {
    bencher
        .with_inputs(|| top_cube_sn_trie(n))
        .bench_local_values(|complex| {
            let mut matching = TopCubicalMatching::new(Some(0), None, None);
            let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

            // Don't optimize away.. (note truncation)
            assert_eq!(morse_complex.dimension(), n as u32);
        });
}
