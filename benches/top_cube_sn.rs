use chomp3rs::{
    CellComplex, ComplexLike, CoreductionMatching, Cube, Cyclic, HashMapGrader, HashMapModule,
    MorseMatching, Orthant, OrthantTrie, TopCubicalMatching,
};
use test_utilities::{top_cube_sn_hashmap, top_cube_sn_trie};

fn main() {
    divan::main();
}

#[divan::bench(args = [4, 5, 6, 7, 8], sample_count = 10)]
fn top_cube_reduce_sn_hashmap(bencher: divan::Bencher, n: usize) {
    bencher
        .with_inputs(|| top_cube_sn_hashmap(n))
        .bench_local_values(|complex| {
            let morse_complex = TopCubicalMatching::<
                HashMapModule<Cube, Cyclic<2>>,
                HashMapGrader<Orthant>,
            >::full_reduce::<
                CoreductionMatching<CellComplex<HashMapModule<u32, Cyclic<2>>>>,
            >(complex)
            .2;

            // Don't optimize away..
            assert_eq!(morse_complex.dimension(), n as u32 + 1);
        });
}

#[divan::bench(args = [4, 5, 6, 7, 8], sample_count = 10)]
fn top_cube_reduce_sn_trie(bencher: divan::Bencher, n: usize) {
    bencher
        .with_inputs(|| top_cube_sn_trie(n))
        .bench_local_values(|complex| {
            let morse_complex =
                TopCubicalMatching::<HashMapModule<Cube, Cyclic<2>>, OrthantTrie>::full_reduce::<
                    CoreductionMatching<CellComplex<HashMapModule<u32, Cyclic<2>>>>,
                >(complex)
                .2;

            // Don't optimize away..
            assert_eq!(morse_complex.dimension(), n as u32 + 1);
        });
}
