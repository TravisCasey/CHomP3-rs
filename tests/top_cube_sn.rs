use chomp3rs::{
    CellComplex, ComplexLike, CoreductionMatching, Cyclic, Grader, HashMapModule, ModuleLike,
    MorseMatching, TopCubicalMatching,
};
use test_utilities::{top_cube_sn_hashmap, top_cube_sn_trie};

const DIMENSIONS: [u32; 4] = [1, 2, 3, 4];

fn verify_morse_complex(n: u32, morse_complex: &CellComplex<HashMapModule<u32, Cyclic<2>>>) {
    let correct_dimensions = vec![0, n];
    let mut dimensions = Vec::new();
    for cell in morse_complex.cell_iter() {
        if morse_complex.grade(&cell) != 0 {
            continue;
        }
        dimensions.push(morse_complex.cell_dimension(&cell));

        assert_eq!(
            morse_complex.cell_boundary_if(&cell, |bd_cell| morse_complex.grade(bd_cell) == 0),
            HashMapModule::new()
        );
        assert_eq!(
            morse_complex.cell_coboundary_if(&cell, |cbd_cell| morse_complex.grade(cbd_cell) == 0),
            HashMapModule::new()
        );
    }
    dimensions.sort();
    assert_eq!(dimensions, correct_dimensions);
}

#[test]
fn top_cube_reduce_sn_hashmap() {
    for n in DIMENSIONS {
        let complex = top_cube_sn_hashmap(n as usize);
        let mut matching = TopCubicalMatching::new();
        let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

        verify_morse_complex(n, &morse_complex);
    }
}

#[test]
fn top_cube_reduce_sn_trie() {
    for n in DIMENSIONS {
        let complex = top_cube_sn_trie(n as usize);
        let mut matching = TopCubicalMatching::new();
        let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

        verify_morse_complex(n, &morse_complex);
    }
}
