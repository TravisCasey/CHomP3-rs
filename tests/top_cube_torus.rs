use chomp3rs::{
    CellComplex, ComplexLike, CoreductionMatching, Cyclic, Grader, HashMapModule, ModuleLike,
    MorseMatching, TopCubicalMatching,
};
use test_utilities::{top_cube_torus_hashmap, top_cube_torus_trie};

fn verify_morse_complex(morse_complex: &CellComplex<HashMapModule<u32, Cyclic<2>>>) {
    let correct_dimensions = vec![0, 1, 1, 2];
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
fn top_cube_reduce_torus_hashmap() {
    let complex = top_cube_torus_hashmap();
    let mut matching = TopCubicalMatching::default();
    let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

    verify_morse_complex(&morse_complex);
}

#[test]
fn top_cube_reduce_torus_trie() {
    let complex = top_cube_torus_trie();
    let mut matching = TopCubicalMatching::default();
    let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

    verify_morse_complex(&morse_complex);
}

fn verify_morse_complex_grade_truncated(
    morse_complex: &CellComplex<HashMapModule<u32, Cyclic<2>>>,
) {
    let correct_dimensions = vec![0, 1, 1, 2];
    let mut dimensions = Vec::new();
    for cell in morse_complex.cell_iter() {
        assert_eq!(morse_complex.grade(&cell), 0);
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
fn top_cube_reduce_torus_hashmap_grade_truncated() {
    let complex = top_cube_torus_hashmap();
    let mut matching = TopCubicalMatching::new(Some(0), None);
    let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

    verify_morse_complex_grade_truncated(&morse_complex);
}

#[test]
fn top_cube_reduce_torus_trie_grade_truncated() {
    let complex = top_cube_torus_trie();
    let mut matching = TopCubicalMatching::new(Some(0), None);
    let morse_complex = matching.full_reduce(CoreductionMatching::new(), complex).1;

    verify_morse_complex_grade_truncated(&morse_complex);
}
