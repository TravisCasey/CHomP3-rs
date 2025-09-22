use chomp3rs::{
    Cube, CubicalComplex, Cyclic, HashMapGrader, HashMapModule, Orthant, OrthantIterator,
    OrthantTrie, TopCubeGrader,
};

pub fn top_cube_sn_hashmap(
    n: usize,
) -> CubicalComplex<HashMapModule<Cube, Cyclic<2>>, TopCubeGrader<HashMapGrader<Orthant>>> {
    let minimum = Orthant::new(vec![0; n + 1]);
    let maximum = Orthant::new(vec![3; n + 1]);

    let maximum_included = Orthant::new(vec![2; n + 1]);
    let grader = TopCubeGrader::new(
        HashMapGrader::uniform(
            OrthantIterator::new(minimum.clone(), maximum_included)
                .filter(|top_cube| *top_cube != Orthant::new(vec![1; n + 1])),
            0,
            1,
        ),
        Some(0),
    );
    CubicalComplex::new(minimum, maximum, grader)
}

pub fn top_cube_sn_trie(
    n: usize,
) -> CubicalComplex<HashMapModule<Cube, Cyclic<2>>, TopCubeGrader<OrthantTrie>> {
    let minimum = Orthant::new(vec![0; n + 1]);
    let maximum = Orthant::new(vec![3; n + 1]);

    let maximum_included = Orthant::new(vec![2; n + 1]);
    let grader = TopCubeGrader::new(
        OrthantTrie::uniform(
            OrthantIterator::new(minimum.clone(), maximum_included)
                .filter(|top_cube| *top_cube != Orthant::new(vec![1; n + 1])),
            0,
            1,
        ),
        Some(0),
    );
    CubicalComplex::new(minimum, maximum, grader)
}
