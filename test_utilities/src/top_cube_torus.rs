use chomp3rs::{
    Cube, CubicalComplex, Cyclic, HashMapGrader, HashMapModule, Orthant, OrthantTrie, TopCubeGrader,
};

fn generate_top_cube_torus_orthants() -> Vec<Orthant> {
    let mut top_cubes = Vec::new();
    for x in 0..7 {
        for y in 0..7 {
            // Create hole in the middle
            if x == 3 && y == 3 {
                continue;
            }
            for z in 0..3 {
                if z == 1 {
                    // Make the torus hollow
                    if (x == 1 || x == 5) && (1..=5).contains(&y) {
                        continue;
                    }
                    if (y == 1 || y == 5) && (1..=5).contains(&x) {
                        continue;
                    }
                }
                top_cubes.push(Orthant::from([x, y, z]));
            }
        }
    }

    top_cubes
}

pub fn top_cube_torus_hashmap()
-> CubicalComplex<HashMapModule<Cube, Cyclic<2>>, TopCubeGrader<HashMapGrader<Orthant>>> {
    let minimum = Orthant::from([0, 0, 0]);
    let maximum = Orthant::from([7, 7, 3]);

    let top_cubes = generate_top_cube_torus_orthants();

    let grader = TopCubeGrader::new(HashMapGrader::uniform(top_cubes, 0, 1), Some(0));

    CubicalComplex::new(minimum, maximum, grader)
}

pub fn top_cube_torus_trie()
-> CubicalComplex<HashMapModule<Cube, Cyclic<2>>, TopCubeGrader<OrthantTrie>> {
    let minimum = Orthant::from([0, 0, 0]);
    let maximum = Orthant::from([7, 7, 3]);

    let top_cubes = generate_top_cube_torus_orthants();

    let grader = TopCubeGrader::new(OrthantTrie::uniform(top_cubes, 0, 1), Some(0));

    CubicalComplex::new(minimum, maximum, grader)
}
