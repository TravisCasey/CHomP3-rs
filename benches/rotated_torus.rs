use std::fs;

use chomp3rs::prelude::*;

fn main() {
    divan::main();
}

/// Load orthants from CSV file (one row per orthant, comma-separated
/// coordinates).
fn load_orthants(path: &str) -> Vec<Orthant> {
    let content = fs::read_to_string(path).expect("Failed to read CSV file");
    content
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let coords: Vec<i16> = line
                .split(',')
                .map(|s| s.trim().parse().expect("Invalid coordinate"))
                .collect();
            Orthant::new(&coords)
        })
        .collect()
}

trait UniformGrader<C>: Clone + Grader<C> {
    fn uniform<I>(cells: I, uniform_grade: u32, default_grade: u32) -> Self
    where
        I: IntoIterator<Item = Orthant>;
}

impl UniformGrader<Orthant> for HashMapGrader<Orthant> {
    fn uniform<I>(cells: I, uniform_grade: u32, default_grade: u32) -> Self
    where
        I: IntoIterator<Item = Orthant>,
    {
        Self::uniform(cells, uniform_grade, default_grade)
    }
}

impl UniformGrader<Orthant> for OrthantTrie {
    fn uniform<I>(cells: I, uniform_grade: u32, default_grade: u32) -> Self
    where
        I: IntoIterator<Item = Orthant>,
    {
        Self::uniform(cells, uniform_grade, default_grade)
    }
}

fn create_complex<G: UniformGrader<Orthant>>(
    orthants: &[Orthant],
) -> CubicalComplex<HashMapModule<Cube, Cyclic<2>>, TopCubeGrader<G>> {
    let dim = orthants[0].ambient_dimension() as usize;
    let mut min_coords = vec![i16::MAX; dim];
    let mut max_coords = vec![i16::MIN; dim];

    for orth in orthants {
        for i in 0..dim {
            min_coords[i] = min_coords[i].min(orth[i]);
            max_coords[i] = max_coords[i].max(orth[i] + 1);
        }
    }

    let minimum = Orthant::new(&min_coords);
    let maximum = Orthant::new(&max_coords);
    let grader = TopCubeGrader::new(G::uniform(orthants.iter().cloned(), 0, 1), Some(0));
    CubicalComplex::new(minimum, maximum, grader)
}

const S1_ARGS: &[(bool, u32, u32)] = &[
    (false, 2, 0),
    (true, 2, 0),
    (false, 3, 0),
    (true, 3, 0),
    (false, 4, 0),
    (true, 4, 0),
    (false, 5, 0),
    (true, 5, 0),
    (true, 6, 0),
    (true, 7, 0),
    (true, 8, 0),
    (false, 2, 1),
    (true, 2, 1),
    (false, 3, 1),
    (true, 3, 1),
    (false, 4, 1),
    (true, 4, 1),
    (false, 5, 1),
    (true, 5, 1),
    (true, 6, 1),
    (true, 7, 1),
    (true, 8, 1),
];

#[divan::bench(types = [HashMapGrader<Orthant>, OrthantTrie], args = S1_ARGS, sample_count = 10)]
fn rotated_s1<G: UniformGrader<Orthant>>(bencher: divan::Bencher, args: (bool, u32, u32)) {
    let (filtered, dim, max_grade) = args;
    let path = format!("data/s1/dim{dim}.csv");
    let orthants = load_orthants(&path);
    bencher
        .with_inputs(|| (create_complex::<G>(&orthants), orthants.clone()))
        .bench_local_values(|(complex, filter_orthants)| {
            let builder = TopCubicalMatching::<
                HashMapModule<Cube, Cyclic<2>>,
                HashMapGrader<Orthant>,
            >::builder()
            .max_grade(max_grade);
            let builder = if filtered {
                builder.filter_orthants(filter_orthants)
            } else {
                builder
            };
            let mut matching = builder.build();
            matching.full_reduce(CoreductionMatching::new(), complex)
        });
}

const TORUS_ARGS: &[(bool, u32, u32)] = &[
    (false, 3, 0),
    (true, 3, 0),
    (false, 4, 0),
    (true, 4, 0),
    (false, 5, 0),
    (true, 5, 0),
    (true, 6, 0),
    (true, 7, 0),
    (true, 8, 0),
    (false, 3, 1),
    (true, 3, 1),
    (false, 4, 1),
    (true, 4, 1),
    (false, 5, 1),
    (true, 5, 1),
    (true, 6, 1),
    (true, 7, 1),
    (true, 8, 1),
];

#[divan::bench(types = [HashMapGrader<Orthant>, OrthantTrie], args = TORUS_ARGS, sample_count = 10)]
fn rotated_torus<G: UniformGrader<Orthant>>(bencher: divan::Bencher, args: (bool, u32, u32)) {
    let (filtered, dim, max_grade) = args;
    let path = format!("data/torus/dim{dim}.csv");
    let orthants = load_orthants(&path);
    bencher
        .with_inputs(|| (create_complex::<G>(&orthants), orthants.clone()))
        .bench_local_values(|(complex, filter_orthants)| {
            let builder = TopCubicalMatching::<
                HashMapModule<Cube, Cyclic<2>>,
                HashMapGrader<Orthant>,
            >::builder()
            .max_grade(max_grade);
            let builder = if filtered {
                builder.filter_orthants(filter_orthants)
            } else {
                builder
            };
            let mut matching = builder.build();
            matching.full_reduce(CoreductionMatching::new(), complex)
        });
}
