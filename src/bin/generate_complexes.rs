// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Executable for generating test complexes (CellComplex and CubicalComplex)
//! for testing purposes.
//!
//! This program generates various test cases of complexes and serializes them
//! to JSON files in the chomp3rs/testing/complexes directory.

use std::error::Error;
use std::fs;

use chomp3rs::{
    CellComplex, Cube, CubicalComplex, Cyclic, HashMapGrader, HashMapModule, Orthant, RingLike,
    TopCubeGrader,
};
use serde::Serialize;

fn serialize_complex<C>(complex_tup: (C, &str)) -> Result<(), Box<dyn Error>>
where
    C: Serialize,
{
    let (complex, name) = complex_tup;
    let filename = format!("testing/complexes/{}_complex.json", name);
    let json = serde_json::to_string_pretty(&complex)?;
    fs::write(&filename, json)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Ensures the directory exists and is empty
    if fs::exists("testing/complexes")? {
        fs::remove_dir_all("testing/complexes")?;
    }
    fs::create_dir_all("testing/complexes")?;

    println!("Generating test complexes...");

    let complexes = (
        (cell_complex_generators::triangle(), "triangle"),
        (cubical_complex_generators::figure_eight(), "figure_eight"),
        (cubical_complex_generators::cube_torus(), "cube_torus"),
    );

    serialize_complex(complexes.0)?;
    serialize_complex(complexes.1)?;
    serialize_complex(complexes.2)?;

    Ok(())
}

/// Module containing generators for CellComplex test cases
mod cell_complex_generators {
    use chomp3rs::ModuleLike;

    use super::*;

    /// Create a triangle complex (3 vertices, 3 edges, 1 face). All grade 0
    pub fn triangle() -> CellComplex<HashMapModule<u32, Cyclic<2>>> {
        // 3 vertices (0,1,2), 3 edges (3,4,5), 1 triangle face (6)
        let cell_dimensions = vec![0, 0, 0, 1, 1, 1, 2];
        let grades = vec![0, 0, 0, 0, 0, 0, 0];

        let mut boundaries = vec![HashMapModule::new(); 7];

        // Edge boundaries: edge = vertex_end - vertex_start
        boundaries[3].insert_or_add(1, Cyclic::one()); // edge 0->1
        boundaries[3].insert_or_add(0, -Cyclic::one());

        boundaries[4].insert_or_add(2, Cyclic::one()); // edge 1->2
        boundaries[4].insert_or_add(1, -Cyclic::one());

        boundaries[5].insert_or_add(0, Cyclic::one()); // edge 2->0
        boundaries[5].insert_or_add(2, -Cyclic::one());

        // Triangle boundary: sum of edges
        boundaries[6].insert_or_add(3, Cyclic::one());
        boundaries[6].insert_or_add(4, Cyclic::one());
        boundaries[6].insert_or_add(5, Cyclic::one());

        let mut coboundaries = vec![HashMapModule::new(); 7];

        // Vertex coboundaries
        coboundaries[0].insert_or_add(3, -Cyclic::one()); // vertex 0 in edge 0->1
        coboundaries[0].insert_or_add(5, Cyclic::one()); // vertex 0 in edge 2->0

        coboundaries[1].insert_or_add(3, Cyclic::one()); // vertex 1 in edge 0->1
        coboundaries[1].insert_or_add(4, -Cyclic::one()); // vertex 1 in edge 1->2

        coboundaries[2].insert_or_add(4, Cyclic::one()); // vertex 2 in edge 1->2
        coboundaries[2].insert_or_add(5, -Cyclic::one()); // vertex 2 in edge 2->0

        // Edge coboundaries (in triangle)
        coboundaries[3].insert_or_add(6, Cyclic::one());
        coboundaries[4].insert_or_add(6, Cyclic::one());
        coboundaries[5].insert_or_add(6, Cyclic::one());

        CellComplex::new(cell_dimensions, grades, boundaries, coboundaries)
    }
}

/// Module containing generators for CubicalComplex test cases
mod cubical_complex_generators {
    use super::*;

    /// Figure eight in two orthants, all included vertices at grade 0.
    /// All other cells (ignored) at grade 1. Ring Cyclic<2>.
    ///       ______ ______
    ///      |      |      |
    ///      |      |      |
    ///      |______|______|
    /// 
    pub fn figure_eight() -> CubicalComplex<HashMapModule<Cube, Cyclic<2>>, HashMapGrader<Cube>> {
        // Define bounds: 3x2 grid of orthants
        let minimum = Orthant::from([0, 0]);
        let maximum = Orthant::from([2, 2]);

        let mut grader = HashMapGrader::uniform(
            [
                Cube::vertex(Orthant::from([0, 0])),
                Cube::vertex(Orthant::from([0, 1])),
                Cube::vertex(Orthant::from([1, 0])),
                Cube::vertex(Orthant::from([1, 1])),
                Cube::vertex(Orthant::from([2, 0])),
                Cube::vertex(Orthant::from([2, 1])),
                Cube::from_extent(Orthant::from([0, 0]), &[true, false]),
                Cube::from_extent(Orthant::from([0, 0]), &[false, true]),
                Cube::from_extent(Orthant::from([1, 0]), &[true, false]),
                Cube::from_extent(Orthant::from([1, 0]), &[false, true]),
                Cube::from_extent(Orthant::from([2, 0]), &[false, true]),
                Cube::from_extent(Orthant::from([0, 1]), &[true, false]),
                Cube::from_extent(Orthant::from([1, 1]), &[true, false]),
            ],
            0,
            1,
        );
        grader.set_default_grade(2);

        CubicalComplex::new(minimum, maximum, grader)
    }

    /// Create a 3D torus top-cell cubical complex
    pub fn cube_torus()
    -> CubicalComplex<HashMapModule<Cube, Cyclic<2>>, TopCubeGrader<HashMapGrader<Orthant>>> {
        // Define bounds: single 3D unit cube
        let minimum = Orthant::from([0, 0, 0]);
        let maximum = Orthant::from([7, 7, 3]);

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
                        if x == 1 || x == 5 || y == 1 || y == 5 {
                            continue;
                        }
                    }
                    top_cubes.push(Orthant::from([x, y, z]));
                }
            }
        }

        let grader = TopCubeGrader::new(HashMapGrader::uniform(top_cubes, 0, 1), Some(0));

        CubicalComplex::new(minimum, maximum, grader)
    }
}
