// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Explicit cell complex with vector-based storage.
//!
//! This module provides [`CellComplex`], a general-purpose cell complex where
//! boundaries and coboundaries are stored explicitly. It is typically used as
//! the output of Morse reduction algorithms.

use crate::{Chain, Complex, Grader, Ring};

/// A concrete implementation of a cell complex with explicit vector-based
/// storage.
///
/// `CellComplex` represents a finite cell complex where cells are identified by
/// `u32` indices. Each cell has associated dimensional information, grade,
/// boundary, and coboundary data explicitly stored in parallel vectors for
/// efficient access.
///
/// ### When to Use
///
/// Use `CellComplex` when:
/// - The complex has a moderate number of cells (memory overhead is acceptable)
/// - You need explicit storage of all boundary and coboundary relationships
/// - The complex is the output of a Morse reduction or other algorithm
///
/// Prefer specialized, implicit structures, such as [`CubicalComplex`], when
/// possible.
///
/// # Examples
///
/// Creating a line segment (two vertices connected by one edge):
///
/// ```rust
/// use chomp3rs::{CellComplex, Chain, Complex, Cyclic, Ring};
///
/// // Cells: 0 = vertex, 1 = vertex, 2 = edge
/// let cell_dimensions = vec![0, 0, 1];
/// let grades = vec![0, 0, 0];
///
/// let mut boundaries: Vec<Chain<u32, Cyclic<5>>> = vec![Chain::new(); 3];
/// // Edge boundary: vertex1 - vertex0
/// boundaries[2].insert_or_add(1, Cyclic::<5>::one());
/// boundaries[2].insert_or_add(0, -Cyclic::<5>::one());
///
/// let mut coboundaries: Vec<Chain<u32, Cyclic<5>>> = vec![Chain::new(); 3];
/// coboundaries[0].insert_or_add(2, -Cyclic::<5>::one());
/// coboundaries[1].insert_or_add(2, Cyclic::<5>::one());
///
/// let complex =
///     CellComplex::new(cell_dimensions, grades, boundaries, coboundaries);
///
/// assert_eq!(complex.dimension(), 1);
/// assert_eq!(complex.cell_count(), 3);
/// assert_eq!(complex.cell_dimension(&2), 1);
///
/// let edge_boundary = complex.cell_boundary(&2);
/// let boundary_of_boundary = complex.boundary(&edge_boundary);
/// assert_eq!(boundary_of_boundary, Chain::new());
/// ```
///
/// [`CubicalComplex`]: crate::CubicalComplex
#[derive(Debug, Clone)]
#[cfg_attr(feature = "mpi", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "mpi",
    serde(bound(
        serialize = "R: serde::Serialize",
        deserialize = "R: serde::de::DeserializeOwned"
    ))
)]
pub struct CellComplex<R: Ring> {
    complex_dimension: u32,
    cell_dimensions: Vec<u32>,
    grades: Vec<u32>,
    boundaries: Vec<Chain<u32, R>>,
    coboundaries: Vec<Chain<u32, R>>,
}

impl<R: Ring> CellComplex<R> {
    /// Creates a new cell complex from the provided data vectors. Each vector
    /// must have the same length, which represents the number of cells in
    /// the complex. The dimension of the cell complex is determined by the
    /// maximum dimension of the cells, or 0 if there are no cells.
    ///
    /// # Panics
    ///
    /// If all four input vectors are not the same length.
    #[must_use]
    pub fn new(
        cell_dimensions: Vec<u32>,
        grades: Vec<u32>,
        boundaries: Vec<Chain<u32, R>>,
        coboundaries: Vec<Chain<u32, R>>,
    ) -> Self {
        let cell_count = cell_dimensions.len();
        assert_eq!(cell_count, grades.len(), "cell count mismatch");
        assert_eq!(cell_count, boundaries.len(), "cell count mismatch");
        assert_eq!(cell_count, coboundaries.len(), "cell count mismatch");

        Self {
            complex_dimension: cell_dimensions.iter().max().copied().unwrap_or_default(),
            cell_dimensions,
            grades,
            boundaries,
            coboundaries,
        }
    }

    /// Returns the number of cells in the complex.
    #[must_use]
    pub fn cell_count(&self) -> u32 {
        self.cell_dimensions.len() as u32
    }
}

impl<R: Ring> Complex for CellComplex<R> {
    type Cell = u32;
    type Ring = R;

    fn cell_boundary(&self, cell: &u32) -> Chain<u32, R> {
        self.boundaries[*cell as usize].clone()
    }

    fn cell_coboundary(&self, cell: &u32) -> Chain<u32, R> {
        self.coboundaries[*cell as usize].clone()
    }

    fn iter(&self) -> impl Iterator<Item = u32> {
        0..self.grades.len() as u32
    }

    fn cell_dimension(&self, cell: &u32) -> u32 {
        self.cell_dimensions[*cell as usize]
    }

    fn dimension(&self) -> u32 {
        self.complex_dimension
    }
}

impl<R: Ring> Grader<u32> for CellComplex<R> {
    fn grade(&self, cell: &u32) -> u32 {
        self.grades[*cell as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cyclic, Ring};

    fn create_line_segment_complex() -> CellComplex<Cyclic<5>> {
        // Two vertices (cells 0, 1) and one edge (cell 2)
        let cell_dimensions = vec![0, 0, 1]; // vertex, vertex, edge
        let grades = vec![0, 0, 0];

        let mut boundaries = vec![Chain::new(); 3];
        // Boundary of edge (cell 2) is vertex1 - vertex0
        boundaries[2].insert_or_add(1, Cyclic::one());
        boundaries[2].insert_or_add(0, -Cyclic::one());

        let mut coboundaries = vec![Chain::new(); 3];
        // Coboundary of vertex 0 includes the edge with coefficient -1
        coboundaries[0].insert_or_add(2, -Cyclic::one());
        // Coboundary of vertex 1 includes the edge with coefficient +1
        coboundaries[1].insert_or_add(2, Cyclic::one());

        CellComplex::new(cell_dimensions, grades, boundaries, coboundaries)
    }

    fn create_graded_triangle_complex() -> CellComplex<Cyclic<5>> {
        // 3 vertices (0,1,2), 3 edges (3,4,5), 1 triangle face (6)
        // The grades correspond to the dimensions of the cells
        let cell_dimensions = vec![0, 0, 0, 1, 1, 1, 2];
        let grades = vec![0, 0, 0, 1, 1, 1, 2];

        let mut boundaries = vec![Chain::new(); 7];
        boundaries[3].insert_or_add(1, Cyclic::one()); // edge 0->1
        boundaries[3].insert_or_add(0, -Cyclic::one());

        boundaries[4].insert_or_add(2, Cyclic::one()); // edge 1->2
        boundaries[4].insert_or_add(1, -Cyclic::one());

        boundaries[5].insert_or_add(0, Cyclic::one()); // edge 2->0
        boundaries[5].insert_or_add(2, -Cyclic::one());

        boundaries[6].insert_or_add(3, Cyclic::one()); // +edge(0->1)
        boundaries[6].insert_or_add(4, Cyclic::one()); // +edge(1->2)
        boundaries[6].insert_or_add(5, Cyclic::one()); // +edge(2->0)

        let mut coboundaries = vec![Chain::new(); 7];
        coboundaries[0].insert_or_add(3, -Cyclic::one());
        coboundaries[0].insert_or_add(5, Cyclic::one());

        coboundaries[1].insert_or_add(4, -Cyclic::one());
        coboundaries[1].insert_or_add(3, Cyclic::one());

        coboundaries[2].insert_or_add(5, -Cyclic::one());
        coboundaries[2].insert_or_add(4, Cyclic::one());

        coboundaries[3].insert_or_add(6, Cyclic::one());
        coboundaries[4].insert_or_add(6, Cyclic::one());
        coboundaries[5].insert_or_add(6, Cyclic::one());

        CellComplex::new(cell_dimensions, grades, boundaries, coboundaries)
    }

    #[test]
    #[should_panic(expected = "cell count mismatch")]
    fn constructor_panics_on_mismatched_grades_length() {
        let cell_dimensions = vec![0, 1, 2];
        let grades = vec![0, 1];
        let boundaries = vec![Chain::new(); 3];
        let coboundaries = vec![Chain::new(); 3];

        let _ = CellComplex::<Cyclic<5>>::new(cell_dimensions, grades, boundaries, coboundaries);
    }

    #[test]
    fn line_segment_structure() {
        let complex = create_line_segment_complex();
        assert_eq!(complex.dimension(), 1);
        assert_eq!(complex.cell_count(), 3);

        // Check cell dimensions
        assert_eq!(complex.cell_dimension(&0), 0);
        assert_eq!(complex.cell_dimension(&1), 0);
        assert_eq!(complex.cell_dimension(&2), 1);

        // Check grades
        assert_eq!(complex.grade(&0), 0);
        assert_eq!(complex.grade(&1), 0);
        assert_eq!(complex.grade(&2), 0);

        // Check boundary of edge
        let edge_boundary = complex.cell_boundary(&2);
        assert_eq!(edge_boundary.coefficient(&0), -Cyclic::one());
        assert_eq!(edge_boundary.coefficient(&1), Cyclic::one());

        let cells: Vec<_> = complex.iter().collect();
        assert_eq!(cells, vec![0, 1, 2]);
    }

    #[test]
    fn triangle_structure() {
        let complex = create_graded_triangle_complex();
        assert_eq!(complex.dimension(), 2);
        assert_eq!(complex.cell_count(), 7);

        // Verify we have all expected cells
        let cells: Vec<_> = complex.iter().collect();
        assert_eq!(cells, vec![0, 1, 2, 3, 4, 5, 6]);

        // Check dimensions of different cell types
        for i in 0..3 {
            assert_eq!(complex.cell_dimension(&i), 0);
        } // vertices
        for i in 3..6 {
            assert_eq!(complex.cell_dimension(&i), 1);
        } // edges
        assert_eq!(complex.cell_dimension(&6), 2); // face
    }

    #[test]
    fn chain_boundary_computation() {
        let complex = create_graded_triangle_complex();

        // Create a chain that is a linear combination of edges: 2*edge3 + 3*edge4 -
        // edge5
        let mut edge_chain: Chain<u32, Cyclic<5>> = Chain::new();
        edge_chain.insert_or_add(3, Cyclic::from(2)); // 2 * edge(0->1)
        edge_chain.insert_or_add(4, Cyclic::from(3)); // 3 * edge(1->2)
        edge_chain.insert_or_add(5, -Cyclic::one()); // -1 * edge(2->0)

        let chain_boundary = complex.boundary(&edge_chain);

        // Expected boundary:
        // 2*(vertex1 - vertex0) + 3*(vertex2 - vertex1) - 1*(vertex0 - vertex2)
        // = -3*vertex0 - vertex1 + 4*vertex2
        let mut expected_boundary = Chain::new();
        expected_boundary.insert_or_add(0, Cyclic::from(2)); // -3 = 2 (mod 5)
        expected_boundary.insert_or_add(1, Cyclic::from(4)); // -1 = 4 (mod 5)
        expected_boundary.insert_or_add(2, Cyclic::from(4)); // 4 = 4 (mod 5)

        assert_eq!(chain_boundary, expected_boundary);
    }

    #[test]
    fn filtered_cell_boundary() {
        let complex = create_graded_triangle_complex();

        // Test boundary of edge 3 (vertex0 -> vertex1), filtering to only vertex1
        let edge_boundary: Chain<u32, Cyclic<5>> = complex
            .cell_boundary(&3)
            .into_iter()
            .filter(|(cell, _)| *cell == 1)
            .collect();

        let mut expected = Chain::new();
        expected.insert_or_add(1, Cyclic::one());

        assert_eq!(edge_boundary, expected);
    }

    #[test]
    fn filtered_boundary() {
        let complex = create_graded_triangle_complex();

        // Create a chain of edges: edge3 + edge4
        let mut edge_chain: Chain<u32, Cyclic<5>> = Chain::new();
        edge_chain.insert_or_add(3, Cyclic::one()); // edge(0->1)
        edge_chain.insert_or_add(4, Cyclic::one()); // edge(1->2)

        // Compute boundary, then filter to only vertices 0 and 2
        let chain_boundary: Chain<u32, Cyclic<5>> = complex
            .boundary(&edge_chain)
            .into_iter()
            .filter(|(cell, _)| *cell == 0 || *cell == 2)
            .collect();

        let mut expected = Chain::new();
        expected.insert_or_add(0, -Cyclic::one()); // From edge3 boundary
        expected.insert_or_add(2, Cyclic::one()); // From edge4 boundary

        assert_eq!(chain_boundary, expected);
    }

    #[test]
    fn cell_coboundary_methods() {
        let complex = create_graded_triangle_complex();

        // Test cell_coboundary for vertex 0
        let vertex0_coboundary = complex.cell_coboundary(&0);

        let mut expected = Chain::new();
        expected.insert_or_add(3, -Cyclic::one()); // edge3 with coefficient -1
        expected.insert_or_add(5, Cyclic::one()); // edge5 with coefficient +1

        assert_eq!(vertex0_coboundary, expected);
    }

    #[test]
    fn filtered_cell_coboundary() {
        let complex = create_graded_triangle_complex();

        // Test coboundary of vertex 0, filtering to only edge 3
        let vertex0_coboundary: Chain<u32, Cyclic<5>> = complex
            .cell_coboundary(&0)
            .into_iter()
            .filter(|(cell, _)| *cell == 3)
            .collect();

        let mut expected = Chain::new();
        expected.insert_or_add(3, -Cyclic::one());

        assert_eq!(vertex0_coboundary, expected);
    }

    #[test]
    fn filtered_coboundary() {
        let complex = create_graded_triangle_complex();

        // Create a chain of vertices: vertex0 + vertex1
        let mut vertex_chain: Chain<u32, Cyclic<5>> = Chain::new();
        vertex_chain.insert_or_add(0, Cyclic::one());
        vertex_chain.insert_or_add(1, Cyclic::one());

        // Compute coboundary, then filter to only edge 3
        let chain_coboundary: Chain<u32, Cyclic<5>> = complex
            .coboundary(&vertex_chain)
            .into_iter()
            .filter(|(cell, _)| *cell == 3)
            .collect();

        // vertex0 contributes -1 and vertex1 contributes +1 to edge3, cancelling
        assert_eq!(chain_coboundary, Chain::new());
    }

    #[test]
    fn chain_coboundary_computation() {
        let complex = create_graded_triangle_complex();

        // Create a chain: vertex0 + 2*vertex1 - vertex2
        let mut vertex_chain: Chain<u32, Cyclic<5>> = Chain::new();
        vertex_chain.insert_or_add(0, Cyclic::one());
        vertex_chain.insert_or_add(1, Cyclic::from(2));
        vertex_chain.insert_or_add(2, -Cyclic::one());

        let chain_coboundary = complex.coboundary(&vertex_chain);

        // Expected coboundary:
        // vertex0: -edge3 + edge5
        // vertex1: -edge4 + edge3 (multiplied by 2)
        // vertex2: -edge5 + edge4 (multiplied by -1)
        // Total: edge3 + 2*edge5 - 3*edge4
        let mut expected_coboundary = Chain::new();
        expected_coboundary.insert_or_add(3, Cyclic::one());
        expected_coboundary.insert_or_add(4, Cyclic::from(2)); // -3 = 2 (mod 5)
        expected_coboundary.insert_or_add(5, Cyclic::from(2));

        assert_eq!(chain_coboundary, expected_coboundary);
    }
}
