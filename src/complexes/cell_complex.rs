// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::{ComplexLike, ModuleLike};

/// A simple concrete implementation of a cell complex with vector-based storage.
///
/// `CellComplex` represents a finite cell complex where cells are identified by `u32` indices.
/// Each cell has associated dimensional information, grade, boundary, and coboundary data
/// explicitly stored in parallel vectors for efficient access. However, there needs to be few
/// enough cells that the memory overhead of storing these vectors is acceptable.
pub struct CellComplex<M>
where
    M: ModuleLike<Cell = u32>,
{
    complex_dimension: u32,
    cell_dimensions: Vec<u32>,
    grades: Vec<u32>,
    boundaries: Vec<M>,
    coboundaries: Vec<M>,
}

pub struct CellRangeIterator {
    next: u32,
    end: u32,
}

impl Iterator for CellRangeIterator {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next < self.end {
            let current = self.next;
            self.next += 1;
            Some(current)
        } else {
            None
        }
    }
}

impl<M> CellComplex<M>
where
    M: ModuleLike<Cell = u32>,
{
    /// Creates a new cell complex from the provided data vectors. Each vector must have the same
    /// length, which represents the number of cells in the complex. The dimension of the cell
    /// complex is determined by the maximum dimension of the cells, or 0 if there are no cells.
    pub fn new(
        cell_dimensions: Vec<u32>,
        grades: Vec<u32>,
        boundaries: Vec<M>,
        coboundaries: Vec<M>,
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
}

impl<M> ComplexLike<M> for CellComplex<M>
where
    M: ModuleLike<Cell = u32>,
{
    type CellIterator = CellRangeIterator;

    fn cell_iter(&self) -> Self::CellIterator {
        CellRangeIterator {
            next: 0,
            end: self.grades.len() as u32,
        }
    }

    fn boundary(&self, cell: &u32) -> M {
        self.boundaries[*cell as usize].clone()
    }

    fn coboundary(&self, cell: &u32) -> M {
        self.coboundaries[*cell as usize].clone()
    }

    fn grade(&self, cell: &u32) -> u32 {
        self.grades[*cell as usize]
    }

    fn cell_dimension(&self, cell: &u32) -> u32 {
        self.cell_dimensions[*cell as usize]
    }

    fn dimension(&self) -> u32 {
        self.complex_dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cyclic, HashMapModule, ModuleLike, RingLike};

    // Type alias to make tests more readable and easily adaptable
    type TestModule = HashMapModule<u32, Cyclic<5>>;

    fn create_empty_complex() -> CellComplex<TestModule> {
        CellComplex::new(vec![], vec![], vec![], vec![])
    }

    fn create_single_vertex_complex() -> CellComplex<TestModule> {
        let cell_dimensions = vec![0]; // One 0-dimensional cell (vertex)
        let grades = vec![0]; // Grade 0
        let boundaries = vec![TestModule::new()];
        let coboundaries = vec![TestModule::new()];

        CellComplex::new(cell_dimensions, grades, boundaries, coboundaries)
    }

    fn create_line_segment_complex() -> CellComplex<TestModule> {
        // Two vertices (cells 0, 1) and one edge (cell 2)
        let cell_dimensions = vec![0, 0, 1]; // vertex, vertex, edge
        let grades = vec![0, 0, 0];

        let mut boundaries = vec![TestModule::new(), TestModule::new(), TestModule::new()];
        // Boundary of edge (cell 2) is vertex1 - vertex0
        boundaries[2].insert_or_add(&1, Cyclic::one());
        boundaries[2].insert_or_add(&0, -Cyclic::one());

        let mut coboundaries = vec![TestModule::new(), TestModule::new(), TestModule::new()];
        // Coboundary of vertex 0 includes the edge with coefficient -1
        coboundaries[0].insert_or_add(&2, -Cyclic::one());
        // Coboundary of vertex 1 includes the edge with coefficient +1
        coboundaries[1].insert_or_add(&2, Cyclic::one());

        CellComplex::new(cell_dimensions, grades, boundaries, coboundaries)
    }

    fn create_graded_triangle_complex() -> CellComplex<TestModule> {
        // 3 vertices (0,1,2), 3 edges (3,4,5), 1 triangle face (6)
        // The grades correspond to the dimensions of the cells
        let cell_dimensions = vec![0, 0, 0, 1, 1, 1, 2];
        let grades = vec![0, 0, 0, 1, 1, 1, 2];

        let mut boundaries = vec![TestModule::new(); 7];
        boundaries[3].insert_or_add(&1, Cyclic::one()); // edge 0->1
        boundaries[3].insert_or_add(&0, -Cyclic::one());

        boundaries[4].insert_or_add(&2, Cyclic::one()); // edge 1->2
        boundaries[4].insert_or_add(&1, -Cyclic::one());

        boundaries[5].insert_or_add(&0, Cyclic::one()); // edge 2->0
        boundaries[5].insert_or_add(&2, -Cyclic::one());

        boundaries[6].insert_or_add(&3, Cyclic::one()); // +edge(0->1)
        boundaries[6].insert_or_add(&4, Cyclic::one()); // +edge(1->2)
        boundaries[6].insert_or_add(&5, Cyclic::one()); // +edge(2->0)

        let mut coboundaries = vec![TestModule::new(); 7];
        coboundaries[0].insert_or_add(&3, -Cyclic::one());
        coboundaries[0].insert_or_add(&5, Cyclic::one());

        coboundaries[1].insert_or_add(&4, -Cyclic::one());
        coboundaries[1].insert_or_add(&3, Cyclic::one());

        coboundaries[2].insert_or_add(&5, -Cyclic::one());
        coboundaries[2].insert_or_add(&4, Cyclic::one());

        coboundaries[3].insert_or_add(&6, Cyclic::one());
        coboundaries[4].insert_or_add(&6, Cyclic::one());
        coboundaries[5].insert_or_add(&6, Cyclic::one());

        CellComplex::new(cell_dimensions, grades, boundaries, coboundaries)
    }

    #[test]
    #[should_panic(expected = "cell count mismatch")]
    fn test_constructor_panics_on_mismatched_grades_length() {
        let cell_dimensions = vec![0, 1, 2];
        let grades = vec![0, 1];
        let boundaries = vec![TestModule::new(); 3];
        let coboundaries = vec![TestModule::new(); 3];

        CellComplex::new(cell_dimensions, grades, boundaries, coboundaries);
    }

    #[test]
    fn test_empty_complex_construction() {
        let complex = create_empty_complex();
        assert_eq!(complex.dimension(), 0);

        let cells: Vec<_> = complex.cell_iter().collect();
        assert!(cells.is_empty());
    }

    #[test]
    fn test_single_vertex_construction() {
        let complex = create_single_vertex_complex();
        assert_eq!(complex.dimension(), 0);
        assert_eq!(complex.grade(&0), 0);
        assert_eq!(complex.cell_dimension(&0), 0);

        let boundary = complex.boundary(&0);
        assert_eq!(boundary, TestModule::new()); // Empty boundary

        let cells: Vec<_> = complex.cell_iter().collect();
        assert_eq!(cells, vec![0]);
    }

    #[test]
    fn test_line_segment_structure() {
        let complex = create_line_segment_complex();
        assert_eq!(complex.dimension(), 1);

        // Check cell dimensions
        assert_eq!(complex.cell_dimension(&0), 0);
        assert_eq!(complex.cell_dimension(&1), 0);
        assert_eq!(complex.cell_dimension(&2), 1);

        // Check grades
        assert_eq!(complex.grade(&0), 0);
        assert_eq!(complex.grade(&1), 0);
        assert_eq!(complex.grade(&2), 0);

        // Check boundary of edge
        let edge_boundary = complex.boundary(&2);
        assert_eq!(edge_boundary.coef(&0), -Cyclic::one());
        assert_eq!(edge_boundary.coef(&1), Cyclic::one());

        let cells: Vec<_> = complex.cell_iter().collect();
        assert_eq!(cells, vec![0, 1, 2]);
    }

    #[test]
    fn test_triangle_structure() {
        let complex = create_graded_triangle_complex();
        assert_eq!(complex.dimension(), 2);

        // Verify we have all expected cells
        let cells: Vec<_> = complex.cell_iter().collect();
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
    fn test_cell_iterator_completeness() {
        let complex = create_graded_triangle_complex();

        let cells: Vec<_> = complex.cell_iter().collect();

        // Should iterate through all cells exactly once
        assert_eq!(cells.len(), 7);
        for i in 0..7 {
            assert!(cells.contains(&i));
        }
    }
}
