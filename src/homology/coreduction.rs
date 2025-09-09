// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::take;

use crate::homology::util::LinkedList;
use crate::{ComplexLike, HashMapModule, MatchResult, ModuleLike, MorseMatching, RingLike};

/// A type implementing an acyclic partial matching (that is, satisfying
/// [`MorseMatching`]) based on alternately excising coreduction pairs and leaf
/// cells from the Hasse diagram of the complex.
///
/// This matching can be applied to any complex (satisfying [`ComplexLike`]) so
/// long as its cell type satisfies the `Hash` trait. The matches for each cell
/// are explicitly stored after computation indivdually querying the full
/// algorithm is inefficient; thus, the complex must be small enough that
/// explicit storage is possible.
///
/// In practice, this method is slower than specialized heuristics for creating
/// partial matching. It's redeeming trait, however, is its applicability to
/// general cell complexes. This matching is commonly computed for Morse
/// complexes obtained after using a more specialized partial matching for the
/// larger top-level complex.
///
/// The implementation is a reformulation of Algorithm 3.6 in Harker,
/// Mischaikow, Mrozek, Nanda, *Discrete Morse Theoretic Algorithms for
/// Computing Homology of Complexes and Maps.*
pub struct CoreductionMatching<C, M = HashMapModule<u32, <C as ComplexLike>::Ring>>
where
    C: ComplexLike,
    C::Cell: Hash,
{
    complex: C,
    critical_cells: Vec<C::Cell>,
    projection: HashMap<C::Cell, u32>,
    matches: HashMap<C::Cell, MatchResult<C::Cell, C::Ring, u32>>,
    lower_module_type: PhantomData<M>,
}

impl<C> CoreductionMatching<C, HashMapModule<u32, <C as ComplexLike>::Ring>>
where
    C: ComplexLike,
    C::Cell: Hash,
{
    /// Functionally identical to the `compute_matching` function of the
    /// [`MorseMatching`] trait implemented by this type, but the compiler will
    /// correctly infer the default module type when using this constructor.
    ///
    /// If the user wants to use a different module type, they can use the
    /// `compute_matching` constructor with type annotations.
    pub fn new(complex: C) -> Self {
        Self::compute_matching(complex)
    }
}

impl<C, M> MorseMatching for CoreductionMatching<C, M>
where
    C: ComplexLike,
    C::Cell: Hash,
    M: ModuleLike<Cell = u32, Ring = C::Ring>,
{
    type LowerModule = M;
    type Priority = u32;
    type Ring = <C as ComplexLike>::Ring;
    type UpperCell = <C as ComplexLike>::Cell;
    type UpperComplex = C;
    type UpperModule = <C as ComplexLike>::Module;

    fn compute_matching(complex: Self::UpperComplex) -> Self {
        let (critical_cells, matches) = CoreductionMatchingImpl::compute_matching(&complex);
        let mut projection = HashMap::with_capacity(critical_cells.len());
        for (lower_cell, upper_cell) in critical_cells.iter().enumerate() {
            projection.insert(upper_cell.clone(), lower_cell as u32);
        }
        Self {
            complex,
            critical_cells,
            projection,
            matches,
            lower_module_type: PhantomData,
        }
    }

    fn get_upper_complex(&self) -> &Self::UpperComplex {
        &self.complex
    }

    fn extract_upper_complex(self) -> Self::UpperComplex {
        self.complex
    }

    fn critical_cells(&self) -> Vec<Self::UpperCell> {
        self.critical_cells.clone()
    }

    fn project_cell(&self, cell: Self::UpperCell) -> Option<u32> {
        self.projection.get(&cell).copied()
    }

    fn include_cell(&self, cell: u32) -> Self::UpperCell {
        self.critical_cells[cell as usize].clone()
    }

    fn match_cell(
        &self,
        cell: &Self::UpperCell,
    ) -> MatchResult<Self::UpperCell, Self::Ring, Self::Priority> {
        self.matches[cell].clone()
    }
}

struct CoreductionNode<T> {
    cofaces: LinkedList,
    faces: LinkedList,
    leaf_index: Option<usize>,
    cell: T,
}

struct CoreductionFace<R> {
    parent: usize,
    parent_face_index: usize,
    child: usize,
    child_coface_index: usize,
    incidence: R,
}

struct CoreductionMatchingImpl<C>
where
    C: ComplexLike,
    C::Cell: Hash,
{
    critical_cells: Vec<C::Cell>,
    matches: HashMap<C::Cell, MatchResult<C::Cell, C::Ring, u32>>,
    indices: HashMap<C::Cell, u32>,
    leaves: LinkedList,
    nodes: Vec<CoreductionNode<C::Cell>>,
    faces: Vec<CoreductionFace<C::Ring>>,
}

impl<C> CoreductionMatchingImpl<C>
where
    C: ComplexLike,
    C::Cell: Hash,
{
    #[allow(clippy::type_complexity)]
    fn compute_matching(
        complex: &C,
    ) -> (
        Vec<C::Cell>,
        HashMap<C::Cell, MatchResult<C::Cell, C::Ring, u32>>,
    ) {
        // Map each cell in `complex` to its index in the iterator from
        // `complex.cell_iter()`.
        // These indices are used to store the cells
        // contiguously for efficient lookup/manipulations during matching.
        let mut indices = HashMap::new();
        for (index, cell) in complex.cell_iter().enumerate() {
            indices.insert(cell, index as u32);
        }
        let cell_count = indices.len();

        // The CoreductionNode class stores a (contiguous) linked list of faces
        // and cofaces as well as the actual cell.
        // We use a linked list so items can be removed in constant time (with
        // a known index of the item). The removed items are connections to
        // cells that have been excised.
        // If the cell is a leaf (no faces remaining), it is stored in the
        // `leaves` linked list; `leaf_index` is its index in `leaves`, if it is
        // present. Should the cell be excised as a coreduction pair, this
        // allows it to be removed from the `leaves` linked list.
        let mut nodes = Vec::with_capacity(cell_count);
        for cell in complex.cell_iter() {
            debug_assert_eq!(indices[&cell] as usize, nodes.len());
            nodes.push(CoreductionNode {
                cofaces: LinkedList::new(),
                faces: LinkedList::new(),
                leaf_index: None,
                cell,
            });
        }

        let mut matching = Self {
            critical_cells: Vec::new(),
            matches: HashMap::with_capacity(cell_count),
            indices,
            leaves: LinkedList::with_capacity(cell_count),
            nodes,
            faces: Vec::new(),
        };

        // Initial coreduction pairs from construction; these are processed
        // first before proceeding to the main (excise leaf, excise coreduction
        // pairs) loop below.
        // It is possible that the paired cell to a cell in this iterator has
        // already been excised; thus, we check that it still has exactly one
        // face.
        for upper_index in matching.initialize_hasse(complex) {
            debug_assert!(
                matching.nodes[upper_index].faces.len() <= 1,
                "initial coreduction pair with greater than one face"
            );

            if matching.nodes[upper_index].faces.len() == 1 {
                matching.match_pair(upper_index);
            }
        }

        while let Some(node_index) = matching.leaves.pop_front() {
            debug_assert!(
                matching.nodes[node_index].faces.is_empty(),
                "cell in leaves list with nonzero number of faces"
            );

            matching.excise_leaf(node_index);
        }

        // Check that all cells are matched, and that their classification is
        // consistent between `matching.critical_cells` and `matching.matches`
        #[cfg(debug_assertions)]
        {
            for cell in complex.cell_iter() {
                let match_result = matching.matches.get(&cell).expect("cell not matched");
                let critical = matching.critical_cells.contains(&cell);
                debug_assert!(!(critical ^ matches!(match_result, MatchResult::Ace { .. })));
            }
        }

        // Check that matchings are consistent, queens <=> kings and aces with
        // themselves.
        #[cfg(debug_assertions)]
        {
            for cell in complex.cell_iter() {
                match &matching.matches[&cell] {
                    MatchResult::King {
                        cell: king,
                        queen,
                        incidence,
                        priority,
                    } => {
                        debug_assert_eq!(cell, *king);
                        debug_assert_eq!(
                            matching.matches[queen],
                            MatchResult::Queen {
                                cell: queen.clone(),
                                king: king.clone(),
                                incidence: incidence.clone(),
                                priority: (priority + 1)
                            }
                        );
                    }
                    MatchResult::Queen {
                        cell: queen,
                        king,
                        incidence,
                        priority,
                    } => {
                        debug_assert_eq!(cell, *queen);
                        debug_assert_eq!(
                            matching.matches[king],
                            MatchResult::King {
                                cell: king.clone(),
                                queen: queen.clone(),
                                incidence: incidence.clone(),
                                priority: (priority - 1)
                            }
                        );
                    }
                    MatchResult::Ace { cell: ace } => {
                        debug_assert_eq!(cell, *ace);
                    }
                };
            }
        }

        (matching.critical_cells, matching.matches)
    }

    fn initialize_hasse(&mut self, complex: &C) -> Vec<usize> {
        let mut initial_pairs = Vec::new();
        for (index, cell) in complex.cell_iter().enumerate() {
            let grade = complex.grade(&cell);

            // All faces with nonzero incidence and identical grade
            for (bd_cell, bd_coef) in
                complex
                    .cell_boundary(&cell)
                    .into_iter()
                    .filter(|(bd_cell, bd_coef)| {
                        *bd_coef != C::Ring::zero() && complex.grade(bd_cell) == grade
                    })
            {
                let bd_index = self.indices[&bd_cell] as usize;
                self.faces.push(CoreductionFace {
                    parent: index,
                    parent_face_index: self.nodes[index].faces.len(),
                    child: bd_index,
                    child_coface_index: self.nodes[bd_index].cofaces.len(),
                    incidence: bd_coef,
                });

                self.nodes[index].faces.push_back(self.faces.len() - 1);
                self.nodes[bd_index].cofaces.push_back(self.faces.len() - 1);
            }

            if self.nodes[index].faces.is_empty() {
                self.nodes[index].leaf_index = Some(self.leaves.push_back(index));
            } else if self.nodes[index].faces.len() == 1 {
                initial_pairs.push(index);
            }
        }

        initial_pairs
    }

    fn match_pair(&mut self, upper_index: usize) {
        debug_assert!(
            self.nodes[upper_index].faces.len() == 1,
            "attempting to match a pair that is not a coreduction pair"
        );

        let upper_face_index = self.nodes[upper_index]
            .faces
            .peek_front()
            .expect("Attempting to match leaf instead of coreduction pair");
        if !self.faces[upper_face_index].incidence.is_invertible() {
            return;
        }

        let lower_index = self.faces[upper_face_index].child;
        if let Some(leaf_index) = self.nodes[lower_index].leaf_index {
            let popped_index = self.leaves.pop(leaf_index);
            debug_assert_eq!(popped_index, Some(lower_index));
        }

        self.excise_pair(upper_index, lower_index);

        self.matches.insert(
            self.nodes[upper_index].cell.clone(),
            MatchResult::King {
                cell: self.nodes[upper_index].cell.clone(),
                queen: self.nodes[lower_index].cell.clone(),
                incidence: self.faces[upper_face_index].incidence.clone(),
                priority: self.matches.len() as u32,
            },
        );
        self.matches.insert(
            self.nodes[lower_index].cell.clone(),
            MatchResult::Queen {
                cell: self.nodes[lower_index].cell.clone(),
                king: self.nodes[upper_index].cell.clone(),
                incidence: self.faces[upper_face_index].incidence.clone(),
                priority: self.matches.len() as u32,
            },
        );
    }

    fn excise_pair(&mut self, upper_index: usize, lower_index: usize) {
        debug_assert!(
            self.nodes[upper_index].faces.len() == 1,
            "attempting to excise a pair that is not a coreduction pair"
        );

        // Remove single edge connecting upper to lower from both
        let upper_face_index = self.nodes[upper_index]
            .faces
            .pop_front()
            .expect("Attempted to excise leaf instead of coreduction pair");
        debug_assert!(self.nodes[upper_index].faces.is_empty());
        self.nodes[lower_index]
            .cofaces
            .pop(self.faces[upper_face_index].child_coface_index);

        // Remove the face connections of the lower node. Does not alter the
        // boundary counts of other cells, so no further work is needed.
        while let Some(lower_face_index) = self.nodes[lower_index].faces.pop_front() {
            let lower_face = &self.faces[lower_face_index];
            debug_assert_eq!(lower_face.parent, lower_index);

            let child_coface_index = self.nodes[lower_face.child]
                .cofaces
                .pop(lower_face.child_coface_index);
            debug_assert_eq!(
                child_coface_index.expect("face of cell not a coface of its child"),
                lower_face_index
            );
        }

        // Remove the coface connections of the lower node. This can create
        // leaves by removing the last boundary face of the coface.
        // Take coface list to satisfy borrow checker on `self.nodes`; if will
        // be emptied anyway.
        let mut lower_node_cofaces = take(&mut self.nodes[lower_index].cofaces);
        for lower_coface_index in lower_node_cofaces.iter() {
            let lower_coface = &self.faces[lower_coface_index];
            debug_assert_eq!(lower_coface.child, lower_index);

            let parent_face_index = self.nodes[lower_coface.parent]
                .faces
                .pop(lower_coface.parent_face_index);
            debug_assert_eq!(
                parent_face_index.expect("coface of cell is not a face of its parent"),
                lower_coface_index
            );

            if self.nodes[lower_coface.parent].faces.is_empty() {
                debug_assert!(self.nodes[lower_coface.parent].leaf_index.is_none());
                self.nodes[lower_coface.parent].leaf_index =
                    Some(self.leaves.push_back(lower_coface.parent));
            }
        }

        // Remove the coface connections of the upper node. This can create
        // leaves by removing the last boundary face of the coface.
        // Take coface list to satisfy borrow checker on `self.nodes`; it will
        // be emptied anyway.
        let mut upper_node_cofaces = take(&mut self.nodes[upper_index].cofaces);
        for upper_coface_index in upper_node_cofaces.iter() {
            let upper_coface = &self.faces[upper_coface_index];
            debug_assert_eq!(upper_coface.child, upper_index);

            let parent_face_index = self.nodes[upper_coface.parent]
                .faces
                .pop(upper_coface.parent_face_index);
            debug_assert_eq!(
                parent_face_index.expect("coface of cell is not a face of its parent"),
                upper_coface_index
            );

            if self.nodes[upper_coface.parent].faces.is_empty() {
                debug_assert!(self.nodes[upper_coface.parent].leaf_index.is_none());
                self.nodes[upper_coface.parent].leaf_index =
                    Some(self.leaves.push_back(upper_coface.parent));
            }
        }

        // At this point, the graph is correct for having excised the pair. The
        // remaining step is to identify nodes that have been reduced to having
        // one face, and match them.

        while let Some(lower_coface_index) = lower_node_cofaces.pop_front() {
            let lower_coface = &self.faces[lower_coface_index];
            if self.nodes[lower_coface.parent].faces.len() == 1 {
                self.match_pair(lower_coface.parent);
            }
        }

        while let Some(upper_coface_index) = upper_node_cofaces.pop_front() {
            let upper_coface = &self.faces[upper_coface_index];
            if self.nodes[upper_coface.parent].faces.len() == 1 {
                self.match_pair(upper_coface.parent);
            }
        }
    }

    fn excise_leaf(&mut self, node_index: usize) {
        debug_assert!(
            self.nodes[node_index].faces.is_empty(),
            "attempting to excise non-leaf cell"
        );

        // Remove the coface connections of the leaf. This can create
        // further leaves by removing the last boundary face of the coface.
        // Take coface list to satisfy borrow checker on `self.nodes`; it will
        // be emptied anyway.
        let mut node_cofaces = take(&mut self.nodes[node_index].cofaces);
        for coface_index in node_cofaces.iter() {
            let coface = &self.faces[coface_index];
            debug_assert_eq!(coface.child, node_index);

            let parent_face_index = self.nodes[coface.parent]
                .faces
                .pop(coface.parent_face_index);
            debug_assert_eq!(
                parent_face_index.expect("coface of cell is not a face of its parent"),
                coface_index
            );

            if self.nodes[coface.parent].faces.is_empty() {
                debug_assert!(self.nodes[coface.parent].leaf_index.is_none());
                self.nodes[coface.parent].leaf_index = Some(self.leaves.push_back(coface.parent));
            }
        }

        while let Some(coface_index) = node_cofaces.pop_front() {
            let coface = &self.faces[coface_index];
            if self.nodes[coface.parent].faces.len() == 1 {
                self.match_pair(coface.parent);
            }
        }

        self.critical_cells
            .push(self.nodes[node_index].cell.clone());
        self.matches.insert(
            self.nodes[node_index].cell.clone(),
            MatchResult::Ace {
                cell: self.nodes[node_index].cell.clone(),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fs;

    use super::*;
    use crate::{
        CellComplex, Cube, CubicalComplex, Cyclic, HashMapGrader, HashMapModule, ModuleLike,
        Orthant, RingLike, TopCubeGrader,
    };

    type TestModule = HashMapModule<u32, Cyclic<5>>;
    type TestCubicalModule = HashMapModule<Cube, Cyclic<5>>;

    #[test]
    fn test_line_segment_cell_complex() {
        // Test a line segment: two vertices and one edge
        // The edge should form a coreduction pair with one vertex
        let cell_dimensions = vec![0, 0, 1]; // vertex0, vertex1, edge
        let grades = vec![0, 0, 0];

        let mut boundaries = vec![TestModule::new(), TestModule::new(), TestModule::new()];
        // Edge boundary: vertex1 - vertex0
        boundaries[2].insert_or_add(1, Cyclic::one());
        boundaries[2].insert_or_add(0, -Cyclic::one());

        let mut coboundaries = vec![TestModule::new(), TestModule::new(), TestModule::new()];
        // Vertex coboundaries
        coboundaries[0].insert_or_add(2, -Cyclic::one());
        coboundaries[1].insert_or_add(2, Cyclic::one());

        let complex = CellComplex::new(cell_dimensions, grades, boundaries, coboundaries);
        let matching = CoreductionMatching::new(complex);

        // Should have one critical cell (contractible to a point)
        assert_eq!(matching.critical_cells().len(), 1);

        // Count the types of matches
        let mut aces = 0;
        let mut kings = 0;
        let mut queens = 0;

        for cell in 0..3 {
            match matching.match_cell(&cell) {
                MatchResult::Ace { .. } => aces += 1,
                MatchResult::King { .. } => kings += 1,
                MatchResult::Queen { .. } => queens += 1,
            }
        }

        // Should have 1 ace, 1 king-queen pair
        assert_eq!(aces, 1);
        assert_eq!(kings, 1);
        assert_eq!(queens, 1);
    }

    #[test]
    fn test_triangle_cell_complex() {
        // Test a triangle: 3 vertices, 3 edges, 1 face
        let cell_dimensions = vec![0, 0, 0, 1, 1, 1, 2]; // 3 vertices, 3 edges, 1 face
        let grades = vec![0, 0, 0, 0, 0, 0, 0];

        let mut boundaries = vec![TestModule::new(); 7];
        // Edge boundaries
        boundaries[3].insert_or_add(1, Cyclic::one()); // edge 0->1
        boundaries[3].insert_or_add(0, -Cyclic::one());

        boundaries[4].insert_or_add(2, Cyclic::one()); // edge 1->2
        boundaries[4].insert_or_add(1, -Cyclic::one());

        boundaries[5].insert_or_add(0, Cyclic::one()); // edge 2->0
        boundaries[5].insert_or_add(2, -Cyclic::one());

        // Face boundary
        boundaries[6].insert_or_add(3, Cyclic::one());
        boundaries[6].insert_or_add(4, Cyclic::one());
        boundaries[6].insert_or_add(5, Cyclic::one());

        let mut coboundaries = vec![TestModule::new(); 7];
        // Vertex coboundaries
        coboundaries[0].insert_or_add(3, -Cyclic::one());
        coboundaries[0].insert_or_add(5, Cyclic::one());

        coboundaries[1].insert_or_add(4, -Cyclic::one());
        coboundaries[1].insert_or_add(3, Cyclic::one());

        coboundaries[2].insert_or_add(5, -Cyclic::one());
        coboundaries[2].insert_or_add(4, Cyclic::one());

        // Edge coboundaries
        coboundaries[3].insert_or_add(6, Cyclic::one());
        coboundaries[4].insert_or_add(6, Cyclic::one());
        coboundaries[5].insert_or_add(6, Cyclic::one());

        let complex = CellComplex::new(cell_dimensions, grades, boundaries, coboundaries);
        let matching = CoreductionMatching::new(complex);

        // Triangle is contractible, should reduce to single point
        assert_eq!(matching.critical_cells().len(), 1);

        // Verify all cells are either critical or matched
        let critical_set: HashSet<u32> = matching.critical_cells().iter().cloned().collect();
        let mut matched_count = 0;

        for cell in 0..7 {
            match matching.match_cell(&cell) {
                MatchResult::Ace { .. } => {
                    assert!(critical_set.contains(&cell));
                }
                MatchResult::King { incidence, .. } => {
                    assert!(incidence.is_invertible());
                    matched_count += 1;
                }
                MatchResult::Queen { incidence, .. } => {
                    assert!(incidence.is_invertible());
                    matched_count += 1;
                }
            }
        }

        assert_eq!(critical_set.len() + matched_count, 7);
    }

    #[test]
    fn test_line_segment_cubical_complex() {
        // Create a 1D line segment from [0] to [1]
        let min = Orthant::new(vec![0]);
        let max = Orthant::new(vec![1]);

        let cells = vec![
            Cube::vertex(Orthant::new(vec![0])),
            Cube::vertex(Orthant::new(vec![1])),
            Cube::from_extent(Orthant::new(vec![0]), &[true]),
        ];
        let grader = HashMapGrader::uniform(cells.into_iter(), 0, 1);
        let complex: CubicalComplex<TestCubicalModule, _> = CubicalComplex::new(min, max, grader);

        let matching = CoreductionMatching::new(complex);

        // Line segment should contract to a point (complex of interest is grade 0)
        assert_eq!(
            matching
                .critical_cells()
                .iter()
                .filter(|cell| matching.get_upper_complex().grade(cell) == 0)
                .count(),
            1
        );

        // Verify the edge forms a coreduction pair with one of its boundary vertices
        let edge = Cube::from_extent(Orthant::new(vec![0]), &[true]);
        let vertex0 = Cube::vertex(Orthant::new(vec![0]));
        let vertex1 = Cube::vertex(Orthant::new(vec![1]));

        let mut found_king_queen_pair = false;
        let cells = vec![vertex0.clone(), vertex1.clone(), edge.clone()];

        for cell in &cells {
            match matching.match_cell(cell) {
                MatchResult::King { queen, .. } => {
                    assert!(cells.contains(&queen));
                    found_king_queen_pair = true;
                }
                MatchResult::Queen { king, .. } => {
                    assert!(cells.contains(&king));
                    found_king_queen_pair = true;
                }
                MatchResult::Ace { .. } => {}
            }
        }

        assert!(
            found_king_queen_pair,
            "Expect to find at least one king-queen pair"
        );
    }

    #[test]
    fn test_empty_square_cubical_complex() {
        // Create a cubical complex on 2x2 grid of orthants
        let min = Orthant::new(vec![0, 0]);
        let max = Orthant::new(vec![2, 1]);

        // Empty square in orthants (0, 0) to (1, 1), with an extra edge
        let cells = [
            Cube::vertex(Orthant::new(vec![0, 0])),
            Cube::vertex(Orthant::new(vec![1, 0])),
            Cube::vertex(Orthant::new(vec![0, 1])),
            Cube::vertex(Orthant::new(vec![1, 1])),
            Cube::vertex(Orthant::new(vec![2, 1])),
            Cube::from_extent(Orthant::new(vec![0, 0]), &[true, false]),
            Cube::from_extent(Orthant::new(vec![1, 0]), &[false, true]),
            Cube::from_extent(Orthant::new(vec![0, 1]), &[true, false]),
            Cube::from_extent(Orthant::new(vec![0, 0]), &[false, true]),
            Cube::from_extent(Orthant::new(vec![1, 1]), &[true, false]),
        ];

        let grader = HashMapGrader::uniform(cells.into_iter(), 0, 1);
        let complex: CubicalComplex<TestCubicalModule, _> = CubicalComplex::new(min, max, grader);

        let matching = CoreductionMatching::new(complex.clone());

        // Empty square has 1 dim 0 critical cell and 1 dim 1 critical cell
        // in grade 0.
        assert_eq!(
            matching
                .critical_cells
                .iter()
                .filter_map(|cell| {
                    if matching.get_upper_complex().grade(cell) == 0 {
                        Some(cell.dimension())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>(),
            vec![0, 1]
        );

        // Verify that the matching preserves the total number of cells
        let all_cells: Vec<_> = complex.cell_iter().collect();
        let critical_count = matching.critical_cells().len();
        let mut matched_count = 0;

        for cell in &all_cells {
            match matching.match_cell(cell) {
                MatchResult::Ace { .. } => {}
                MatchResult::King { .. } | MatchResult::Queen { .. } => {
                    matched_count += 1;
                }
            }
        }

        assert_eq!(critical_count + matched_count, all_cells.len());
    }

    #[test]
    fn full_reduce_triangle_complex() {
        let serialized_complex = fs::read_to_string("testing/complexes/triangle_complex.json")
            .expect("Testing complex file not found.");
        let complex: CellComplex<HashMapModule<u32, Cyclic<2>>> =
            serde_json::from_str(&serialized_complex)
                .expect("Testing complex could not be deserialized.");
        let (_top_matching, _further_matchings, morse_complex) = CoreductionMatching::full_reduce::<
            CoreductionMatching<CellComplex<HashMapModule<u32, Cyclic<2>>>>,
        >(complex);

        let mut cells_by_dimension = [Vec::new(), Vec::new(), Vec::new()];
        for cell in morse_complex.cell_iter() {
            if morse_complex.grade(&cell) == 0 {
                cells_by_dimension[morse_complex.cell_dimension(&cell) as usize].push(cell);
                assert_eq!(morse_complex.cell_boundary(&cell), HashMapModule::new());
            }
        }
        assert_eq!(cells_by_dimension[0].len(), 1, "0-dimensional cells");
        assert_eq!(cells_by_dimension[1].len(), 0, "1-dimensional cells");
        assert_eq!(cells_by_dimension[2].len(), 0, "2-dimensional cells");
    }

    #[test]
    fn full_reduce_cube_torus_complex() {
        let serialized_complex = fs::read_to_string("testing/complexes/cube_torus_complex.json")
            .expect("Testing complex file not found.");
        let complex: CubicalComplex<
            HashMapModule<Cube, Cyclic<2>>,
            TopCubeGrader<HashMapGrader<Orthant>>,
        > = serde_json::from_str(&serialized_complex)
            .expect("Testing complex could not be deserialized.");
        let (_top_matching, _further_matchings, morse_complex) = CoreductionMatching::full_reduce::<
            CoreductionMatching<CellComplex<HashMapModule<u32, Cyclic<2>>>>,
        >(complex);

        let mut cells_by_dimension = [Vec::new(), Vec::new(), Vec::new()];
        for cell in morse_complex.cell_iter() {
            if morse_complex.grade(&cell) == 0 {
                cells_by_dimension[morse_complex.cell_dimension(&cell) as usize].push(cell);
                assert_eq!(morse_complex.cell_boundary(&cell), HashMapModule::new());
            }
        }
        assert_eq!(cells_by_dimension[0].len(), 1, "0-dimensional cells");
        assert_eq!(cells_by_dimension[1].len(), 2, "1-dimensional cells");
        assert_eq!(cells_by_dimension[2].len(), 1, "2-dimensional cells");
    }

    #[test]
    fn full_reduce_figure_eight_complex() {
        let serialized_complex = fs::read_to_string("testing/complexes/figure_eight_complex.json")
            .expect("Testing complex file not found.");
        let complex: CubicalComplex<HashMapModule<Cube, Cyclic<2>>, HashMapGrader<Cube>> =
            serde_json::from_str(&serialized_complex)
                .expect("Testing complex could not be deserialized.");
        let (_top_matching, _further_matchings, morse_complex) = CoreductionMatching::full_reduce::<
            CoreductionMatching<CellComplex<HashMapModule<u32, Cyclic<2>>>>,
        >(complex);

        let mut cells_by_dimension = [Vec::new(), Vec::new(), Vec::new()];
        for cell in morse_complex.cell_iter() {
            if morse_complex.grade(&cell) == 0 {
                cells_by_dimension[morse_complex.cell_dimension(&cell) as usize].push(cell);
                assert_eq!(morse_complex.cell_boundary(&cell), HashMapModule::new());
            }
        }
        assert_eq!(cells_by_dimension[0].len(), 1, "0-dimensional cells");
        assert_eq!(cells_by_dimension[1].len(), 2, "1-dimensional cells");
        assert_eq!(cells_by_dimension[2].len(), 0, "2-dimensional cells");
    }
}
