// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Coreduction-based discrete Morse matching.
//!
//! This module provides [`CoreductionMatching`], a general-purpose algorithm
//! for computing acyclic partial matchings on arbitrary cell complexes. It
//! works by alternately excising coreduction pairs and leaf cells from the
//! Hasse diagram.
//!
//! For cubical complexes, prefer
//! [`TopCubicalMatching`](super::TopCubicalMatching) which exploits the grid
//! structure for better performance.

use std::{collections::HashMap, mem::take};

use tracing::{debug, info, trace};

use crate::{
    CellMatch, Complex, MorseMatching, Ring, homology::linked_list::LinkedList,
    logging::ProgressTracker,
};

/// A general-purpose acyclic partial matching based on coreduction.
///
/// This implementation computes a discrete Morse matching by alternately
/// excising coreduction pairs (cells with exactly one face) and leaf cells
/// (cells with no faces) from the Hasse diagram of the complex.
///
/// # When to Use
///
/// - **General cell complexes**: Works on any type implementing [`Complex`]
///   with hashable cells.
/// - **Morse complex reduction**: Used as the generic matching in
///   [`MorseMatching::full_reduce`] to further reduce Morse complexes.
/// - **Small to medium complexes**: Since matches are stored explicitly, the
///   complex must fit in memory.
///
/// # Algorithm
///
/// The algorithm maintains the Hasse diagram as a graph where:
/// - Nodes represent cells
/// - Edges represent face/coface relationships with nonzero incidence
///
/// It repeatedly:
/// 1. Identifies cells with exactly one face (coreduction pairs)
/// 2. Matches and excises these pairs
/// 3. Identifies cells with no faces (leaves)
/// 4. Marks leaves as critical (ace) cells and excises them
///
/// This process continues until all cells are either matched or marked
/// critical.
///
/// # Example
///
/// ```
/// use chomp3rs::{
///     CellComplex, Chain, CoreductionMatching, F2, MorseMatching, Ring,
/// };
///
/// // Create a line segment: two vertices (cells 0, 1) and one edge (cell 2)
/// let mut boundaries: Vec<Chain<u32, F2>> = vec![Chain::new(); 3];
/// boundaries[2].insert_or_add(1, F2::one()); // edge: v1 - v0
/// boundaries[2].insert_or_add(0, -F2::one());
///
/// let mut coboundaries: Vec<Chain<u32, F2>> = vec![Chain::new(); 3];
/// coboundaries[0].insert_or_add(2, -F2::one());
/// coboundaries[1].insert_or_add(2, F2::one());
///
/// let complex = CellComplex::new(
///     vec![0, 0, 1], // dimensions
///     vec![0, 0, 0], // grades
///     boundaries,
///     coboundaries,
/// );
///
/// let matching: CoreductionMatching<_> = CoreductionMatching::new(complex);
///
/// // Line segment is contractible -> one critical cell (a point)
/// assert_eq!(matching.critical_cells().len(), 1);
/// ```
///
/// # References
///
/// The implementation is based on Algorithm 3.6 in Harker, Mischaikow, Mrozek,
/// and Nanda, *Discrete Morse Theoretic Algorithms for Computing Homology of
/// Complexes and Maps*.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "C: serde::Serialize, C::Cell: serde::Serialize, C::Ring: serde::Serialize",
        deserialize = "C: serde::de::DeserializeOwned, C::Cell: serde::de::DeserializeOwned + \
                       std::hash::Hash, C::Ring: serde::de::DeserializeOwned"
    ))
)]
pub struct CoreductionMatching<C>
where
    C: Complex,
{
    complex: C,
    critical_cells: Vec<C::Cell>,
    projection: HashMap<C::Cell, u32>,
    matches: HashMap<C::Cell, CellMatch<C::Cell, C::Ring, u32>>,
}

impl<C> CoreductionMatching<C>
where
    C: Complex,
{
    /// Compute a coreduction matching on the given cell complex.
    ///
    /// Consumes `complex`, computes the acyclic partial matching immediately,
    /// and returns the completed matching. All [`MorseMatching`] trait methods
    /// are available on the returned value.
    #[must_use]
    pub fn new(complex: C) -> Self {
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
        }
    }
}

impl<C> MorseMatching for CoreductionMatching<C>
where
    C: Complex,
{
    type Priority = u32;
    type Ring = C::Ring;
    type UpperCell = C::Cell;
    type UpperComplex = C;

    fn upper_complex(&self) -> &Self::UpperComplex {
        &self.complex
    }

    fn into_upper_complex(self) -> Self::UpperComplex {
        self.complex
    }

    fn critical_cells(&self) -> &[Self::UpperCell] {
        &self.critical_cells
    }

    fn project_cell(&self, cell: &Self::UpperCell) -> Option<u32> {
        self.projection.get(cell).copied()
    }

    fn include_cell(&self, cell: u32) -> Self::UpperCell {
        self.critical_cells[cell as usize].clone()
    }

    fn match_cell(
        &self,
        cell: &Self::UpperCell,
    ) -> CellMatch<Self::UpperCell, Self::Ring, Self::Priority> {
        self.matches[cell].clone()
    }
}

/// A node in the Hasse diagram representing a single cell.
///
/// Tracks the cell's faces (lower-dimensional neighbors) and cofaces
/// (higher-dimensional neighbors) via linked lists for O(1) removal.
struct CoreductionNode<T> {
    cofaces: LinkedList,
    faces: LinkedList,
    leaf_index: Option<usize>,
    cell: T,
}

/// An edge in the Hasse diagram representing a face/coface relationship.
///
/// Connects a parent (higher-dimensional cell) to a child (face) with an
/// incidence coefficient.
struct CoreductionFace<R> {
    parent: usize,
    parent_face_index: usize,
    child: usize,
    child_coface_index: usize,
    incidence: R,
}

/// Internal implementation of the coreduction algorithm.
///
/// This type holds the mutable state during matching computation and is
/// consumed to produce the final matching result.
struct CoreductionMatchingImpl<C>
where
    C: Complex,
{
    critical_cells: Vec<C::Cell>,
    matches: HashMap<C::Cell, CellMatch<C::Cell, C::Ring, u32>>,
    indices: HashMap<C::Cell, u32>,
    leaves: LinkedList,
    nodes: Vec<CoreductionNode<C::Cell>>,
    faces: Vec<CoreductionFace<C::Ring>>,
}

impl<C> CoreductionMatchingImpl<C>
where
    C: Complex,
{
    #[allow(clippy::type_complexity)]
    fn compute_matching(
        complex: &C,
    ) -> (
        Vec<C::Cell>,
        HashMap<C::Cell, CellMatch<C::Cell, C::Ring, u32>>,
    ) {
        // Map each cell in `complex` to its index in the iterator from
        // `complex.iter()`.
        // These indices are used to store the cells
        // contiguously for efficient lookup/manipulations during matching.
        let mut indices = HashMap::new();
        for (index, cell) in complex.iter().enumerate() {
            indices.insert(cell, index as u32);
        }
        let cell_count = indices.len();

        info!("Coreduction: beginning matching on a complex of {cell_count} cells");

        // The CoreductionNode class stores a (contiguous) linked list of faces
        // and cofaces as well as the actual cell.
        //
        // We use a linked list so items can be removed in constant time (with
        // a known index of the item). The removed items are connections to
        // cells that have been excised.
        //
        // If the cell is a leaf (no faces remaining), it is stored in the
        // `leaves` linked list; `leaf_index` is its index in `leaves`, if it is
        // present. Should the cell be excised as a coreduction pair, this
        // allows it to be removed from the `leaves` linked list.
        let mut nodes = Vec::with_capacity(cell_count);
        for cell in complex.iter() {
            debug_assert_eq!(indices[&cell] as usize, nodes.len());
            nodes.push(CoreductionNode {
                cofaces: LinkedList::new(),
                faces: LinkedList::new(),
                leaf_index: None,
                cell,
            });
        }

        debug!("Coreduction: node list created");

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
        let initial_pairs = matching.initialize_hasse(complex);
        let initial_pair_count = initial_pairs.len();
        debug!("Coreduction: found {initial_pair_count} initial coreduction pairs");

        let mut initial_matched = 0usize;
        for upper_index in initial_pairs {
            debug_assert!(
                matching.nodes[upper_index].faces.len() <= 1,
                "initial coreduction pair with greater than one face"
            );

            if matching.nodes[upper_index].faces.len() == 1 {
                matching.match_pair(upper_index);
                initial_matched += 1;
            }
        }
        debug!("Coreduction: matched {initial_matched} of {initial_pair_count} initial pairs");

        let mut leaves_processed = 0usize;
        let mut progress = ProgressTracker::new("Coreduction", cell_count).with_interval(10);
        while let Some(node_index) = matching.leaves.pop_front() {
            debug_assert!(
                matching.nodes[node_index].faces.is_empty(),
                "cell in leaves list with nonzero number of faces"
            );

            trace!("Coreduction: excising leaf at node index {node_index}");
            matching.excise_leaf(node_index);
            leaves_processed += 1;
            progress.set(matching.matches.len());
        }
        progress.finish();
        debug!("Coreduction: processed {leaves_processed} leaves during main loop");

        let critical_count = matching.critical_cells.len();
        let matched_pairs = (matching.matches.len() - critical_count) / 2;
        info!(
            "Coreduction: matching complete: {critical_count} critical cells, {matched_pairs} \
             matched pairs"
        );

        (matching.critical_cells, matching.matches)
    }

    fn initialize_hasse(&mut self, complex: &C) -> Vec<usize> {
        let mut initial_pairs = Vec::new();
        for (index, cell) in complex.iter().enumerate() {
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
            trace!(
                "Coreduction: skipping pair at node {upper_index}: incidence coefficient is not \
                 invertible"
            );
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
            CellMatch::King {
                cell: self.nodes[upper_index].cell.clone(),
                queen: self.nodes[lower_index].cell.clone(),
                incidence: self.faces[upper_face_index].incidence.clone(),
                priority: self.matches.len() as u32,
            },
        );
        self.matches.insert(
            self.nodes[lower_index].cell.clone(),
            CellMatch::Queen {
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
            CellMatch::Ace {
                cell: self.nodes[node_index].cell.clone(),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::{
        CellComplex, Chain, Complex, Cube, CubicalComplex, Cyclic, Grader, HashGrader, Orthant,
        Ring,
    };

    /// Build a minimal S^1 as a CW complex: two 0-cells and two 1-cells.
    ///
    /// `v0 `and `v1` are vertices. `e0` runs from `v0` to `v1`, `e1` runs `v1`
    /// to `v0`. Together they form a circle.
    fn circle_complex() -> CellComplex<Cyclic<5>> {
        // Cells: v0=0, v1=1, e0=2, e1=3
        let cell_dimensions = vec![0, 0, 1, 1];
        let grades = vec![0, 0, 0, 0];

        let mut boundaries: Vec<Chain<u32, Cyclic<5>>> = vec![Chain::new(); 4];
        // e0: boundary = +v1 - v0
        boundaries[2].insert_or_add(1, Cyclic::one());
        boundaries[2].insert_or_add(0, -Cyclic::one());
        // e1: boundary = +v0 - v1
        boundaries[3].insert_or_add(0, Cyclic::one());
        boundaries[3].insert_or_add(1, -Cyclic::one());

        let mut coboundaries: Vec<Chain<u32, Cyclic<5>>> = vec![Chain::new(); 4];
        // v0: coboundary = -e0 + e1
        coboundaries[0].insert_or_add(2, -Cyclic::one());
        coboundaries[0].insert_or_add(3, Cyclic::one());
        // v1: coboundary = +e0 - e1
        coboundaries[1].insert_or_add(2, Cyclic::one());
        coboundaries[1].insert_or_add(3, -Cyclic::one());

        CellComplex::new(cell_dimensions, grades, boundaries, coboundaries)
    }

    #[test]
    fn line_segment_cell_complex() {
        let cell_dimensions = vec![0, 0, 1]; // vertex0, vertex1, edge
        let grades = vec![0, 0, 0];

        let mut boundaries: Vec<Chain<u32, Cyclic<5>>> = vec![Chain::new(); 3];
        boundaries[2].insert_or_add(1, Cyclic::one());
        boundaries[2].insert_or_add(0, -Cyclic::one());

        let mut coboundaries: Vec<Chain<u32, Cyclic<5>>> = vec![Chain::new(); 3];
        coboundaries[0].insert_or_add(2, -Cyclic::one());
        coboundaries[1].insert_or_add(2, Cyclic::one());

        let complex = CellComplex::new(cell_dimensions, grades, boundaries, coboundaries);
        let matching = CoreductionMatching::new(complex);

        // Should have one critical cell (contractible to a point)
        assert_eq!(matching.critical_cells().len(), 1);

        let mut aces = 0;
        let mut kings = 0;
        let mut queens = 0;

        for cell in 0..3 {
            match matching.match_cell(&cell) {
                CellMatch::Ace { .. } => aces += 1,
                CellMatch::King { .. } => kings += 1,
                CellMatch::Queen { .. } => queens += 1,
            }
        }

        assert_eq!(aces, 1);
        assert_eq!(kings, 1);
        assert_eq!(queens, 1);
    }

    #[test]
    fn triangle_cell_complex() {
        let cell_dimensions = vec![0, 0, 0, 1, 1, 1, 2];
        let grades = vec![0, 0, 0, 0, 0, 0, 0];

        let mut boundaries: Vec<Chain<u32, Cyclic<5>>> = vec![Chain::new(); 7];
        boundaries[3].insert_or_add(1, Cyclic::one());
        boundaries[3].insert_or_add(0, -Cyclic::one());
        boundaries[4].insert_or_add(2, Cyclic::one());
        boundaries[4].insert_or_add(1, -Cyclic::one());
        boundaries[5].insert_or_add(0, Cyclic::one());
        boundaries[5].insert_or_add(2, -Cyclic::one());
        boundaries[6].insert_or_add(3, Cyclic::one());
        boundaries[6].insert_or_add(4, Cyclic::one());
        boundaries[6].insert_or_add(5, Cyclic::one());

        let mut coboundaries: Vec<Chain<u32, Cyclic<5>>> = vec![Chain::new(); 7];
        coboundaries[0].insert_or_add(3, -Cyclic::one());
        coboundaries[0].insert_or_add(5, Cyclic::one());
        coboundaries[1].insert_or_add(4, -Cyclic::one());
        coboundaries[1].insert_or_add(3, Cyclic::one());
        coboundaries[2].insert_or_add(5, -Cyclic::one());
        coboundaries[2].insert_or_add(4, Cyclic::one());
        coboundaries[3].insert_or_add(6, Cyclic::one());
        coboundaries[4].insert_or_add(6, Cyclic::one());
        coboundaries[5].insert_or_add(6, Cyclic::one());

        let complex = CellComplex::new(cell_dimensions, grades, boundaries, coboundaries);
        let matching = CoreductionMatching::new(complex);

        assert_eq!(matching.critical_cells().len(), 1);

        let critical_set: HashSet<u32> = matching.critical_cells().iter().copied().collect();
        let mut matched_count = 0;

        for cell in 0..7 {
            match matching.match_cell(&cell) {
                CellMatch::Ace { .. } => {
                    assert!(critical_set.contains(&cell));
                },
                CellMatch::King { incidence, .. } | CellMatch::Queen { incidence, .. } => {
                    assert!(incidence.is_invertible());
                    matched_count += 1;
                },
            }
        }

        assert_eq!(critical_set.len() + matched_count, 7);
    }

    #[test]
    fn line_segment_cubical_complex() {
        let min = Orthant::from([0]);
        let max = Orthant::from([1]);

        let cells = vec![
            Cube::vertex(Orthant::from([0])),
            Cube::vertex(Orthant::from([1])),
            Cube::from_extent(Orthant::from([0]), &[true]),
        ];
        let grader = HashGrader::uniform(cells, 0, 1);
        let complex: CubicalComplex<Cyclic<5>, _> = CubicalComplex::new(min, max, grader);

        let matching = CoreductionMatching::new(complex);

        assert_eq!(
            matching
                .critical_cells()
                .iter()
                .filter(|cell| matching.upper_complex().grade(cell) == 0)
                .count(),
            1
        );

        let edge = Cube::from_extent(Orthant::from([0]), &[true]);
        let vertex0 = Cube::vertex(Orthant::from([0]));
        let vertex1 = Cube::vertex(Orthant::from([1]));

        let mut found_king_queen_pair = false;
        let cells = vec![vertex0.clone(), vertex1.clone(), edge.clone()];

        for cell in &cells {
            match matching.match_cell(cell) {
                CellMatch::King { queen, .. } => {
                    assert!(cells.contains(&queen));
                    found_king_queen_pair = true;
                },
                CellMatch::Queen { king, .. } => {
                    assert!(cells.contains(&king));
                    found_king_queen_pair = true;
                },
                CellMatch::Ace { .. } => {},
            }
        }

        assert!(
            found_king_queen_pair,
            "Expect to find at least one king-queen pair"
        );
    }

    #[test]
    fn empty_square_cubical_complex() {
        let min = Orthant::from([0, 0]);
        let max = Orthant::from([2, 1]);

        let cells = [
            Cube::vertex(Orthant::from([0, 0])),
            Cube::vertex(Orthant::from([1, 0])),
            Cube::vertex(Orthant::from([0, 1])),
            Cube::vertex(Orthant::from([1, 1])),
            Cube::vertex(Orthant::from([2, 1])),
            Cube::from_extent(Orthant::from([0, 0]), &[true, false]),
            Cube::from_extent(Orthant::from([1, 0]), &[false, true]),
            Cube::from_extent(Orthant::from([0, 1]), &[true, false]),
            Cube::from_extent(Orthant::from([0, 0]), &[false, true]),
            Cube::from_extent(Orthant::from([1, 1]), &[true, false]),
        ];

        let grader = HashGrader::uniform(cells, 0, 1);
        let complex: CubicalComplex<Cyclic<5>, _> = CubicalComplex::new(min, max, grader);

        let matching = CoreductionMatching::new(complex.clone());

        // Empty square has 1 dim 0 critical cell and 1 dim 1 critical cell in grade 0.
        assert_eq!(
            matching
                .critical_cells()
                .iter()
                .filter_map(|cell| {
                    if matching.upper_complex().grade(cell) == 0 {
                        Some(cell.dimension())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>(),
            vec![0, 1]
        );

        let all_cells: Vec<_> = complex.iter().collect();
        let critical_count = matching.critical_cells().len();
        let mut matched_count = 0;

        for cell in &all_cells {
            match matching.match_cell(cell) {
                CellMatch::Ace { .. } => {},
                CellMatch::King { .. } | CellMatch::Queen { .. } => {
                    matched_count += 1;
                },
            }
        }

        assert_eq!(critical_count + matched_count, all_cells.len());
    }

    #[test]
    fn empty_complex() {
        let complex: CellComplex<Cyclic<5>> = CellComplex::new(vec![], vec![], vec![], vec![]);
        let matching = CoreductionMatching::new(complex);
        assert!(matching.critical_cells().is_empty());
    }

    #[test]
    fn single_cell_complex() {
        let complex: CellComplex<Cyclic<5>> =
            CellComplex::new(vec![0], vec![0], vec![Chain::new()], vec![Chain::new()]);

        let matching = CoreductionMatching::new(complex);

        assert_eq!(matching.critical_cells().len(), 1);
        assert_eq!(matching.critical_cells()[0], 0);
        assert!(matching.match_cell(&0).is_ace());
    }

    #[test]
    fn project_include_roundtrip() {
        let matching = CoreductionMatching::new(circle_complex());

        for (i, &cell) in matching.critical_cells().iter().enumerate() {
            // project_cell(include_cell(i)) == Some(i)
            assert_eq!(
                matching.project_cell(&matching.include_cell(i as u32)),
                Some(i as u32)
            );
            // project_cell of a critical cell returns its index
            assert_eq!(matching.project_cell(&cell), Some(i as u32));
            // include_cell(project_cell(cell)) == cell
            assert_eq!(
                matching.include_cell(matching.project_cell(&cell).unwrap()),
                cell
            );
        }
    }

    #[test]
    fn lower_lift_roundtrip() {
        let matching = CoreductionMatching::new(circle_complex());

        for (i, _) in matching.critical_cells().iter().enumerate() {
            let morse_chain: Chain<u32, Cyclic<5>> = Chain::from(i as u32);
            let lifted = matching.lift(morse_chain.clone());
            let lowered = matching.lower(lifted);
            assert_eq!(
                morse_chain, lowered,
                "lower(lift(chain)) != chain for Morse cell {i}"
            );
        }
    }

    #[test]
    fn colift_colower_roundtrip() {
        let matching = CoreductionMatching::new(circle_complex());

        for (i, _) in matching.critical_cells().iter().enumerate() {
            let morse_cochain: Chain<u32, Cyclic<5>> = Chain::from(i as u32);
            let colifted = matching.colift(morse_cochain.clone());
            let colowered = matching.colower(colifted);
            assert_eq!(
                morse_cochain, colowered,
                "colower(colift(cochain)) != cochain for Morse cell {i}"
            );
        }
    }

    #[test]
    fn flow_operations_are_chain_maps() {
        let complex = circle_complex();
        let matching = CoreductionMatching::new(complex.clone());
        let morse_complex = matching.construct_morse_complex();

        // lower(bd(c)) == bd_morse(lower(c)) for all cells
        for cell in complex.iter() {
            let chain: Chain<u32, Cyclic<5>> = Chain::from(cell);
            let left = matching.lower(complex.boundary(&chain));
            let right = morse_complex.boundary(&matching.lower(chain));
            assert_eq!(left, right, "lower is not a chain map for cell {cell}");
        }

        // lift(bd_morse(m)) == bd(lift(m)) for all Morse cells
        for morse_cell in morse_complex.iter() {
            let chain: Chain<u32, Cyclic<5>> = Chain::from(morse_cell);
            let left = complex.boundary(&matching.lift(chain.clone()));
            let right = matching.lift(morse_complex.boundary(&chain));
            assert_eq!(
                left, right,
                "lift is not a chain map for Morse cell {morse_cell}"
            );
        }

        // colower(cbd(c)) == cbd_morse(colower(c)) for all cells
        for cell in complex.iter() {
            let cochain: Chain<u32, Cyclic<5>> = Chain::from(cell);
            let left = matching.colower(complex.coboundary(&cochain));
            let right = morse_complex.coboundary(&matching.colower(cochain));
            assert_eq!(left, right, "colower is not a cochain map for cell {cell}");
        }

        // colift(cbd_morse(m)) == cbd(colift(m)) for all Morse cells
        for morse_cell in morse_complex.iter() {
            let cochain: Chain<u32, Cyclic<5>> = Chain::from(morse_cell);
            let left = complex.coboundary(&matching.colift(cochain.clone()));
            let right = matching.colift(morse_complex.coboundary(&cochain));
            assert_eq!(
                left, right,
                "colift is not a cochain map for Morse cell {morse_cell}"
            );
        }
    }

    #[test]
    fn morse_complex_boundary_squared_is_zero() {
        let matching = CoreductionMatching::new(circle_complex());
        let morse_complex = matching.construct_morse_complex();

        for cell in morse_complex.iter() {
            let bd = morse_complex.cell_boundary(&cell);
            let bd2 = morse_complex.boundary(&bd);
            assert_eq!(bd2, Chain::new(), "d^2 != 0 for Morse cell {cell}");
        }
    }

    #[test]
    fn capped_methods_match_uncapped_at_neutral_bounds() {
        // All cells have grade 0, so min_grade=0 / max_grade=u32::MAX should
        // produce the same result as the uncapped variants.
        let matching = CoreductionMatching::new(circle_complex());

        for (i, _) in matching.critical_cells().iter().enumerate() {
            let chain: Chain<u32, Cyclic<5>> = Chain::from(i as u32);

            let lifted = matching.lift(chain.clone());
            let lifted_capped = matching.lift_capped(chain.clone(), 0);
            assert_eq!(lifted, lifted_capped, "lift_capped != lift at min_grade=0");

            let lowered = matching.lower(lifted.clone());
            let lowered_capped = matching.lower_capped(lifted.clone(), 0);
            assert_eq!(
                lowered, lowered_capped,
                "lower_capped != lower at min_grade=0"
            );

            let colifted = matching.colift(chain.clone());
            let colifted_capped = matching.colift_capped(chain.clone(), u32::MAX);
            assert_eq!(
                colifted, colifted_capped,
                "colift_capped != colift at max_grade=u32::MAX"
            );

            let colowered = matching.colower(colifted.clone());
            let colowered_capped = matching.colower_capped(colifted.clone(), u32::MAX);
            assert_eq!(
                colowered, colowered_capped,
                "colower_capped != colower at max_grade=u32::MAX"
            );
        }
    }

    #[test]
    fn full_reduce_circle() {
        let matching = CoreductionMatching::new(circle_complex());
        let (_, morse_complex) = matching.full_reduce(CoreductionMatching::new);

        // S^1: exactly one 0-cell and one 1-cell
        let mut cells_by_dim = [0usize; 2];
        for cell in morse_complex.iter() {
            let d = morse_complex.cell_dimension(&cell) as usize;
            assert!(d < 2, "unexpected cell dimension {d}");
            cells_by_dim[d] += 1;
        }
        assert_eq!(cells_by_dim[0], 1, "expected 1 critical 0-cell (H0=Z)");
        assert_eq!(cells_by_dim[1], 1, "expected 1 critical 1-cell (H1=Z)");

        // d^2 = 0 in the final Morse complex
        for cell in morse_complex.iter() {
            let bd = morse_complex.cell_boundary(&cell);
            let bd2 = morse_complex.boundary(&bd);
            assert_eq!(bd2, Chain::new(), "d^2 != 0 in final Morse complex");
        }

        // The 1-cell's boundary must be zero (otherwise H1 would be trivial)
        for cell in morse_complex.iter() {
            if morse_complex.cell_dimension(&cell) == 1 {
                assert_eq!(
                    morse_complex.cell_boundary(&cell),
                    Chain::new(),
                    "1-cell boundary is not zero; H1 would be trivial"
                );
            }
        }
    }
}
