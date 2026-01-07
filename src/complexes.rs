// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Cell complexes and cubical complex implementations.
//!
//! This module provides the core abstractions and implementations for working
//! with cell complexes in computational homology. The primary components are:
//!
//! - **Traits**: [`ComplexLike`] and [`Grader`] define the expected behavior
//!   for complexes and grading functions.
//! - **General complexes**: [`CellComplex`] provides an explicit vector-based
//!   implementation suitable for moderate-sized complexes.
//! - **Cubical complexes**: The [`cubical`] module provides specialized
//!   implementations for efficient high-dimensional cubical complexes.
//! - **Grading**: [`HashMapGrader`] provides a general-purpose filtration
//!   function for any complex.
//!
//! # Cell Complexes
//!
//! A cell complex is a mathematical structure composed of cells (vertices,
//! edges, faces, etc.) with well-defined boundary and coboundary relationships.
//! The [`ComplexLike`] trait provides the interface for computing boundaries,
//! coboundaries, and iterating over cells.
//!
//! ## Explicit vs Implicit Complexes
//!
//! - **Explicit**: [`CellComplex`] stores all cells, boundaries, and
//!   coboundaries in memory. Suitable for moderate-sized complexes where
//!   explicit storage is feasible.
//! - **Implicit**: [`CubicalComplex`] computes boundaries and coboundaries
//!   on-the-fly without storing all cells. Suitable for very large
//!   high-dimensional cubical complexes.
//!
//! # Grading and Filtrations
//!
//! The [`Grader`] trait assigns grades (non-negative integers) to cells,
//! defining a filtration of the complex. Filtrations are used in:
//!
//! - Discrete Morse theory with grade-based matching,
//! - Sublevel set and distance-based filtrations,
//! - Persistent homology computation.
//!
//! # Examples
//!
//! ## Creating an Explicit Cell Complex
//!
//! ```rust
//! use chomp3rs::{CellComplex, Cyclic, HashMapModule, ModuleLike, RingLike};
//!
//! // Create a simple line segment: two vertices (0, 1) and one edge (2)
//! let cell_dimensions = vec![0, 0, 1];
//! let grades = vec![0, 0, 0];
//!
//! let mut boundaries: Vec<HashMapModule<u32, Cyclic<5>>> = vec![HashMapModule::new(); 3];
//! // Boundary of edge (cell 2) is vertex1 - vertex0
//! boundaries[2].insert_or_add(1, Cyclic::<5>::one());
//! boundaries[2].insert_or_add(0, -Cyclic::<5>::one());
//!
//! let coboundaries = vec![HashMapModule::new(); 3];
//! // (Coboundaries would be filled in for a complete example)
//!
//! let complex =
//!     CellComplex::new(cell_dimensions, grades, boundaries, coboundaries);
//! ```
//!
//! ## Working with Cubical Complexes
//!
//! ```rust
//! use std::collections::HashMap;
//!
//! use chomp3rs::{
//!     ComplexLike, Cube, CubicalComplex, Cyclic, HashMapGrader,
//!     HashMapModule, ModuleLike, Orthant,
//! };
//!
//! // Create a 2D cubical complex
//! let min = Orthant::from([0, 0]);
//! let max = Orthant::from([2, 2]);
//!
//! // Set up grading for filtration
//! let vertex = Cube::vertex(Orthant::from([1, 1]));
//! let mut grades = HashMap::new();
//! grades.insert(vertex.clone(), 1);
//! let grader = HashMapGrader::from_map(grades, 0);
//!
//! let complex: CubicalComplex<HashMapModule<Cube, Cyclic<7>>, _> =
//!     CubicalComplex::new(min, max, grader);
//!
//! // Compute boundary of a cube
//! let boundary = complex.cell_boundary(&vertex);
//! assert_eq!(boundary, HashMapModule::new()); // Vertices have empty boundary
//! ```
//!
//! ## Custom Grader Implementation
//!
//! ```rust
//! use chomp3rs::Grader;
//!
//! struct DistanceGrader {
//!     origin: (i32, i32),
//! }
//!
//! impl Grader<(i32, i32)> for DistanceGrader {
//!     fn grade(&self, cell: &(i32, i32)) -> u32 {
//!         let dx = (cell.0 - self.origin.0).abs();
//!         let dy = (cell.1 - self.origin.1).abs();
//!         (dx + dy) as u32
//!     }
//! }
//!
//! let grader = DistanceGrader { origin: (0, 0) };
//! assert_eq!(grader.grade(&(3, 4)), 7); // Manhattan distance
//! ```

use std::fmt::Debug;

pub use cell_complex::CellComplex;
pub use cubical::{
    Cube, CubeIterator, CubicalComplex, Orthant, OrthantIterator, OrthantTrie, TopCubeGrader,
};
pub use grading::HashMapGrader;

use crate::{ModuleLike, RingLike};

pub mod cell_complex;
pub mod cubical;
pub mod grading;

/// Trait for types representing cell complexes over modules.
///
/// A cell complex is a mathematical structure composed of cells with defined
/// boundary and coboundary relationships. This trait provides the core
/// interface for working with such complexes, enabling computation of homology
/// and other topological invariants.
///
/// # Mathematical Properties
///
/// Implementations should satisfy the boundary property: the boundary of a
/// boundary is zero. This is fundamental to the definition of a chain complex
/// and is required for homology computation.
///
/// # Examples
///
/// Implementing `ComplexLike` for a custom complex type (a line segment):
///
/// ```rust
/// use std::ops::Range;
///
/// use chomp3rs::{
///     ComplexLike, Cyclic, Grader, HashMapModule, ModuleLike, RingLike,
/// };
///
/// // A line segment: vertices 0, 1 and edge 2
/// struct LineSegment;
///
/// impl Grader<u32> for LineSegment {
///     fn grade(&self, _cell: &u32) -> u32 {
///         0
///     }
/// }
///
/// impl ComplexLike for LineSegment {
///     type Cell = u32;
///     type CellIterator = Range<u32>;
///     type Module = HashMapModule<u32, Cyclic<5>>;
///     type Ring = Cyclic<5>;
///
///     fn cell_boundary_if(
///         &self,
///         cell: &u32,
///         predicate: impl Fn(&u32) -> bool,
///     ) -> Self::Module {
///         let mut boundary = HashMapModule::new();
///         match cell {
///             2 => {
///                 // Edge boundary: vertex1 - vertex0
///                 if predicate(&1) {
///                     boundary.insert_or_add(1, Cyclic::one());
///                 }
///                 if predicate(&0) {
///                     boundary.insert_or_add(0, -Cyclic::one());
///                 }
///             },
///             _ => {}, // Vertices have empty boundary
///         }
///         boundary
///     }
///
///     fn cell_coboundary_if(
///         &self,
///         cell: &u32,
///         predicate: impl Fn(&u32) -> bool,
///     ) -> Self::Module {
///         let mut coboundary = HashMapModule::new();
///         match cell {
///             0 if predicate(&2) => {
///                 coboundary.insert_or_add(2, -Cyclic::one());
///             },
///             1 if predicate(&2) => {
///                 coboundary.insert_or_add(2, Cyclic::one());
///             },
///             _ => {},
///         }
///         coboundary
///     }
///
///     fn iter(&self) -> Self::CellIterator {
///         0..3
///     }
///
///     fn dimension(&self) -> u32 {
///         1
///     }
///
///     fn cell_dimension(&self, cell: &u32) -> u32 {
///         if *cell == 2 { 1 } else { 0 }
///     }
/// }
///
/// let complex = LineSegment;
/// let edge_boundary = complex.cell_boundary(&2);
/// let boundary_of_boundary = complex.boundary(&edge_boundary);
/// assert_eq!(boundary_of_boundary, HashMapModule::new());
/// ```
pub trait ComplexLike: Grader<Self::Cell> {
    /// Cell type of the complex. Must be equivalent to
    /// `<Self::Module as ModuleLike>::Cell`.
    type Cell: Clone + Debug + Eq;

    /// Ring type of chains emitted by the complex. Must be equivalent to
    /// `<Self::Module as ModuleLike>::Ring`.
    type Ring: RingLike;

    /// The type of chains and cochains accepted by and output by the complex.
    /// Its associated types `Cell` and `Ring` must be equivalent to the
    /// `Cell` and `Ring` associated types of the complex.
    type Module: ModuleLike<Cell = Self::Cell, Ring = Self::Ring>;

    /// Iterator type for traversing all cells in the complex.
    type CellIterator: Iterator<Item = Self::Cell>;

    /// Return the boundary chain of a cell as a module element, including only
    /// those boundary cells that satisfy the predicate.
    ///
    /// The boundary of a cell is a formal linear combination of its faces
    /// (lower-dimensional boundary cells).
    fn cell_boundary_if(
        &self,
        cell: &Self::Cell,
        predicate: impl Fn(&Self::Cell) -> bool,
    ) -> Self::Module;

    /// Return the boundary chain of a cell as a module element.
    ///
    /// This is a provided method that calls `cell_boundary_if` with a predicate
    /// that accepts all cells.
    fn cell_boundary(&self, cell: &Self::Cell) -> Self::Module {
        self.cell_boundary_if(cell, |_| true)
    }

    /// Compute the boundary of a chain (formal linear combination of cells),
    /// including only those boundary cells that satisfy the predicate.
    ///
    /// This is a provided method that applies the predicate to each boundary
    /// cell before including it in the result.
    fn boundary_if(
        &self,
        chain: &Self::Module,
        predicate: impl Fn(&Self::Cell) -> bool,
    ) -> Self::Module {
        chain
            .iter()
            .fold(Self::Module::new(), |acc, (cell, coefficient)| {
                acc + self
                    .cell_boundary_if(cell, &predicate)
                    .scalar_mul(coefficient.clone())
            })
    }

    /// Compute the boundary of a chain (formal linear combination of cells).
    ///
    /// This is a provided method that computes the full boundary without
    /// filtering.
    fn boundary(&self, chain: &Self::Module) -> Self::Module {
        chain
            .iter()
            .fold(Self::Module::new(), |acc, (cell, coefficient)| {
                acc + self.cell_boundary(cell).scalar_mul(coefficient.clone())
            })
    }

    /// Return the coboundary chain of a cell as a module element, including
    /// only those coboundary cells that satisfy the predicate.
    ///
    /// The coboundary of a cell is a formal linear combination of cells that
    /// have this cell as a face.
    fn cell_coboundary_if(
        &self,
        cell: &Self::Cell,
        predicate: impl Fn(&Self::Cell) -> bool,
    ) -> Self::Module;

    /// Return the coboundary chain of a cell as a module element.
    ///
    /// This is a provided method that calls `cell_coboundary_if` with a
    /// predicate that accepts all cells.
    fn cell_coboundary(&self, cell: &Self::Cell) -> Self::Module {
        self.cell_coboundary_if(cell, |_| true)
    }

    /// Compute the coboundary of a cochain (formal linear combination of
    /// cells), including only those coboundary cells that satisfy the
    /// predicate.
    ///
    /// This is a provided method that applies the predicate to each coboundary
    /// cell before including it in the result.
    fn coboundary_if(
        &self,
        chain: &Self::Module,
        predicate: impl Fn(&Self::Cell) -> bool,
    ) -> Self::Module {
        chain
            .iter()
            .fold(Self::Module::new(), |acc, (cell, coefficient)| {
                acc + self
                    .cell_coboundary_if(cell, &predicate)
                    .scalar_mul(coefficient.clone())
            })
    }

    /// Compute the coboundary of a cochain (formal linear combination of
    /// cells).
    ///
    /// This is a provided method that computes the full coboundary without
    /// filtering.
    fn coboundary(&self, cochain: &Self::Module) -> Self::Module {
        cochain
            .iter()
            .fold(Self::Module::new(), |acc, (cell, coefficient)| {
                acc + self.cell_coboundary(cell).scalar_mul(coefficient.clone())
            })
    }

    /// Returns an iterator over all cells in the complex.
    fn iter(&self) -> Self::CellIterator;

    /// Returns the dimension of the complex.
    ///
    /// This is typically the maximum cell dimension, but is only required to
    /// be at least the dimension of any cell in the complex.
    fn dimension(&self) -> u32;

    /// Returns the topological dimension of a specific cell.
    fn cell_dimension(&self, cell: &Self::Cell) -> u32;
}

/// Trait for assigning grades (filtration levels) to cells in a complex.
///
/// Graders assign a non-negative integer grade to each cell, defining a
/// filtration of the complex.
///
/// # Examples
///
/// Implementing a custom grader:
///
/// ```rust
/// use chomp3rs::Grader;
///
/// struct EuclideanGrader;
///
/// impl Grader<(f64, f64)> for EuclideanGrader {
///     fn grade(&self, cell: &(f64, f64)) -> u32 {
///         let distance = (cell.0 * cell.0 + cell.1 * cell.1).sqrt();
///         (distance * 100.0) as u32 // Scale and convert to integer
///     }
/// }
///
/// let grader = EuclideanGrader;
/// assert_eq!(grader.grade(&(3.0, 4.0)), 500); // Distance 5.0, grade 500
/// ```
pub trait Grader<C> {
    /// Returns the grade (filtration level) of the specified cell.
    fn grade(&self, cell: &C) -> u32;
}
