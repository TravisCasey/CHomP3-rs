// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Cell complex abstractions and implementations for computational homology.
//!
//! The [`Complex`] trait provides the core interface for cell complexes with
//! boundary and coboundary operations, and [`Grader`] assigns filtration levels
//! to cells. Two [`Complex`] implementations are provided: [`CellComplex`] for
//! explicit vector-based storage (typically used as Morse reduction output),
//! and [`CubicalComplex`] for implicit high-dimensional cubical complexes that
//! compute boundaries on-the-fly.

use std::{fmt::Debug, hash::Hash};

pub use cell_complex::CellComplex;
pub use cubical::{
    Cube, CubeIterator, CubicalComplex, Orthant, OrthantIterator, OrthantTrie, TopCubeGrader,
};
pub use grading::HashGrader;

use crate::{Chain, Ring};

pub mod cell_complex;
pub mod cubical;
pub mod grading;

/// Trait for types representing cell complexes.
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
/// Implementing `Complex` for a custom complex type (a line segment):
///
/// ```rust
/// use chomp3rs::{Chain, Complex, Cyclic, Grader, Ring};
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
/// impl Complex for LineSegment {
///     type Cell = u32;
///     type Ring = Cyclic<5>;
///
///     fn cell_boundary(&self, cell: &u32) -> Chain<u32, Cyclic<5>> {
///         let mut boundary = Chain::new();
///         match cell {
///             2 => {
///                 // Edge boundary: vertex1 - vertex0
///                 boundary.insert_or_add(1, Cyclic::one());
///                 boundary.insert_or_add(0, -Cyclic::one());
///             },
///             _ => {}, // Vertices have empty boundary
///         }
///         boundary
///     }
///
///     fn cell_coboundary(&self, cell: &u32) -> Chain<u32, Cyclic<5>> {
///         let mut coboundary = Chain::new();
///         match cell {
///             0 => {
///                 coboundary.insert_or_add(2, -Cyclic::one());
///             },
///             1 => {
///                 coboundary.insert_or_add(2, Cyclic::one());
///             },
///             _ => {},
///         }
///         coboundary
///     }
///
///     fn iter(&self) -> impl Iterator<Item = u32> {
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
/// assert_eq!(boundary_of_boundary, Chain::new());
/// ```
pub trait Complex: Grader<Self::Cell> {
    /// Cell type of the complex.
    type Cell: Clone + Debug + Eq + Hash;

    /// Ring type of chains emitted by the complex.
    type Ring: Ring;

    /// Return the boundary chain of a cell.
    ///
    /// The boundary of a cell is a formal linear combination of its faces
    /// (lower-dimensional boundary cells).
    fn cell_boundary(&self, cell: &Self::Cell) -> Chain<Self::Cell, Self::Ring>;

    /// Return the coboundary chain of a cell.
    ///
    /// The coboundary of a cell is a formal linear combination of cells that
    /// have this cell as a face.
    fn cell_coboundary(&self, cell: &Self::Cell) -> Chain<Self::Cell, Self::Ring>;

    /// Returns an iterator over all cells in the complex.
    fn iter(&self) -> impl Iterator<Item = Self::Cell>;

    /// Returns the dimension of the complex.
    ///
    /// This is typically the maximum cell dimension, but is only required to
    /// be at least the dimension of any cell in the complex.
    fn dimension(&self) -> u32;

    /// Returns the topological dimension of a specific cell.
    fn cell_dimension(&self, cell: &Self::Cell) -> u32;

    /// Compute the boundary of a chain (formal linear combination of cells).
    ///
    /// The default implementation calls [`cell_boundary`](Self::cell_boundary)
    /// for each term and accumulates the result via scalar multiplication and
    /// addition. Cost is proportional to the number of terms times the
    /// boundary size per cell.
    fn boundary<'a>(
        &self,
        chain: impl IntoIterator<Item = (&'a Self::Cell, &'a Self::Ring)>,
    ) -> Chain<Self::Cell, Self::Ring>
    where
        Self::Cell: 'a,
        Self::Ring: 'a,
    {
        chain
            .into_iter()
            .fold(Chain::new(), |acc, (cell, coefficient)| {
                acc + self.cell_boundary(cell).scalar_mul(coefficient)
            })
    }

    /// Compute the coboundary of a cochain (formal linear combination of
    /// cells).
    ///
    /// The default implementation calls
    /// [`cell_coboundary`](Self::cell_coboundary) for each term and
    /// accumulates the result via scalar multiplication and addition. Cost is
    /// proportional to the number of terms times the coboundary size per cell.
    fn coboundary<'a>(
        &self,
        chain: impl IntoIterator<Item = (&'a Self::Cell, &'a Self::Ring)>,
    ) -> Chain<Self::Cell, Self::Ring>
    where
        Self::Cell: 'a,
        Self::Ring: 'a,
    {
        chain
            .into_iter()
            .fold(Chain::new(), |acc, (cell, coefficient)| {
                acc + self.cell_coboundary(cell).scalar_mul(coefficient)
            })
    }
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
