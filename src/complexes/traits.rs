// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::{ModuleLike, RingLike};

/// Trait for types representing cell complexes over modules.
///
/// A cell complex is a mathematical structure composed of cells with defined
/// boundary and coboundary relationships. This trait provides the core
/// interface for working with such complexes.
pub trait ComplexLike {
    /// Cell type of the complex. Must be equivalent to
    /// `<Self::Module as ModuleLike>::Cell`
    type Cell;

    /// Ring type of chains emitted by the complex. Must be equivalent to
    /// `<Self::Module as ModuleLike>::Ring`
    type Ring: RingLike;

    /// The type of chains and cochains accepted by and output by the complex.
    /// Its associated types `Cell` and `Ring` must be equivalent to the
    /// `Cell` and `Ring` associated types of the complex.
    type Module: ModuleLike<Cell = Self::Cell, Ring = Self::Ring>;

    /// Iterator type for traversing all cells in the complex.
    type CellIterator: Iterator<Item = Self::Cell>;

    /// Return the boundary chain of a cell as a module element.
    ///
    /// The boundary of a cell is a formal linear combination of its faces
    /// (lower-dimensional boundary cells).
    fn boundary_of_cell(&self, cell: &Self::Cell) -> Self::Module;

    /// A provided method to compute the boundary of a chain (formal linear
    /// combination of cells) using the required `boundary_of_cell` method.
    fn boundary(&self, chain: &Self::Module) -> Self::Module {
        chain.iter().fold(Self::Module::new(), |acc, (cell, coef)| {
            acc + self.boundary_of_cell(cell).scalar_mul(coef.clone())
        })
    }

    /// Return the coboundary chain of a cell as a module element.
    ///
    /// The coboundary of a cell is a formal linear combination of cells that
    /// have this cell as a face.
    fn coboundary_of_cell(&self, cell: &Self::Cell) -> Self::Module;

    /// A provided method to compute the coboundary of a cochain (formal linear
    /// combination of cells) using the required `coboundary_of_cell` method.
    fn coboundary(&self, cochain: &Self::Module) -> Self::Module {
        cochain
            .iter()
            .fold(Self::Module::new(), |acc, (cell, coef)| {
                acc + self.coboundary_of_cell(cell).scalar_mul(coef.clone())
            })
    }

    /// Returns an iterator over all cells in the complex.
    fn cell_iter(&self) -> Self::CellIterator;

    /// Returns the dimension of the complex.
    ///
    /// This is typically the maximum cell dimension, but is only required to
    /// be at least the dimension of any cell in the complex.
    fn dimension(&self) -> u32;

    /// Returns the topological dimension of a specific cell.
    fn cell_dimension(&self, cell: &Self::Cell) -> u32;

    /// Returns the grade (filtration level) of the specified cell.
    ///
    /// Grades are used in filtered complexes for persistent homology
    /// computations.
    fn grade(&self, cell: &Self::Cell) -> u32;
}

/// Trait for assigning grades (filtration levels) to cells in a complex.
///
/// Graders are used to assign filtration levels to cells, enabling the
/// computation of persistent homology and other filtered homological
/// invariants.
pub trait Grader<C> {
    /// Returns the grade (filtration level) of the specified cell.
    fn grade(&self, cell: &C) -> u32;
}
