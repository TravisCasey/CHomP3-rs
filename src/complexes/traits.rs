// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::ModuleLike;

/// Trait for types representing cell complexes over modules.
///
/// A cell complex is a mathematical structure composed of cells with defined boundary and
/// coboundary relationships. This trait provides the core interface for working with such complexes
/// in homological computations.
pub trait ComplexLike<M>
where
    M: ModuleLike,
{
    /// Iterator type for traversing all cells in the complex.
    type CellIterator: Iterator<Item = M::Cell>;

    /// Return the boundary chain of a cell as a module element.
    fn boundary(&self, cell: &M::Cell) -> M;
    /// Return the coboundary chain of a cell as a module element.
    fn coboundary(&self, cell: &M::Cell) -> M;
    /// Returns an iterator over all cells in the complex.
    fn cell_iter(&self) -> Self::CellIterator;
    /// Returns the dimension of the complex. This is typically the maximum cell dimension, but is
    /// only required to be at least the dimension of any cell in the complex.
    fn dimension(&self) -> u32;
    /// Returns the algebraic dimension of a specific cell.
    fn cell_dimension(&self, cell: &M::Cell) -> u32;
    /// Returns the grade (filtration level) of the specified cell.
    fn grade(&self, cell: &M::Cell) -> u32;
}
