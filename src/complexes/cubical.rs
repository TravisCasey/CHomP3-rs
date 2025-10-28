// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Implementation of a cubical complex with more cells than can be explicitly
//! stored. The cubical complex is comprised of cubes in n-dimensional real
//! space with vertices on the integer lattice. Cubes of interest are
//! differentiated by their grade, from the grader provided to the complex.
//!
//! The key types are as follows:
//! - `Orthant`: An interval of cubical cells in the ambient space. See below.
//! - `Cube`: Represents a cube defined by the base orthant it is in and the
//!   dual orthant. See below.
//! - `CubicalComplex`: A complex satisfying the `ComplexLike` trait with `Cube`
//!   instances as cells.
//!
//! ## Orthants
//!
//! An orthant is an interval of cubical cells in the ambient space between a
//! vertex and the cube it is a face of that is greater along each axis.
//! Orthants partition space into convenient regions; the cubical complex
//! implemented here comprises a rectangular grid of orthants between
//! (inclusive) the provided minimum and maximum orthants.
//!
//! ## Extent System
//!
//! Cubes are defined by their extent along each axis using base and dual
//! orthants:
//! - **Base orthant**: The unique orthant (interval of cells) containing the
//!   cube
//! - **Dual orthant**: Encodes which axes the cube extends along
//!   - If `dual[i] == base[i]`, the cube extends along axis `i` (extent =
//!     `true`)
//!   - If `dual[i] == base[i] - 1`, the cube doesn't extend along axis `i`
//!     (extent = `false`)
//!
//! ### Examples in 3D space:
//! - Vertex (extent 000): `base=[1,1,1]`, `dual=[0,0,0]`
//! - Line along X-axis (extent 100): `base=[1,1,1]`, `dual=[1,0,0]`
//! - Square in XY-plane (extent 110): `base=[1,1,1]`, `dual=[1,1,0]`
//! - 3-cube (extent 111): `base=[1,1,1]`, `dual=[1,1,1]`
//!
//! # Examples
//!
//! ```rust
//! use chomp3rs::{ComplexLike, Cube, CubicalComplex, Cyclic, HashMapGrader,
//!     HashMapModule, ModuleLike, Orthant, RingLike};
//! use std::collections::HashMap;
//!
//! // Create orthants and cubes
//! let vertex = Cube::vertex(Orthant::new(vec![1, 1]));                    // 0D vertex
//! let edge = Cube::from_extent(Orthant::new(vec![1, 1]), &[true, false]); // 1D edge
//!
//! // Check extent patterns
//! assert_eq!(vertex.extent(), vec![false, false]); // extent 00
//! assert_eq!(edge.extent(), vec![true, false]);    // extent 10
//!
//! // Create grader with custom grades
//! let mut grades = HashMap::new();
//! grades.insert(vertex.clone(), 1);
//! grades.insert(edge.clone(), 3);
//! let grader = HashMapGrader::from_map(grades);
//!
//! // Create a 2D cubical complex
//! let min = Orthant::new(vec![0, 0]);
//! let max = Orthant::new(vec![1, 1]);
//! let complex: CubicalComplex<HashMapModule<Cube, Cyclic<5>>, _> =
//!     CubicalComplex::new(min, max, grader);
//!
//! assert_eq!(complex.cell_boundary(&vertex), HashMapModule::new());
//! assert_eq!(complex.cell_boundary(&edge), HashMapModule::from([(vertex, -Cyclic::one())]));
//! ```

use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::iter::{FromIterator, zip};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::slice::{Iter, IterMut};

use crate::{ComplexLike, CubeIterator, Grader, ModuleLike, RingLike, TopCubeGrader};

/// An orthant is an interval of cubical cells in the ambient space between a
/// vertex and the top-dimensional cube it is a face of that is greater along
/// each axis. Each [`Cube`] is defined by two `Orthant` instances.
///
/// The maximum ambient dimension of an `Orthant` (as well as that of [`Cube`]
/// and [`CubicalComplex`] instances) is 32. The interface is otherwise
/// similar to an array with size fixed after construction.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Orthant {
    dimension: usize,
    coordinates: [i16; 32],
}

impl Orthant {
    /// Create a new orthant with given coordinates.
    ///
    /// Panics if the length of `coordinates` exceeds 32, which is the maximum
    /// ambient dimension of `Orthant` instances.
    #[must_use]
    pub fn new(coordinates: Vec<i16>) -> Self {
        assert!(
            coordinates.len() <= 32,
            "Cubical complex ambient dimension cannot exceed 32"
        );
        let mut array_coordinates = [0; 32];
        array_coordinates.as_mut_slice()[..coordinates.len()]
            .copy_from_slice(coordinates.as_slice());
        Self {
            dimension: coordinates.len(),
            coordinates: array_coordinates,
        }
    }

    /// Create an orthant with all coordinates set to zero.
    #[must_use]
    pub fn zeros(dimension: usize) -> Self {
        Self {
            dimension,
            coordinates: [0; 32],
        }
    }

    /// Get the dimension of this orthant.
    #[must_use]
    pub fn ambient_dimension(&self) -> u32 {
        self.dimension as u32
    }

    /// Get a reference to the coordinates as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[i16] {
        &self.coordinates[..self.dimension]
    }

    /// Get a mutable reference to the coordinates as a slice.
    ///
    /// # Warning
    /// Modifying the length of the returned slice may cause unexpected behavior
    /// and should be avoided. Only modify the values of existing elements.
    pub fn as_mut_slice(&mut self) -> &mut [i16] {
        &mut self.coordinates[..self.dimension]
    }

    /// Create an iterator over the coordinates.
    pub fn iter(&self) -> Iter<'_, i16> {
        self.coordinates[..self.dimension].iter()
    }

    /// Create a mutable iterator over the coordinates.
    pub fn iter_mut(&mut self) -> IterMut<'_, i16> {
        self.coordinates[..self.dimension].iter_mut()
    }

    /// Safely get a reference to the coordinate at the given index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&i16> {
        self.coordinates[..self.dimension].get(index)
    }

    /// Safely get a mutable reference to the coordinate at the given index.
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut i16> {
        self.coordinates[..self.dimension].get_mut(index)
    }
}

impl Index<usize> for Orthant {
    type Output = i16;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coordinates[index]
    }
}

impl IndexMut<usize> for Orthant {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coordinates[index]
    }
}

impl FromIterator<i16> for Orthant {
    fn from_iter<T: IntoIterator<Item = i16>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

impl<const N: usize> From<[i16; N]> for Orthant {
    fn from(array: [i16; N]) -> Self {
        Self::new(array.to_vec())
    }
}

impl<const N: usize> From<&[i16; N]> for Orthant {
    fn from(array: &[i16; N]) -> Self {
        Self::new(array.to_vec())
    }
}

impl<const N: usize> From<&mut [i16; N]> for Orthant {
    fn from(array: &mut [i16; N]) -> Self {
        Self::new(array.to_vec())
    }
}

impl From<&[i16]> for Orthant {
    fn from(slice: &[i16]) -> Self {
        Self::new(slice.to_vec())
    }
}

impl From<&mut [i16]> for Orthant {
    fn from(slice: &mut [i16]) -> Self {
        Self::new(slice.to_vec())
    }
}

impl Display for Orthant {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, coord) in self.coordinates[..self.dimension].iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coord)?;
        }
        write!(f, ")")
    }
}

impl PartialOrd for Orthant {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Orthant {
    fn cmp(&self, other: &Self) -> Ordering {
        self.coordinates[..self.dimension].cmp(&other.coordinates[..other.dimension])
    }
}

/// A `Cube` instance represents a (hyper)cube of arbitrary topological
/// dimension (retrived by the [`Cube::dimension`] method) in a `n`-dimensional
/// (retrieved by the [`Cube::ambient_dimension`] method) cubical complex.
/// Each cube is uniquely defined by two [`Orthant] instances: a base orthant
/// the cube exists in and a dual orthant that encodes which axes the cube
/// extends along.
///
/// The cube's extent is determined by comparing coordinates:
/// - If `dual[axis] == base[axis]`, the cube extends along `axis`
/// - If `dual[axis] == base[axis] - 1`, the cube does not extend along `axis`
///
/// See the module-level documentation for detailed information about the extent
/// and orthant systems with examples.
///
/// # Examples
///
/// ```rust
/// use chomp3rs::{Cube, Orthant};
///
/// // Create a vertex in 2D space at position (1,1)
/// let vertex = Cube::vertex(Orthant::new(vec![1, 1]));
/// assert_eq!(vertex.dimension(), 0);
/// assert_eq!(vertex.extent(), vec![false, false]); // extent 00
///
/// // Create an edge starting at (1,1)
/// let edge = Cube::from_extent(Orthant::new(vec![1, 1]), &[true, false]);
/// assert_eq!(edge.dimension(), 1);
/// assert_eq!(edge.extent(), vec![true, false]); // extent 10
///
/// // Create a 2D square with base at (1,1)
/// let square = Cube::top_cube(Orthant::new(vec![1, 1]));
/// assert_eq!(square.dimension(), 2);
/// assert_eq!(square.extent(), vec![true, true]); // extent 11
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Cube {
    base_orthant: Orthant,
    dual_orthant: Orthant,
}

impl Cube {
    /// Create a new cube from base and dual orthants.
    #[must_use]
    pub fn new(base_orthant: Orthant, dual_orthant: Orthant) -> Self {
        #[cfg(debug_assertions)]
        {
            if base_orthant.ambient_dimension() != dual_orthant.ambient_dimension() {
                panic!("Base and dual orthants must have the same dimension");
            }
            for axis in 0..base_orthant.ambient_dimension() {
                let base_coord = base_orthant[axis as usize];
                let dual_coord = dual_orthant[axis as usize];
                if dual_coord != base_coord && dual_coord != base_coord - 1 {
                    panic!(
                        "Dual orthant coordinate at axis {} must equal base coordinate {} or be exactly one less",
                        axis, base_coord
                    );
                }
            }
        }

        Self {
            base_orthant,
            dual_orthant,
        }
    }

    /// Create a cube from a base orthant and an extent bitstring.
    ///
    /// The dual orthant is a clone of the base orthant, except for axes along
    /// which the extent is `false`; there, the dual coordinate is one less
    /// than the base coordinate.
    #[must_use]
    pub fn from_extent(base_orthant: Orthant, extent: &[bool]) -> Self {
        #[cfg(debug_assertions)]
        {
            if base_orthant.ambient_dimension() != extent.len() as u32 {
                panic!("Base orthant dimension must match extent length");
            }
        }

        let dual_orthant = zip(base_orthant.iter(), extent.iter())
            .map(|(coord, extends)| if *extends { *coord } else { *coord - 1 })
            .collect::<Orthant>();

        Self {
            base_orthant,
            dual_orthant,
        }
    }

    /// Create a vertex (0-dimensional) cube from a base orthant.
    ///
    /// The dual orthant has each coordinate one less than the base orthant.
    #[must_use]
    pub fn vertex(base_orthant: Orthant) -> Self {
        let mut dual_orthant = base_orthant.clone();
        for coord in dual_orthant.as_mut_slice() {
            *coord -= 1;
        }
        Self {
            base_orthant,
            dual_orthant,
        }
    }

    /// Create a top cube from a base orthant.
    ///
    /// The dual orthant is a clone of the base orthant.
    #[must_use]
    pub fn top_cube(base_orthant: Orthant) -> Self {
        Self {
            dual_orthant: base_orthant.clone(),
            base_orthant,
        }
    }

    /// Get a reference to the base orthant.
    #[must_use]
    pub fn base(&self) -> &Orthant {
        &self.base_orthant
    }

    /// Get a mutable reference to the base orthant.
    pub fn base_mut(&mut self) -> &mut Orthant {
        &mut self.base_orthant
    }

    /// Get a reference to the dual orthant.
    #[must_use]
    pub fn dual(&self) -> &Orthant {
        &self.dual_orthant
    }

    /// Get a mutable reference to the dual orthant.
    pub fn dual_mut(&mut self) -> &mut Orthant {
        &mut self.dual_orthant
    }

    /// Safely get the base coordinate at specific index.
    #[must_use]
    pub fn base_coord(&self, index: usize) -> Option<i16> {
        self.base_orthant.get(index).copied()
    }

    /// Safely get the dual coordinate at specific index.
    #[must_use]
    pub fn dual_coord(&self, index: usize) -> Option<i16> {
        self.dual_orthant.get(index).copied()
    }

    /// Calculate the extent as a bitstring showing which axes the cube extends
    /// along.
    ///
    /// Returns a `Vec<bool>` where `true` indicates the cube extends along
    /// that axis.
    #[must_use]
    pub fn extent(&self) -> Vec<bool> {
        let mut result = Vec::with_capacity(self.ambient_dimension() as usize);
        for (base, dual) in zip(self.base_orthant.iter(), self.dual_orthant.iter()) {
            result.push(*base == *dual);
        }
        result
    }

    /// Calculate the topological dimension of the cube (number of axes with
    /// extent).
    ///
    /// This is equivalent to the number of `true` values in the extent
    /// bitstring.
    #[must_use]
    pub fn dimension(&self) -> u32 {
        zip(self.base_orthant.iter(), self.dual_orthant.iter())
            .filter(|(base, dual)| *base == *dual)
            .count() as u32
    }

    /// Get the dimension of the ambient space this cube is embedded in.
    #[must_use]
    pub fn ambient_dimension(&self) -> u32 {
        self.base_orthant.ambient_dimension()
    }
}

impl Display for Cube {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cube[base: {}, dual: {}, extent: ",
            self.base_orthant, self.dual_orthant
        )?;
        let extent = self.extent();

        for extends in extent.iter().rev() {
            write!(f, "{}", if *extends { "1" } else { "0" })?;
        }
        write!(f, "]")
    }
}

/// A `CubicalComplex` represents a finite cubical complex embedded in a
/// rectangular region in n-dimensional space. The complex contains all cubes
/// within the bounds defined by minimum and maximum orthants, with each cube
/// graded by a user-provided grading function.
///
/// See the module-level documentation for examples and further information
/// about this implementation of a cubical complex.
///
/// # Type Parameters
///
/// - `M`: Module type for representing linear combinations of cubes (namely,
///   results of `boundary` and `coboundary` calls). Must implement
///   [`ModuleLike`].
/// - `G`: Grader type for assigning filtration levels to cubes. All cubes in
///   the rectangular region are in this cubical complex; to select a subset of
///   these cubes for consideration, grade them lower than ignored cubes. Must
///   implement [`Grader`].
///
/// # Examples
///
/// ```rust
/// use std::collections::HashMap;
///
/// use chomp3rs::{
///     ComplexLike, Cube, CubicalComplex, Cyclic, HashMapGrader,
///     HashMapModule, Orthant,
/// };
///
/// // Create a simple 2x2 cubical complex
/// let min = Orthant::new(vec![0, 0]);
/// let max = Orthant::new(vec![1, 1]);
///
/// // Set up grading (vertices get grade 5, others get default 0)
/// let vertex = Cube::vertex(Orthant::new(vec![1, 1]));
/// let mut grades = HashMap::new();
/// grades.insert(vertex.clone(), 5);
/// let grader = HashMapGrader::from_map(grades);
///
/// let complex: CubicalComplex<HashMapModule<Cube, Cyclic<7>>, _> =
///     CubicalComplex::new(min, max, grader);
///
/// // Use the complex for homological computations
/// assert_eq!(complex.dimension(), 2);
/// assert!(complex.contains_cube(&vertex));
/// let boundary = complex.cell_boundary(&vertex);
/// ```
#[derive(Clone, Debug)]
pub struct CubicalComplex<M, G> {
    minimum_orthant: Orthant,
    maximum_orthant: Orthant,
    grading_function: G,
    module_type: PhantomData<M>,
}

impl<M, G> CubicalComplex<M, G> {
    /// Create a new cubical complex with custom grading function.
    ///
    /// # Panics
    /// Panics if the minimum and maximum orthants have different dimensions,
    /// or if any coordinate of the minimum orthant is greater than the
    /// corresponding coordinate of the maximum orthant.
    #[must_use]
    pub fn new(minimum_orthant: Orthant, maximum_orthant: Orthant, grading_function: G) -> Self {
        if minimum_orthant.ambient_dimension() != maximum_orthant.ambient_dimension() {
            panic!("Minimum and maximum orthants must have the same dimension");
        }

        for (min_coord, max_coord) in zip(minimum_orthant.iter(), maximum_orthant.iter()) {
            if min_coord > max_coord {
                panic!(
                    "Each coordinate of the minimum orthant must be less than or equal to that of the maximum orthant"
                );
            }
        }

        Self {
            minimum_orthant,
            maximum_orthant,
            grading_function,
            module_type: PhantomData,
        }
    }

    /// Get the minimum orthant of the rectangular region.
    #[must_use]
    pub fn minimum(&self) -> &Orthant {
        &self.minimum_orthant
    }

    /// Get the maximum orthant of the rectangular region.
    #[must_use]
    pub fn maximum(&self) -> &Orthant {
        &self.maximum_orthant
    }

    /// Get the dimension of the ambient space this cubical complex is embedded
    /// in.
    #[must_use]
    pub fn ambient_dimension(&self) -> u32 {
        self.minimum_orthant.ambient_dimension()
    }

    /// Check if a cube is within the complex bounds.
    #[must_use]
    pub fn contains_cube(&self, cube: &Cube) -> bool {
        if cube.ambient_dimension() != self.ambient_dimension() {
            return false;
        }

        self.minimum_orthant.as_slice() <= cube.base().as_slice()
            && cube.base().as_slice() <= self.maximum_orthant.as_slice()
    }

    /// Return an immutable reference to the grading function.
    pub fn grader(&self) -> &G {
        &self.grading_function
    }
}

impl<R: RingLike, M: ModuleLike<Cell = Cube, Ring = R>, G: Grader<Cube>> ComplexLike
    for CubicalComplex<M, G>
{
    type Cell = Cube;
    type CellIterator = CubeIterator;
    type Module = M;
    type Ring = R;

    fn cell_boundary_if(&self, cell: &Cube, predicate: impl Fn(&Cube) -> bool) -> M {
        debug_assert!(
            self.contains_cube(cell),
            "Cube is not in the complex bounds"
        );
        debug_assert!(
            cell.ambient_dimension() == self.ambient_dimension(),
            "Cube dimension mismatch"
        );

        let mut result = M::new();
        let mut coef = R::one();
        for (axis, (base_coord, dual_coord)) in
            zip(cell.base().iter(), cell.dual().iter()).enumerate()
        {
            // If dual[axis] == base[axis], cube extends along axis
            if *dual_coord == *base_coord {
                // Create two boundary faces by removing extent along axis
                let mut outer_base = cell.base().clone();
                outer_base[axis] += 1;
                if *base_coord < self.maximum()[axis] {
                    let outer_cube = Cube::new(outer_base, cell.dual().clone());
                    if predicate(&outer_cube) {
                        result.insert_or_add(outer_cube, coef.clone());
                    }
                }

                let mut inner_dual = cell.dual().clone();
                inner_dual[axis] -= 1;
                let inner_cube = Cube::new(cell.base().clone(), inner_dual);
                if predicate(&inner_cube) {
                    result.insert_or_add(inner_cube, -coef.clone());
                }

                coef = -coef;
            }
        }

        result
    }

    fn cell_coboundary_if(&self, cell: &Cube, predicate: impl Fn(&Cube) -> bool) -> M {
        debug_assert!(
            self.contains_cube(cell),
            "Cube is not in the complex bounds"
        );
        debug_assert!(
            cell.ambient_dimension() == self.ambient_dimension(),
            "Cube dimension mismatch"
        );

        let mut result = M::new();
        let mut coef = R::one();
        for (axis, (base_coord, dual_coord)) in
            zip(cell.base().iter(), cell.dual().iter()).enumerate()
        {
            // If dual[axis] == base[axis] - 1, cube doesn't extend along axis
            if *dual_coord == *base_coord - 1 {
                // Create two coboundary faces by adding extent along axis
                let mut outer_dual = cell.dual().clone();
                outer_dual[axis] += 1;
                let outer_cube = Cube::new(cell.base().clone(), outer_dual);
                if predicate(&outer_cube) {
                    result.insert_or_add(outer_cube, coef.clone());
                }

                let mut inner_base = cell.base().clone();
                inner_base[axis] -= 1;
                if *base_coord > self.minimum()[axis] {
                    let inner_cube = Cube::new(inner_base, cell.dual().clone());
                    if predicate(&inner_cube) {
                        result.insert_or_add(inner_cube, -coef.clone());
                    }
                }

                coef = -coef;
            }
        }

        result
    }

    fn cell_iter(&self) -> Self::CellIterator {
        CubeIterator::new(self.minimum_orthant.clone(), self.maximum_orthant.clone())
    }

    fn dimension(&self) -> u32 {
        self.ambient_dimension()
    }

    fn cell_dimension(&self, cell: &Cube) -> u32 {
        cell.dimension()
    }
}

impl<M, G: Grader<Cube>> Grader<Cube> for CubicalComplex<M, G> {
    fn grade(&self, cell: &Cube) -> u32 {
        self.grading_function.grade(cell)
    }
}

impl<M, G: Grader<Orthant>> Grader<Orthant> for CubicalComplex<M, TopCubeGrader<G>> {
    fn grade(&self, cell: &Orthant) -> u32 {
        self.grading_function.grade(cell)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{Cyclic, HashMapGrader, HashMapModule, ModuleLike, TopCubeGrader};

    #[test]
    fn test_orthant_creation_and_access() {
        // Test creation and basic access methods
        let orthant = Orthant::new(vec![1, 2, 3]);
        assert_eq!(orthant.ambient_dimension(), 3);

        // Test indexing and slice access
        assert_eq!(orthant[0], 1);
        assert_eq!(orthant[1], 2);
        assert_eq!(orthant[2], 3);
        assert_eq!(orthant.as_slice(), &[1, 2, 3]);

        // Test safe access methods
        assert_eq!(orthant.get(1), Some(&2));
        assert_eq!(orthant.get(10), None);

        // Test zeros constructor
        let zeros = Orthant::zeros(4);
        assert_eq!(zeros.ambient_dimension(), 4);
        assert_eq!(zeros.as_slice(), &[0, 0, 0, 0]);
    }

    #[test]
    fn test_orthant_mutation() {
        let mut orthant = Orthant::new(vec![1, 2, 3]);

        // Test IndexMut
        orthant[1] = 42;
        assert_eq!(orthant[1], 42);

        // Test get_mut
        if let Some(coord) = orthant.get_mut(2) {
            *coord = 100;
        }
        assert_eq!(orthant[2], 100);

        // Test iter_mut
        for coord in orthant.iter_mut() {
            *coord *= 2;
        }
        assert_eq!(orthant.as_slice(), &[2, 84, 200]);

        // Test as_mut_slice
        let mut_slice = orthant.as_mut_slice();
        mut_slice[0] = 5;
        assert_eq!(orthant[0], 5);
    }

    #[test]
    fn test_orthant_from_traits() {
        let arr = [1, 2, 3];
        let orthant1: Orthant = arr.into();
        assert_eq!(orthant1.as_slice(), &[1, 2, 3]);

        let orthant2: Orthant = (&arr).into();
        assert_eq!(orthant2.as_slice(), &[1, 2, 3]);

        let slice = &[4, 5, 6][..];
        let orthant3: Orthant = slice.into();
        assert_eq!(orthant3.as_slice(), &[4, 5, 6]);

        // Test FromIterator
        let orthant4: Orthant = (0..4).collect();
        assert_eq!(orthant4.as_slice(), &[0, 1, 2, 3]);

        let orthant5: Orthant = vec![10, 20, 30].into_iter().collect();
        assert_eq!(orthant5.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_cube_creation_and_properties() {
        let base = Orthant::new(vec![1, 1]);
        let dual = Orthant::new(vec![1, 0]);
        let cube = Cube::new(base, dual);

        assert_eq!(cube.dimension(), 1); // One axis with extent
        assert_eq!(cube.ambient_dimension(), 2);
        assert_eq!(cube.extent(), vec![true, false]); // extent 10

        // Test coordinate access
        assert_eq!(cube.base()[0], 1);
        assert_eq!(cube.base()[1], 1);
        assert_eq!(cube.dual()[0], 1); // base == dual, so extends along axis 0
        assert_eq!(cube.dual()[1], 0); // base - 1 == dual, so no extent along axis 1

        // Test checked coordinate getters
        assert_eq!(cube.base_coord(0), Some(1));
        assert_eq!(cube.dual_coord(1), Some(0));
        assert_eq!(cube.base_coord(10), None);
    }

    #[test]
    fn test_cube_constructors() {
        let base = Orthant::new(vec![1, 2, 3]);

        let vertex = Cube::vertex(base.clone());
        assert_eq!(vertex.dimension(), 0);
        assert_eq!(vertex.ambient_dimension(), 3);
        assert_eq!(vertex.dual().as_slice(), &[0, 1, 2]); // base - 1 for each coordinate
        assert_eq!(vertex.extent(), vec![false, false, false]);

        let top_cube = Cube::top_cube(base.clone());
        assert_eq!(top_cube.dimension(), 3);
        assert_eq!(top_cube.dual().as_slice(), base.as_slice()); // dual == base
        assert_eq!(top_cube.extent(), vec![true, true, true]);

        let edge_x = Cube::from_extent(base.clone(), &[true, false, false]);
        assert_eq!(edge_x.dimension(), 1);
        assert_eq!(edge_x.extent(), vec![true, false, false]);
        assert_eq!(edge_x.dual().as_slice(), &[1, 1, 2]); // [1, 2-1, 3-1]
    }

    #[test]
    fn test_cube_extent_patterns() {
        // Test all possible extent patterns in 2D
        let base = Orthant::new(vec![2, 3]);

        let patterns = vec![
            (vec![false, false], 0, Orthant::new(vec![1, 2])),
            (vec![true, false], 1, Orthant::new(vec![2, 2])),
            (vec![false, true], 1, Orthant::new(vec![1, 3])),
            (vec![true, true], 2, Orthant::new(vec![2, 3])),
        ];

        for (extent, expected_dim, expected_dual) in patterns {
            let cube = Cube::from_extent(base.clone(), &extent);
            assert_eq!(cube.extent(), extent);
            assert_eq!(cube.dimension(), expected_dim);
            assert_eq!(cube.dual().clone(), expected_dual);
        }
    }

    #[test]
    fn test_cubical_complex_creation() {
        let min = Orthant::new(vec![0, 0]);
        let max = Orthant::new(vec![2, 2]);

        let vertex = Cube::vertex(Orthant::new(vec![1, 1]));
        let edge = Cube::new(Orthant::new(vec![1, 1]), Orthant::new(vec![1, 0]));
        let mut grades = HashMap::new();
        grades.insert(vertex.clone(), 1);
        grades.insert(edge.clone(), 3);
        let grader = HashMapGrader::from_map_with_default(grades, 5);

        let complex: CubicalComplex<HashMapModule<Cube, Cyclic<7>>, _> =
            CubicalComplex::new(min, max, grader);

        assert_eq!(complex.ambient_dimension(), 2);

        // cube containment
        let cube_inside = Cube::new(Orthant::new(vec![1, 1]), Orthant::new(vec![1, 0]));
        assert!(complex.contains_cube(&cube_inside));

        let cube_outside = Cube::new(Orthant::new(vec![3, 1]), Orthant::new(vec![2, 0]));
        assert!(!complex.contains_cube(&cube_outside));

        // complex properties
        assert_eq!(complex.dimension(), 2);
        assert_eq!(complex.minimum().as_slice(), &[0, 0]);
        assert_eq!(complex.maximum().as_slice(), &[2, 2]);
    }

    #[test]
    fn test_boundary_and_coboundary() {
        let min = Orthant::new(vec![0, 0]);
        let max = Orthant::new(vec![2, 2]);

        // Create orthant grader for surrounding top cube grader
        let mut orthant_grades = HashMap::new();
        orthant_grades.insert(Orthant::new(vec![0, 0]), 4);
        orthant_grades.insert(Orthant::new(vec![1, 0]), 2);
        orthant_grades.insert(Orthant::new(vec![0, 1]), 3);
        orthant_grades.insert(Orthant::new(vec![1, 1]), 1);
        let orthant_grader = HashMapGrader::from_map(orthant_grades);
        let grader = TopCubeGrader::new(orthant_grader, Some(1));

        let complex: CubicalComplex<HashMapModule<Cube, Cyclic<7>>, _> =
            CubicalComplex::new(min, max, grader);

        // Test boundary of a 1-cube (edge) - horizontal edge
        let edge = Cube::from_extent(Orthant::new(vec![1, 1]), &[true, false]);
        let boundary = complex.cell_boundary(&edge);
        assert_eq!(
            boundary,
            HashMapModule::from([
                (Cube::vertex(Orthant::new(vec![2, 1])), Cyclic::one()),
                (Cube::vertex(Orthant::new(vec![1, 1])), -Cyclic::one())
            ])
        );

        // Test coboundary of a 0-cube (vertex)
        let vertex = Cube::vertex(Orthant::new(vec![1, 1]));
        let coboundary = complex.cell_coboundary(&vertex);
        assert_eq!(
            coboundary,
            HashMapModule::from([
                (
                    Cube::new(Orthant::new(vec![1, 1]), Orthant::new(vec![1, 0])),
                    Cyclic::one()
                ),
                (
                    Cube::new(Orthant::new(vec![0, 1]), Orthant::new(vec![0, 0])),
                    -Cyclic::one()
                ),
                (
                    Cube::new(Orthant::new(vec![1, 1]), Orthant::new(vec![0, 1])),
                    -Cyclic::one()
                ),
                (
                    Cube::new(Orthant::new(vec![1, 0]), Orthant::new(vec![0, 0])),
                    Cyclic::one()
                )
            ])
        );
    }

    #[test]
    #[should_panic(expected = "Base and dual orthants must have the same dimension")]
    fn test_cube_dimension_mismatch_panic() {
        let base = Orthant::new(vec![0, 0]);
        let dual = Orthant::new(vec![1, 0, 0]); // Different dimension
        let _cube = Cube::new(base, dual); // Should panic
    }

    #[test]
    #[should_panic(expected = "Base orthant dimension must match extent length")]
    fn test_from_extent_dimension_mismatch() {
        let base = Orthant::new(vec![1, 2]);
        let extent = &[true, false, true]; // Different length
        let _cube = Cube::from_extent(base, extent);
    }

    #[test]
    fn test_comprehensive_3d_system() {
        // Comprehensive test of 3D cubical system
        let min = Orthant::new(vec![0, 0, 0]);
        let max = Orthant::new(vec![2, 2, 2]);

        // Create orthant grader for TopCubeGrader with varied grades
        let mut orthant_grades = HashMap::new();
        orthant_grades.insert(Orthant::new(vec![0, 0, 0]), 8);
        orthant_grades.insert(Orthant::new(vec![1, 0, 0]), 3);
        orthant_grades.insert(Orthant::new(vec![0, 1, 0]), 6);
        orthant_grades.insert(Orthant::new(vec![1, 1, 0]), 1);
        orthant_grades.insert(Orthant::new(vec![0, 0, 1]), 7);
        orthant_grades.insert(Orthant::new(vec![1, 0, 1]), 4);
        orthant_grades.insert(Orthant::new(vec![0, 1, 1]), 2);
        orthant_grades.insert(Orthant::new(vec![1, 1, 1]), 9);
        let orthant_grader = HashMapGrader::from_map(orthant_grades);
        let grader = TopCubeGrader::new(orthant_grader, Some(1));

        let complex: CubicalComplex<HashMapModule<Cube, Cyclic<7>>, _> =
            CubicalComplex::new(min, max, grader);

        // Test vertex (0-cube)
        let vertex = Cube::vertex(Orthant::new(vec![1, 1, 1]));
        assert_eq!(vertex.dimension(), 0);
        assert_eq!(vertex.extent(), vec![false, false, false]);
        assert_eq!(complex.cell_dimension(&vertex), 0);
        assert_eq!(complex.grade(&vertex), 1); // Min grade from surrounding cubes

        // Test edge (1-cube)
        let edge = Cube::new(Orthant::new(vec![1, 1, 1]), Orthant::new(vec![1, 0, 0]));
        assert_eq!(edge.dimension(), 1);
        assert_eq!(edge.extent(), vec![true, false, false]);
        assert_eq!(complex.cell_dimension(&edge), 1);
        assert_eq!(complex.grade(&edge), 1); // Short-circuited to min grade

        // Test face (2-cube)
        let face = Cube::new(Orthant::new(vec![1, 1, 1]), Orthant::new(vec![1, 1, 0]));
        assert_eq!(face.dimension(), 2);
        assert_eq!(face.extent(), vec![true, true, false]);
        assert_eq!(complex.cell_dimension(&face), 2);
        assert_eq!(complex.grade(&face), 1); // Short-circuited to min grade

        // Test 3-cube
        let cube_3d = Cube::top_cube(Orthant::new(vec![1, 1, 1]));
        assert_eq!(cube_3d.dimension(), 3);
        assert_eq!(cube_3d.extent(), vec![true, true, true]);
        assert_eq!(complex.cell_dimension(&cube_3d), 3);
        assert_eq!(complex.grade(&cube_3d), 9); // Direct grade from orthant (1,1,1)

        // Test boundary relations (vertex has empty boundary)
        let vertex_boundary: HashMapModule<Cube, Cyclic<7>> = complex.cell_boundary(&vertex);
        assert_eq!(vertex_boundary, HashMapModule::new());

        // Test that all cubes are contained
        assert!(complex.contains_cube(&vertex));
        assert!(complex.contains_cube(&edge));
        assert!(complex.contains_cube(&face));
        assert!(complex.contains_cube(&cube_3d));
    }

    #[test]
    fn test_display_implementations() {
        let orthant_2d = Orthant::new(vec![1, 2]);
        assert_eq!(orthant_2d.to_string(), "(1, 2)");

        let orthant_3d = Orthant::new(vec![-1, 0, 3]);
        assert_eq!(orthant_3d.to_string(), "(-1, 0, 3)");

        let empty_orthant = Orthant::new(vec![]);
        assert_eq!(empty_orthant.to_string(), "()");

        let single_orthant = Orthant::new(vec![42]);
        assert_eq!(single_orthant.to_string(), "(42)");

        let vertex = Cube::vertex(Orthant::new(vec![1, 1]));
        assert_eq!(
            vertex.to_string(),
            "Cube[base: (1, 1), dual: (0, 0), extent: 00]"
        );

        let edge = Cube::from_extent(Orthant::new(vec![1, 1]), &[true, false]);
        assert_eq!(
            edge.to_string(),
            "Cube[base: (1, 1), dual: (1, 0), extent: 01]"
        );

        let square = Cube::top_cube(Orthant::new(vec![1, 1]));
        assert_eq!(
            square.to_string(),
            "Cube[base: (1, 1), dual: (1, 1), extent: 11]"
        );

        let cube_3d = Cube::from_extent(Orthant::new(vec![2, 3, 4]), &[true, false, true]);
        assert_eq!(
            cube_3d.to_string(),
            "Cube[base: (2, 3, 4), dual: (2, 2, 4), extent: 101]"
        );
    }

    #[test]
    fn test_chain_boundary_computation() {
        let min = Orthant::new(vec![0, 0]);
        let max = Orthant::new(vec![2, 2]);

        // Create orthant grader for TopCubeGrader
        let mut orthant_grades = HashMap::new();
        orthant_grades.insert(Orthant::new(vec![0, 0]), 4);
        orthant_grades.insert(Orthant::new(vec![1, 0]), 2);
        orthant_grades.insert(Orthant::new(vec![0, 1]), 3);
        orthant_grades.insert(Orthant::new(vec![1, 1]), 1);
        let orthant_grader = HashMapGrader::from_map(orthant_grades);
        let grader = TopCubeGrader::new(orthant_grader, Some(1));

        let complex: CubicalComplex<HashMapModule<Cube, Cyclic<7>>, _> =
            CubicalComplex::new(min, max, grader);

        // Create a chain that is a linear combination of edges:
        // 2 * horizontal_edge + 3 * vertical_edge
        let horizontal_edge = Cube::from_extent(Orthant::new(vec![1, 1]), &[true, false]);
        let vertical_edge = Cube::from_extent(Orthant::new(vec![1, 1]), &[false, true]);

        let mut edge_chain = HashMapModule::new();
        edge_chain.insert_or_add(horizontal_edge, Cyclic::from(2));
        edge_chain.insert_or_add(vertical_edge, Cyclic::from(3));

        // Compute boundary of the chain using BoundaryComputer trait
        let chain_boundary = complex.boundary(&edge_chain);

        // Expected boundary:
        // 2 * boundary(horizontal_edge) + 3 * boundary(vertical_edge)
        // = 2 * (vertex(2,1) - vertex(1,1)) + 3 * (vertex(1,2) - vertex(1,1))
        // = 2*vertex(2,1) - 2*vertex(1,1) + 3*vertex(1,2) - 3*vertex(1,1)
        // = 2*vertex(2,1) + 3*vertex(1,2) - 5*vertex(1,1)
        let vertex_11 = Cube::vertex(Orthant::new(vec![1, 1]));
        let vertex_21 = Cube::vertex(Orthant::new(vec![2, 1]));
        let vertex_12 = Cube::vertex(Orthant::new(vec![1, 2]));

        assert_eq!(chain_boundary.coef(&vertex_11), Cyclic::from(2)); // -5 = 2 (mod 7)
        assert_eq!(chain_boundary.coef(&vertex_21), Cyclic::from(2));
        assert_eq!(chain_boundary.coef(&vertex_12), Cyclic::from(3));
    }

    #[test]
    fn test_chain_coboundary_computation() {
        let min = Orthant::new(vec![0, 0]);
        let max = Orthant::new(vec![2, 2]);

        // Create orthant grader for TopCubeGrader
        let mut orthant_grades = HashMap::new();
        orthant_grades.insert(Orthant::new(vec![0, 0]), 4);
        orthant_grades.insert(Orthant::new(vec![1, 0]), 2);
        orthant_grades.insert(Orthant::new(vec![0, 1]), 3);
        orthant_grades.insert(Orthant::new(vec![1, 1]), 1);
        let orthant_grader = HashMapGrader::from_map(orthant_grades);
        let grader = TopCubeGrader::new(orthant_grader, Some(1));

        let complex: CubicalComplex<HashMapModule<Cube, Cyclic<7>>, _> =
            CubicalComplex::new(min, max, grader);

        // Create a chain that is a linear combination of vertices:
        // vertex(1,1) + 2*vertex(1,2) - vertex(2,1)
        let vertex_11 = Cube::vertex(Orthant::new(vec![1, 1]));
        let vertex_12 = Cube::vertex(Orthant::new(vec![1, 2]));
        let vertex_21 = Cube::vertex(Orthant::new(vec![2, 1]));

        let mut vertex_chain = HashMapModule::new();
        vertex_chain.insert_or_add(vertex_11, Cyclic::one());
        vertex_chain.insert_or_add(vertex_12, Cyclic::from(2));
        vertex_chain.insert_or_add(vertex_21, -Cyclic::one());

        // Compute coboundary of the chain using CoboundaryComputer trait
        let chain_coboundary = complex.coboundary(&vertex_chain);

        // The coboundary should be a linear combination of edges that have these
        // vertices as faces. We expect edges connecting these vertices to
        // appear with appropriate coefficients based on the chain coefficients
        // and orientations.

        // Check that some specific edges appear in the coboundary
        let horizontal_edge = Cube::from_extent(Orthant::new(vec![1, 1]), &[true, false]);
        let vertical_edge = Cube::from_extent(Orthant::new(vec![1, 1]), &[false, true]);

        // The exact coefficients depend on the specific orientations and combinations,
        // but we can verify that the coboundary is non-empty and contains expected
        // edges
        let coboundary_vertices: Vec<_> = chain_coboundary
            .iter()
            .map(|(cube, _)| cube.clone())
            .collect();
        assert!(!coboundary_vertices.is_empty());

        // At least one of the expected edges should be in the coboundary
        let contains_expected_edges = coboundary_vertices.contains(&horizontal_edge)
            || coboundary_vertices.contains(&vertical_edge);
        assert!(contains_expected_edges);
    }

    #[test]
    fn test_cell_boundary_if_with_predicate() {
        let min = Orthant::new(vec![0, 0]);
        let max = Orthant::new(vec![2, 2]);

        let mut orthant_grades = HashMap::new();
        orthant_grades.insert(Orthant::new(vec![0, 0]), 4);
        orthant_grades.insert(Orthant::new(vec![1, 0]), 2);
        orthant_grades.insert(Orthant::new(vec![0, 1]), 3);
        orthant_grades.insert(Orthant::new(vec![1, 1]), 1);
        let orthant_grader = HashMapGrader::from_map(orthant_grades);
        let grader = TopCubeGrader::new(orthant_grader, Some(1));

        let complex: CubicalComplex<HashMapModule<Cube, Cyclic<7>>, _> =
            CubicalComplex::new(min, max, grader);

        // Test boundary of horizontal edge with predicate that only includes vertices
        // at x=1
        let horizontal_edge = Cube::from_extent(Orthant::new(vec![1, 1]), &[true, false]);
        let boundary_filtered =
            complex.cell_boundary_if(&horizontal_edge, |cube| cube.base()[0] == 1);

        // Only vertex(1,1) should be included, not vertex(2,1)
        let vertex_11 = Cube::vertex(Orthant::new(vec![1, 1]));
        let mut expected = HashMapModule::new();
        expected.insert_or_add(vertex_11, -Cyclic::one());

        assert_eq!(boundary_filtered, expected);
    }

    #[test]
    fn test_boundary_if_with_predicate() {
        let min = Orthant::new(vec![0, 0]);
        let max = Orthant::new(vec![2, 2]);

        let mut orthant_grades = HashMap::new();
        orthant_grades.insert(Orthant::new(vec![0, 0]), 4);
        orthant_grades.insert(Orthant::new(vec![1, 0]), 2);
        orthant_grades.insert(Orthant::new(vec![0, 1]), 3);
        orthant_grades.insert(Orthant::new(vec![1, 1]), 1);
        let orthant_grader = HashMapGrader::from_map(orthant_grades);
        let grader = TopCubeGrader::new(orthant_grader, Some(1));

        let complex: CubicalComplex<HashMapModule<Cube, Cyclic<7>>, _> =
            CubicalComplex::new(min, max, grader);

        let horizontal_edge = Cube::from_extent(Orthant::new(vec![1, 1]), &[true, false]);
        let vertical_edge = Cube::from_extent(Orthant::new(vec![1, 1]), &[false, true]);

        let mut edge_chain = HashMapModule::new();
        edge_chain.insert_or_add(horizontal_edge, Cyclic::one());
        edge_chain.insert_or_add(vertical_edge, Cyclic::one());

        // Compute boundary with predicate that only includes vertices with y=1
        let boundary_filtered = complex.boundary_if(&edge_chain, |cube| cube.base()[1] == 1);

        // Should include vertex(1,1) from both edges and vertex(2,1) from horizontal
        // edge
        let vertex_11 = Cube::vertex(Orthant::new(vec![1, 1]));
        let vertex_21 = Cube::vertex(Orthant::new(vec![2, 1]));

        let mut expected = HashMapModule::new();
        expected.insert_or_add(vertex_11, Cyclic::from(5)); // -1 + -1 = -2 = 5 (mod 7)
        expected.insert_or_add(vertex_21, Cyclic::one());

        assert_eq!(boundary_filtered, expected);
    }

    #[test]
    fn test_cell_coboundary_if_with_predicate() {
        let min = Orthant::new(vec![0, 0]);
        let max = Orthant::new(vec![2, 2]);

        let mut orthant_grades = HashMap::new();
        orthant_grades.insert(Orthant::new(vec![0, 0]), 4);
        orthant_grades.insert(Orthant::new(vec![1, 0]), 2);
        orthant_grades.insert(Orthant::new(vec![0, 1]), 3);
        orthant_grades.insert(Orthant::new(vec![1, 1]), 1);
        let orthant_grader = HashMapGrader::from_map(orthant_grades);
        let grader = TopCubeGrader::new(orthant_grader, Some(1));

        let complex: CubicalComplex<HashMapModule<Cube, Cyclic<7>>, _> =
            CubicalComplex::new(min, max, grader);

        // Test coboundary of vertex with predicate that only includes horizontal edges
        let vertex = Cube::vertex(Orthant::new(vec![1, 1]));
        let coboundary_filtered =
            complex.cell_coboundary_if(&vertex, |cube| cube.extent() == vec![true, false]);

        // Should only include horizontal edges that have this vertex as a face
        let horizontal_edge1 = Cube::new(Orthant::new(vec![1, 1]), Orthant::new(vec![1, 0]));
        let horizontal_edge2 = Cube::new(Orthant::new(vec![0, 1]), Orthant::new(vec![0, 0]));

        // Check that only horizontal edges are included
        for (cube, _) in coboundary_filtered.iter() {
            assert_eq!(cube.extent(), vec![true, false]);
        }

        // Should contain at least the expected horizontal edges
        assert!(
            coboundary_filtered.coef(&horizontal_edge1) != Cyclic::zero()
                || coboundary_filtered.coef(&horizontal_edge2) != Cyclic::zero()
        );
    }

    #[test]
    fn test_coboundary_if_with_predicate() {
        let min = Orthant::new(vec![0, 0]);
        let max = Orthant::new(vec![2, 2]);

        let mut orthant_grades = HashMap::new();
        orthant_grades.insert(Orthant::new(vec![0, 0]), 4);
        orthant_grades.insert(Orthant::new(vec![1, 0]), 2);
        orthant_grades.insert(Orthant::new(vec![0, 1]), 3);
        orthant_grades.insert(Orthant::new(vec![1, 1]), 1);
        let orthant_grader = HashMapGrader::from_map(orthant_grades);
        let grader = TopCubeGrader::new(orthant_grader, Some(1));

        let complex: CubicalComplex<HashMapModule<Cube, Cyclic<7>>, _> =
            CubicalComplex::new(min, max, grader);

        let vertex_11 = Cube::vertex(Orthant::new(vec![1, 1]));
        let vertex_21 = Cube::vertex(Orthant::new(vec![2, 1]));

        let mut vertex_chain = HashMapModule::new();
        vertex_chain.insert_or_add(vertex_11, Cyclic::one());
        vertex_chain.insert_or_add(vertex_21, Cyclic::one());

        // Compute coboundary with predicate that only includes vertical edges
        let coboundary_filtered =
            complex.coboundary_if(&vertex_chain, |cube| cube.extent() == vec![false, true]);

        // Check that only vertical edges are included in the result
        for (cube, _) in coboundary_filtered.iter() {
            assert_eq!(cube.extent(), vec![false, true]);
        }

        // The result should be non-empty since these vertices should have vertical
        // edges in their coboundaries
        assert!(coboundary_filtered != HashMapModule::new());
    }
}
