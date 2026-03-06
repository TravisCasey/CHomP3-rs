// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Cubical complex implementation for high-dimensional spaces.
//!
//! A [`CubicalComplex`] represents a rectangular grid of cubes in n-dimensional
//! space with vertices on the integer lattice. Each [`Cube`] is defined by a
//! base [`Orthant`] and a dual [`Orthant`] encoding which axes the cube extends
//! along (the extent system).
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
//!   - If `dual[i] == base[i]`, the cube extends along axis `i` (extent `true`)
//!   - If `dual[i] == base[i] - 1`, the cube does not extend along axis `i`
//!     (extent `false`)
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
//! use std::collections::HashMap;
//!
//! use chomp3rs::{Chain, Complex, Cube, CubicalComplex, Cyclic, HashGrader,
//!     Orthant, Ring};
//!
//! // Create orthants and cubes
//! let vertex = Cube::vertex(Orthant::from([1, 1]));                    // 0D vertex
//! let edge = Cube::from_extent(Orthant::from([1, 1]), &[true, false]); // 1D edge
//!
//! // Check extent patterns
//! assert_eq!(vertex.extent(), vec![false, false]); // extent 00
//! assert_eq!(edge.extent(), vec![true, false]);    // extent 10
//!
//! // Create grader with custom grades
//! let mut grades = HashMap::new();
//! grades.insert(vertex.clone(), 1);
//! grades.insert(edge.clone(), 3);
//! let grader = HashGrader::from_map(grades, 0);
//!
//! // Create a 2D cubical complex
//! let min = Orthant::from([0, 0]);
//! let max = Orthant::from([1, 1]);
//! let complex: CubicalComplex<Cyclic<5>, _> =
//!     CubicalComplex::new(min, max, grader);
//!
//! assert_eq!(complex.cell_boundary(&vertex), Chain::new());
//! assert_eq!(complex.cell_boundary(&edge), Chain::from([(vertex, -Cyclic::one())]));
//! ```

use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display, Formatter},
    hash::Hash,
    iter::zip,
    marker::PhantomData,
    ops::{Index, IndexMut},
    slice::{Iter, IterMut},
};

pub use graders::{OrthantTrie, TopCubeGrader};
pub use iterators::{CubeIterator, OrthantIterator};

use crate::{Chain, Complex, Grader, Ring};

mod graders;
mod iterators;

/// Maximum ambient dimension supported by cubical types.
///
/// This limit applies to [`Orthant`], [`Cube`], and [`CubicalComplex`].
const MAX_DIMENSION: usize = 32;

/// An orthant is an interval of cubical cells in the ambient space between a
/// vertex and the top-dimensional cube it is a face of that is greater along
/// each axis. Each [`Cube`] is defined by two `Orthant` instances.
///
/// The maximum ambient dimension of an `Orthant` (as well as that of [`Cube`]
/// and [`CubicalComplex`] instances) is [`MAX_DIMENSION`](Self::MAX_DIMENSION).
/// The interface is otherwise similar to an array with size fixed after
/// construction.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Orthant {
    dimension: usize,
    coordinates: [i16; MAX_DIMENSION],
}

impl Orthant {
    /// Maximum ambient dimension supported by cubical types.
    ///
    /// This limit applies to [`Orthant`], [`Cube`], and [`CubicalComplex`].
    pub const MAX_DIMENSION: usize = MAX_DIMENSION;

    /// Create a new orthant with given coordinates.
    ///
    /// # Panics
    /// If the length of `coordinates` exceeds
    /// [`MAX_DIMENSION`](Self::MAX_DIMENSION).
    #[must_use]
    pub fn new(coordinates: &[i16]) -> Self {
        assert!(
            coordinates.len() <= Self::MAX_DIMENSION,
            "ambient dimension cannot exceed {}, got {}",
            Self::MAX_DIMENSION,
            coordinates.len()
        );
        let mut array_coordinates = [0i16; Self::MAX_DIMENSION];
        array_coordinates[..coordinates.len()].copy_from_slice(coordinates);
        Self {
            dimension: coordinates.len(),
            coordinates: array_coordinates,
        }
    }

    /// Create an orthant with all coordinates set to zero.
    ///
    /// # Panics
    /// If `dimension` exceeds [`MAX_DIMENSION`](Self::MAX_DIMENSION).
    #[must_use]
    pub fn zeros(dimension: usize) -> Self {
        assert!(
            dimension <= Self::MAX_DIMENSION,
            "ambient dimension cannot exceed {}, got {dimension}",
            Self::MAX_DIMENSION,
        );
        Self {
            dimension,
            coordinates: [0; Self::MAX_DIMENSION],
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
    pub fn as_mut_slice(&mut self) -> &mut [i16] {
        &mut self.coordinates[..self.dimension]
    }

    /// Create an iterator over the coordinates.
    pub fn iter(&self) -> Iter<'_, i16> {
        self.into_iter()
    }

    /// Create a mutable iterator over the coordinates.
    pub fn iter_mut(&mut self) -> IterMut<'_, i16> {
        self.into_iter()
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
        &self.coordinates[..self.dimension][index]
    }
}

impl IndexMut<usize> for Orthant {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coordinates[..self.dimension][index]
    }
}

impl<'a> IntoIterator for &'a Orthant {
    type IntoIter = std::slice::Iter<'a, i16>;
    type Item = &'a i16;

    fn into_iter(self) -> Self::IntoIter {
        self.coordinates[..self.dimension].iter()
    }
}

impl<'a> IntoIterator for &'a mut Orthant {
    type IntoIter = std::slice::IterMut<'a, i16>;
    type Item = &'a mut i16;

    fn into_iter(self) -> Self::IntoIter {
        self.coordinates[..self.dimension].iter_mut()
    }
}

impl FromIterator<i16> for Orthant {
    fn from_iter<T: IntoIterator<Item = i16>>(iter: T) -> Self {
        let mut coordinates = [0i16; Self::MAX_DIMENSION];
        let mut dimension = 0;
        for (index, value) in iter.into_iter().enumerate() {
            assert!(
                index < Self::MAX_DIMENSION,
                "ambient dimension cannot exceed {}, got at least {}",
                Self::MAX_DIMENSION,
                index + 1,
            );
            coordinates[index] = value;
            dimension += 1;
        }

        Orthant {
            dimension,
            coordinates,
        }
    }
}

impl<const N: usize> From<[i16; N]> for Orthant {
    fn from(array: [i16; N]) -> Self {
        Self::new(array.as_slice())
    }
}

impl<const N: usize> From<&[i16; N]> for Orthant {
    fn from(array: &[i16; N]) -> Self {
        Self::new(array.as_slice())
    }
}

impl<const N: usize> From<&mut [i16; N]> for Orthant {
    fn from(array: &mut [i16; N]) -> Self {
        Self::new(array.as_slice())
    }
}

impl From<&[i16]> for Orthant {
    fn from(slice: &[i16]) -> Self {
        Self::new(slice)
    }
}

impl From<&mut [i16]> for Orthant {
    fn from(slice: &mut [i16]) -> Self {
        Self::new(slice)
    }
}

impl Debug for Orthant {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Orthant {{ coordinates: [")?;
        let mut first = true;
        for coord in self {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{coord}")?;
        }
        write!(f, "] }}")?;
        Ok(())
    }
}

impl Display for Orthant {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, coord) in self.coordinates[..self.dimension].iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{coord}")?;
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

impl Hash for Orthant {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coordinates[..self.dimension].hash(state);
    }
}

/// A `Cube` instance represents a (hyper)cube of arbitrary topological
/// dimension (retrieved by the [`Cube::dimension`] method) in a `n`-dimensional
/// (retrieved by the [`Cube::ambient_dimension`] method) cubical complex.
/// Each cube is uniquely defined by two [`Orthant`] instances: a base orthant
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
/// let vertex = Cube::vertex(Orthant::from([1, 1]));
/// assert_eq!(vertex.dimension(), 0);
/// assert_eq!(vertex.extent(), vec![false, false]); // extent 00
///
/// // Create an edge starting at (1,1)
/// let edge = Cube::from_extent(Orthant::from([1, 1]), &[true, false]);
/// assert_eq!(edge.dimension(), 1);
/// assert_eq!(edge.extent(), vec![true, false]); // extent 10
///
/// // Create a 2D square with base at (1,1)
/// let square = Cube::top_cube(Orthant::from([1, 1]));
/// assert_eq!(square.dimension(), 2);
/// assert_eq!(square.extent(), vec![true, true]); // extent 11
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cube {
    base_orthant: Orthant,
    dual_orthant: Orthant,
}

impl Cube {
    /// Create a new cube from base and dual orthants.
    ///
    /// # Panics
    ///
    /// If `base_orthant` is not of the same ambient dimension as `dual_orthant`
    /// or if any of the dual orthant coordinates are not equal to the same
    /// base orthant coordinate or one less.
    #[must_use]
    pub fn new(base_orthant: Orthant, dual_orthant: Orthant) -> Self {
        assert!(
            base_orthant.ambient_dimension() == dual_orthant.ambient_dimension(),
            "Base and dual orthants must have the same dimension: base has dimension {}, dual has \
             dimension {}",
            base_orthant.ambient_dimension(),
            dual_orthant.ambient_dimension()
        );

        for axis in 0..base_orthant.ambient_dimension() {
            let base_coord = base_orthant[axis as usize];
            let dual_coord = dual_orthant[axis as usize];
            assert!(
                dual_coord == base_coord || dual_coord == base_coord - 1,
                "Dual orthant coordinate at axis {axis} must equal base coordinate {base_coord} \
                 or be exactly one less, got dual coordinate {dual_coord}",
            );
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
    ///
    /// # Panics
    ///
    /// If the dimension of `base_orthant` does not match the length of
    /// `extent`.
    #[must_use]
    pub fn from_extent(base_orthant: Orthant, extent: &[bool]) -> Self {
        assert!(
            base_orthant.ambient_dimension() == extent.len() as u32,
            "Base orthant dimension must match extent length: base has dimension {}, extent has \
             length {}",
            base_orthant.ambient_dimension(),
            extent.len()
        );

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
        let dual_orthant = base_orthant.iter().map(|coord| coord - 1).collect();
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
        zip(self.base_orthant.iter(), self.dual_orthant.iter())
            .map(|(base, dual)| base == dual)
            .collect()
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
/// - `R`: Coefficient ring for chains produced by boundary and coboundary
///   operations. Must implement [`Ring`].
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
///     Chain, Complex, Cube, CubicalComplex, Cyclic, HashGrader, Orthant,
/// };
///
/// // Create a simple 2x2 cubical complex
/// let min = Orthant::from([0, 0]);
/// let max = Orthant::from([1, 1]);
///
/// // Set up grading (vertices get grade 5, others get default 0)
/// let vertex = Cube::vertex(Orthant::from([1, 1]));
/// let mut grades = HashMap::new();
/// grades.insert(vertex.clone(), 5);
/// let grader = HashGrader::from_map(grades, 0);
///
/// let complex: CubicalComplex<Cyclic<7>, _> =
///     CubicalComplex::new(min, max, grader);
///
/// // Use the complex for homological computations
/// assert_eq!(complex.dimension(), 2);
/// assert!(complex.contains_cube(&vertex));
/// let boundary = complex.cell_boundary(&vertex);
/// ```
#[derive(Clone, Debug)]
pub struct CubicalComplex<R, G> {
    minimum_orthant: Orthant,
    maximum_orthant: Orthant,
    grading_function: G,
    ring_type: PhantomData<R>,
}

impl<R, G> CubicalComplex<R, G> {
    /// Create a new cubical complex with custom grading function.
    ///
    /// # Panics
    /// Panics if the minimum and maximum orthants have different dimensions,
    /// or if any coordinate of the minimum orthant is greater than the
    /// corresponding coordinate of the maximum orthant.
    #[must_use]
    pub fn new(minimum_orthant: Orthant, maximum_orthant: Orthant, grading_function: G) -> Self {
        assert!(
            minimum_orthant.ambient_dimension() == maximum_orthant.ambient_dimension(),
            "Minimum and maximum orthants must have the same dimension: minimum has dimension {}, \
             maximum has dimension {}",
            minimum_orthant.ambient_dimension(),
            maximum_orthant.ambient_dimension()
        );

        for (axis, (min_coord, max_coord)) in
            zip(minimum_orthant.iter(), maximum_orthant.iter()).enumerate()
        {
            assert!(
                min_coord <= max_coord,
                "Each coordinate of the minimum orthant must be less than or equal to that of the \
                 maximum orthant: at axis {axis}, minimum is {min_coord} but maximum is \
                 {max_coord}",
            );
        }

        Self {
            minimum_orthant,
            maximum_orthant,
            grading_function,
            ring_type: PhantomData,
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
    #[must_use]
    pub fn grader(&self) -> &G {
        &self.grading_function
    }
}

impl<R: Ring, G: Grader<Cube>> Complex for CubicalComplex<R, G> {
    type Cell = Cube;
    type Ring = R;

    fn cell_boundary(&self, cell: &Cube) -> Chain<Cube, R> {
        debug_assert!(
            self.contains_cube(cell),
            "Cube {:?} is not within the complex bounds [{:?}, {:?}]",
            cell,
            self.minimum(),
            self.maximum()
        );
        debug_assert!(
            cell.ambient_dimension() == self.ambient_dimension(),
            "Cube dimension mismatch: cube has ambient dimension {}, complex has ambient \
             dimension {}",
            cell.ambient_dimension(),
            self.ambient_dimension()
        );

        let mut result = Chain::new();
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
                    result.insert_or_add(outer_cube, coef.clone());
                }

                let mut inner_dual = cell.dual().clone();
                inner_dual[axis] -= 1;
                let inner_cube = Cube::new(cell.base().clone(), inner_dual);
                result.insert_or_add(inner_cube, -coef.clone());

                coef = -coef;
            }
        }

        result
    }

    fn cell_coboundary(&self, cell: &Cube) -> Chain<Cube, R> {
        debug_assert!(
            self.contains_cube(cell),
            "Cube {:?} is not within the complex bounds [{:?}, {:?}]",
            cell,
            self.minimum(),
            self.maximum()
        );
        debug_assert!(
            cell.ambient_dimension() == self.ambient_dimension(),
            "Cube dimension mismatch: cube has ambient dimension {}, complex has ambient \
             dimension {}",
            cell.ambient_dimension(),
            self.ambient_dimension()
        );

        let mut result = Chain::new();
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
                result.insert_or_add(outer_cube, coef.clone());

                let mut inner_base = cell.base().clone();
                inner_base[axis] -= 1;
                if *base_coord > self.minimum()[axis] {
                    let inner_cube = Cube::new(inner_base, cell.dual().clone());
                    result.insert_or_add(inner_cube, -coef.clone());
                }

                coef = -coef;
            }
        }

        result
    }

    fn iter(&self) -> impl Iterator<Item = Cube> {
        CubeIterator::new(self.minimum_orthant.clone(), self.maximum_orthant.clone())
    }

    fn dimension(&self) -> u32 {
        self.ambient_dimension()
    }

    fn cell_dimension(&self, cell: &Cube) -> u32 {
        cell.dimension()
    }
}

impl<R, G: Grader<Cube>> Grader<Cube> for CubicalComplex<R, G> {
    fn grade(&self, cell: &Cube) -> u32 {
        self.grading_function.grade(cell)
    }
}

impl<R, G: Grader<Orthant>> Grader<Orthant> for CubicalComplex<R, TopCubeGrader<G>> {
    fn grade(&self, cell: &Orthant) -> u32 {
        self.grading_function.grade(cell)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{Cyclic, HashGrader, Ring, TopCubeGrader};

    /// Build the 2D graded cubical complex used by several tests.
    ///
    /// The complex spans `[0,0]` to `[2,2]` with orthant grades:
    /// `(0,0)→4`, `(1,0)→2`, `(0,1)→3`, `(1,1)→1`.
    fn create_2d_graded_complex() -> CubicalComplex<Cyclic<7>, TopCubeGrader<HashGrader<Orthant>>> {
        let mut orthant_grades = HashMap::new();
        orthant_grades.insert(Orthant::from([0, 0]), 4);
        orthant_grades.insert(Orthant::from([1, 0]), 2);
        orthant_grades.insert(Orthant::from([0, 1]), 3);
        orthant_grades.insert(Orthant::from([1, 1]), 1);
        let grader = TopCubeGrader::new(HashGrader::from_map(orthant_grades, 0), Some(1));
        CubicalComplex::new(Orthant::from([0, 0]), Orthant::from([2, 2]), grader)
    }

    #[test]
    fn orthant_creation_and_access() {
        // Test creation and basic access methods
        let orthant = Orthant::new(&[1, 2, 3]);
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
    fn orthant_mutation() {
        let mut orthant = Orthant::from([1, 2, 3]);

        // Test IndexMut
        orthant[1] = 42;
        assert_eq!(orthant[1], 42);

        // Test get_mut
        if let Some(coord) = orthant.get_mut(2) {
            *coord = 100;
        }
        assert_eq!(orthant[2], 100);

        // Test iter_mut
        for coord in &mut orthant {
            *coord *= 2;
        }
        assert_eq!(orthant.as_slice(), &[2, 84, 200]);

        // Test as_mut_slice
        let mut_slice = orthant.as_mut_slice();
        mut_slice[0] = 5;
        assert_eq!(orthant[0], 5);
    }

    #[test]
    fn orthant_from_traits() {
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
    fn cube_creation_and_properties() {
        let base = Orthant::from([1, 1]);
        let dual = Orthant::from([1, 0]);
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
    fn cube_constructors() {
        let base = Orthant::from([1, 2, 3]);

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
    fn cube_extent_patterns() {
        // Test all possible extent patterns in 2D
        let base = Orthant::from([2, 3]);

        let patterns = vec![
            (vec![false, false], 0, Orthant::from([1, 2])),
            (vec![true, false], 1, Orthant::from([2, 2])),
            (vec![false, true], 1, Orthant::from([1, 3])),
            (vec![true, true], 2, Orthant::from([2, 3])),
        ];

        for (extent, expected_dim, expected_dual) in patterns {
            let cube = Cube::from_extent(base.clone(), &extent);
            assert_eq!(cube.extent(), extent);
            assert_eq!(cube.dimension(), expected_dim);
            assert_eq!(cube.dual().clone(), expected_dual);
        }
    }

    #[test]
    fn cubical_complex_creation() {
        let min = Orthant::from([0, 0]);
        let max = Orthant::from([2, 2]);

        let vertex = Cube::vertex(Orthant::from([1, 1]));
        let edge = Cube::new(Orthant::from([1, 1]), Orthant::from([1, 0]));
        let mut cell_grades = HashMap::new();
        cell_grades.insert(vertex.clone(), 1);
        cell_grades.insert(edge.clone(), 3);
        let grader = HashGrader::from_map(cell_grades, 5);

        let complex: CubicalComplex<Cyclic<7>, _> = CubicalComplex::new(min, max, grader);

        assert_eq!(complex.ambient_dimension(), 2);

        // cube containment
        let cube_inside = Cube::new(Orthant::from([1, 1]), Orthant::from([1, 0]));
        assert!(complex.contains_cube(&cube_inside));

        let cube_outside = Cube::new(Orthant::from([3, 1]), Orthant::from([2, 0]));
        assert!(!complex.contains_cube(&cube_outside));

        // complex properties
        assert_eq!(complex.dimension(), 2);
        assert_eq!(complex.minimum().as_slice(), &[0, 0]);
        assert_eq!(complex.maximum().as_slice(), &[2, 2]);
    }

    #[test]
    fn boundary_and_coboundary() {
        let complex = create_2d_graded_complex();

        // Test boundary of a 1-cube (edge) - horizontal edge
        let edge = Cube::from_extent(Orthant::from([1, 1]), &[true, false]);
        let boundary = complex.cell_boundary(&edge);
        assert_eq!(
            boundary,
            Chain::from([
                (Cube::vertex(Orthant::from([2, 1])), Cyclic::one()),
                (Cube::vertex(Orthant::from([1, 1])), -Cyclic::one())
            ])
        );

        // Test coboundary of a 0-cube (vertex)
        let vertex = Cube::vertex(Orthant::from([1, 1]));
        let coboundary = complex.cell_coboundary(&vertex);
        assert_eq!(
            coboundary,
            Chain::from([
                (
                    Cube::new(Orthant::from([1, 1]), Orthant::from([1, 0])),
                    Cyclic::one()
                ),
                (
                    Cube::new(Orthant::from([0, 1]), Orthant::from([0, 0])),
                    -Cyclic::one()
                ),
                (
                    Cube::new(Orthant::from([1, 1]), Orthant::from([0, 1])),
                    -Cyclic::one()
                ),
                (
                    Cube::new(Orthant::from([1, 0]), Orthant::from([0, 0])),
                    Cyclic::one()
                )
            ])
        );
    }

    #[test]
    #[should_panic(expected = "Base and dual orthants must have the same dimension")]
    fn cube_dimension_mismatch_panic() {
        let base = Orthant::from([0, 0]);
        let dual = Orthant::from([1, 0, 0]); // Different dimension
        let _cube = Cube::new(base, dual); // Should panic
    }

    #[test]
    #[should_panic(expected = "Base orthant dimension must match extent length")]
    fn from_extent_dimension_mismatch() {
        let base = Orthant::from([1, 2]);
        let extent = &[true, false, true]; // Different length
        let _cube = Cube::from_extent(base, extent);
    }

    #[test]
    fn comprehensive_3d_system() {
        // Comprehensive test of 3D cubical system
        let min = Orthant::from([0, 0, 0]);
        let max = Orthant::from([2, 2, 2]);

        // Create orthant grader for TopCubeGrader with varied grades
        let mut orthant_grades = HashMap::new();
        orthant_grades.insert(Orthant::from([0, 0, 0]), 8);
        orthant_grades.insert(Orthant::from([1, 0, 0]), 3);
        orthant_grades.insert(Orthant::from([0, 1, 0]), 6);
        orthant_grades.insert(Orthant::from([1, 1, 0]), 1);
        orthant_grades.insert(Orthant::from([0, 0, 1]), 7);
        orthant_grades.insert(Orthant::from([1, 0, 1]), 4);
        orthant_grades.insert(Orthant::from([0, 1, 1]), 2);
        orthant_grades.insert(Orthant::from([1, 1, 1]), 9);
        let grader = TopCubeGrader::new(HashGrader::from_map(orthant_grades, 0), Some(1));

        let complex: CubicalComplex<Cyclic<7>, _> = CubicalComplex::new(min, max, grader);

        // Test vertex (0-cube)
        let vertex = Cube::vertex(Orthant::from([1, 1, 1]));
        assert_eq!(vertex.dimension(), 0);
        assert_eq!(vertex.extent(), vec![false, false, false]);
        assert_eq!(complex.cell_dimension(&vertex), 0);
        assert_eq!(complex.grade(&vertex), 1); // Min grade from surrounding cubes

        // Test edge (1-cube)
        let edge = Cube::new(Orthant::from([1, 1, 1]), Orthant::from([1, 0, 0]));
        assert_eq!(edge.dimension(), 1);
        assert_eq!(edge.extent(), vec![true, false, false]);
        assert_eq!(complex.cell_dimension(&edge), 1);
        assert_eq!(complex.grade(&edge), 1); // Short-circuited to min grade

        // Test face (2-cube)
        let face = Cube::new(Orthant::from([1, 1, 1]), Orthant::from([1, 1, 0]));
        assert_eq!(face.dimension(), 2);
        assert_eq!(face.extent(), vec![true, true, false]);
        assert_eq!(complex.cell_dimension(&face), 2);
        assert_eq!(complex.grade(&face), 1); // Short-circuited to min grade

        // Test 3-cube
        let cube_3d = Cube::top_cube(Orthant::from([1, 1, 1]));
        assert_eq!(cube_3d.dimension(), 3);
        assert_eq!(cube_3d.extent(), vec![true, true, true]);
        assert_eq!(complex.cell_dimension(&cube_3d), 3);
        assert_eq!(complex.grade(&cube_3d), 9); // Direct grade from orthant (1,1,1)

        // Test boundary relations (vertex has empty boundary)
        let vertex_boundary = complex.cell_boundary(&vertex);
        assert_eq!(vertex_boundary, Chain::new());

        // Test that all cubes are contained
        assert!(complex.contains_cube(&vertex));
        assert!(complex.contains_cube(&edge));
        assert!(complex.contains_cube(&face));
        assert!(complex.contains_cube(&cube_3d));
    }

    #[test]
    fn orthant_display() {
        let orthant_2d = Orthant::from([1, 2]);
        assert_eq!(orthant_2d.to_string(), "(1, 2)");

        let orthant_3d = Orthant::from([-1, 0, 3]);
        assert_eq!(orthant_3d.to_string(), "(-1, 0, 3)");

        let empty_orthant = Orthant::from([]);
        assert_eq!(empty_orthant.to_string(), "()");

        let single_orthant = Orthant::from([42]);
        assert_eq!(single_orthant.to_string(), "(42)");
    }

    #[test]
    fn orthant_debug() {
        // Zero entries (empty orthant)
        let empty = Orthant::from([]);
        assert_eq!(format!("{empty:?}"), "Orthant { coordinates: [] }");

        // One entry
        let single = Orthant::from([42]);
        assert_eq!(format!("{single:?}"), "Orthant { coordinates: [42] }");

        // Multiple entries (8 dimensions)
        let eight_dim = Orthant::from([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            format!("{eight_dim:?}"),
            "Orthant { coordinates: [1, 2, 3, 4, 5, 6, 7, 8] }"
        );
    }

    #[test]
    fn cube_display() {
        let vertex = Cube::vertex(Orthant::from([1, 1]));
        assert_eq!(
            vertex.to_string(),
            "Cube[base: (1, 1), dual: (0, 0), extent: 00]"
        );

        let edge = Cube::from_extent(Orthant::from([1, 1]), &[true, false]);
        assert_eq!(
            edge.to_string(),
            "Cube[base: (1, 1), dual: (1, 0), extent: 01]"
        );

        let square = Cube::top_cube(Orthant::from([1, 1]));
        assert_eq!(
            square.to_string(),
            "Cube[base: (1, 1), dual: (1, 1), extent: 11]"
        );

        let cube_3d = Cube::from_extent(Orthant::from([2, 3, 4]), &[true, false, true]);
        assert_eq!(
            cube_3d.to_string(),
            "Cube[base: (2, 3, 4), dual: (2, 2, 4), extent: 101]"
        );
    }

    #[test]
    fn chain_boundary_computation() {
        let complex = create_2d_graded_complex();

        // Create a chain: 2 * horizontal_edge + 3 * vertical_edge
        let horizontal_edge = Cube::from_extent(Orthant::from([1, 1]), &[true, false]);
        let vertical_edge = Cube::from_extent(Orthant::from([1, 1]), &[false, true]);

        let mut edge_chain: Chain<Cube, Cyclic<7>> = Chain::new();
        edge_chain.insert_or_add(horizontal_edge, Cyclic::from(2));
        edge_chain.insert_or_add(vertical_edge, Cyclic::from(3));

        let chain_boundary = complex.boundary(&edge_chain);

        // Expected: 2*vertex(2,1) + 3*vertex(1,2) - 5*vertex(1,1)
        let vertex_11 = Cube::vertex(Orthant::from([1, 1]));
        let vertex_21 = Cube::vertex(Orthant::from([2, 1]));
        let vertex_12 = Cube::vertex(Orthant::from([1, 2]));

        assert_eq!(chain_boundary.coefficient(&vertex_11), Cyclic::from(2)); // -5 = 2 (mod 7)
        assert_eq!(chain_boundary.coefficient(&vertex_21), Cyclic::from(2));
        assert_eq!(chain_boundary.coefficient(&vertex_12), Cyclic::from(3));
    }

    #[test]
    fn chain_coboundary_computation() {
        let complex = create_2d_graded_complex();

        // Create a chain: vertex(1,1) + 2*vertex(1,2) - vertex(2,1)
        let vertex_11 = Cube::vertex(Orthant::from([1, 1]));
        let vertex_12 = Cube::vertex(Orthant::from([1, 2]));
        let vertex_21 = Cube::vertex(Orthant::from([2, 1]));

        let mut vertex_chain: Chain<Cube, Cyclic<7>> = Chain::new();
        vertex_chain.insert_or_add(vertex_11, Cyclic::one());
        vertex_chain.insert_or_add(vertex_12, Cyclic::from(2));
        vertex_chain.insert_or_add(vertex_21, -Cyclic::one());

        let chain_coboundary = complex.coboundary(&vertex_chain);

        // Coboundary should contain edges incident to these vertices
        let horizontal_edge = Cube::from_extent(Orthant::from([1, 1]), &[true, false]);
        let vertical_edge = Cube::from_extent(Orthant::from([1, 1]), &[false, true]);

        let coboundary_cells: Vec<_> = chain_coboundary
            .iter()
            .map(|(cube, _)| cube.clone())
            .collect();
        assert!(!coboundary_cells.is_empty());
        assert!(
            coboundary_cells.contains(&horizontal_edge)
                || coboundary_cells.contains(&vertical_edge)
        );
    }

    #[test]
    fn filtered_cell_boundary() {
        let complex = create_2d_graded_complex();

        // Boundary of horizontal edge, filtered to vertices at x=1
        let horizontal_edge = Cube::from_extent(Orthant::from([1, 1]), &[true, false]);
        let boundary_filtered: Chain<Cube, Cyclic<7>> = complex
            .cell_boundary(&horizontal_edge)
            .into_iter()
            .filter(|(cube, _)| cube.base()[0] == 1)
            .collect();

        let vertex_11 = Cube::vertex(Orthant::from([1, 1]));
        let mut expected = Chain::new();
        expected.insert_or_add(vertex_11, -Cyclic::one());

        assert_eq!(boundary_filtered, expected);
    }

    #[test]
    fn filtered_boundary() {
        let complex = create_2d_graded_complex();

        let horizontal_edge = Cube::from_extent(Orthant::from([1, 1]), &[true, false]);
        let vertical_edge = Cube::from_extent(Orthant::from([1, 1]), &[false, true]);

        let mut edge_chain: Chain<Cube, Cyclic<7>> = Chain::new();
        edge_chain.insert_or_add(horizontal_edge, Cyclic::one());
        edge_chain.insert_or_add(vertical_edge, Cyclic::one());

        // Compute boundary, then filter to vertices with y=1
        let boundary_filtered: Chain<Cube, Cyclic<7>> = complex
            .boundary(&edge_chain)
            .into_iter()
            .filter(|(cube, _)| cube.base()[1] == 1)
            .collect();

        let vertex_11 = Cube::vertex(Orthant::from([1, 1]));
        let vertex_21 = Cube::vertex(Orthant::from([2, 1]));

        let mut expected = Chain::new();
        expected.insert_or_add(vertex_11, Cyclic::from(5)); // -1 + -1 = -2 = 5 (mod 7)
        expected.insert_or_add(vertex_21, Cyclic::one());

        assert_eq!(boundary_filtered, expected);
    }

    #[test]
    fn filtered_cell_coboundary() {
        let complex = create_2d_graded_complex();

        // Coboundary of vertex, filtered to only horizontal edges
        let vertex = Cube::vertex(Orthant::from([1, 1]));
        let coboundary_filtered: Chain<Cube, Cyclic<7>> = complex
            .cell_coboundary(&vertex)
            .into_iter()
            .filter(|(cube, _)| cube.extent() == vec![true, false])
            .collect();

        let horizontal_edge1 = Cube::new(Orthant::from([1, 1]), Orthant::from([1, 0]));
        let horizontal_edge2 = Cube::new(Orthant::from([0, 1]), Orthant::from([0, 0]));

        for (cube, _) in &coboundary_filtered {
            assert_eq!(cube.extent(), vec![true, false]);
        }

        assert!(
            coboundary_filtered.coefficient(&horizontal_edge1) != Cyclic::zero()
                || coboundary_filtered.coefficient(&horizontal_edge2) != Cyclic::zero()
        );
    }

    #[test]
    fn filtered_coboundary() {
        let complex = create_2d_graded_complex();

        let vertex_11 = Cube::vertex(Orthant::from([1, 1]));
        let vertex_21 = Cube::vertex(Orthant::from([2, 1]));

        let mut vertex_chain: Chain<Cube, Cyclic<7>> = Chain::new();
        vertex_chain.insert_or_add(vertex_11, Cyclic::one());
        vertex_chain.insert_or_add(vertex_21, Cyclic::one());

        // Compute coboundary, then filter to only vertical edges
        let coboundary_filtered: Chain<Cube, Cyclic<7>> = complex
            .coboundary(&vertex_chain)
            .into_iter()
            .filter(|(cube, _)| cube.extent() == vec![false, true])
            .collect();

        for (cube, _) in &coboundary_filtered {
            assert_eq!(cube.extent(), vec![false, true]);
        }

        assert!(coboundary_filtered != Chain::new());
    }

    #[test]
    #[should_panic(expected = "ambient dimension cannot exceed")]
    fn zeros_dimension_overflow() {
        let _orthant = Orthant::zeros(Orthant::MAX_DIMENSION + 1);
    }

    #[test]
    #[should_panic(expected = "ambient dimension cannot exceed")]
    fn from_iter_dimension_overflow() {
        let _orthant: Orthant = (0..=Orthant::MAX_DIMENSION).map(|i| i as i16).collect();
    }
}
