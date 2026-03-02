//! Benchmark data generation for rotated manifolds and S^n constructions.
//!
//! This module handles the complete data generation pipeline used by the
//! benchmark suite:
//!
//! 1. **Sample** the manifold surface parametrically in its native dimension
//! 2. **Embed** into n-D via a random orthonormal basis (QR decomposition of a
//!    random Gaussian matrix)
//! 3. **Discretize** to integer grid coordinates via bounding-box normalization
//! 4. **Thicken** (S^1 only) to create a tubular neighborhood
//!
//! A dimension-dependent grid resolution ([`manifold_resolution`]) ensures that
//! the effective torus minor radius satisfies `r_eff > 2*sqrt(n)`, which
//! guarantees the discretized tube remains hollow and the center hole is
//! preserved.
//!
//! ## Circle (S^1)
//!
//! The circle is sampled parametrically and then **thickened** by adding all
//! neighboring cubes within an L-infinity ball of radius 1. This creates a
//! tubular neighborhood that deformation-retracts onto the curve, preserving
//! its homology (Betti numbers [1, 1]). Thickening is safe for 1-manifolds
//! and provides meaningful complex sizes that scale with ambient dimension.
//!
//! ## Torus (T^2)
//!
//! The torus is sampled parametrically using angular coordinates (theta, phi)
//! and discretized **without thickening**. Unlike S^1, thickening a 2-manifold
//! fills the surface patches solid, collapsing the interior cavity and
//! destroying `H_2` (producing Betti [1, 1, 0] instead of the correct
//! [1, 2, 1]). The dimension-dependent resolution ensures sufficient
//! oversampling for the discretized surface to have the correct topology.
//!
//! ## Reproducibility
//!
//! Generation uses a fixed seed ([`GENERATION_SEED`]) for reproducibility.
//! The same seed produces identical data across runs and platforms.

// Not all functions are used by every bench binary.
#![allow(dead_code)]

use std::{
    collections::BTreeSet,
    f64::consts::PI,
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};

use chomp3rs::{complexes::OrthantIterator, prelude::*};
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// Fixed seed for reproducible data generation.
///
/// Change this value to generate different random rotations.
const GENERATION_SEED: u64 = 9171998;

/// Thickness for S^1 surface thickening (L-infinity ball radius).
///
/// Creates a tubular neighborhood around the discretized curve, ensuring the
/// cubical complex is connected. This is safe for 1-manifolds: the tubular
/// neighborhood deformation-retracts onto S^1, preserving homology.
const THICKNESS: i16 = 1;

/// Circle radius in source (2D) space.
const S1_RADIUS: f64 = 3.0;

/// Oversampling factor for S^1 parametric sampling.
///
/// Controls sampling density of the base curve before thickening. Since
/// thickening already ensures connectivity, a moderate oversampling suffices.
const S1_OVERSAMPLING: f64 = 2.0;

/// Torus major radius in source (3D) space.
const TORUS_MAJOR_RADIUS: f64 = 6.0;

/// Torus minor radius in source (3D) space.
const TORUS_MINOR_RADIUS: f64 = 3.0;

/// Oversampling factor for T^2 parametric sampling.
///
/// Ensures adjacent sample points are less than 1 grid unit apart after
/// discretization, which guarantees the surface cubes form a connected complex
/// with the correct topology.
const TORUS_OVERSAMPLING: f64 = 3.0;

// --- Data Loading ---

/// Load orthants from a CSV file, generating the data if it doesn't exist.
///
/// Each line in the CSV contains comma-separated integer coordinates for one
/// orthant.
///
/// # Panics
///
/// Panics if the file cannot be read or contains invalid coordinate data.
#[must_use]
pub fn load_orthants(path: &str) -> Vec<Orthant> {
    if !Path::new(path).exists() {
        eprintln!("Benchmark data not found at {path}, generating...");
        generate_all_data();
    }

    let content = fs::read_to_string(path).expect("Failed to read CSV file");
    content
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let coords: Vec<i16> = line
                .split(',')
                .map(|s| s.trim().parse().expect("Invalid coordinate"))
                .collect();
            Orthant::new(&coords)
        })
        .collect()
}

/// Generate orthants for S^n (n-sphere) embedded in (n+1)-dimensional space.
///
/// The sphere is constructed as the boundary of an (n+1)-cube with a hole in
/// the center.
#[must_use]
pub fn generate_sn_orthants(n: usize) -> Vec<Orthant> {
    let minimum = Orthant::new(&vec![0; n + 1]);
    let maximum_included = Orthant::new(&vec![2; n + 1]);
    let hole = Orthant::new(&vec![1; n + 1]);

    OrthantIterator::new(minimum, maximum_included)
        .filter(|orth| *orth != hole)
        .collect()
}

/// Compute the grid resolution for a given ambient dimension.
///
/// Returns `ceil(12 * sqrt(dim)) + 2`, which ensures that after bounding-box
/// normalization with a 2:1 aspect ratio torus, the effective minor radius
/// `r_eff = (res - 1) / 6` exceeds `2 * sqrt(dim)`. This guarantees the
/// discretized torus tube is hollow and the center hole is preserved.
fn manifold_resolution(dim: usize) -> i16 {
    (12.0 * (dim as f64).sqrt()).ceil() as i16 + 2
}

/// Generate all benchmark data (S^1 and T^2 in various dimensions).
fn generate_all_data() {
    eprintln!("Generating benchmark data with seed {GENERATION_SEED}...");
    generate_s1_data(GENERATION_SEED);
    generate_torus_data(GENERATION_SEED.wrapping_add(1000));
    eprintln!("Benchmark data generation complete.");
}

/// Generate S^1 (circle) data for dimensions 2-7.
fn generate_s1_data(base_seed: u64) {
    let n_points = (2.0 * PI * S1_RADIUS * S1_OVERSAMPLING).ceil() as usize;
    let base_points = generate_circle_points(n_points, S1_RADIUS);

    for target_dim in 2..=7 {
        let resolution = manifold_resolution(target_dim);
        let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(target_dim as u64));

        let points: Vec<Vec<f64>> = if target_dim == 2 {
            base_points.iter().map(|p| p.to_vec()).collect()
        } else {
            let basis = random_orthonormal_basis(2, target_dim, &mut rng);
            base_points
                .iter()
                .map(|p| rotate_point(p, &basis))
                .collect()
        };

        let orthants = discretize_with_thickening(&points, resolution, THICKNESS);
        let path = format!("data/s1/dim{target_dim}.csv");
        write_orthants(Path::new(&path), &orthants);
        eprintln!("  Generated {path}: {} orthants", orthants.len());
    }
}

/// Generate T^2 (torus) data for dimensions 3-7.
///
/// Samples the torus surface parametrically, embeds into higher-dimensional
/// space via a random orthonormal basis, and discretizes via bounding-box
/// normalization. No thickening is applied (see module docs for rationale).
fn generate_torus_data(base_seed: u64) {
    let base_points =
        generate_torus_surface_points(TORUS_MAJOR_RADIUS, TORUS_MINOR_RADIUS, TORUS_OVERSAMPLING);

    for target_dim in 3..=7 {
        let resolution = manifold_resolution(target_dim);
        let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(target_dim as u64));

        let points: Vec<Vec<f64>> = if target_dim == 3 {
            base_points.iter().map(|p| p.to_vec()).collect()
        } else {
            let basis = random_orthonormal_basis(3, target_dim, &mut rng);
            base_points
                .iter()
                .map(|p| rotate_point(p, &basis))
                .collect()
        };

        let orthants = discretize_points(&points, resolution);
        let path = format!("data/torus/dim{target_dim}.csv");
        write_orthants(Path::new(&path), &orthants);
        eprintln!("  Generated {path}: {} orthants", orthants.len());
    }
}

/// Generate points on a circle (S^1) in 2D.
fn generate_circle_points(n_points: usize, radius: f64) -> Vec<[f64; 2]> {
    (0..n_points)
        .map(|i| {
            let theta = 2.0 * PI * (i as f64) / (n_points as f64);
            [radius * theta.cos(), radius * theta.sin()]
        })
        .collect()
}

/// Sample points on a torus surface parametrically.
///
/// Returns 3D points on the torus defined by `major_radius` (distance from
/// center to tube center) and `minor_radius` (tube radius). The `oversampling`
/// factor controls sampling density: angular step sizes are chosen so that
/// adjacent samples are less than `1 / oversampling` source-space units apart.
fn generate_torus_surface_points(
    major_radius: f64,
    minor_radius: f64,
    oversampling: f64,
) -> Vec<[f64; 3]> {
    let n_theta = (2.0 * PI * (major_radius + minor_radius) * oversampling).ceil() as usize;
    let n_phi = (2.0 * PI * minor_radius * oversampling).ceil() as usize;

    let mut points = Vec::with_capacity(n_theta * n_phi);
    for i in 0..n_theta {
        let theta = 2.0 * PI * (i as f64) / (n_theta as f64);
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        for j in 0..n_phi {
            let phi = 2.0 * PI * (j as f64) / (n_phi as f64);
            let cos_phi = phi.cos();
            let sin_phi = phi.sin();

            points.push([
                (major_radius + minor_radius * cos_phi) * cos_theta,
                (major_radius + minor_radius * cos_phi) * sin_theta,
                minor_radius * sin_phi,
            ]);
        }
    }
    points
}

/// Generate a random orthonormal basis for embedding `from_dim` into `to_dim`.
///
/// Uses QR decomposition of a random Gaussian matrix. Returns a matrix where
/// each row is an orthonormal basis vector in the target dimension.
fn random_orthonormal_basis(from_dim: usize, to_dim: usize, rng: &mut ChaCha8Rng) -> DMatrix<f64> {
    let random_matrix = DMatrix::from_fn(to_dim, from_dim, |_, _| StandardNormal.sample(rng));
    let qr = random_matrix.qr();
    qr.q().columns(0, from_dim).into_owned()
}

/// Apply rotation matrix to embed a point into higher-dimensional space.
fn rotate_point<const N: usize>(point: &[f64; N], basis: &DMatrix<f64>) -> Vec<f64> {
    let mut result = vec![0.0; basis.nrows()];
    for (i, &coord) in point.iter().enumerate() {
        for j in 0..basis.nrows() {
            result[j] += coord * basis[(j, i)];
        }
    }
    result
}

/// Discretize continuous points to integer grid coordinates (no thickening).
///
/// Uses uniform scaling across all axes (based on the maximum axis range) to
/// preserve geometric proportions. Points are mapped to `[0, resolution-1]`
/// and truncated. Returns unique orthants only.
fn discretize_points(points: &[Vec<f64>], resolution: i16) -> Vec<Vec<i16>> {
    if points.is_empty() {
        return Vec::new();
    }

    let dim = points[0].len();

    // Find bounding box
    let mut min_coords = vec![f64::INFINITY; dim];
    let mut max_coords = vec![f64::NEG_INFINITY; dim];
    for point in points {
        for (i, &coord) in point.iter().enumerate() {
            min_coords[i] = min_coords[i].min(coord);
            max_coords[i] = max_coords[i].max(coord);
        }
    }

    // Use uniform scale: the maximum range across all axes
    let max_range = min_coords
        .iter()
        .zip(max_coords.iter())
        .map(|(&min, &max)| max - min)
        .fold(0.0_f64, f64::max)
        .max(1e-10);

    // Discretize and deduplicate
    let mut orthant_set = BTreeSet::new();
    for point in points {
        let orth: Vec<i16> = point
            .iter()
            .enumerate()
            .map(|(i, &coord)| {
                let normalized = (coord - min_coords[i]) / max_range;
                let cell = (normalized * f64::from(resolution - 1)) as i16;
                cell.clamp(0, resolution - 1)
            })
            .collect();
        orthant_set.insert(orth);
    }

    orthant_set.into_iter().collect()
}

/// Discretize continuous points to integer grid coordinates with thickening.
///
/// Thickening adds neighboring cells within the specified thickness, ensuring
/// the resulting cubical complex is connected.
fn discretize_with_thickening(
    points: &[Vec<f64>],
    resolution: i16,
    thickness: i16,
) -> Vec<Vec<i16>> {
    if points.is_empty() {
        return Vec::new();
    }

    let dim = points[0].len();

    // Find bounding box
    let mut min_coords = vec![f64::INFINITY; dim];
    let mut max_coords = vec![f64::NEG_INFINITY; dim];
    for point in points {
        for (i, &coord) in point.iter().enumerate() {
            min_coords[i] = min_coords[i].min(coord);
            max_coords[i] = max_coords[i].max(coord);
        }
    }

    // Compute range (avoid division by zero)
    let range_coords: Vec<f64> = min_coords
        .iter()
        .zip(max_coords.iter())
        .map(|(&min, &max)| {
            let r = max - min;
            if r < 1e-10 { 1.0 } else { r }
        })
        .collect();

    // Discretize base points (truncate, not round, to match standard practice)
    let base_orthants: Vec<Vec<i16>> = points
        .iter()
        .map(|point| {
            point
                .iter()
                .enumerate()
                .map(|(i, &coord)| {
                    let normalized = (coord - min_coords[i]) / range_coords[i];
                    (normalized * (resolution - 1) as f64) as i16
                })
                .collect()
        })
        .collect();

    // Generate neighbor offsets and collect unique thickened orthants
    let offsets = generate_offsets(dim, thickness);
    let mut orthant_set = BTreeSet::new();

    for orth in &base_orthants {
        for offset in &offsets {
            let new_orth: Vec<i16> = orth
                .iter()
                .zip(offset.iter())
                .map(|(&coord, &off)| coord + off)
                .collect();

            if new_orth.iter().all(|&c| c >= 0 && c < resolution) {
                orthant_set.insert(new_orth);
            }
        }
    }

    orthant_set.into_iter().collect()
}

/// Generate all offset combinations within the given thickness (L-infinity
/// ball).
fn generate_offsets(dim: usize, thickness: i16) -> Vec<Vec<i16>> {
    let mut offsets = vec![vec![]];
    for _ in 0..dim {
        offsets = offsets
            .into_iter()
            .flat_map(|offset| {
                (-thickness..=thickness).map(move |delta| {
                    let mut new_offset = offset.clone();
                    new_offset.push(delta);
                    new_offset
                })
            })
            .collect();
    }
    offsets
}

/// Write orthants to a CSV file.
fn write_orthants(path: &Path, orthants: &[Vec<i16>]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("Failed to create data directory");
    }

    let file = File::create(path).expect("Failed to create data file");
    let mut writer = BufWriter::new(file);

    for orth in orthants {
        let line: String = orth
            .iter()
            .map(i16::to_string)
            .collect::<Vec<_>>()
            .join(",");
        writeln!(writer, "{line}").expect("Failed to write data");
    }
}
