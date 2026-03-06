# CHomP3-rs

**C**omputational **Hom**ology **P**roject 3 is a Rust library for computing the homology of cell complexes using discrete Morse theory. It targets high-dimensional complexes where size or complexity may preclude standard computational methods.

The general approach is:

1. Represent a topological space as a cell complex (currently limited to cubical complexes).
2. Compute an acyclic Morse matching to identify critical cells.
3. Build a reduced Morse complex containing only critical cells.
4. Compute homology of the much smaller reduced complex.
5. Repeat as needed.

This project is inspired by and partially built upon previous iterations of CHomP. See the [NOTICE](./NOTICE) for attribution.

## Status

This crate is in an early developmental state. The public API is unstable and may change at any point. Documentation is incomplete and examples are limited.

## Getting Started

The prelude re-exports the core types and traits needed for a typical workflow:

```rust
use chomp3rs::prelude::*;
```

This includes coefficient types (`F2`, `Cyclic`), complex types (`CubicalComplex`, `Cube`, `Orthant`), chain types (`Chain`, `OrderedChain`), and Morse matching types (`CoreductionMatching`, `TopCubicalMatching`). See the [API documentation](https://docs.rs/chomp3rs) for details.

## Feature Flags

- **`mpi`**: Enables MPI-based distributed computation for large cubical complexes. Requires an MPI implementation installed on the system and adds serde bounds on generic type parameters.

## Contributing

All CI checks are defined in `scripts/ci.sh`, which is also the script used by GitHub Actions. Run it locally before submitting changes.
