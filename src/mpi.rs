// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Re-exports of MPI types for user-facing MPI setup.
//!
//! Users who enable the `mpi` feature should use these re-exports for
//! initializing MPI and obtaining communicators. The actual parallel
//! execution logic lives in the internal [`parallel`](crate::parallel)
//! module.

pub use mpi::{
    environment::{Universe, initialize, processor_name},
    topology::SimpleCommunicator,
    traits::Communicator,
};
