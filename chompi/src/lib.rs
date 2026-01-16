// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! MPI-based work-stealing executor for distributed computation.
//!
//! This crate provides [`MpiExecutor`], which distributes work items across
//! MPI processes using a work-stealing pattern. The root process (rank 0)
//! lazily iterates work items and assigns them to workers on demand.
//!
//! # Example
//!
//! ```ignore
//! use chompi::MpiExecutor;
//!
//! if let Some(mut executor) = MpiExecutor::from_world() {
//!     let results = executor.execute("Compute squares", 0..100, |x| Some(x * x));
//!
//!     if executor.is_root() {
//!         println!("Results: {:?}", results);
//!     }
//! }
//! ```

mod executor;

pub use executor::MpiExecutor;
pub use mpi::{topology::SimpleCommunicator, traits::Communicator};
pub use serde::{Serialize, de::DeserializeOwned};
