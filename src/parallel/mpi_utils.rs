// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Shared MPI utilities for parallel execution backends.

use mpi::{
    topology::SimpleCommunicator,
    traits::{Communicator, Root},
};
use postcard::{from_bytes, to_allocvec};
use serde::{Serialize, de::DeserializeOwned};

/// Message tags for the MPI work-stealing protocol.
///
/// These integer values are part of the wire protocol between root and workers
/// and must remain stable.
#[derive(Clone, Copy)]
#[repr(i32)]
pub(crate) enum MPITag {
    /// Worker to root: ready for work.
    WorkRequest = 1,
    /// Root to worker: batch of items to process.
    WorkAssignment = 2,
    /// Worker to root: computed results.
    ResultSubmission = 3,
    /// Root to worker: no more work, shut down.
    Shutdown = 4,
}

/// Broadcast a value from root (rank 0) to all processes.
///
/// All processes must call this function. The root process provides the data,
/// and all processes (including root) receive the broadcast value.
///
/// # Panics
///
/// - If serialization or deserialization fails.
/// - If the serialized payload exceeds `i32::MAX` bytes.
pub(crate) fn broadcast<T>(comm: &SimpleCommunicator, data: &T) -> T
where
    T: Serialize + DeserializeOwned,
{
    let encoded = if comm.rank() == 0 {
        to_allocvec(data).expect("failed to encode for broadcast")
    } else {
        Vec::new()
    };

    let mut len: i32 = encoded
        .len()
        .try_into()
        .expect("broadcast payload exceeds i32::MAX bytes");
    comm.process_at_rank(0).broadcast_into(&mut len);

    let mut buffer = if comm.rank() == 0 {
        encoded
    } else {
        vec![0u8; len as usize]
    };
    comm.process_at_rank(0).broadcast_into(&mut buffer);

    from_bytes(&buffer).expect("failed to decode broadcast data")
}
