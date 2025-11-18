// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::mem::{replace, take};
use std::rc::Rc;

use bincode::error::{DecodeError, EncodeError};
use bincode::serde::{decode_from_slice, encode_to_vec};
use mpi::collective::CommunicatorCollectives;
use mpi::datatype::PartitionMut;
use mpi::topology::{Color, SimpleCommunicator};
use mpi::traits::{Communicator, Destination, Source};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tracing::{error, info, trace, warn};

use crate::homology::cubical::gradient::TopCubicalGradientPropagator;
use crate::homology::cubical::subgrid::{GridSubdivision, Subgrid};
use crate::homology::cubical::top_cubical::TopCubicalMatchingConfig;
use crate::{Cube, CubicalComplex, Grader, ModuleLike, Orthant, TopCubeGrader};

/// Arbitrary choice of tags for each type of parllel communication.
enum MPITag {
    Config = 106732,
    ConfigCheckResult = 106733,
    AssignSubgrid = 106734,
    CriticalCellResult = 106735,
    AssignCriticalCell = 106737,
    GradientResult = 106738,
}

/// Error type pertaining to parallel computation communication and validation
/// failures.
#[derive(Debug)]
pub enum CHomPMultiprocessingError {
    /// The config for a worker process does not match that of the root
    /// controlling process.
    RejectedConfiguration(String),

    /// After validating child worker processes, none remained.
    NoValidChildren,

    /// Error while attemtping to serialize an object for inter-process
    /// communication.
    SerializationError(String),
}

impl Display for CHomPMultiprocessingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RejectedConfiguration(config) => {
                write!(
                    f,
                    "The following child process TopCubicalMatching configuration was rejected by \
                    the root process:\n{config:?}"
                )
            }
            Self::NoValidChildren => {
                write!(
                    f,
                    "After excluding processes with invalid configs or encoder/decoder errors, \
                    only the root process remainded."
                )
            }
            Self::SerializationError(message) => {
                write!(
                    f,
                    "The following error occurred while serializing or deserializing data for \
                    inter-process communication:\n{message}"
                )
            }
        }
    }
}

impl Error for CHomPMultiprocessingError {}

/// Interface of the root and child forms of the parallel handler structs
/// (which are [`TopCubicalParallelRoot`] and [`TopCubicalParallelChild`],
/// respectively) which are called through this trait without knowledge of
/// whether the calling process is the root or a child.
pub(super) trait TopCubicalParallelHandler<LM> {
    /// Verify `config` between the root process and the child worker processes.
    /// Returns [`CHomPMultiprocessingError::NoValidChildren`] on the root
    /// process if no children passed validation. Returns
    /// [`CHomPMultiprocessingError::RejectedConfiguration`] on the child
    /// process if it did not pass verification, and
    /// [`CHomPMultiprocessingError::SerializationError`] if there was an
    /// issue encoding the configuration for transmittal.
    fn verify_configuration(
        &mut self,
        config: &TopCubicalMatchingConfig,
    ) -> Result<(), CHomPMultiprocessingError>;

    /// Compute and store the critical cells, and sync them between all
    /// procceses.
    fn compute_critical_cells(&mut self);

    /// Compute and store the gradient (boundary) of each critical cell of the
    /// previous step, and sync them between all processes. Returns a
    /// [`CHomPMultiprocessingError::SerializationError`] if there was an issue
    /// encoding the boundary chains for transmittal.
    fn compute_gradient(&mut self) -> Result<(), CHomPMultiprocessingError>;

    /// Consume the handler and return the critical cells, projection map,
    /// and boundaries of each critical cell.
    fn finalize(&mut self) -> (Vec<Cube>, HashMap<Cube, u32>, Vec<LM>);
}

pub(super) struct TopCubicalParallelRoot<LM> {
    parent_comm: Rc<dyn Communicator>,
    comm: Option<SimpleCommunicator>,
    subdivision: GridSubdivision,
    critical_cells: Vec<Cube>,
    projection: HashMap<Cube, u32>,
    boundaries: Vec<LM>,
}

impl<LM> TopCubicalParallelRoot<LM> {
    /// Create a parallel handler for the root, intended to be used aside from
    /// this method by the trait it implements, [`TopCubicalParallelHandler`].
    /// `parent_comm` is an MPI communicator for processes to use; if any of
    /// them fail to verify their configurations, it will ignore them for the
    /// remainder of computation.
    pub(super) fn new<UM, G>(
        parent_comm: Rc<dyn Communicator>,
        complex: &CubicalComplex<UM, TopCubeGrader<G>>,
        config: &TopCubicalMatchingConfig,
    ) -> Self
    where
        G: Grader<Orthant>,
    {
        let subdivision = GridSubdivision::new(
            complex.minimum().clone(),
            complex.maximum().clone(),
            config.subgrid_shape.clone(),
            None,
        );

        Self {
            parent_comm,
            comm: None,
            subdivision,
            critical_cells: Vec::new(),
            projection: HashMap::new(),
            boundaries: Vec::new(),
        }
    }

    fn decode_config(encoded_config: &[u8]) -> Result<TopCubicalMatchingConfig, DecodeError> {
        decode_from_slice::<TopCubicalMatchingConfig, _>(
            encoded_config,
            bincode::config::standard(),
        )
        .map(|(config, _)| config)
    }
}

impl<LM, R> TopCubicalParallelHandler<LM> for TopCubicalParallelRoot<LM>
where
    LM: ModuleLike<Cell = u32, Ring = R>,
    R: DeserializeOwned + Serialize + Clone,
{
    fn verify_configuration(
        &mut self,
        config: &TopCubicalMatchingConfig,
    ) -> Result<(), CHomPMultiprocessingError> {
        let comm_size = self.parent_comm.size();
        info!("Verifying TopCubicalMatching configurations on {comm_size} processes");

        // skip root process
        for _ in 1..comm_size {
            let (encoded_config, status) = self
                .parent_comm
                .any_process()
                .receive_vec_with_tag::<u8>(MPITag::Config as i32);
            let source_rank = status.source_rank();
            let config_verified = match Self::decode_config(&encoded_config) {
                Ok(decoded_config) => {
                    if decoded_config != *config {
                        warn!(
                            "Could not verify TopCubicalMatching config from process rank \
                            {source_rank}. Continuing with other processes."
                        );
                        false
                    } else {
                        info!(
                            "Successfully verified TopCubicalMatching configuration of process \
                            rank {source_rank}."
                        );
                        true
                    }
                }
                Err(decode_error) => {
                    warn!(
                        "Encountered the following error while decoding TopCubicalMatching \
                        configuration from process {source_rank}:\n{decode_error}\nContinuing with \
                        other processes."
                    );
                    false
                }
            };

            // Let the other process know it was verified so that it can also join the new
            // communicator.
            self.parent_comm
                .process_at_rank(source_rank)
                .send_with_tag(&config_verified, MPITag::ConfigCheckResult as i32);
        }
        self.comm = self
            .parent_comm
            .split_by_color_with_key(Color::with_value(0), self.parent_comm.rank());
        let new_comm_size = self.comm.as_ref().unwrap().size();
        info!("Created new communicator on {new_comm_size} processes");

        if new_comm_size <= 1 {
            error!("No non-root processes validated.");
            Err(CHomPMultiprocessingError::NoValidChildren)
        } else {
            Ok(())
        }
    }

    fn compute_critical_cells(&mut self) {
        let comm = self.comm.as_ref().unwrap();
        let comm_size = comm.size();
        info!("Beginning critical cell computation with {comm_size} processes...");

        let mut processes_active = comm_size - 1;
        let mut subgrid_iter = self.subdivision.nonempty_subgrids();

        // Assign subgrids to each child worker process to find critical cells
        // within. They return `true` if the computation was successful and
        // `false` otherwise. An empty message indicates computation is over.
        while processes_active != 0 {
            let (success, status) = comm
                .any_process()
                .receive_with_tag::<bool>(MPITag::CriticalCellResult as i32);
            if !success {
                warn!(
                    "Error value received while computing critical cells from process rank {}.",
                    status.source_rank()
                );
            }

            let message = match subgrid_iter.next() {
                Some((subgrid_min_orthant, subgrid_max_orthant)) => {
                    vec![subgrid_min_orthant.clone(), subgrid_max_orthant.clone()]
                }
                None => {
                    processes_active -= 1;
                    Vec::new()
                }
            };
            comm.process_at_rank(status.source_rank())
                .send_with_tag(&message, MPITag::AssignSubgrid as i32);
        }

        info!("Critical cell computation complete.");
        info!("Synchronizing critical cells...");
        sync_critical_cells(comm, &mut self.critical_cells, &mut self.projection);
        info!("Critical cell synchronization complete.");
    }

    fn compute_gradient(&mut self) -> Result<(), CHomPMultiprocessingError> {
        let comm = self.comm.as_ref().unwrap();
        let comm_size = comm.size();
        self.boundaries.resize(self.critical_cells.len(), LM::new());
        info!("Beginning gradient computation with {comm_size} processes...");

        let mut processes_active = comm_size - 1;
        let mut critical_cell_index = 0u32;

        // Assign a critical cell index (all processes have synchronized
        // critical cell vectors at this point) to child worker processes. They
        // return a boolean value representing if the computation was
        // successful or not.
        while processes_active != 0 {
            let (success, status) = comm
                .any_process()
                .receive_with_tag::<bool>(MPITag::GradientResult as i32);
            if !success {
                warn!(
                    "Error value received while computing gradient on process rank {}",
                    status.source_rank()
                );
            }

            if critical_cell_index as usize >= self.critical_cells.len() {
                processes_active -= 1;
            }
            comm.process_at_rank(status.source_rank())
                .send_with_tag(&critical_cell_index, MPITag::AssignCriticalCell as i32);

            critical_cell_index += 1;
        }

        info!("Gradient computation complete.");
        info!("Synchronizing gradients...");
        sync_gradient(comm, &[], &mut self.boundaries)?;
        info!("Gradient synchronization complete.");
        Ok(())
    }

    fn finalize(&mut self) -> (Vec<Cube>, HashMap<Cube, u32>, Vec<LM>) {
        (
            take(&mut self.critical_cells),
            take(&mut self.projection),
            take(&mut self.boundaries),
        )
    }
}

pub(super) struct TopCubicalParallelChild<'a, UM, G, LM>
where
    UM: ModuleLike<Cell = Cube>,
    G: Grader<Orthant>,
{
    parent_comm: Rc<dyn Communicator>,
    comm: Option<SimpleCommunicator>,
    subgrid: Subgrid<G>,
    critical_cells: Vec<Cube>,
    projection: HashMap<Cube, u32>,
    gradient_computer: TopCubicalGradientPropagator<'a, UM, G>,
    boundaries: Vec<LM>,
}

impl<'a, UM, G, LM> TopCubicalParallelChild<'a, UM, G, LM>
where
    UM: ModuleLike<Cell = Cube>,
    G: Grader<Orthant> + Clone,
{
    pub(super) fn new(
        parent_comm: Rc<dyn Communicator>,
        complex: &'a CubicalComplex<UM, TopCubeGrader<G>>,
        config: &TopCubicalMatchingConfig,
    ) -> Self
    where
        G: Grader<Orthant>,
    {
        let subgrid = Subgrid::new(
            complex,
            config.maximum_critical_grade,
            config.maximum_critical_dimension,
        );
        let gradient_computer = TopCubicalGradientPropagator::new(complex);

        Self {
            parent_comm,
            comm: None,
            subgrid,
            critical_cells: Vec::new(),
            projection: HashMap::new(),
            gradient_computer,
            boundaries: Vec::new(),
        }
    }

    fn encode_config(config: &TopCubicalMatchingConfig) -> Result<Vec<u8>, EncodeError> {
        encode_to_vec(config, bincode::config::standard())
    }
}

impl<'a, UM, G, LM, R> TopCubicalParallelHandler<LM> for TopCubicalParallelChild<'a, UM, G, LM>
where
    UM: ModuleLike<Cell = Cube, Ring = R>,
    G: Grader<Orthant> + Clone,
    LM: ModuleLike<Cell = u32, Ring = R>,
    R: Serialize + DeserializeOwned + Clone,
{
    fn verify_configuration(
        &mut self,
        config: &TopCubicalMatchingConfig,
    ) -> Result<(), CHomPMultiprocessingError> {
        info!("Attempting to get TopCubicalMatching config verified");
        let mut result = Ok(());
        let color = match Self::encode_config(config) {
            Ok(encoded_config) => {
                self.parent_comm
                    .process_at_rank(0)
                    .send_with_tag(&encoded_config, MPITag::Config as i32);
                match self
                    .parent_comm
                    .process_at_rank(0)
                    .receive_with_tag(MPITag::ConfigCheckResult as i32)
                    .0
                {
                    true => {
                        info!("TopCubicalMatching configuration was successfully verified.");
                        Color::with_value(0)
                    }
                    false => {
                        error!("TopCubicalMatching configuration was rejected.");
                        result = Err(CHomPMultiprocessingError::RejectedConfiguration(format!(
                            "{:?}",
                            config
                        )));
                        Color::undefined()
                    }
                }
            }
            Err(encode_error) => {
                error!(
                    "Encountered the following error while encoding TopCubicalMatching \
                    configuration:\n{encode_error}"
                );
                result = Err(CHomPMultiprocessingError::SerializationError(
                    encode_error.to_string(),
                ));
                Color::undefined()
            }
        };

        self.comm = self
            .parent_comm
            .split_by_color_with_key(color, self.parent_comm.rank());
        result
    }

    fn compute_critical_cells(&mut self) {
        let comm = self.comm.as_ref().unwrap();

        let mut success = true;
        loop {
            comm.process_at_rank(0)
                .send_with_tag(&success, MPITag::CriticalCellResult as i32);
            success = true;
            let subgrid_vec = comm
                .process_at_rank(0)
                .receive_vec_with_tag::<Orthant>(MPITag::AssignSubgrid as i32)
                .0;

            if subgrid_vec.is_empty() {
                break;
            }
            if subgrid_vec.len() != 2 {
                error!(
                    "Received invalid subgrid specification containing {} orthants instead of two. \
                    Skipping.",
                    subgrid_vec.len()
                );
                success = false;
                continue;
            }

            trace!(
                "Beginning critical cell computation on subgrid:\n{{{}, {}}}",
                subgrid_vec[0], subgrid_vec[1]
            );

            let bcm = self
                .subgrid
                .match_subgrid(subgrid_vec[0].clone(), subgrid_vec[1].clone());
            for (_, mut critical, _) in bcm {
                self.critical_cells.append(&mut critical);
            }
        }

        sync_critical_cells(comm, &mut self.critical_cells, &mut self.projection);
    }

    fn compute_gradient(&mut self) -> Result<(), CHomPMultiprocessingError> {
        let comm = self.comm.as_ref().unwrap();
        let mut boundary_pairs = Vec::new();
        self.boundaries.resize(self.critical_cells.len(), LM::new());

        let mut success = true;
        loop {
            comm.process_at_rank(0)
                .send_with_tag(&success, MPITag::GradientResult as i32);
            success = true;
            let cell_index = comm
                .process_at_rank(0)
                .receive_with_tag::<u32>(MPITag::AssignCriticalCell as i32)
                .0;

            if cell_index as usize >= self.critical_cells.len() {
                break;
            }

            trace!("Beginning gradient computation on cell {cell_index}",);

            let upper_gradient_chain = self
                .gradient_computer
                .compute_gradient(&self.critical_cells[cell_index as usize]);
            boundary_pairs.push((
                cell_index,
                upper_gradient_chain
                    .into_iter()
                    .map(|(cube, coef)| (self.projection[&cube], coef))
                    .collect(),
            ));
        }

        sync_gradient(comm, &boundary_pairs, &mut self.boundaries)
    }

    fn finalize(&mut self) -> (Vec<Cube>, HashMap<Cube, u32>, Vec<LM>) {
        (
            take(&mut self.critical_cells),
            take(&mut self.projection),
            take(&mut self.boundaries),
        )
    }
}

/// Essentially a wrapper for MPI Allgather varcount to synchronize critical
/// cells across all processes.
fn sync_critical_cells(
    comm: &SimpleCommunicator,
    critical_cells: &mut Vec<Cube>,
    projection: &mut HashMap<Cube, u32>,
) {
    let comm_size = comm.size();

    let mut critical_cells_per_process = vec![0i32; comm_size as usize];
    comm.all_gather_into(
        &(critical_cells.len() as i32),
        &mut critical_cells_per_process,
    );
    let displs: Vec<i32> = critical_cells_per_process
        .iter()
        .scan(0, |acc, x| {
            let tmp = *acc;
            *acc += *x;
            Some(tmp)
        })
        .collect();
    let num_critical_cells = displs.last().unwrap() + critical_cells_per_process.last().unwrap();
    let send_critical_cells = replace(
        critical_cells,
        vec![Cube::vertex(Orthant::zeros(0)); num_critical_cells as usize],
    );

    let mut partition = PartitionMut::new(critical_cells, critical_cells_per_process, displs);
    comm.all_gather_varcount_into(&send_critical_cells, &mut partition);

    for (cell_index, cell) in critical_cells.iter().enumerate() {
        projection.insert(cell.clone(), cell_index as u32);
    }
}

/// Essentially a wrapper for MPI Allgather varcount to synchronize boundaries
/// across all processes.
fn sync_gradient<LM, R>(
    comm: &SimpleCommunicator,
    boundary_pairs: &[(u32, Vec<(u32, R)>)],
    boundaries: &mut Vec<LM>,
) -> Result<(), CHomPMultiprocessingError>
where
    LM: ModuleLike<Cell = u32, Ring = R>,
    R: DeserializeOwned + Serialize + Clone,
{
    let comm_size = comm.size();

    // For transmitting, we express the boundary as pairs (cell index, vector of
    // boundary cell and coefficient).
    let mut result = Ok(());
    let encoded_boundary_pairs = match encode_to_vec(boundary_pairs, bincode::config::standard()) {
        Ok(encoded) => encoded,
        Err(encode_error) => {
            error!(
                "Encountered the following error while encoding gradient pairs:\n{encode_error}"
            );
            result = Err(CHomPMultiprocessingError::SerializationError(
                encode_error.to_string(),
            ));
            Vec::new()
        }
    };

    // First get the number of bytes each process will transmit.
    let mut bytes_per_process = vec![0i32; comm_size as usize];
    comm.all_gather_into(
        &(encoded_boundary_pairs.len() as i32),
        &mut bytes_per_process,
    );
    let displs: Vec<i32> = bytes_per_process
        .iter()
        .scan(0, |acc, x| {
            let tmp = *acc;
            *acc += *x;
            Some(tmp)
        })
        .collect();
    let num_bytes = displs.last().unwrap() + bytes_per_process.last().unwrap();
    let mut gathered_encoded_boundary_pairs = vec![0u8; num_bytes as usize];

    let mut partition = PartitionMut::new(
        &mut gathered_encoded_boundary_pairs,
        bytes_per_process.clone(),
        displs,
    );
    comm.all_gather_varcount_into(&encoded_boundary_pairs, &mut partition);

    // Now decode the byte stream into boundary chains
    let mut remaining_slice = gathered_encoded_boundary_pairs.as_slice();
    let mut encoded_slice;
    for num_bytes in bytes_per_process {
        (encoded_slice, remaining_slice) = remaining_slice.split_at(num_bytes as usize);
        let encoded_vec = encoded_slice.to_vec();
        match decode_from_slice::<Vec<(u32, Vec<(u32, R)>)>, _>(
            &encoded_vec,
            bincode::config::standard(),
        ) {
            Ok((decoded_pairs, _)) => {
                for (critical_cell_index, boundary_pairs) in decoded_pairs {
                    if boundaries.len() <= critical_cell_index as usize {
                        boundaries.resize(critical_cell_index as usize, LM::new());
                    }
                    boundaries[critical_cell_index as usize] =
                        LM::from_iter(boundary_pairs.into_iter())
                }
            }
            Err(decode_error) => {
                error!(
                    "Encountered the following error while decoding gradient pairs:\n{decode_error}"
                );
                result = match result {
                    Err(CHomPMultiprocessingError::SerializationError(message)) => {
                        Err(CHomPMultiprocessingError::SerializationError(
                            decode_error.to_string() + "\n" + &message,
                        ))
                    }
                    _ => Err(CHomPMultiprocessingError::SerializationError(
                        decode_error.to_string(),
                    )),
                };
            }
        }
    }

    result
}
