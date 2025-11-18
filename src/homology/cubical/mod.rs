// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#[cfg(feature = "parallel")]
pub use parallel::CHomPMultiprocessingError;
pub use top_cubical::{TopCubicalMatching, TopCubicalMatchingBuilder};

mod gradient;
mod orthant_matching;
#[cfg(feature = "parallel")]
mod parallel;
mod subgrid;
mod top_cubical;
