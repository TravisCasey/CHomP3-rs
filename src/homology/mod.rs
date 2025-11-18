// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

pub use coreduction::CoreductionMatching;
#[cfg(feature = "parallel")]
pub use cubical::CHomPMultiprocessingError;
pub use cubical::{TopCubicalMatching, TopCubicalMatchingBuilder};
pub use morse::MatchResult;
pub use traits::MorseMatching;

mod coreduction;
mod cubical;
mod morse;
mod traits;
mod util;
