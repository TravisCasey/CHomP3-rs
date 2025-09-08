// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

pub use coreduction::CoreductionMatching;
pub use morse::MatchResult;
pub use traits::MorseMatching;

mod coreduction;
mod morse;
mod traits;
mod util;
