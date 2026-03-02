// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Frontier data structure for distributed wavefront tracking.

use std::collections::{BTreeMap, BTreeSet};

use super::sequential::{WavefrontConfig, cube_to_extent};
use crate::{Chain, Cube, Orthant, Ring, homology::cubical::OrthantMatching};

/// State of an orthant's matching computation.
#[derive(Clone, Debug)]
pub(super) enum MatchingState {
    /// Needs matching, not yet dispatched to a worker.
    Pending,
    /// Dispatched to a worker, awaiting result.
    Dispatched,
    /// Matching computed and ready.
    Ready(OrthantMatching),
}

impl MatchingState {
    pub fn is_pending(&self) -> bool {
        matches!(self, Self::Pending)
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready(_))
    }

    /// Take the matching, replacing with `Pending`.
    ///
    /// # Panics
    /// Panics if not `Ready`.
    pub fn take(&mut self) -> OrthantMatching {
        match std::mem::replace(self, Self::Pending) {
            Self::Ready(matching) => matching,
            Self::Pending => panic!("matching is pending"),
            Self::Dispatched => panic!("matching is dispatched"),
        }
    }
}

/// Entry in the frontier.
struct FrontierEntry<R: Ring> {
    chain: Chain<u32, R>,
    state: MatchingState,
}

/// Frontier of pending orthants with matching state tracking.
pub(super) struct Frontier<R: Ring> {
    entries: BTreeMap<Orthant, FrontierEntry<R>>,
    pending: BTreeSet<Orthant>,
    config: WavefrontConfig,
    /// Number of orthants with precomputed `Ready` matchings (for memory
    /// limiting).
    precomputed_count: usize,
}

impl<R: Ring> Frontier<R> {
    pub fn for_boundary() -> Self {
        Self {
            entries: BTreeMap::new(),
            pending: BTreeSet::new(),
            config: WavefrontConfig::Boundary,
            precomputed_count: 0,
        }
    }

    pub fn for_coboundary() -> Self {
        Self {
            entries: BTreeMap::new(),
            pending: BTreeSet::new(),
            config: WavefrontConfig::Coboundary,
            precomputed_count: 0,
        }
    }

    pub fn config(&self) -> WavefrontConfig {
        self.config
    }

    /// Add a cell to the frontier. New orthants start in `Pending` state.
    pub fn push(&mut self, orthant: Orthant, extent: u32, coef: R) {
        if coef == R::zero() {
            return;
        }

        self.entries
            .entry(orthant)
            .or_insert_with_key(|orthant| {
                self.pending.insert(orthant.clone());
                FrontierEntry {
                    chain: Chain::new(),
                    state: MatchingState::Pending,
                }
            })
            .chain
            .insert_or_add(extent, coef);
    }

    /// Seed the frontier from a chain of cubes.
    pub fn seed(&mut self, chain: impl IntoIterator<Item = (Cube, R)>) {
        for (cube, coef) in chain {
            let extent = cube_to_extent(&cube);
            self.push(cube.base().clone(), extent, coef);
        }
    }

    /// Peek at the next orthant to process.
    pub fn peek_next(&self) -> Option<&Orthant> {
        match self.config {
            WavefrontConfig::Boundary => self.entries.keys().next(),
            WavefrontConfig::Coboundary => self.entries.keys().next_back(),
        }
    }

    /// Get the next orthant in `Pending` state.
    pub fn next_pending(&self) -> Option<&Orthant> {
        match self.config {
            WavefrontConfig::Boundary => self.pending.first(),
            WavefrontConfig::Coboundary => self.pending.last(),
        }
    }

    /// Get the matching state of an orthant.
    pub fn state(&self, orthant: &Orthant) -> Option<&MatchingState> {
        self.entries.get(orthant).map(|e| &e.state)
    }

    /// Mark an orthant as dispatched.
    pub fn mark_dispatched(&mut self, orthant: &Orthant) {
        if let Some(entry) = self.entries.get_mut(orthant) {
            debug_assert!(entry.state.is_pending());
            entry.state = MatchingState::Dispatched;
            self.pending.remove(orthant);
            self.precomputed_count += 1;
        }
    }

    /// Attach a computed matching.
    ///
    /// Returns `Some(matching)` if orthant no longer in frontier.
    pub fn attach_matching(
        &mut self,
        orthant: &Orthant,
        matching: OrthantMatching,
    ) -> Option<OrthantMatching> {
        if let Some(entry) = self.entries.get_mut(orthant) {
            debug_assert!(!entry.state.is_pending());
            entry.state = MatchingState::Ready(matching);
            None
        } else {
            Some(matching)
        }
    }

    /// Check if the next orthant is `Ready`.
    pub fn next_is_ready(&self) -> bool {
        self.peek_next()
            .and_then(|o| self.entries.get(o))
            .is_some_and(|e| e.state.is_ready())
    }

    /// Pop the next orthant with its chain and matching.
    ///
    /// # Panics
    /// Panics if the matching is not `Ready`.
    pub fn pop_next(&mut self) -> Option<(Orthant, Chain<u32, R>, OrthantMatching)> {
        let (orthant, mut entry) = match self.config {
            WavefrontConfig::Boundary => self.entries.pop_first()?,
            WavefrontConfig::Coboundary => self.entries.pop_last()?,
        };

        self.pending.remove(&orthant);
        self.precomputed_count -= 1;
        let matching = entry.state.take();
        Some((orthant, entry.chain, matching))
    }

    /// Number of orthants with precomputed `Ready` matchings.
    pub fn precomputed_count(&self) -> usize {
        self.precomputed_count
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cyclic;

    fn make_matching(extent: u32) -> OrthantMatching {
        OrthantMatching::Critical {
            ace_dual_orthant: Orthant::from([0, 0]),
            ace_extent: extent,
            grade: 0,
        }
    }

    #[test]
    fn matching_state_transitions_and_take() {
        // Pending state
        let mut state = MatchingState::Pending;
        assert!(state.is_pending());
        assert!(!state.is_ready());

        // Dispatched state
        state = MatchingState::Dispatched;
        assert!(!state.is_pending());
        assert!(!state.is_ready());

        // Ready state and take
        state = MatchingState::Ready(make_matching(0b01));
        assert!(!state.is_pending());
        assert!(state.is_ready());

        let matching = state.take();
        assert!(matches!(
            matching,
            OrthantMatching::Critical {
                ace_extent: 0b01,
                ..
            }
        ));

        // Take from non-ready panics
        let mut pending = MatchingState::Pending;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pending.take()));
        assert!(result.is_err());

        let mut dispatched = MatchingState::Dispatched;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dispatched.take()));
        assert!(result.is_err());
    }

    #[test]
    fn pending_index_invariant() {
        let mut frontier = Frontier::<Cyclic<7>>::for_boundary();
        let o1 = Orthant::from([1, 0]);
        let o2 = Orthant::from([2, 0]);

        // Push creates pending entries
        frontier.push(o1.clone(), 0b01, Cyclic::one());
        frontier.push(o2.clone(), 0b10, Cyclic::one());
        assert_eq!(frontier.pending.len(), 2);
        assert!(frontier.pending.contains(&o1));
        assert!(frontier.pending.contains(&o2));

        // Dispatch removes from pending
        frontier.mark_dispatched(&o1);
        assert_eq!(frontier.pending.len(), 1);
        assert!(!frontier.pending.contains(&o1));
        assert!(frontier.pending.contains(&o2));

        // Dispatch increments precomputed_count
        assert_eq!(frontier.precomputed_count(), 1); // o1 was dispatched above

        // attach_matching doesn't change precomputed_count (already counted at
        // dispatch)
        frontier.attach_matching(&o1, make_matching(0b01));
        assert_eq!(frontier.pending.len(), 1);
        assert_eq!(frontier.precomputed_count(), 1);

        // Dispatch o2 increments precomputed_count
        frontier.mark_dispatched(&o2);
        assert_eq!(frontier.precomputed_count(), 2);
        frontier.attach_matching(&o2, make_matching(0b10));
        assert_eq!(frontier.precomputed_count(), 2); // unchanged by attach

        // Pop decrements precomputed_count
        frontier.pop_next();
        assert_eq!(frontier.precomputed_count(), 1);
        frontier.pop_next();
        assert_eq!(frontier.precomputed_count(), 0);
        assert!(frontier.pending.is_empty());
        assert!(frontier.is_empty());
    }

    #[test]
    fn boundary_ascending_order() {
        let mut frontier = Frontier::<Cyclic<7>>::for_boundary();
        let o_high = Orthant::from([5, 5]);
        let o_low = Orthant::from([1, 1]);
        let o_mid = Orthant::from([3, 3]);

        // Insert out of order
        frontier.push(o_high.clone(), 0b01, Cyclic::one());
        frontier.push(o_low.clone(), 0b01, Cyclic::one());
        frontier.push(o_mid.clone(), 0b01, Cyclic::one());

        // peek_next and next_pending return lowest
        assert_eq!(frontier.peek_next(), Some(&o_low));
        assert_eq!(frontier.next_pending(), Some(&o_low));

        // Make all ready and pop in ascending order
        for o in [&o_low, &o_mid, &o_high] {
            frontier.mark_dispatched(o);
            frontier.attach_matching(o, make_matching(0b01));
        }

        let (first, ..) = frontier.pop_next().unwrap();
        let (second, ..) = frontier.pop_next().unwrap();
        let (third, ..) = frontier.pop_next().unwrap();
        assert_eq!([first, second, third], [o_low, o_mid, o_high]);
    }

    #[test]
    fn coboundary_descending_order() {
        let mut frontier = Frontier::<Cyclic<7>>::for_coboundary();
        let o_high = Orthant::from([5, 5]);
        let o_low = Orthant::from([1, 1]);
        let o_mid = Orthant::from([3, 3]);

        frontier.push(o_low.clone(), 0b01, Cyclic::one());
        frontier.push(o_high.clone(), 0b01, Cyclic::one());
        frontier.push(o_mid.clone(), 0b01, Cyclic::one());

        // peek_next and next_pending return highest
        assert_eq!(frontier.peek_next(), Some(&o_high));
        assert_eq!(frontier.next_pending(), Some(&o_high));

        for o in [&o_low, &o_mid, &o_high] {
            frontier.mark_dispatched(o);
            frontier.attach_matching(o, make_matching(0b01));
        }

        let (first, ..) = frontier.pop_next().unwrap();
        let (second, ..) = frontier.pop_next().unwrap();
        let (third, ..) = frontier.pop_next().unwrap();
        assert_eq!([first, second, third], [o_high, o_mid, o_low]);
    }

    #[test]
    fn attach_matching_returns_orphan() {
        let mut frontier = Frontier::<Cyclic<7>>::for_boundary();
        let o1 = Orthant::from([1, 0]);
        let orphan = Orthant::from([9, 9]);

        frontier.push(o1.clone(), 0b01, Cyclic::one());
        frontier.mark_dispatched(&o1);

        // Attach to existing entry returns None
        assert!(frontier.attach_matching(&o1, make_matching(0b01)).is_none());

        // Attach to non-existent entry returns Some
        let result = frontier.attach_matching(&orphan, make_matching(0b11));
        assert!(matches!(
            result,
            Some(OrthantMatching::Critical {
                ace_extent: 0b11,
                ..
            })
        ));
    }

    #[test]
    fn next_is_ready() {
        let mut frontier = Frontier::<Cyclic<7>>::for_boundary();
        let o1 = Orthant::from([1, 0]);

        frontier.push(o1.clone(), 0b01, Cyclic::one());
        assert!(!frontier.next_is_ready());

        frontier.mark_dispatched(&o1);
        assert!(!frontier.next_is_ready());

        frontier.attach_matching(&o1, make_matching(0b01));
        assert!(frontier.next_is_ready());
    }

    #[test]
    fn push_accumulates_and_ignores_zero() {
        let mut frontier = Frontier::<Cyclic<7>>::for_boundary();
        let o1 = Orthant::from([1, 0]);

        // Zero coefficient is ignored
        frontier.push(o1.clone(), 0b01, Cyclic::zero());
        assert!(frontier.is_empty());

        // Multiple pushes accumulate
        frontier.push(o1.clone(), 0b01, Cyclic::from(3));
        frontier.push(o1.clone(), 0b01, Cyclic::from(4));
        frontier.push(o1.clone(), 0b10, Cyclic::from(2));

        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier.pending.len(), 1);

        frontier.mark_dispatched(&o1);
        frontier.attach_matching(&o1, make_matching(0b01));
        let (_, chain, _) = frontier.pop_next().unwrap();

        // 3 + 4 = 7 ≡ 0 (mod 7)
        assert_eq!(chain.coefficient(&0b01), Cyclic::zero());
        assert_eq!(chain.coefficient(&0b10), Cyclic::from(2));
    }

    #[test]
    fn pop_requires_ready_state() {
        let mut frontier = Frontier::<Cyclic<7>>::for_boundary();
        let o1 = Orthant::from([1, 0]);

        // Pop pending panics
        frontier.push(o1.clone(), 0b01, Cyclic::one());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| frontier.pop_next()));
        assert!(result.is_err());

        // Pop dispatched panics
        let mut frontier = Frontier::<Cyclic<7>>::for_boundary();
        frontier.push(o1.clone(), 0b01, Cyclic::one());
        frontier.mark_dispatched(&o1);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| frontier.pop_next()));
        assert!(result.is_err());
    }
}
