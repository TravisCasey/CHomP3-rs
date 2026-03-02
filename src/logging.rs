// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Internal progress tracking utilities for long-running operations.

use std::time::{Duration, Instant};

use tracing::{Level, debug, error, info, trace, warn};

/// Tracks progress for long-running operations with periodic logging.
///
/// Logs progress at configurable percentage intervals, including current
/// count, total, and estimated time remaining. Designed for imperative loops
/// and event-driven contexts where iterator adapters aren't suitable.
pub(crate) struct ProgressTracker {
    label: String,
    total: usize,
    done: bool,
    current: usize,
    interval_percent: usize,
    last_logged_percent: usize,
    start_time: Instant,
    level: Level,
}

impl ProgressTracker {
    /// Create a new progress tracker at INFO level.
    ///
    /// Logs progress at each percent.
    pub(crate) fn new(label: impl Into<String>, total: usize) -> Self {
        Self {
            label: label.into(),
            total,
            done: false,
            current: 0,
            interval_percent: 1,
            last_logged_percent: 0,
            start_time: Instant::now(),
            level: Level::INFO,
        }
    }

    /// Set the logging interval as a percentage (clamped to `1..=100`).
    #[must_use]
    pub(crate) fn with_interval(mut self, percent: usize) -> Self {
        self.interval_percent = percent.clamp(1, 100);
        self
    }

    /// Set the tracing log level (default: INFO).
    #[must_use]
    #[cfg_attr(not(feature = "mpi"), expect(dead_code))]
    pub(crate) fn with_level(mut self, level: Level) -> Self {
        self.level = level;
        self
    }

    /// Increment progress by one item.
    pub(crate) fn increment(&mut self) {
        self.increment_by(1);
    }

    /// Increment progress by a custom amount.
    pub(crate) fn increment_by(&mut self, n: usize) {
        self.current = (self.current + n).min(self.total);
        self.maybe_log();
    }

    /// Set progress to a specific count.
    pub(crate) fn set(&mut self, count: usize) {
        self.current = count.min(self.total);
        self.maybe_log();
    }

    /// Log final progress and mark the tracker as complete.
    ///
    /// Emits a 100% completion message unconditionally (unless `total` is
    /// zero). Subsequent calls to `increment`, `increment_by`, `set`, or
    /// `finish` are no-ops.
    pub(crate) fn finish(&mut self) {
        if self.done {
            return;
        }
        self.current = self.total;
        self.done = true;
        if self.total > 0 {
            self.log_progress(100, "done");
        }
    }

    fn maybe_log(&mut self) {
        if self.done || self.total == 0 {
            return;
        }

        let percent = (self.current * 100) / self.total;
        let threshold = self.last_logged_percent + self.interval_percent;

        if percent >= threshold {
            if self.current >= self.total {
                self.done = true;
            }
            let eta = self.calculate_eta();
            self.log_progress(percent, &eta);
            self.last_logged_percent = percent - (percent % self.interval_percent);
        }
    }

    fn calculate_eta(&self) -> String {
        if self.current == 0 {
            return String::from("calculating...");
        }

        let remaining = self.total.saturating_sub(self.current);
        if remaining == 0 {
            return String::from("done");
        }

        let elapsed = self.start_time.elapsed();
        let nanos_per_item = elapsed.as_nanos() / self.current as u128;
        let remaining_nanos = nanos_per_item * remaining as u128;
        let remaining_nanos = remaining_nanos.min(u128::from(u64::MAX)) as u64;

        format_duration(Duration::from_nanos(remaining_nanos))
    }

    fn log_progress(&self, percent: usize, eta: &str) {
        match self.level {
            Level::TRACE => trace!(
                "{}: {}% ({}/{}) - ETA: {}",
                self.label, percent, self.current, self.total, eta
            ),
            Level::DEBUG => debug!(
                "{}: {}% ({}/{}) - ETA: {}",
                self.label, percent, self.current, self.total, eta
            ),
            Level::INFO => info!(
                "{}: {}% ({}/{}) - ETA: {}",
                self.label, percent, self.current, self.total, eta
            ),
            Level::WARN => warn!(
                "{}: {}% ({}/{}) - ETA: {}",
                self.label, percent, self.current, self.total, eta
            ),
            Level::ERROR => error!(
                "{}: {}% ({}/{}) - ETA: {}",
                self.label, percent, self.current, self.total, eta
            ),
        }
    }
}

/// Format a duration as a human-readable string.
fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs == 0 {
        String::from("<1s")
    } else if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_duration_ranges() {
        // Sub-second
        assert_eq!(format_duration(Duration::from_millis(500)), "<1s");
        assert_eq!(format_duration(Duration::from_secs(0)), "<1s");

        // Seconds
        assert_eq!(format_duration(Duration::from_secs(1)), "1s");
        assert_eq!(format_duration(Duration::from_secs(59)), "59s");

        // Minutes
        assert_eq!(format_duration(Duration::from_secs(60)), "1m 0s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(3599)), "59m 59s");

        // Hours
        assert_eq!(format_duration(Duration::from_secs(3600)), "1h 0m");
        assert_eq!(format_duration(Duration::from_secs(5400)), "1h 30m");
        assert_eq!(format_duration(Duration::from_secs(86400)), "24h 0m");
    }

    #[test]
    fn increment_and_set() {
        let mut t = ProgressTracker::new("test", 100);
        assert_eq!(t.current, 0);
        t.increment();
        assert_eq!(t.current, 1);
        t.increment_by(10);
        assert_eq!(t.current, 11);
        t.set(50);
        assert_eq!(t.current, 50);
    }

    #[test]
    fn increment_by_clamps_to_total() {
        let mut t = ProgressTracker::new("test", 100);
        t.increment_by(200);
        assert_eq!(t.current, 100);
    }

    #[test]
    fn set_clamps_to_total() {
        let mut t = ProgressTracker::new("test", 100);
        t.set(999);
        assert_eq!(t.current, 100);
    }

    #[test]
    fn interval_clamps_to_valid_range() {
        assert_eq!(
            ProgressTracker::new("t", 100)
                .with_interval(0)
                .interval_percent,
            1
        );
        assert_eq!(
            ProgressTracker::new("t", 100)
                .with_interval(200)
                .interval_percent,
            100
        );
        assert_eq!(
            ProgressTracker::new("t", 100)
                .with_interval(50)
                .interval_percent,
            50
        );
    }

    #[test]
    fn finish_sets_done_and_current() {
        let mut t = ProgressTracker::new("test", 100);
        t.increment_by(50);
        t.finish();
        assert!(t.done);
        assert_eq!(t.current, 100);
    }

    #[test]
    fn finish_is_idempotent() {
        let mut t = ProgressTracker::new("test", 10);
        t.finish();
        assert!(t.done);
        assert_eq!(t.current, 10);
        t.finish();
        assert!(t.done);
        assert_eq!(t.current, 10);
    }

    #[test]
    fn zero_total_does_not_panic() {
        let mut t = ProgressTracker::new("test", 0);
        t.increment();
        t.increment_by(5);
        t.set(10);
        t.finish();
        assert!(t.done);
    }

    #[test]
    fn threshold_crossing_updates_last_logged() {
        let mut t = ProgressTracker::new("test", 100).with_interval(10);
        t.increment_by(5);
        assert_eq!(t.last_logged_percent, 0); // 5% < 10% threshold
        t.increment_by(5);
        assert_eq!(t.last_logged_percent, 10); // 10% crosses threshold
        t.increment_by(5);
        assert_eq!(t.last_logged_percent, 10); // 15% < 20% threshold
        t.increment_by(10);
        assert_eq!(t.last_logged_percent, 20); // 25% crosses 20% threshold, aligns to 20
    }

    #[test]
    fn done_prevents_further_logging() {
        let mut t = ProgressTracker::new("test", 100).with_interval(10);
        t.finish();
        let logged = t.last_logged_percent;
        // done is set, so increment_by is a no-op for logging
        t.increment_by(50);
        assert_eq!(t.last_logged_percent, logged);
    }
}
