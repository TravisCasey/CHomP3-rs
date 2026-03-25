// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Internal progress tracking utilities for long-running operations.

use std::{
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use tracing::{Level, debug, error, info, trace, warn};

/// Tracks progress for long-running operations with periodic logging.
///
/// Logs progress at configurable percentage intervals, including current
/// count, total, and estimated time remaining. All mutation methods take
/// `&self` (using atomics internally), so the tracker can be shared across
/// threads for parallel progress reporting.
pub(crate) struct ProgressTracker {
    label: String,
    total: usize,
    done: AtomicBool,
    current: AtomicUsize,
    interval_percent: usize,
    last_logged_percent: AtomicUsize,
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
            done: AtomicBool::new(false),
            current: AtomicUsize::new(0),
            interval_percent: 1,
            last_logged_percent: AtomicUsize::new(0),
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
    pub(crate) fn with_level(mut self, level: Level) -> Self {
        self.level = level;
        self
    }

    /// Increment progress by one item.
    pub(crate) fn increment(&self) {
        self.increment_by(1);
    }

    /// Increment progress by a custom amount.
    pub(crate) fn increment_by(&self, n: usize) {
        let prev = self.current.fetch_add(n, Ordering::Relaxed);
        let new = (prev + n).min(self.total);
        // Correct if we overshot total
        if prev + n > self.total {
            self.current.store(self.total, Ordering::Relaxed);
        }
        self.maybe_log(new);
    }

    /// Set progress to a specific count.
    pub(crate) fn set(&self, count: usize) {
        let clamped = count.min(self.total);
        self.current.store(clamped, Ordering::Relaxed);
        self.maybe_log(clamped);
    }

    /// Log final progress and mark the tracker as complete.
    ///
    /// Emits a 100% completion message unconditionally (unless `total` is
    /// zero). Subsequent calls to `increment`, `increment_by`, `set`, or
    /// `finish` are no-ops.
    pub(crate) fn finish(&self) {
        if self.done.swap(true, Ordering::Relaxed) {
            return;
        }
        self.current.store(self.total, Ordering::Relaxed);
        if self.total > 0 {
            self.log_progress(100, "done");
        }
    }

    fn maybe_log(&self, current: usize) {
        if self.done.load(Ordering::Relaxed) || self.total == 0 {
            return;
        }

        let percent = (current * 100) / self.total;
        let last = self.last_logged_percent.load(Ordering::Relaxed);
        let threshold = last + self.interval_percent;

        if percent >= threshold {
            if current >= self.total {
                self.done.store(true, Ordering::Relaxed);
            }
            let eta = self.calculate_eta(current);
            self.log_progress(percent, &eta);
            let aligned = percent - (percent % self.interval_percent);
            self.last_logged_percent.store(aligned, Ordering::Relaxed);
        }
    }

    fn calculate_eta(&self, current: usize) -> String {
        if current == 0 {
            return String::from("calculating...");
        }

        let remaining = self.total.saturating_sub(current);
        if remaining == 0 {
            return String::from("done");
        }

        let elapsed = self.start_time.elapsed();
        let nanos_per_item = elapsed.as_nanos() / current as u128;
        let remaining_nanos = nanos_per_item * remaining as u128;
        let remaining_nanos = remaining_nanos.min(u128::from(u64::MAX)) as u64;

        format_duration(Duration::from_nanos(remaining_nanos))
    }

    fn log_progress(&self, percent: usize, eta: &str) {
        match self.level {
            Level::TRACE => trace!(
                "{}: {}% ({}/{}) - ETA: {}",
                self.label,
                percent,
                self.current.load(Ordering::Relaxed),
                self.total,
                eta
            ),
            Level::DEBUG => debug!(
                "{}: {}% ({}/{}) - ETA: {}",
                self.label,
                percent,
                self.current.load(Ordering::Relaxed),
                self.total,
                eta
            ),
            Level::INFO => info!(
                "{}: {}% ({}/{}) - ETA: {}",
                self.label,
                percent,
                self.current.load(Ordering::Relaxed),
                self.total,
                eta
            ),
            Level::WARN => warn!(
                "{}: {}% ({}/{}) - ETA: {}",
                self.label,
                percent,
                self.current.load(Ordering::Relaxed),
                self.total,
                eta
            ),
            Level::ERROR => error!(
                "{}: {}% ({}/{}) - ETA: {}",
                self.label,
                percent,
                self.current.load(Ordering::Relaxed),
                self.total,
                eta
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
        let t = ProgressTracker::new("test", 100);
        assert_eq!(t.current.load(Ordering::Relaxed), 0);
        t.increment();
        assert_eq!(t.current.load(Ordering::Relaxed), 1);
        t.increment_by(10);
        assert_eq!(t.current.load(Ordering::Relaxed), 11);
        t.set(50);
        assert_eq!(t.current.load(Ordering::Relaxed), 50);
    }

    #[test]
    fn increment_by_clamps_to_total() {
        let t = ProgressTracker::new("test", 100);
        t.increment_by(200);
        assert_eq!(t.current.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn set_clamps_to_total() {
        let t = ProgressTracker::new("test", 100);
        t.set(999);
        assert_eq!(t.current.load(Ordering::Relaxed), 100);
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
        let t = ProgressTracker::new("test", 100);
        t.increment_by(50);
        t.finish();
        assert!(t.done.load(Ordering::Relaxed));
        assert_eq!(t.current.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn finish_is_idempotent() {
        let t = ProgressTracker::new("test", 10);
        t.finish();
        assert!(t.done.load(Ordering::Relaxed));
        assert_eq!(t.current.load(Ordering::Relaxed), 10);
        t.finish();
        assert!(t.done.load(Ordering::Relaxed));
        assert_eq!(t.current.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn zero_total_does_not_panic() {
        let t = ProgressTracker::new("test", 0);
        t.increment();
        t.increment_by(5);
        t.set(10);
        t.finish();
        assert!(t.done.load(Ordering::Relaxed));
    }

    #[test]
    fn threshold_crossing_updates_last_logged() {
        let t = ProgressTracker::new("test", 100).with_interval(10);
        t.increment_by(5);
        assert_eq!(t.last_logged_percent.load(Ordering::Relaxed), 0); // 5% < 10% threshold
        t.increment_by(5);
        assert_eq!(t.last_logged_percent.load(Ordering::Relaxed), 10); // 10% crosses threshold
        t.increment_by(5);
        assert_eq!(t.last_logged_percent.load(Ordering::Relaxed), 10); // 15% < 20% threshold
        t.increment_by(10);
        assert_eq!(t.last_logged_percent.load(Ordering::Relaxed), 20); // 25% crosses 20% threshold, aligns to 20
    }

    #[test]
    fn done_prevents_further_logging() {
        let t = ProgressTracker::new("test", 100).with_interval(10);
        t.finish();
        let logged = t.last_logged_percent.load(Ordering::Relaxed);
        // done is set, so increment_by is a no-op for logging
        t.increment_by(50);
        assert_eq!(t.last_logged_percent.load(Ordering::Relaxed), logged);
    }
}
