// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

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

    /// Set the logging interval as a percentage (minimum one percent).
    #[must_use]
    pub(crate) fn with_interval(mut self, percent: usize) -> Self {
        self.interval_percent = percent.max(1);
        self
    }

    /// Set the tracing log level (default: INFO).
    #[must_use]
    #[expect(dead_code)]
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
        self.current += n;
        self.maybe_log();
    }

    /// Set progress to a specific count.
    pub(crate) fn set(&mut self, count: usize) {
        self.current = count;
        self.maybe_log();
    }

    /// Force a final progress log message.
    pub(crate) fn finish(&mut self) {
        if !self.done {
            self.done = true;
            self.maybe_log();
        }
    }

    fn maybe_log(&mut self) {
        if self.total == 0 {
            return;
        }

        let percent = (self.current * 100) / self.total;
        let threshold = self.last_logged_percent + self.interval_percent;

        if percent >= threshold {
            if self.done {
                return;
            } else if self.current >= self.total {
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
    fn progress_tracker_increment() {
        let mut tracker = ProgressTracker::new("test", 100);
        assert_eq!(tracker.current, 0);

        tracker.increment();
        assert_eq!(tracker.current, 1);

        tracker.increment_by(10);
        assert_eq!(tracker.current, 11);

        tracker.set(50);
        assert_eq!(tracker.current, 50);
    }

    #[test]
    fn progress_tracker_interval_clamped() {
        let tracker = ProgressTracker::new("test", 100).with_interval(0);
        assert_eq!(tracker.interval_percent, 1);
    }
}
