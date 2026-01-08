//! Daemon Metrics Collection
//!
//! Tracks performance and resource metrics for the daemon.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Atomic counter for thread-safe incrementing
#[derive(Debug, Default)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    /// Create a new counter
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Increment the counter by 1
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the counter by a value
    pub fn add(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Get the current value
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
}

/// Atomic gauge for thread-safe value tracking
#[derive(Debug, Default)]
pub struct Gauge {
    value: AtomicU64,
}

impl Gauge {
    /// Create a new gauge
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Set the gauge value
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Increment the gauge
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the gauge
    pub fn dec(&self) {
        self.value.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get the current value
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
}

/// Simple histogram for latency tracking
#[derive(Debug)]
pub struct Histogram {
    /// Bucket boundaries in microseconds
    buckets: Vec<u64>,
    /// Count per bucket
    counts: Vec<AtomicU64>,
    /// Sum of all values (for mean calculation)
    sum: AtomicU64,
    /// Total count
    count: AtomicU64,
}

impl Histogram {
    /// Create a new histogram with default latency buckets (in ms)
    pub fn new_latency() -> Self {
        // Buckets: 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 5s
        let buckets = vec![1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 5000000];
        let counts = buckets.iter().map(|_| AtomicU64::new(0)).collect();

        Self {
            buckets,
            counts,
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Record a duration
    pub fn observe(&self, duration: Duration) {
        let micros = duration.as_micros() as u64;
        self.sum.fetch_add(micros, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);

        // Increment the appropriate bucket
        for (i, &boundary) in self.buckets.iter().enumerate() {
            if micros <= boundary {
                self.counts[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        // Greater than all buckets - add to last bucket
        if let Some(last) = self.counts.last() {
            last.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get the count of observations
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the mean value in microseconds
    pub fn mean_micros(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.sum.load(Ordering::Relaxed) as f64 / count as f64
    }

    /// Get the mean value in milliseconds
    pub fn mean_ms(&self) -> f64 {
        self.mean_micros() / 1000.0
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new_latency()
    }
}

/// All daemon metrics
#[derive(Debug, Default)]
pub struct DaemonMetrics {
    // Query metrics
    pub queries_total: Counter,
    pub query_latency: Histogram,
    pub queries_failed: Counter,

    // Write metrics
    pub chunks_indexed: Counter,
    pub documents_indexed: Counter,
    pub commits_total: Counter,
    pub commit_latency: Histogram,
    pub writes_failed: Counter,

    // Job metrics
    pub jobs_started: Counter,
    pub jobs_completed: Counter,
    pub jobs_failed: Counter,
    pub jobs_cancelled: Counter,

    // Connection metrics
    pub connections_total: Counter,
    pub active_connections: Gauge,

    // Resource metrics
    pub memory_usage_bytes: Gauge,
}

impl DaemonMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a shareable metrics instance
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Take a snapshot of all metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            queries_total: self.queries_total.get(),
            query_latency_ms: self.query_latency.mean_ms(),
            queries_failed: self.queries_failed.get(),

            chunks_indexed: self.chunks_indexed.get(),
            documents_indexed: self.documents_indexed.get(),
            commits_total: self.commits_total.get(),
            commit_latency_ms: self.commit_latency.mean_ms(),
            writes_failed: self.writes_failed.get(),

            jobs_started: self.jobs_started.get(),
            jobs_completed: self.jobs_completed.get(),
            jobs_failed: self.jobs_failed.get(),
            jobs_cancelled: self.jobs_cancelled.get(),

            connections_total: self.connections_total.get(),
            active_connections: self.active_connections.get(),

            memory_usage_bytes: self.memory_usage_bytes.get(),
        }
    }

    /// Update memory usage from system
    pub fn update_memory_usage(&self) {
        if let Some(usage) = get_memory_usage() {
            self.memory_usage_bytes.set(usage);
        }
    }
}

/// Point-in-time snapshot of all metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    // Query metrics
    pub queries_total: u64,
    pub query_latency_ms: f64,
    pub queries_failed: u64,

    // Write metrics
    pub chunks_indexed: u64,
    pub documents_indexed: u64,
    pub commits_total: u64,
    pub commit_latency_ms: f64,
    pub writes_failed: u64,

    // Job metrics
    pub jobs_started: u64,
    pub jobs_completed: u64,
    pub jobs_failed: u64,
    pub jobs_cancelled: u64,

    // Connection metrics
    pub connections_total: u64,
    pub active_connections: u64,

    // Resource metrics
    pub memory_usage_bytes: u64,
}

impl MetricsSnapshot {
    /// Format as human-readable string
    pub fn to_display_string(&self) -> String {
        format!(
            "Queries: {} (avg {:.1}ms, {} failed)\n\
             Indexed: {} chunks, {} documents, {} commits (avg {:.1}ms)\n\
             Jobs: {} started, {} completed, {} failed, {} cancelled\n\
             Connections: {} total, {} active\n\
             Memory: {:.1} MB",
            self.queries_total,
            self.query_latency_ms,
            self.queries_failed,
            self.chunks_indexed,
            self.documents_indexed,
            self.commits_total,
            self.commit_latency_ms,
            self.jobs_started,
            self.jobs_completed,
            self.jobs_failed,
            self.jobs_cancelled,
            self.connections_total,
            self.active_connections,
            self.memory_usage_bytes as f64 / 1_048_576.0,
        )
    }
}

/// Helper for timing operations
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Start a new timer
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get elapsed duration
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Record to histogram and return elapsed
    pub fn record(self, histogram: &Histogram) -> Duration {
        let elapsed = self.elapsed();
        histogram.observe(elapsed);
        elapsed
    }
}

/// Get current process memory usage in bytes
fn get_memory_usage() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(rss) = content.split_whitespace().nth(1) {
                if let Ok(pages) = rss.parse::<u64>() {
                    // Page size is typically 4KB
                    return Some(pages * 4096);
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let counter = Counter::new();
        assert_eq!(counter.get(), 0);

        counter.inc();
        assert_eq!(counter.get(), 1);

        counter.add(5);
        assert_eq!(counter.get(), 6);
    }

    #[test]
    fn test_gauge() {
        let gauge = Gauge::new();
        assert_eq!(gauge.get(), 0);

        gauge.set(10);
        assert_eq!(gauge.get(), 10);

        gauge.inc();
        assert_eq!(gauge.get(), 11);

        gauge.dec();
        assert_eq!(gauge.get(), 10);
    }

    #[test]
    fn test_histogram() {
        let histogram = Histogram::new_latency();

        histogram.observe(Duration::from_millis(5));
        histogram.observe(Duration::from_millis(10));
        histogram.observe(Duration::from_millis(15));

        assert_eq!(histogram.count(), 3);
        assert!(histogram.mean_ms() > 9.0 && histogram.mean_ms() < 11.0);
    }

    #[test]
    fn test_timer() {
        let histogram = Histogram::new_latency();
        let timer = Timer::start();

        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.record(&histogram);

        assert!(elapsed.as_millis() >= 10);
        assert_eq!(histogram.count(), 1);
    }

    #[test]
    fn test_metrics_snapshot() {
        let metrics = DaemonMetrics::new();
        metrics.queries_total.add(100);
        metrics.chunks_indexed.add(500);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.queries_total, 100);
        assert_eq!(snapshot.chunks_indexed, 500);
    }
}
