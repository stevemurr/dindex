//! Daemon Metrics Collection
//!
//! Tracks performance and resource metrics for the daemon.

use std::fmt::Write as FmtWrite;
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
    /// Overflow count (values exceeding all bucket boundaries)
    overflow: AtomicU64,
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
            overflow: AtomicU64::new(0),
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
        // Greater than all buckets - increment overflow
        self.overflow.fetch_add(1, Ordering::Relaxed);
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

    /// Get bucket boundaries in microseconds
    pub fn bucket_boundaries(&self) -> &[u64] {
        &self.buckets
    }

    /// Get bucket counts (non-cumulative)
    pub fn bucket_counts(&self) -> Vec<u64> {
        self.counts.iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }

    /// Get the overflow count
    pub fn overflow_count(&self) -> u64 {
        self.overflow.load(Ordering::Relaxed)
    }

    /// Get the sum of all observed values in microseconds
    pub fn sum_micros(&self) -> u64 {
        self.sum.load(Ordering::Relaxed)
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

    // Embedding metrics
    pub embedding_requests_total: Counter,
    pub embedding_latency: Histogram,
    pub embedding_errors_total: Counter,

    // Scraping metrics
    pub scrape_fetch_total: Counter,
    pub scrape_fetch_latency: Histogram,
    pub scrape_fetch_errors_total: Counter,
    pub scrape_pages_indexed: Counter,

    // P2P metrics
    pub p2p_connected_peers: Gauge,
    pub p2p_queries_received: Counter,
    pub p2p_queries_sent: Counter,

    // HTTP metrics
    pub http_requests_total: Counter,
    pub http_request_latency: Histogram,
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

            embedding_requests_total: self.embedding_requests_total.get(),
            embedding_latency_ms: self.embedding_latency.mean_ms(),
            embedding_errors_total: self.embedding_errors_total.get(),

            scrape_fetch_total: self.scrape_fetch_total.get(),
            scrape_fetch_errors_total: self.scrape_fetch_errors_total.get(),
            scrape_pages_indexed: self.scrape_pages_indexed.get(),

            p2p_connected_peers: self.p2p_connected_peers.get(),
            p2p_queries_received: self.p2p_queries_received.get(),
            p2p_queries_sent: self.p2p_queries_sent.get(),

            http_requests_total: self.http_requests_total.get(),
        }
    }

    /// Update memory usage from system
    pub fn update_memory_usage(&self) {
        if let Some(usage) = get_memory_usage() {
            self.memory_usage_bytes.set(usage);
        }
    }

    /// Export all metrics in Prometheus exposition format
    pub fn to_prometheus(&self) -> String {
        let mut out = String::with_capacity(4096);

        // Query metrics
        write_counter(&mut out, "dindex_queries_total", "Total number of search queries", self.queries_total.get());
        write_histogram(&mut out, "dindex_query_latency_seconds", "Query latency in seconds", &self.query_latency);
        write_counter(&mut out, "dindex_queries_failed_total", "Total number of failed queries", self.queries_failed.get());

        // Write metrics
        write_counter(&mut out, "dindex_chunks_indexed_total", "Total number of chunks indexed", self.chunks_indexed.get());
        write_counter(&mut out, "dindex_documents_indexed_total", "Total number of documents indexed", self.documents_indexed.get());
        write_counter(&mut out, "dindex_commits_total", "Total number of index commits", self.commits_total.get());
        write_histogram(&mut out, "dindex_commit_latency_seconds", "Commit latency in seconds", &self.commit_latency);
        write_counter(&mut out, "dindex_writes_failed_total", "Total number of failed writes", self.writes_failed.get());

        // Job metrics
        write_counter(&mut out, "dindex_jobs_started_total", "Total number of jobs started", self.jobs_started.get());
        write_counter(&mut out, "dindex_jobs_completed_total", "Total number of jobs completed", self.jobs_completed.get());
        write_counter(&mut out, "dindex_jobs_failed_total", "Total number of jobs failed", self.jobs_failed.get());
        write_counter(&mut out, "dindex_jobs_cancelled_total", "Total number of jobs cancelled", self.jobs_cancelled.get());

        // Connection metrics
        write_counter(&mut out, "dindex_connections_total", "Total number of connections", self.connections_total.get());
        write_gauge(&mut out, "dindex_active_connections", "Number of active connections", self.active_connections.get());

        // Resource metrics
        write_gauge(&mut out, "dindex_memory_usage_bytes", "Current memory usage in bytes", self.memory_usage_bytes.get());

        // Embedding metrics
        write_counter(&mut out, "dindex_embedding_requests_total", "Total embedding requests", self.embedding_requests_total.get());
        write_histogram(&mut out, "dindex_embedding_latency_seconds", "Embedding request latency in seconds", &self.embedding_latency);
        write_counter(&mut out, "dindex_embedding_errors_total", "Total embedding errors", self.embedding_errors_total.get());

        // Scraping metrics
        write_counter(&mut out, "dindex_scrape_fetch_total", "Total scrape fetches", self.scrape_fetch_total.get());
        write_histogram(&mut out, "dindex_scrape_fetch_latency_seconds", "Scrape fetch latency in seconds", &self.scrape_fetch_latency);
        write_counter(&mut out, "dindex_scrape_fetch_errors_total", "Total scrape fetch errors", self.scrape_fetch_errors_total.get());
        write_counter(&mut out, "dindex_scrape_pages_indexed_total", "Total pages indexed by scraper", self.scrape_pages_indexed.get());

        // P2P metrics
        write_gauge(&mut out, "dindex_p2p_connected_peers", "Number of connected P2P peers", self.p2p_connected_peers.get());
        write_counter(&mut out, "dindex_p2p_queries_received_total", "Total P2P queries received", self.p2p_queries_received.get());
        write_counter(&mut out, "dindex_p2p_queries_sent_total", "Total P2P queries sent", self.p2p_queries_sent.get());

        // HTTP metrics
        write_counter(&mut out, "dindex_http_requests_total", "Total HTTP requests", self.http_requests_total.get());
        write_histogram(&mut out, "dindex_http_request_latency_seconds", "HTTP request latency in seconds", &self.http_request_latency);

        out
    }
}

/// Write a counter metric in Prometheus exposition format
fn write_counter(out: &mut String, name: &str, help: &str, value: u64) {
    let _ = writeln!(out, "# HELP {} {}", name, help);
    let _ = writeln!(out, "# TYPE {} counter", name);
    let _ = writeln!(out, "{} {}", name, value);
    let _ = writeln!(out);
}

/// Write a gauge metric in Prometheus exposition format
fn write_gauge(out: &mut String, name: &str, help: &str, value: u64) {
    let _ = writeln!(out, "# HELP {} {}", name, help);
    let _ = writeln!(out, "# TYPE {} gauge", name);
    let _ = writeln!(out, "{} {}", name, value);
    let _ = writeln!(out);
}

/// Write a histogram metric in Prometheus exposition format
fn write_histogram(out: &mut String, name: &str, help: &str, hist: &Histogram) {
    let _ = writeln!(out, "# HELP {} {}", name, help);
    let _ = writeln!(out, "# TYPE {} histogram", name);

    let boundaries = hist.bucket_boundaries();
    let counts = hist.bucket_counts();

    // Produce cumulative bucket counts (each le bucket includes all lower buckets)
    let mut cumulative: u64 = 0;
    for (i, &boundary) in boundaries.iter().enumerate() {
        cumulative += counts[i];
        // Convert microsecond boundaries to seconds
        let le_seconds = boundary as f64 / 1_000_000.0;
        let _ = writeln!(out, "{}_bucket{{le=\"{:.3}\"}} {}", name, le_seconds, cumulative);
    }
    // +Inf bucket uses total count
    let total_count = hist.count();
    let _ = writeln!(out, "{}_bucket{{le=\"+Inf\"}} {}", name, total_count);

    // Sum in seconds
    let sum_seconds = hist.sum_micros() as f64 / 1_000_000.0;
    let _ = writeln!(out, "{}_sum {:.6}", name, sum_seconds);
    let _ = writeln!(out, "{}_count {}", name, total_count);
    let _ = writeln!(out);
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

    // Embedding metrics
    pub embedding_requests_total: u64,
    pub embedding_latency_ms: f64,
    pub embedding_errors_total: u64,

    // Scraping metrics
    pub scrape_fetch_total: u64,
    pub scrape_fetch_errors_total: u64,
    pub scrape_pages_indexed: u64,

    // P2P metrics
    pub p2p_connected_peers: u64,
    pub p2p_queries_received: u64,
    pub p2p_queries_sent: u64,

    // HTTP metrics
    pub http_requests_total: u64,
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
pub(super) fn get_memory_usage() -> Option<u64> {
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
    fn test_histogram_overflow() {
        let histogram = Histogram::new_latency();

        // Observe a value beyond all bucket boundaries (>5s)
        histogram.observe(Duration::from_secs(10));
        assert_eq!(histogram.count(), 1);
        assert_eq!(histogram.overflow_count(), 1);

        // The last finite bucket should NOT have been incremented
        let counts = histogram.bucket_counts();
        assert_eq!(*counts.last().unwrap(), 0);
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

    #[test]
    fn test_prometheus_output() {
        let metrics = DaemonMetrics::new();
        metrics.queries_total.add(42);
        metrics.query_latency.observe(Duration::from_millis(50));
        metrics.query_latency.observe(Duration::from_millis(200));
        metrics.memory_usage_bytes.set(1_048_576);

        let output = metrics.to_prometheus();

        // Verify key Prometheus format elements
        assert!(output.contains("# HELP dindex_queries_total"));
        assert!(output.contains("# TYPE dindex_queries_total counter"));
        assert!(output.contains("dindex_queries_total 42"));

        assert!(output.contains("# TYPE dindex_query_latency_seconds histogram"));
        assert!(output.contains("dindex_query_latency_seconds_bucket{le=\"+Inf\"} 2"));
        assert!(output.contains("dindex_query_latency_seconds_count 2"));

        assert!(output.contains("# TYPE dindex_memory_usage_bytes gauge"));
        assert!(output.contains("dindex_memory_usage_bytes 1048576"));

        // Verify cumulative bucket ordering (50ms observation should be in 0.050 bucket)
        assert!(output.contains("dindex_query_latency_seconds_bucket{le=\"0.050\"} 1"));
        // Both observations should be in 0.250 bucket (cumulative)
        assert!(output.contains("dindex_query_latency_seconds_bucket{le=\"0.250\"} 2"));
    }
}
