//! Progress tracking for bulk imports

use super::source::{ImportCheckpoint, ImportStats};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Progress tracker for import operations
pub struct ImportProgress {
    /// Progress bar (None if running in quiet mode)
    progress_bar: Option<ProgressBar>,
    /// Start time
    start_time: Instant,
    /// Documents processed
    docs_processed: AtomicUsize,
    /// Documents imported
    docs_imported: AtomicUsize,
    /// Documents skipped
    docs_skipped: AtomicUsize,
    /// Documents errored
    docs_errored: AtomicUsize,
    /// Chunks created
    chunks_created: AtomicUsize,
    /// Bytes processed
    bytes_processed: AtomicU64,
    /// Last checkpoint time
    last_checkpoint: std::sync::Mutex<Instant>,
    /// Checkpoint interval (documents)
    checkpoint_interval: usize,
    /// Checkpoint path
    checkpoint_path: Option<PathBuf>,
    /// Source path for checkpoint
    source_path: PathBuf,
    /// Cancelled flag
    cancelled: AtomicBool,
}

impl ImportProgress {
    /// Create a new progress tracker
    pub fn new(
        source_path: PathBuf,
        total_expected: Option<u64>,
        checkpoint_interval: usize,
        checkpoint_path: Option<PathBuf>,
        quiet: bool,
    ) -> Self {
        let progress_bar = if !quiet {
            let pb = if let Some(total) = total_expected {
                ProgressBar::new(total)
            } else {
                ProgressBar::new_spinner()
            };

            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")
                    .unwrap_or_else(|_| ProgressStyle::default_bar())
                    .progress_chars("#>-"),
            );

            Some(pb)
        } else {
            None
        };

        Self {
            progress_bar,
            start_time: Instant::now(),
            docs_processed: AtomicUsize::new(0),
            docs_imported: AtomicUsize::new(0),
            docs_skipped: AtomicUsize::new(0),
            docs_errored: AtomicUsize::new(0),
            chunks_created: AtomicUsize::new(0),
            bytes_processed: AtomicU64::new(0),
            last_checkpoint: std::sync::Mutex::new(Instant::now()),
            checkpoint_interval,
            checkpoint_path,
            source_path,
            cancelled: AtomicBool::new(false),
        }
    }

    /// Update progress after processing a document
    pub fn document_processed(&self, title: &str, imported: bool, chunks: usize, bytes: u64) {
        let processed = self.docs_processed.fetch_add(1, Ordering::Relaxed) + 1;

        if imported {
            self.docs_imported.fetch_add(1, Ordering::Relaxed);
            self.chunks_created.fetch_add(chunks, Ordering::Relaxed);
        } else {
            self.docs_skipped.fetch_add(1, Ordering::Relaxed);
        }

        self.bytes_processed.fetch_add(bytes, Ordering::Relaxed);

        if let Some(ref pb) = self.progress_bar {
            pb.set_position(processed as u64);

            // Calculate rate
            let elapsed = self.start_time.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 {
                processed as f64 / elapsed
            } else {
                0.0
            };

            // Show last document title (truncated safely for UTF-8)
            let display_title = if title.chars().count() > 30 {
                let truncated: String = title.chars().take(27).collect();
                format!("{}...", truncated)
            } else {
                title.to_string()
            };

            pb.set_message(format!("{:.1} docs/s | {}", rate, display_title));
        }

        // Check if we should checkpoint
        if self.checkpoint_path.is_some() && processed % self.checkpoint_interval == 0 {
            self.maybe_checkpoint();
        }
    }

    /// Record an error
    pub fn document_error(&self, _error: &str) {
        self.docs_errored.fetch_add(1, Ordering::Relaxed);
    }

    /// Check if we should create a checkpoint
    fn maybe_checkpoint(&self) {
        if let Some(ref path) = self.checkpoint_path {
            // Use unwrap_or_else to recover from poisoned mutex
            let mut last = self.last_checkpoint.lock().unwrap_or_else(|e| e.into_inner());
            if last.elapsed() >= Duration::from_secs(30) {
                // Create checkpoint
                let stats = self.get_stats();
                let checkpoint = ImportCheckpoint::new(
                    self.source_path.clone(),
                    self.bytes_processed.load(Ordering::Relaxed),
                    &stats,
                );
                if let Err(e) = checkpoint.save(path) {
                    tracing::warn!("Failed to save checkpoint: {}", e);
                }
                *last = Instant::now();
            }
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> ImportStats {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let processed = self.docs_processed.load(Ordering::Relaxed);

        let mut stats = ImportStats {
            documents_processed: processed,
            documents_imported: self.docs_imported.load(Ordering::Relaxed),
            documents_skipped: self.docs_skipped.load(Ordering::Relaxed),
            documents_errored: self.docs_errored.load(Ordering::Relaxed),
            chunks_created: self.chunks_created.load(Ordering::Relaxed),
            bytes_processed: self.bytes_processed.load(Ordering::Relaxed),
            elapsed_seconds: elapsed,
            docs_per_second: 0.0,
        };
        stats.update_rate();
        stats
    }

    /// Check if import has been cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Cancel the import
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        if let Some(ref pb) = self.progress_bar {
            pb.abandon_with_message("Cancelled");
        }
    }

    /// Finish the progress bar
    pub fn finish(&self) {
        if let Some(ref pb) = self.progress_bar {
            let stats = self.get_stats();
            pb.finish_with_message(format!(
                "Done! {} imported, {} skipped, {} errors, {:.1} docs/s",
                stats.documents_imported,
                stats.documents_skipped,
                stats.documents_errored,
                stats.docs_per_second
            ));
        }
    }

    /// Get bytes processed
    pub fn bytes_processed(&self) -> u64 {
        self.bytes_processed.load(Ordering::Relaxed)
    }

    /// Print summary to console
    pub fn print_summary(&self) {
        let stats = self.get_stats();

        println!("\nImport Summary");
        println!("==============");
        println!("Documents processed: {}", stats.documents_processed);
        println!("Documents imported:  {}", stats.documents_imported);
        println!("Documents skipped:   {}", stats.documents_skipped);
        println!("Documents errored:   {}", stats.documents_errored);
        println!("Chunks created:      {}", stats.chunks_created);
        println!("Bytes processed:     {} MB", stats.bytes_processed / 1_000_000);
        println!("Elapsed time:        {:.1}s", stats.elapsed_seconds);
        println!("Processing rate:     {:.1} docs/s", stats.docs_per_second);
    }
}

impl Drop for ImportProgress {
    fn drop(&mut self) {
        // Create final checkpoint if enabled
        if let Some(ref path) = self.checkpoint_path {
            let stats = self.get_stats();
            let checkpoint = ImportCheckpoint::new(
                self.source_path.clone(),
                self.bytes_processed.load(Ordering::Relaxed),
                &stats,
            );
            if let Err(e) = checkpoint.save(path) {
                eprintln!("Warning: failed to save final import checkpoint: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_progress_tracking() {
        let progress = ImportProgress::new(
            PathBuf::from("/tmp/test.xml.bz2"),
            Some(100),
            10,
            None,
            true, // quiet mode for tests
        );

        progress.document_processed("Test Article 1", true, 5, 1000);
        progress.document_processed("Test Article 2", true, 3, 800);
        progress.document_processed("Skipped Article", false, 0, 200);

        let stats = progress.get_stats();
        assert_eq!(stats.documents_processed, 3);
        assert_eq!(stats.documents_imported, 2);
        assert_eq!(stats.documents_skipped, 1);
        assert_eq!(stats.chunks_created, 8);
        assert_eq!(stats.bytes_processed, 2000);
    }

    #[test]
    fn test_cancellation() {
        let progress = ImportProgress::new(
            PathBuf::from("/tmp/test.xml.bz2"),
            None,
            100,
            None,
            true,
        );

        assert!(!progress.is_cancelled());
        progress.cancel();
        assert!(progress.is_cancelled());
    }
}
