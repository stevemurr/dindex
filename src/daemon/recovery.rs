//! Crash Recovery
//!
//! Handles recovery from unclean shutdowns and index integrity verification.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Recovery state file name
const RECOVERY_STATE_FILE: &str = ".dindex_recovery";

/// Write-ahead log state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalState {
    /// Clean state - no pending operations
    Clean,
    /// Currently writing chunks
    Writing {
        stream_id: String,
        chunks_written: usize,
        started_at: u64,
    },
    /// Commit in progress
    Committing {
        started_at: u64,
    },
}

impl Default for WalState {
    fn default() -> Self {
        Self::Clean
    }
}

/// Recovery state persisted to disk
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecoveryState {
    /// Current WAL state
    pub wal_state: WalState,
    /// Last successful commit timestamp
    pub last_commit: Option<u64>,
    /// Last integrity check timestamp
    pub last_integrity_check: Option<u64>,
    /// Version of the recovery format
    pub version: u32,
}

impl RecoveryState {
    const CURRENT_VERSION: u32 = 1;

    /// Load recovery state from disk
    pub fn load(data_dir: &Path) -> Result<Self> {
        let state_path = data_dir.join(RECOVERY_STATE_FILE);

        if !state_path.exists() {
            return Ok(Self::default());
        }

        let mut file = File::open(&state_path)?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;

        let state: RecoveryState = bincode::deserialize(&contents)?;

        // Check version compatibility
        if state.version > Self::CURRENT_VERSION {
            warn!(
                "Recovery state version {} is newer than supported {}",
                state.version,
                Self::CURRENT_VERSION
            );
        }

        Ok(state)
    }

    /// Save recovery state to disk
    pub fn save(&self, data_dir: &Path) -> Result<()> {
        let state_path = data_dir.join(RECOVERY_STATE_FILE);

        let encoded = bincode::serialize(&RecoveryState {
            version: Self::CURRENT_VERSION,
            ..self.clone()
        })?;

        // Write atomically using temp file
        let temp_path = state_path.with_extension("tmp");
        let mut file = File::create(&temp_path)?;
        file.write_all(&encoded)?;
        file.sync_all()?;

        fs::rename(temp_path, state_path)?;
        Ok(())
    }

    /// Mark write operation started
    pub fn mark_writing(&mut self, stream_id: &str) {
        self.wal_state = WalState::Writing {
            stream_id: stream_id.to_string(),
            chunks_written: 0,
            started_at: current_timestamp(),
        };
    }

    /// Update chunk count during write
    pub fn update_chunk_count(&mut self, count: usize) {
        if let WalState::Writing { chunks_written, .. } = &mut self.wal_state {
            *chunks_written = count;
        }
    }

    /// Mark operation complete (clean state)
    pub fn mark_complete(&mut self) {
        self.wal_state = WalState::Clean;
        self.last_commit = Some(current_timestamp());
    }

    /// Check if recovery is needed
    pub fn needs_recovery(&self) -> bool {
        !matches!(self.wal_state, WalState::Clean)
    }
}

/// Recovery manager for crash recovery operations
pub struct RecoveryManager {
    data_dir: PathBuf,
    state: RecoveryState,
}

impl RecoveryManager {
    /// Create a new recovery manager
    pub fn new(data_dir: PathBuf) -> Result<Self> {
        let state = RecoveryState::load(&data_dir)?;
        Ok(Self { data_dir, state })
    }

    /// Perform recovery if needed
    pub fn recover(&mut self) -> Result<RecoveryResult> {
        if !self.state.needs_recovery() {
            return Ok(RecoveryResult::NoRecoveryNeeded);
        }

        info!("Recovery needed, checking state...");

        // Extract info from wal_state before modifying
        let recovery_info = match &self.state.wal_state {
            WalState::Clean => return Ok(RecoveryResult::NoRecoveryNeeded),
            WalState::Writing { stream_id, chunks_written, started_at } => {
                warn!(
                    "Found incomplete write: stream={}, chunks={}, started={}",
                    stream_id, chunks_written, started_at
                );
                Some((stream_id.clone(), *chunks_written))
            }
            WalState::Committing { started_at } => {
                warn!("Found incomplete commit started at {}", started_at);
                None
            }
        };

        // Now we can modify the state
        self.state.mark_complete();
        self.state.save(&self.data_dir)?;

        // Return appropriate result
        match recovery_info {
            Some((stream_id, chunks)) => Ok(RecoveryResult::RolledBackWrite { stream_id, chunks }),
            None => Ok(RecoveryResult::CompletedCommit),
        }
    }

    /// Get the current recovery state
    pub fn state(&self) -> &RecoveryState {
        &self.state
    }

    /// Save current state
    pub fn save(&self) -> Result<()> {
        self.state.save(&self.data_dir)
    }
}

/// Result of recovery operation
#[derive(Debug)]
pub enum RecoveryResult {
    /// No recovery was needed
    NoRecoveryNeeded,
    /// Rolled back an incomplete write
    RolledBackWrite { stream_id: String, chunks: usize },
    /// Completed an interrupted commit
    CompletedCommit,
}

/// Result of integrity verification
#[derive(Debug)]
pub enum IntegrityResult {
    /// No issues found
    Ok,
    /// Issues were found
    Issues(Vec<String>),
}

/// Get current timestamp as seconds since epoch
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_recovery_state_save_load() {
        let temp_dir = TempDir::new().unwrap();

        let mut state = RecoveryState::default();
        state.mark_writing("test-stream");
        state.update_chunk_count(42);

        state.save(temp_dir.path()).unwrap();

        let loaded = RecoveryState::load(temp_dir.path()).unwrap();
        match loaded.wal_state {
            WalState::Writing { stream_id, chunks_written, .. } => {
                assert_eq!(stream_id, "test-stream");
                assert_eq!(chunks_written, 42);
            }
            _ => panic!("Expected Writing state"),
        }
    }

    #[test]
    fn test_recovery_manager_no_recovery_needed() {
        let temp_dir = TempDir::new().unwrap();

        let mut manager = RecoveryManager::new(temp_dir.path().to_path_buf()).unwrap();
        let result = manager.recover().unwrap();

        assert!(matches!(result, RecoveryResult::NoRecoveryNeeded));
    }

    #[test]
    fn test_recovery_manager_incomplete_write() {
        let temp_dir = TempDir::new().unwrap();

        // Create incomplete write state
        let mut state = RecoveryState::default();
        state.mark_writing("test-stream");
        state.update_chunk_count(10);
        state.save(temp_dir.path()).unwrap();

        // Recover
        let mut manager = RecoveryManager::new(temp_dir.path().to_path_buf()).unwrap();
        let result = manager.recover().unwrap();

        match result {
            RecoveryResult::RolledBackWrite { stream_id, chunks } => {
                assert_eq!(stream_id, "test-stream");
                assert_eq!(chunks, 10);
            }
            _ => panic!("Expected RolledBackWrite"),
        }
    }
}
