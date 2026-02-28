//! Persistent Document Registry for deduplication
//!
//! Provides content-based document identity and deduplication that works
//! uniformly across all input sources (Wikipedia import, web scraping, local files).
//!
//! Uses sled for entry persistence with in-memory indices for fast lookups.

use crate::types::{ChunkId, ContentHash, ContentId, DocumentIdentity};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Entry in the document registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentEntry {
    /// Content-based document ID (SimHash hex)
    pub content_id: ContentId,
    /// Exact content hash (SHA256 hex)
    pub content_hash: ContentHash,
    /// Raw SimHash value for Hamming distance calculations
    pub simhash: u64,
    /// All URLs mapping to this content
    pub urls: HashSet<String>,
    /// Document title
    pub title: Option<String>,
    /// When the document was first indexed
    pub first_seen: DateTime<Utc>,
    /// When the document was last updated
    pub last_updated: DateTime<Utc>,
    /// Source type (e.g., "wikipedia", "web", "local")
    pub source_type: String,
    /// Source-specific identifiers (e.g., {"wikipedia_id": "12345"})
    pub source_ids: HashMap<String, String>,
    /// Chunk IDs belonging to this document
    pub chunk_ids: Vec<ChunkId>,
}

impl DocumentEntry {
    /// Create a new document entry
    pub fn new(
        identity: DocumentIdentity,
        title: Option<String>,
        source_type: String,
    ) -> Self {
        let now = Utc::now();
        Self {
            content_id: identity.content_id,
            content_hash: identity.content_hash,
            simhash: identity.simhash,
            urls: HashSet::new(),
            title,
            first_seen: now,
            last_updated: now,
            source_type,
            source_ids: HashMap::new(),
            chunk_ids: Vec::new(),
        }
    }

    /// Add a URL to this document
    pub fn add_url(&mut self, url: String) {
        self.urls.insert(url);
        self.last_updated = Utc::now();
    }

    /// Add a source ID (e.g., wikipedia_id)
    pub fn add_source_id(&mut self, key: String, value: String) {
        self.source_ids.insert(key, value);
    }

    /// Update chunk IDs
    pub fn set_chunk_ids(&mut self, chunk_ids: Vec<ChunkId>) {
        self.chunk_ids = chunk_ids;
        self.last_updated = Utc::now();
    }

    /// Update content hash (for content updates)
    pub fn update_content(&mut self, new_hash: ContentHash) {
        self.content_hash = new_hash;
        self.last_updated = Utc::now();
    }
}

/// Result of checking for duplicates
#[derive(Debug, Clone)]
pub enum DuplicateCheckResult {
    /// Content is new, not seen before
    New,
    /// Exact match - same content hash
    ExactMatch { entry: DocumentEntry },
    /// Near-duplicate - similar content (within Hamming distance threshold)
    NearDuplicate {
        entry: DocumentEntry,
        hamming_distance: u32,
    },
}

/// Number of LSH bands for near-duplicate detection.
/// Each band uses an 8-bit slice of the 64-bit simhash.
/// Using 8 bands improves near-duplicate recall from ~67% to ~97%.
const NUM_LSH_BANDS: usize = 8;

/// Extract the bucket key for a given LSH band
fn band_key(simhash: u64, band: usize) -> u8 {
    (simhash >> (band * 8)) as u8
}

/// Inner data protected by a single lock to prevent TOCTOU races
struct RegistryInner {
    /// Content ID to entry mapping (in-memory cache)
    entries: HashMap<String, DocumentEntry>,
    /// SimHash to content ID mapping (for exact simhash lookups)
    simhash_index: HashMap<u64, String>,
    /// Multi-band LSH buckets for fast near-duplicate search
    /// Each band uses a different 8-bit slice of the 64-bit simhash
    simhash_bands: Vec<HashMap<u8, HashSet<String>>>,
    /// URL to content ID mapping (for URL-based lookups)
    url_index: HashMap<String, String>,
}

/// Persistent document registry for deduplication
pub struct DocumentRegistry {
    /// All in-memory indices under a single lock
    inner: RwLock<RegistryInner>,
    /// Sled database for persistent storage
    db: Option<sled::Db>,
    /// Storage path
    data_dir: PathBuf,
    /// Near-duplicate threshold (max Hamming distance)
    distance_threshold: u32,
}

impl DocumentRegistry {
    /// Create a new registry
    pub fn new(data_dir: impl AsRef<Path>, distance_threshold: u32) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;

        Ok(Self {
            inner: RwLock::new(RegistryInner {
                entries: HashMap::new(),
                simhash_index: HashMap::new(),
                simhash_bands: (0..NUM_LSH_BANDS).map(|_| HashMap::new()).collect(),
                url_index: HashMap::new(),
            }),
            db: None,
            data_dir,
            distance_threshold,
        })
    }

    /// Load registry from disk (auto-migrates from JSON if needed)
    pub fn load(data_dir: impl AsRef<Path>, distance_threshold: u32) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        let json_path = data_dir.join("document_registry.json");
        let sled_path = data_dir.join("document_registry.sled");

        // Check if we need to migrate from JSON
        if json_path.exists() && !sled_path.exists() {
            info!("Migrating document registry from JSON to sled...");
            let data = std::fs::read_to_string(&json_path)
                .context("Failed to read document registry")?;
            let saved: SavedRegistry =
                serde_json::from_str(&data).context("Failed to parse document registry")?;

            // Create sled database
            let db = sled::open(&sled_path).context("Failed to open registry database")?;

            // Write all entries to sled
            for (content_id, entry) in &saved.entries {
                let data = bincode::serialize(&entry)?;
                db.insert(content_id.as_bytes(), data)?;
            }
            db.flush()?;

            // Build in-memory registry
            let mut registry = Self::new(&data_dir, distance_threshold)?;
            registry.db = Some(db);

            for (content_id, entry) in saved.entries {
                registry.add_entry_internal(content_id, entry);
            }

            // Backup old JSON file
            let backup_path = data_dir.join("document_registry.json.backup");
            std::fs::rename(&json_path, &backup_path)?;
            info!(
                "Migration complete. Old JSON backed up to {:?}",
                backup_path
            );

            return Ok(registry);
        }

        // Load from sled if it exists
        if sled_path.exists() {
            let db = sled::open(&sled_path).context("Failed to open registry database")?;

            let mut registry = Self::new(&data_dir, distance_threshold)?;

            // Load all entries and rebuild indices
            for result in db.iter() {
                let (k, v) = result?;
                let content_id = String::from_utf8(k.to_vec())
                    .context("Invalid content ID in registry")?;
                let entry: DocumentEntry = bincode::deserialize(&v)
                    .context("Failed to deserialize registry entry")?;
                registry.add_entry_internal(content_id, entry);
            }

            info!("Loaded document registry with {} entries from sled", registry.len());
            registry.db = Some(db);

            return Ok(registry);
        }

        // No existing registry
        Self::new(data_dir, distance_threshold)
    }

    /// Save registry to disk
    pub fn save(&self) -> Result<()> {
        let sled_path = self.data_dir.join("document_registry.sled");

        let db = if let Some(ref db) = self.db {
            db.clone()
        } else {
            sled::open(&sled_path).context("Failed to open registry database")?
        };

        // Write all entries to sled
        let inner = self.inner.read();
        for (content_id, entry) in inner.entries.iter() {
            let data = bincode::serialize(entry)?;
            db.insert(content_id.as_bytes(), data)?;
        }

        db.flush().context("Failed to flush registry database")?;

        info!("Saved document registry with {} entries", inner.entries.len());

        Ok(())
    }

    /// Check if content is a duplicate
    pub fn check_duplicate(&self, identity: &DocumentIdentity) -> DuplicateCheckResult {
        let content_id_str = identity.content_id.as_str();
        let inner = self.inner.read();

        // Check for exact SimHash match first
        if let Some(entry) = inner.entries.get(content_id_str) {
            if entry.content_hash == identity.content_hash {
                return DuplicateCheckResult::ExactMatch {
                    entry: entry.clone(),
                };
            } else {
                return DuplicateCheckResult::NearDuplicate {
                    entry: entry.clone(),
                    hamming_distance: 0,
                };
            }
        }

        // Check for near-duplicates using multi-band LSH buckets
        // Collect unique candidates from all bands
        let mut checked = HashSet::new();
        for band in 0..NUM_LSH_BANDS {
            let key = band_key(identity.simhash, band);
            if let Some(bucket) = inner.simhash_bands[band].get(&key) {
                for candidate_id in bucket {
                    if !checked.insert(candidate_id.clone()) {
                        continue; // Already checked this candidate
                    }
                    if let Some(entry) = inner.entries.get(candidate_id.as_str()) {
                        let distance = (entry.simhash ^ identity.simhash).count_ones();
                        if distance <= self.distance_threshold {
                            if entry.content_hash == identity.content_hash {
                                return DuplicateCheckResult::ExactMatch {
                                    entry: entry.clone(),
                                };
                            } else {
                                return DuplicateCheckResult::NearDuplicate {
                                    entry: entry.clone(),
                                    hamming_distance: distance,
                                };
                            }
                        }
                    }
                }
            }
        }

        DuplicateCheckResult::New
    }

    /// Check for duplicate by URL
    pub fn check_url(&self, url: &str) -> Option<DocumentEntry> {
        let inner = self.inner.read();
        inner
            .url_index
            .get(url)
            .and_then(|content_id| inner.entries.get(content_id).cloned())
    }

    /// Register a new document
    pub fn register(
        &self,
        identity: DocumentIdentity,
        title: Option<String>,
        url: Option<String>,
        source_type: &str,
        source_id: Option<(&str, &str)>,
        chunk_ids: Vec<ChunkId>,
    ) -> DocumentEntry {
        let content_id_str = identity.content_id.as_str().to_string();
        let simhash = identity.simhash;

        let mut entry = DocumentEntry::new(identity, title, source_type.to_string());

        if let Some((key, value)) = source_id {
            entry.add_source_id(key.to_string(), value.to_string());
        }

        entry.set_chunk_ids(chunk_ids);

        // Single lock acquisition for all index updates
        {
            let mut inner = self.inner.write();

            if let Some(ref u) = url {
                entry.add_url(u.clone());
                inner.url_index.insert(u.clone(), content_id_str.clone());
            }

            inner.simhash_index.insert(simhash, content_id_str.clone());

            for band in 0..NUM_LSH_BANDS {
                let key = band_key(simhash, band);
                inner.simhash_bands[band]
                    .entry(key)
                    .or_default()
                    .insert(content_id_str.clone());
            }

            inner.entries.insert(content_id_str.clone(), entry.clone());
        }

        // Persist to sled (outside lock)
        if let Some(ref db) = self.db {
            match bincode::serialize(&entry) {
                Ok(data) => {
                    if let Err(e) = db.insert(content_id_str.as_bytes(), data) {
                        warn!("Failed to persist registry entry {}: {}", content_id_str, e);
                    }
                }
                Err(e) => warn!("Failed to serialize registry entry {}: {}", content_id_str, e),
            }
        }

        debug!("Registered new document: {}", entry.content_id);

        entry
    }

    /// Update an existing document's metadata
    pub fn update_metadata(
        &self,
        content_id: &ContentId,
        url: Option<String>,
        source_id: Option<(&str, &str)>,
    ) -> Option<DocumentEntry> {
        let content_id_str = content_id.as_str();
        let mut inner = self.inner.write();

        // Update URL index first (before borrowing entries mutably)
        if let Some(ref u) = url {
            inner.url_index.insert(u.clone(), content_id_str.to_string());
        }

        if let Some(entry) = inner.entries.get_mut(content_id_str) {
            if let Some(u) = url {
                entry.add_url(u);
            }

            if let Some((key, value)) = source_id {
                entry.add_source_id(key.to_string(), value.to_string());
            }

            entry.last_updated = Utc::now();
            let result = entry.clone();

            // Persist to sled
            if let Some(ref db) = self.db {
                match bincode::serialize(&result) {
                    Ok(data) => {
                        if let Err(e) = db.insert(content_id_str.as_bytes(), data) {
                            warn!("Failed to persist registry entry {}: {}", content_id_str, e);
                        }
                    }
                    Err(e) => warn!("Failed to serialize registry entry {}: {}", content_id_str, e),
                }
            }

            Some(result)
        } else {
            // Entry doesn't exist — remove the URL index entry we just added
            if let Some(ref u) = url {
                inner.url_index.remove(u);
            }
            None
        }
    }

    /// Update an existing document's content (for near-duplicate updates)
    pub fn update_content(
        &self,
        content_id: &ContentId,
        new_identity: DocumentIdentity,
        chunk_ids: Vec<ChunkId>,
    ) -> Option<(DocumentEntry, Vec<ChunkId>)> {
        let content_id_str = content_id.as_str();
        let mut inner = self.inner.write();

        if let Some(entry) = inner.entries.get_mut(content_id_str) {
            let old_chunk_ids = std::mem::replace(&mut entry.chunk_ids, chunk_ids);
            entry.update_content(new_identity.content_hash);
            let result = entry.clone();

            // Persist to sled
            if let Some(ref db) = self.db {
                match bincode::serialize(&result) {
                    Ok(data) => {
                        if let Err(e) = db.insert(content_id_str.as_bytes(), data) {
                            warn!("Failed to persist registry entry {}: {}", content_id_str, e);
                        }
                    }
                    Err(e) => warn!("Failed to serialize registry entry {}: {}", content_id_str, e),
                }
            }

            Some((result, old_chunk_ids))
        } else {
            None
        }
    }

    /// Remove a document from the registry
    pub fn remove(&self, content_id: &ContentId) -> Option<DocumentEntry> {
        let content_id_str = content_id.as_str();
        let mut inner = self.inner.write();

        let entry = inner.entries.remove(content_id_str)?;

        // Clean up indices under the same lock
        inner.simhash_index.remove(&entry.simhash);

        for band in 0..NUM_LSH_BANDS {
            let key = band_key(entry.simhash, band);
            if let Some(bucket) = inner.simhash_bands[band].get_mut(&key) {
                bucket.remove(content_id_str);
            }
        }

        for url in &entry.urls {
            inner.url_index.remove(url);
        }

        // Remove from sled
        if let Some(ref db) = self.db {
            if let Err(e) = db.remove(content_id_str.as_bytes()) {
                warn!("Failed to remove registry entry {} from database: {}", content_id_str, e);
            }
        }

        Some(entry)
    }

    /// Get a document entry by content ID
    pub fn get(&self, content_id: &ContentId) -> Option<DocumentEntry> {
        self.inner.read().entries.get(content_id.as_str()).cloned()
    }

    /// Get the number of registered documents
    pub fn len(&self) -> usize {
        self.inner.read().entries.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.inner.read().entries.is_empty()
    }

    /// Internal method to add an entry and update all indices (used during load)
    fn add_entry_internal(&mut self, content_id: String, entry: DocumentEntry) {
        let inner = self.inner.get_mut();

        inner.simhash_index.insert(entry.simhash, content_id.clone());

        for band in 0..NUM_LSH_BANDS {
            let key = band_key(entry.simhash, band);
            inner.simhash_bands[band]
                .entry(key)
                .or_default()
                .insert(content_id.clone());
        }

        for url in &entry.urls {
            inner.url_index.insert(url.clone(), content_id.clone());
        }

        inner.entries.insert(content_id, entry);
    }

    /// Get statistics about the registry
    pub fn stats(&self) -> RegistryStats {
        let inner = self.inner.read();
        let total_docs = inner.entries.len();
        let total_chunks: usize = inner.entries.values().map(|e| e.chunk_ids.len()).sum();
        let total_urls = inner.url_index.len();
        let buckets_used = inner.simhash_bands.iter().map(|b| b.len()).sum();

        let source_counts: HashMap<String, usize> = inner.entries
            .values()
            .fold(HashMap::new(), |mut acc, e| {
                *acc.entry(e.source_type.clone()).or_insert(0) += 1;
                acc
            });

        RegistryStats {
            total_documents: total_docs,
            total_chunks,
            total_urls,
            buckets_used,
            source_counts,
        }
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub total_documents: usize,
    pub total_chunks: usize,
    pub total_urls: usize,
    pub buckets_used: usize,
    pub source_counts: HashMap<String, usize>,
}

/// Serializable registry for JSON format (used only for migration from old format)
#[derive(Serialize, Deserialize)]
struct SavedRegistry {
    entries: HashMap<String, DocumentEntry>,
    #[allow(dead_code)]
    version: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_registry_new_document() {
        let temp_dir = TempDir::new().unwrap();
        let registry = DocumentRegistry::new(temp_dir.path(), 3).unwrap();

        let content = "This is a test document for the registry";
        let identity = DocumentIdentity::compute(content);

        // Check should return New
        let result = registry.check_duplicate(&identity);
        assert!(matches!(result, DuplicateCheckResult::New));

        // Register the document
        let entry = registry.register(
            identity.clone(),
            Some("Test Document".to_string()),
            Some("https://example.com/test".to_string()),
            "test",
            Some(("test_id", "123")),
            vec!["chunk1".to_string(), "chunk2".to_string()],
        );

        assert_eq!(entry.title, Some("Test Document".to_string()));
        assert!(entry.urls.contains("https://example.com/test"));
        assert_eq!(entry.chunk_ids.len(), 2);

        // Check should now return ExactMatch
        let result = registry.check_duplicate(&identity);
        assert!(matches!(result, DuplicateCheckResult::ExactMatch { .. }));
    }

    #[test]
    fn test_registry_exact_match() {
        let temp_dir = TempDir::new().unwrap();
        let registry = DocumentRegistry::new(temp_dir.path(), 3).unwrap();

        let content = "This is the exact same content";
        let identity1 = DocumentIdentity::compute(content);
        let identity2 = DocumentIdentity::compute(content);

        // Register first
        registry.register(
            identity1.clone(),
            Some("First".to_string()),
            None,
            "test",
            None,
            vec![],
        );

        // Check second should be exact match
        let result = registry.check_duplicate(&identity2);
        match result {
            DuplicateCheckResult::ExactMatch { entry } => {
                assert_eq!(entry.title, Some("First".to_string()));
            }
            _ => panic!("Expected ExactMatch"),
        }
    }

    #[test]
    fn test_registry_near_duplicate() {
        let temp_dir = TempDir::new().unwrap();
        let registry = DocumentRegistry::new(temp_dir.path(), 10).unwrap();

        let content1 = "This is a test document with some content for testing purposes";
        let content2 = "This is a test document with some content for testing purposes today";

        let identity1 = DocumentIdentity::compute(content1);
        let identity2 = DocumentIdentity::compute(content2);

        // Verify they're similar but not identical
        let distance = identity1.hamming_distance(&identity2);
        assert!(distance > 0, "Contents should produce different hashes");
        assert!(distance <= 10, "Contents should be similar");

        // Register first
        registry.register(identity1.clone(), Some("First".to_string()), None, "test", None, vec![]);

        // Check second
        let result = registry.check_duplicate(&identity2);
        match result {
            DuplicateCheckResult::NearDuplicate { entry, hamming_distance } => {
                assert_eq!(entry.title, Some("First".to_string()));
                assert!(hamming_distance <= 10);
            }
            DuplicateCheckResult::ExactMatch { .. } => {
                // Also acceptable if hashes happened to match
            }
            DuplicateCheckResult::New => {
                panic!("Expected NearDuplicate or ExactMatch, got New");
            }
        }
    }

    #[test]
    fn test_registry_url_lookup() {
        let temp_dir = TempDir::new().unwrap();
        let registry = DocumentRegistry::new(temp_dir.path(), 3).unwrap();

        let identity = DocumentIdentity::compute("Test content");

        registry.register(
            identity,
            Some("Test".to_string()),
            Some("https://example.com/page".to_string()),
            "web",
            None,
            vec![],
        );

        // Lookup by URL
        let entry = registry
            .check_url("https://example.com/page")
            .expect("URL should be found in registry");
        assert_eq!(entry.title, Some("Test".to_string()));

        // Non-existent URL
        let entry = registry.check_url("https://other.com/page");
        assert!(entry.is_none());
    }

    #[test]
    fn test_registry_persistence() {
        let temp_dir = TempDir::new().unwrap();

        // Create and populate registry
        {
            let registry = DocumentRegistry::new(temp_dir.path(), 3).unwrap();
            let identity = DocumentIdentity::compute("Persistent content");
            registry.register(
                identity,
                Some("Persistent Doc".to_string()),
                Some("https://example.com/persist".to_string()),
                "test",
                None,
                vec!["chunk1".to_string()],
            );
            registry.save().unwrap();
        }

        // Load and verify
        {
            let registry = DocumentRegistry::load(temp_dir.path(), 3).unwrap();
            assert_eq!(registry.len(), 1);

            let identity = DocumentIdentity::compute("Persistent content");
            let result = registry.check_duplicate(&identity);
            match result {
                DuplicateCheckResult::ExactMatch { entry } => {
                    assert_eq!(entry.title, Some("Persistent Doc".to_string()));
                }
                _ => panic!("Expected ExactMatch after reload"),
            }
        }

        // Verify sled file exists (not JSON)
        assert!(temp_dir.path().join("document_registry.sled").exists());
        assert!(!temp_dir.path().join("document_registry.json").exists());
    }

    #[test]
    fn test_registry_update_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let registry = DocumentRegistry::new(temp_dir.path(), 3).unwrap();

        let identity = DocumentIdentity::compute("Test content");
        let entry = registry.register(
            identity.clone(),
            Some("Original".to_string()),
            Some("https://example.com/first".to_string()),
            "test",
            None,
            vec![],
        );

        // Update with new URL
        let updated = registry.update_metadata(
            &entry.content_id,
            Some("https://example.com/second".to_string()),
            Some(("extra_id", "456")),
        );

        let updated = updated.expect("update_metadata should succeed");
        assert!(updated.urls.contains("https://example.com/first"));
        assert!(updated.urls.contains("https://example.com/second"));
        assert_eq!(updated.source_ids.get("extra_id"), Some(&"456".to_string()));
    }

    #[test]
    fn test_registry_remove() {
        let temp_dir = TempDir::new().unwrap();
        let registry = DocumentRegistry::new(temp_dir.path(), 3).unwrap();

        let identity = DocumentIdentity::compute("Content to remove");
        let entry = registry.register(
            identity.clone(),
            Some("Removable".to_string()),
            Some("https://example.com/remove".to_string()),
            "test",
            None,
            vec!["chunk1".to_string()],
        );

        assert_eq!(registry.len(), 1);

        // Remove
        let removed = registry.remove(&entry.content_id);
        assert!(removed.is_some());
        assert_eq!(registry.len(), 0);

        // Check it's gone
        let result = registry.check_duplicate(&identity);
        assert!(matches!(result, DuplicateCheckResult::New));

        // URL lookup should also fail
        let url_lookup = registry.check_url("https://example.com/remove");
        assert!(url_lookup.is_none());
    }

    #[test]
    fn test_registry_stats() {
        let temp_dir = TempDir::new().unwrap();
        let registry = DocumentRegistry::new(temp_dir.path(), 3).unwrap();

        // Add some documents from different sources
        registry.register(
            DocumentIdentity::compute("Wikipedia article"),
            Some("Wiki".to_string()),
            Some("https://en.wikipedia.org/wiki/Test".to_string()),
            "wikipedia",
            Some(("wikipedia_id", "123")),
            vec!["c1".to_string(), "c2".to_string(), "c3".to_string()],
        );

        registry.register(
            DocumentIdentity::compute("Web page content"),
            Some("Web".to_string()),
            Some("https://example.com/page".to_string()),
            "web",
            None,
            vec!["c4".to_string(), "c5".to_string()],
        );

        registry.register(
            DocumentIdentity::compute("Local file content"),
            Some("Local".to_string()),
            None,
            "local",
            None,
            vec!["c6".to_string()],
        );

        let stats = registry.stats();
        assert_eq!(stats.total_documents, 3);
        assert_eq!(stats.total_chunks, 6);
        assert_eq!(stats.total_urls, 2);
        assert_eq!(stats.source_counts.get("wikipedia"), Some(&1));
        assert_eq!(stats.source_counts.get("web"), Some(&1));
        assert_eq!(stats.source_counts.get("local"), Some(&1));
    }
}
