//! Persistent Document Registry for deduplication
//!
//! Provides content-based document identity and deduplication that works
//! uniformly across all input sources (Wikipedia import, web scraping, local files).

use crate::types::{ChunkId, ContentHash, ContentId, DocumentIdentity};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

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

/// Persistent document registry for deduplication
pub struct DocumentRegistry {
    /// Content ID to entry mapping
    entries: RwLock<HashMap<String, DocumentEntry>>,
    /// SimHash to content ID mapping (for exact simhash lookups)
    simhash_index: RwLock<HashMap<u64, String>>,
    /// LSH buckets for fast near-duplicate search (top 8 bits of simhash)
    simhash_buckets: RwLock<HashMap<u8, HashSet<String>>>,
    /// URL to content ID mapping (for URL-based lookups)
    url_index: RwLock<HashMap<String, String>>,
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
            entries: RwLock::new(HashMap::new()),
            simhash_index: RwLock::new(HashMap::new()),
            simhash_buckets: RwLock::new(HashMap::new()),
            url_index: RwLock::new(HashMap::new()),
            data_dir,
            distance_threshold,
        })
    }

    /// Load registry from disk
    pub fn load(data_dir: impl AsRef<Path>, distance_threshold: u32) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        let registry_path = data_dir.join("document_registry.json");

        let registry = if registry_path.exists() {
            let data = std::fs::read_to_string(&registry_path)
                .context("Failed to read document registry")?;
            let saved: SavedRegistry =
                serde_json::from_str(&data).context("Failed to parse document registry")?;

            info!("Loaded document registry with {} entries", saved.entries.len());

            let mut new_registry = Self::new(&data_dir, distance_threshold)?;

            // Rebuild all indices from loaded entries
            for (content_id, entry) in saved.entries {
                new_registry.add_entry_internal(content_id, entry);
            }

            new_registry
        } else {
            Self::new(data_dir, distance_threshold)?
        };

        Ok(registry)
    }

    /// Save registry to disk
    pub fn save(&self) -> Result<()> {
        let registry_path = self.data_dir.join("document_registry.json");

        let saved = SavedRegistry {
            entries: self.entries.read().clone(),
            version: 1,
        };

        let data = serde_json::to_string_pretty(&saved)?;
        std::fs::write(&registry_path, data)?;

        info!(
            "Saved document registry with {} entries",
            saved.entries.len()
        );

        Ok(())
    }

    /// Check if content is a duplicate
    pub fn check_duplicate(&self, identity: &DocumentIdentity) -> DuplicateCheckResult {
        let content_id_str = identity.content_id.as_str();

        // Check for exact SimHash match first
        {
            let entries = self.entries.read();
            if let Some(entry) = entries.get(content_id_str) {
                // Check if content hash matches exactly
                if entry.content_hash == identity.content_hash {
                    return DuplicateCheckResult::ExactMatch {
                        entry: entry.clone(),
                    };
                } else {
                    // Same simhash but different content - this is a near-duplicate
                    return DuplicateCheckResult::NearDuplicate {
                        entry: entry.clone(),
                        hamming_distance: 0,
                    };
                }
            }
        }

        // Check for near-duplicates using LSH buckets
        let bucket_id = (identity.simhash >> 56) as u8;

        {
            let buckets = self.simhash_buckets.read();
            let entries = self.entries.read();

            if let Some(bucket) = buckets.get(&bucket_id) {
                for candidate_id in bucket {
                    if let Some(entry) = entries.get(candidate_id) {
                        let distance = (entry.simhash ^ identity.simhash).count_ones();
                        if distance <= self.distance_threshold {
                            // Check if it's an exact match by content hash
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

        // No duplicate found
        DuplicateCheckResult::New
    }

    /// Check for duplicate by URL
    pub fn check_url(&self, url: &str) -> Option<DocumentEntry> {
        let url_index = self.url_index.read();
        let entries = self.entries.read();

        url_index
            .get(url)
            .and_then(|content_id| entries.get(content_id).cloned())
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

        if let Some(u) = url {
            entry.add_url(u.clone());
            self.url_index.write().insert(u, content_id_str.clone());
        }

        if let Some((key, value)) = source_id {
            entry.add_source_id(key.to_string(), value.to_string());
        }

        entry.set_chunk_ids(chunk_ids);

        // Update indices
        self.simhash_index
            .write()
            .insert(simhash, content_id_str.clone());

        let bucket_id = (simhash >> 56) as u8;
        self.simhash_buckets
            .write()
            .entry(bucket_id)
            .or_default()
            .insert(content_id_str.clone());

        self.entries.write().insert(content_id_str, entry.clone());

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
        let mut entries = self.entries.write();

        if let Some(entry) = entries.get_mut(content_id_str) {
            if let Some(u) = url {
                entry.add_url(u.clone());
                self.url_index.write().insert(u, content_id_str.to_string());
            }

            if let Some((key, value)) = source_id {
                entry.add_source_id(key.to_string(), value.to_string());
            }

            entry.last_updated = Utc::now();
            Some(entry.clone())
        } else {
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
        let mut entries = self.entries.write();

        if let Some(entry) = entries.get_mut(content_id_str) {
            let old_chunk_ids = std::mem::replace(&mut entry.chunk_ids, chunk_ids);
            entry.update_content(new_identity.content_hash);
            Some((entry.clone(), old_chunk_ids))
        } else {
            None
        }
    }

    /// Remove a document from the registry
    pub fn remove(&self, content_id: &ContentId) -> Option<DocumentEntry> {
        let content_id_str = content_id.as_str();

        let entry = self.entries.write().remove(content_id_str)?;

        // Clean up indices
        self.simhash_index.write().remove(&entry.simhash);

        let bucket_id = (entry.simhash >> 56) as u8;
        if let Some(bucket) = self.simhash_buckets.write().get_mut(&bucket_id) {
            bucket.remove(content_id_str);
        }

        for url in &entry.urls {
            self.url_index.write().remove(url);
        }

        Some(entry)
    }

    /// Get a document entry by content ID
    pub fn get(&self, content_id: &ContentId) -> Option<DocumentEntry> {
        self.entries.read().get(content_id.as_str()).cloned()
    }

    /// Get all entries
    pub fn all_entries(&self) -> Vec<DocumentEntry> {
        self.entries.read().values().cloned().collect()
    }

    /// Get the number of registered documents
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Internal method to add an entry and update all indices
    fn add_entry_internal(&mut self, content_id: String, entry: DocumentEntry) {
        // Update simhash index
        self.simhash_index
            .write()
            .insert(entry.simhash, content_id.clone());

        // Update bucket
        let bucket_id = (entry.simhash >> 56) as u8;
        self.simhash_buckets
            .write()
            .entry(bucket_id)
            .or_default()
            .insert(content_id.clone());

        // Update URL index
        for url in &entry.urls {
            self.url_index
                .write()
                .insert(url.clone(), content_id.clone());
        }

        // Add entry
        self.entries.write().insert(content_id, entry);
    }

    /// Get statistics about the registry
    pub fn stats(&self) -> RegistryStats {
        let entries = self.entries.read();
        let total_docs = entries.len();
        let total_chunks: usize = entries.values().map(|e| e.chunk_ids.len()).sum();
        let total_urls = self.url_index.read().len();
        let buckets_used = self.simhash_buckets.read().len();

        let source_counts: HashMap<String, usize> = entries
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

/// Serializable registry for persistence
#[derive(Serialize, Deserialize)]
struct SavedRegistry {
    entries: HashMap<String, DocumentEntry>,
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
        let entry = registry.check_url("https://example.com/page");
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().title, Some("Test".to_string()));

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

        assert!(updated.is_some());
        let updated = updated.unwrap();
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
