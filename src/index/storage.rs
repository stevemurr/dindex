//! Index storage and persistence utilities
//!
//! Uses sled embedded database for efficient on-disk storage with lazy loading.
//! Chunks are stored and retrieved on-demand without loading everything into memory.

use crate::types::{Chunk, IndexedChunk};
use anyhow::{Context, Result};

#[cfg(test)]
use crate::types::ChunkMetadata;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Storage for chunk metadata and content using sled embedded database
pub struct ChunkStorage {
    /// Sled database for chunk storage
    db: sled::Db,
    /// Secondary index: document_id -> chunk_ids (for document queries)
    doc_index: sled::Tree,
}

/// Stored chunk data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredChunk {
    pub chunk: Chunk,
    pub embedding: Vec<f32>,
    pub index_key: u64,
}

impl ChunkStorage {
    /// Create new chunk storage
    pub fn new(data_dir: impl AsRef<Path>) -> Result<Self> {
        Self::open(data_dir)
    }

    /// Load storage from disk (automatically migrates from JSON if needed)
    pub fn load(data_dir: impl AsRef<Path>) -> Result<Self> {
        let data_dir = data_dir.as_ref();

        // Check if old JSON format exists and migrate
        if let Some(migrated) = Self::migrate_from_json(data_dir)? {
            return Ok(migrated);
        }

        Self::open(data_dir)
    }

    /// Open or create the sled database
    fn open(data_dir: impl AsRef<Path>) -> Result<Self> {
        let db_path = data_dir.as_ref().join("chunks.sled");
        let db = sled::open(&db_path)
            .with_context(|| format!("Failed to open chunk database at {:?}", db_path))?;

        let doc_index = db
            .open_tree("doc_index")
            .context("Failed to open document index tree")?;

        Ok(Self { db, doc_index })
    }

    /// Save storage to disk (flushes sled buffers)
    pub fn save(&self) -> Result<()> {
        self.db.flush().context("Failed to flush chunk database")?;
        Ok(())
    }

    /// Store a chunk
    pub fn store(&self, chunk: &IndexedChunk) {
        let stored = StoredChunk {
            chunk: chunk.chunk.clone(),
            embedding: chunk.embedding.clone(),
            index_key: chunk.index_key,
        };

        let chunk_id = &chunk.chunk.metadata.chunk_id;
        let doc_id = &chunk.chunk.metadata.document_id;

        // Serialize and store the chunk
        if let Ok(data) = bincode::serialize(&stored) {
            let _ = self.db.insert(chunk_id.as_bytes(), data);

            // Update document index
            self.add_to_doc_index(doc_id, chunk_id);
        }
    }

    /// Add chunk_id to document index
    fn add_to_doc_index(&self, doc_id: &str, chunk_id: &str) {
        let key = doc_id.as_bytes();
        let mut chunk_ids: Vec<String> = self
            .doc_index
            .get(key)
            .ok()
            .flatten()
            .and_then(|data| bincode::deserialize(&data).ok())
            .unwrap_or_default();

        if !chunk_ids.contains(&chunk_id.to_string()) {
            chunk_ids.push(chunk_id.to_string());
            if let Ok(data) = bincode::serialize(&chunk_ids) {
                let _ = self.doc_index.insert(key, data);
            }
        }
    }

    /// Remove chunk_id from document index
    fn remove_from_doc_index(&self, doc_id: &str, chunk_id: &str) {
        let key = doc_id.as_bytes();
        if let Ok(Some(data)) = self.doc_index.get(key) {
            if let Ok(mut chunk_ids) = bincode::deserialize::<Vec<String>>(&data) {
                chunk_ids.retain(|id| id != chunk_id);
                if chunk_ids.is_empty() {
                    let _ = self.doc_index.remove(key);
                } else if let Ok(data) = bincode::serialize(&chunk_ids) {
                    let _ = self.doc_index.insert(key, data);
                }
            }
        }
    }

    /// Get a chunk by ID
    pub fn get(&self, chunk_id: &str) -> Option<StoredChunk> {
        self.db
            .get(chunk_id.as_bytes())
            .ok()
            .flatten()
            .and_then(|data| bincode::deserialize(&data).ok())
    }

    /// Get multiple chunks by ID
    pub fn get_batch(&self, chunk_ids: &[String]) -> Vec<StoredChunk> {
        chunk_ids.iter().filter_map(|id| self.get(id)).collect()
    }

    /// Remove a chunk
    pub fn remove(&self, chunk_id: &str) -> Option<StoredChunk> {
        let stored = self.get(chunk_id)?;
        let doc_id = &stored.chunk.metadata.document_id;

        self.remove_from_doc_index(doc_id, chunk_id);
        let _ = self.db.remove(chunk_id.as_bytes());

        Some(stored)
    }

    /// Get all chunk IDs
    pub fn chunk_ids(&self) -> Vec<String> {
        self.db
            .iter()
            .keys()
            .filter_map(|r| r.ok())
            .filter_map(|k| String::from_utf8(k.to_vec()).ok())
            .collect()
    }

    /// Get total count
    pub fn len(&self) -> usize {
        self.db.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.db.is_empty()
    }

    /// Get all embeddings (for centroid computation)
    /// Note: This still loads all embeddings - consider streaming for very large indexes
    pub fn all_embeddings(&self) -> Vec<(String, Vec<f32>)> {
        self.db
            .iter()
            .filter_map(|r| r.ok())
            .filter_map(|(k, v)| {
                let id = String::from_utf8(k.to_vec()).ok()?;
                let stored: StoredChunk = bincode::deserialize(&v).ok()?;
                Some((id, stored.embedding))
            })
            .collect()
    }

    /// Get chunks by document ID (uses secondary index)
    pub fn get_by_document(&self, document_id: &str) -> Vec<StoredChunk> {
        let chunk_ids: Vec<String> = self
            .doc_index
            .get(document_id.as_bytes())
            .ok()
            .flatten()
            .and_then(|data| bincode::deserialize(&data).ok())
            .unwrap_or_default();

        self.get_batch(&chunk_ids)
    }

    /// Count unique documents using the document index
    pub fn document_count(&self) -> usize {
        self.doc_index.len()
    }

    /// Migrate from old JSON storage format if it exists
    pub fn migrate_from_json(data_dir: impl AsRef<Path>) -> Result<Option<Self>> {
        use std::collections::HashMap;

        let data_dir = data_dir.as_ref();
        let json_path = data_dir.join("chunks.json");

        if !json_path.exists() {
            return Ok(None);
        }

        tracing::info!("Migrating chunk storage from JSON to sled...");

        // Load old JSON data
        let data = std::fs::read_to_string(&json_path)?;
        let old_chunks: HashMap<String, StoredChunk> = serde_json::from_str(&data)?;

        // Create new sled storage
        let storage = Self::new(data_dir)?;

        // Migrate all chunks
        for (_chunk_id, stored) in old_chunks {
            let indexed = IndexedChunk {
                chunk: stored.chunk,
                embedding: stored.embedding,
                lsh_signature: None,
                index_key: stored.index_key,
            };
            storage.store(&indexed);
        }

        storage.save()?;

        // Rename old file as backup
        let backup_path = data_dir.join("chunks.json.backup");
        std::fs::rename(&json_path, &backup_path)?;
        tracing::info!(
            "Migration complete. Old JSON backed up to {:?}",
            backup_path
        );

        Ok(Some(storage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_chunk_storage() {
        let temp_dir = TempDir::new().unwrap();
        let storage = ChunkStorage::new(temp_dir.path()).unwrap();

        let metadata = ChunkMetadata::new("chunk1".to_string(), "doc1".to_string());
        let chunk = Chunk {
            metadata,
            content: "test content".to_string(),
            token_count: 2,
        };

        let indexed = IndexedChunk {
            chunk,
            embedding: vec![1.0, 2.0, 3.0],
            lsh_signature: None,
            index_key: 0,
        };

        storage.store(&indexed);
        assert_eq!(storage.len(), 1);

        let retrieved = storage.get("chunk1").unwrap();
        assert_eq!(retrieved.chunk.content, "test content");
    }

    #[test]
    fn test_chunk_storage_document_count() {
        let temp_dir = TempDir::new().unwrap();
        let storage = ChunkStorage::new(temp_dir.path()).unwrap();

        // Add chunks from two different documents
        for i in 0..3u64 {
            let metadata = ChunkMetadata::new(format!("chunk-doc1-{}", i), "doc1".to_string());
            let indexed = IndexedChunk {
                chunk: Chunk {
                    metadata,
                    content: format!("content {}", i),
                    token_count: 2,
                },
                embedding: vec![1.0; 3],
                lsh_signature: None,
                index_key: i,
            };
            storage.store(&indexed);
        }

        for i in 0..2u64 {
            let metadata = ChunkMetadata::new(format!("chunk-doc2-{}", i), "doc2".to_string());
            let indexed = IndexedChunk {
                chunk: Chunk {
                    metadata,
                    content: format!("content {}", i),
                    token_count: 2,
                },
                embedding: vec![2.0; 3],
                lsh_signature: None,
                index_key: 10 + i,
            };
            storage.store(&indexed);
        }

        assert_eq!(storage.len(), 5);
        assert_eq!(storage.document_count(), 2);

        // Test get_by_document
        let doc1_chunks = storage.get_by_document("doc1");
        assert_eq!(doc1_chunks.len(), 3);

        let doc2_chunks = storage.get_by_document("doc2");
        assert_eq!(doc2_chunks.len(), 2);
    }

    #[test]
    fn test_chunk_storage_persistence() {
        let temp_dir = TempDir::new().unwrap();

        // Create and store
        {
            let storage = ChunkStorage::new(temp_dir.path()).unwrap();
            let metadata = ChunkMetadata::new("chunk1".to_string(), "doc1".to_string());
            let indexed = IndexedChunk {
                chunk: Chunk {
                    metadata,
                    content: "persistent content".to_string(),
                    token_count: 2,
                },
                embedding: vec![1.0, 2.0, 3.0],
                lsh_signature: None,
                index_key: 42,
            };
            storage.store(&indexed);
            storage.save().unwrap();
        }

        // Reload and verify
        {
            let storage = ChunkStorage::load(temp_dir.path()).unwrap();
            assert_eq!(storage.len(), 1);
            let retrieved = storage.get("chunk1").unwrap();
            assert_eq!(retrieved.chunk.content, "persistent content");
            assert_eq!(retrieved.index_key, 42);
        }
    }
}
