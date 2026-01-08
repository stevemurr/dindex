//! Index storage and persistence utilities

use crate::types::{Chunk, IndexedChunk};
use anyhow::Result;

#[cfg(test)]
use crate::types::ChunkMetadata;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Storage for chunk metadata and content
pub struct ChunkStorage {
    /// In-memory storage
    chunks: RwLock<HashMap<String, StoredChunk>>,
    /// Storage directory
    data_dir: PathBuf,
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
        let data_dir = data_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;

        Ok(Self {
            chunks: RwLock::new(HashMap::new()),
            data_dir,
        })
    }

    /// Load storage from disk
    pub fn load(data_dir: impl AsRef<Path>) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        let storage_path = data_dir.join("chunks.json");

        let chunks = if storage_path.exists() {
            let data = std::fs::read_to_string(&storage_path)?;
            serde_json::from_str(&data)?
        } else {
            HashMap::new()
        };

        Ok(Self {
            chunks: RwLock::new(chunks),
            data_dir,
        })
    }

    /// Save storage to disk
    pub fn save(&self) -> Result<()> {
        let storage_path = self.data_dir.join("chunks.json");
        let data = serde_json::to_string_pretty(&*self.chunks.read())?;
        std::fs::write(&storage_path, data)?;
        Ok(())
    }

    /// Store a chunk
    pub fn store(&self, chunk: &IndexedChunk) {
        let stored = StoredChunk {
            chunk: chunk.chunk.clone(),
            embedding: chunk.embedding.clone(),
            index_key: chunk.index_key,
        };
        self.chunks
            .write()
            .insert(chunk.chunk.metadata.chunk_id.clone(), stored);
    }

    /// Get a chunk by ID
    pub fn get(&self, chunk_id: &str) -> Option<StoredChunk> {
        self.chunks.read().get(chunk_id).cloned()
    }

    /// Get multiple chunks by ID
    pub fn get_batch(&self, chunk_ids: &[String]) -> Vec<StoredChunk> {
        let chunks = self.chunks.read();
        chunk_ids
            .iter()
            .filter_map(|id| chunks.get(id).cloned())
            .collect()
    }

    /// Remove a chunk
    pub fn remove(&self, chunk_id: &str) -> Option<StoredChunk> {
        self.chunks.write().remove(chunk_id)
    }

    /// Get all chunk IDs
    pub fn chunk_ids(&self) -> Vec<String> {
        self.chunks.read().keys().cloned().collect()
    }

    /// Get total count
    pub fn len(&self) -> usize {
        self.chunks.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.chunks.read().is_empty()
    }

    /// Get all embeddings (for centroid computation)
    pub fn all_embeddings(&self) -> Vec<(String, Vec<f32>)> {
        self.chunks
            .read()
            .iter()
            .map(|(id, stored)| (id.clone(), stored.embedding.clone()))
            .collect()
    }

    /// Get chunks by document ID
    pub fn get_by_document(&self, document_id: &str) -> Vec<StoredChunk> {
        self.chunks
            .read()
            .values()
            .filter(|stored| stored.chunk.metadata.document_id == document_id)
            .cloned()
            .collect()
    }
}

/// Document storage for managing full documents
pub struct DocumentStorage {
    /// Document metadata
    documents: RwLock<HashMap<String, DocumentMetadata>>,
    /// Storage directory
    data_dir: PathBuf,
}

/// Document metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub id: String,
    pub title: Option<String>,
    pub url: Option<String>,
    pub chunk_ids: Vec<String>,
    pub indexed_at: chrono::DateTime<chrono::Utc>,
}

impl DocumentStorage {
    /// Create new document storage
    pub fn new(data_dir: impl AsRef<Path>) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;

        Ok(Self {
            documents: RwLock::new(HashMap::new()),
            data_dir,
        })
    }

    /// Load from disk
    pub fn load(data_dir: impl AsRef<Path>) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        let storage_path = data_dir.join("documents.json");

        let documents = if storage_path.exists() {
            let data = std::fs::read_to_string(&storage_path)?;
            serde_json::from_str(&data)?
        } else {
            HashMap::new()
        };

        Ok(Self {
            documents: RwLock::new(documents),
            data_dir,
        })
    }

    /// Save to disk
    pub fn save(&self) -> Result<()> {
        let storage_path = self.data_dir.join("documents.json");
        let data = serde_json::to_string_pretty(&*self.documents.read())?;
        std::fs::write(&storage_path, data)?;
        Ok(())
    }

    /// Add document metadata
    pub fn add(&self, metadata: DocumentMetadata) {
        self.documents
            .write()
            .insert(metadata.id.clone(), metadata);
    }

    /// Get document metadata
    pub fn get(&self, document_id: &str) -> Option<DocumentMetadata> {
        self.documents.read().get(document_id).cloned()
    }

    /// Remove document
    pub fn remove(&self, document_id: &str) -> Option<DocumentMetadata> {
        self.documents.write().remove(document_id)
    }

    /// List all document IDs
    pub fn document_ids(&self) -> Vec<String> {
        self.documents.read().keys().cloned().collect()
    }

    /// Get document count
    pub fn len(&self) -> usize {
        self.documents.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.documents.read().is_empty()
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
}
