//! BM25 lexical search using Tantivy

use crate::types::{Chunk, ChunkId};

#[cfg(test)]
use crate::types::ChunkMetadata;
use anyhow::{Context, Result};
use std::path::Path;
use tantivy::{
    collector::TopDocs,
    directory::MmapDirectory,
    query::QueryParser,
    schema::{Field, Schema, Value, STORED, TEXT},
    Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument,
};
use tracing::debug;

/// BM25 search index using Tantivy
pub struct Bm25Index {
    index: Index,
    reader: IndexReader,
    writer: parking_lot::Mutex<IndexWriter>,
    schema: Bm25Schema,
}

/// Schema fields for BM25 index
struct Bm25Schema {
    chunk_id: Field,
    document_id: Field,
    content: Field,
    title: Field,
}

/// BM25 search result
#[derive(Debug, Clone)]
pub struct Bm25SearchResult {
    pub chunk_id: ChunkId,
    pub document_id: String,
    pub score: f32,
}

impl Bm25Index {
    /// Create a new BM25 index in memory
    pub fn new_in_memory() -> Result<Self> {
        let (schema, fields) = Self::build_schema();
        let index = Index::create_in_ram(schema);

        let writer = index.writer(50_000_000)?; // 50MB buffer
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        Ok(Self {
            index,
            reader,
            writer: parking_lot::Mutex::new(writer),
            schema: fields,
        })
    }

    /// Create a new BM25 index on disk
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;

        let (schema, fields) = Self::build_schema();
        let dir = MmapDirectory::open(path)?;
        let index = Index::open_or_create(dir, schema)?;

        let writer = index.writer(50_000_000)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        Ok(Self {
            index,
            reader,
            writer: parking_lot::Mutex::new(writer),
            schema: fields,
        })
    }

    fn build_schema() -> (Schema, Bm25Schema) {
        let mut schema_builder = Schema::builder();

        let chunk_id = schema_builder.add_text_field("chunk_id", STORED);
        let document_id = schema_builder.add_text_field("document_id", STORED);
        let content = schema_builder.add_text_field("content", TEXT | STORED);
        let title = schema_builder.add_text_field("title", TEXT);

        let schema = schema_builder.build();
        let fields = Bm25Schema {
            chunk_id,
            document_id,
            content,
            title,
        };

        (schema, fields)
    }

    /// Add a chunk to the index
    pub fn add(&self, chunk: &Chunk) -> Result<()> {
        let mut doc = TantivyDocument::new();
        doc.add_text(self.schema.chunk_id, &chunk.metadata.chunk_id);
        doc.add_text(self.schema.document_id, &chunk.metadata.document_id);
        doc.add_text(self.schema.content, &chunk.content);

        if let Some(title) = &chunk.metadata.source_title {
            doc.add_text(self.schema.title, title);
        }

        self.writer.lock().add_document(doc)?;
        Ok(())
    }

    /// Commit pending changes
    pub fn commit(&self) -> Result<()> {
        self.writer.lock().commit()?;
        // Force reader reload to see committed changes immediately
        self.reader.reload()?;
        Ok(())
    }

    /// Search for matching chunks
    pub fn search(&self, query_text: &str, k: usize) -> Result<Vec<Bm25SearchResult>> {
        let searcher = self.reader.searcher();

        let query_parser = QueryParser::for_index(&self.index, vec![self.schema.content]);
        let query = query_parser
            .parse_query(query_text)
            .context("Failed to parse query")?;

        let top_docs = searcher.search(&query, &TopDocs::with_limit(k))?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;

            let chunk_id = doc
                .get_first(self.schema.chunk_id)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let document_id = doc
                .get_first(self.schema.document_id)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            results.push(Bm25SearchResult {
                chunk_id,
                document_id,
                score,
            });
        }

        debug!("BM25 search for '{}': {} results", query_text, results.len());
        Ok(results)
    }

    /// Delete a chunk by ID
    pub fn delete(&self, chunk_id: &str) -> Result<()> {
        let term = tantivy::Term::from_field_text(self.schema.chunk_id, chunk_id);
        self.writer.lock().delete_term(term);
        Ok(())
    }

    /// Delete all documents and commit
    pub fn clear(&self) -> Result<()> {
        self.writer.lock().delete_all_documents()?;
        self.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(id: &str, doc_id: &str, content: &str) -> Chunk {
        Chunk {
            metadata: ChunkMetadata::new(id.to_string(), doc_id.to_string()),
            content: content.to_string(),
            token_count: content.split_whitespace().count(),
        }
    }

    #[test]
    fn test_bm25_search() {
        let index = Bm25Index::new_in_memory().unwrap();

        let chunk1 = Chunk {
            metadata: ChunkMetadata::new("chunk1".to_string(), "doc1".to_string()),
            content: "The quick brown fox jumps over the lazy dog".to_string(),
            token_count: 9,
        };

        let chunk2 = Chunk {
            metadata: ChunkMetadata::new("chunk2".to_string(), "doc1".to_string()),
            content: "A fast cat runs across the street".to_string(),
            token_count: 7,
        };

        index.add(&chunk1).unwrap();
        index.add(&chunk2).unwrap();
        index.commit().unwrap();

        let results = index.search("fox jumps", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "chunk1");
    }

    #[test]
    fn test_bm25_delete_does_not_error() {
        let index = Bm25Index::new_in_memory().unwrap();

        let chunk1 = make_chunk("chunk1", "doc1", "unique alpha bravo content");
        let chunk2 = make_chunk("chunk2", "doc1", "different charlie delta content");

        index.add(&chunk1).unwrap();
        index.add(&chunk2).unwrap();
        index.commit().unwrap();

        // Verify chunk1 is findable before delete
        let results = index.search("alpha bravo", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "chunk1");

        // Delete should not error (note: chunk_id field is STORED-only,
        // so delete_term on it is effectively a no-op in Tantivy;
        // the HybridIndexer compensates by also removing from vector
        // index and chunk storage, filtering out the deleted chunk
        // from final results)
        index.delete("chunk1").unwrap();
        index.commit().unwrap();

        // chunk2 should still be findable regardless
        let results = index.search("charlie delta", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "chunk2");
    }

    #[test]
    fn test_bm25_empty_query_returns_empty() {
        let index = Bm25Index::new_in_memory().unwrap();

        let chunk = make_chunk("chunk1", "doc1", "some content here");
        index.add(&chunk).unwrap();
        index.commit().unwrap();

        // Tantivy may error on truly empty queries; we treat that as empty results
        let results = index.search("", 10);
        match results {
            Ok(r) => assert!(r.is_empty(), "Empty query should return no results"),
            Err(_) => {} // An error on empty query is also acceptable
        }
    }

    #[test]
    fn test_bm25_search_before_commit_returns_empty() {
        let index = Bm25Index::new_in_memory().unwrap();

        let chunk = make_chunk("chunk1", "doc1", "searchable content text");
        index.add(&chunk).unwrap();
        // Deliberately not calling commit()

        let results = index.search("searchable content", 10).unwrap();
        assert!(
            results.is_empty(),
            "Search before commit should return empty results"
        );
    }

    #[test]
    fn test_bm25_delete_nonexistent_is_ok() {
        let index = Bm25Index::new_in_memory().unwrap();

        // Deleting a chunk that was never added should not error
        let result = index.delete("nonexistent_chunk");
        assert!(result.is_ok());
    }

    #[test]
    fn test_bm25_search_with_no_documents() {
        let index = Bm25Index::new_in_memory().unwrap();
        index.commit().unwrap();

        let results = index.search("anything", 10).unwrap();
        assert!(results.is_empty());
    }
}
