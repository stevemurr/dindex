# CLI Reference

Full command-line usage for DIndex.

## Initialization

```bash
# Generate default dindex.toml in the current directory
dindex init
```

## Indexing Documents

```bash
# Index a single file
dindex index ./path/to/document.txt

# Index with metadata
dindex index ./paper.pdf --title "Research Paper" --url "https://example.com/paper"

# Index a directory
dindex index ./documents/
```

## Searching

```bash
# Basic search
dindex search "semantic search concepts"

# JSON output for LLM consumption
dindex search "semantic search" --format json --top-k 20

# Export results to file
dindex search "query" --format json > results.json
```

## Web Scraping

```bash
# Scrape a site with depth control
dindex scrape https://example.com --depth 2 --stay-on-domain

# Scrape multiple seeds with rate limiting
dindex scrape https://site1.com https://site2.com --max-pages 1000 --delay-ms 1000

# View scraping statistics
dindex scrape-stats
```

## Bulk Import

```bash
# Import Wikipedia dump (auto-detects format)
dindex import ./wiki.xml.bz2 --batch-size 100

# Resume an interrupted import
dindex import ./wiki.xml.bz2 --resume --checkpoint ./checkpoint.json

# Check import progress
dindex import-status ./checkpoint.json
```

## P2P Network

```bash
# Start node (daemonizes by default)
dindex start

# Start in foreground
dindex start --foreground

# Start with custom listen address
dindex start --listen /ip4/0.0.0.0/udp/4001/quic-v1

# Connect to bootstrap peers
dindex start --bootstrap /ip4/1.2.3.4/udp/4001/quic-v1/p2p/QmPeerId

# Daemon management
dindex daemon status
dindex daemon stop
dindex daemon restart
```

## Statistics & Export

```bash
# Show index statistics
dindex stats

# Show document registry statistics
dindex registry-stats

# Export for LLM consumption
dindex export ./output.jsonl --format jsonl
```

## Output Format

Retrieved chunks include rich metadata for LLM consumption:

```json
{
  "chunk_id": "doc123_0",
  "document_id": "doc123",
  "content": "The chunk content...",
  "source_url": "https://example.com/doc",
  "source_title": "Document Title",
  "timestamp": "2025-01-08T00:00:00Z",
  "position_in_doc": 0.35,
  "section_hierarchy": ["Chapter 2", "Section 2.1"],
  "preceding_chunk_id": "doc123_prev",
  "following_chunk_id": "doc123_next",
  "relevance_score": 0.89,
  "matched_by": ["dense", "bm25"]
}
```
