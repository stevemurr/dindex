# HTTP API

DIndex includes a REST API server for programmatic access. Enable it in your config:

```toml
[http]
enabled = true
listen_addr = "0.0.0.0:8080"
api_keys = []       # Empty = no auth required; add keys to require Bearer token
cors_enabled = true
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/status` | Daemon status |
| `GET` | `/api/v1/stats` | Index statistics |
| `POST` | `/api/v1/search` | Search with optional filters |
| `POST` | `/api/v1/index` | Index documents |
| `POST` | `/api/v1/index/commit` | Force commit pending writes |
| `POST` | `/api/v1/index/clear` | Clear all entries from the index |
| `DELETE` | `/api/v1/documents` | Delete documents by IDs |

## Search

```bash
curl -s http://localhost:8081/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "top_k": 10}' | jq
```

### Search with Metadata Filtering

```bash
curl -X POST http://localhost:8081/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "top_k": 10,
    "filters": {
      "source_url_prefix": "https://arxiv.org",
      "metadata_equals": {"source": "arxiv"},
      "metadata_contains": {"category": ["ml", "ai"]}
    }
  }'
```

Filter types:

- **`source_url_prefix`**: Match documents whose URL starts with the given prefix
- **`metadata_equals`**: All specified key-value pairs must match exactly
- **`metadata_contains`**: Value must appear in the metadata field (supports comma-separated values in stored metadata)

## Index Documents

```bash
curl -X POST http://localhost:8081/api/v1/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "content": "Document text here...",
      "title": "My Document",
      "url": "https://example.com/doc",
      "metadata": {"category": "tech", "author": "Jane"}
    }]
  }'
```

## Delete Documents

```bash
# Delete specific documents by ID
curl -X DELETE http://localhost:8081/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{"document_ids": ["doc123", "doc456"]}'
```

## Clear Index

```bash
curl -X POST http://localhost:8081/api/v1/index/clear
```

## Force Commit

```bash
curl -X POST http://localhost:8081/api/v1/index/commit
```

## Authentication

When `api_keys` is non-empty in the config, all requests require a Bearer token:

```bash
curl -H "Authorization: Bearer your-api-key" http://localhost:8081/api/v1/health
```
