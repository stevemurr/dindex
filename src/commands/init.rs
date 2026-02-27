use anyhow::Result;
use dindex::config::Config;
use std::path::PathBuf;

pub async fn init_config(path: PathBuf) -> Result<()> {
    let config = Config::default();
    let config_path = path.join("dindex.toml");

    // Generate TOML config
    let toml_content = format!(
        r#"# DIndex Configuration

[node]
listen_addr = "{}"
data_dir = ".dindex"
enable_mdns = true
replication_factor = 3
query_timeout_secs = 10

[embedding]
model_name = "{}"
dimensions = {}
truncated_dimensions = {}
max_sequence_length = {}
quantize_int8 = true
# GPU acceleration - auto-enabled when built with cuda feature
# Set to false to force CPU mode
# use_gpu = true
# gpu_device_id = 0

[index]
hnsw_m = {}
hnsw_ef_construction = {}
hnsw_ef_search = {}
memory_mapped = true
max_capacity = {}

[chunking]
chunk_size = {}
overlap_fraction = {}
min_chunk_size = {}
max_chunk_size = {}

[retrieval]
enable_dense = true
enable_bm25 = true
rrf_k = {}
candidate_count = {}
enable_reranking = true

[routing]
num_centroids = {}
lsh_bits = {}
lsh_num_hashes = {}
bloom_bits_per_item = {}
candidate_nodes = {}
"#,
        config.node.listen_addr,
        config.embedding.model_name,
        config.embedding.dimensions,
        config.embedding.truncated_dimensions,
        config.embedding.max_sequence_length,
        config.index.hnsw_m,
        config.index.hnsw_ef_construction,
        config.index.hnsw_ef_search,
        config.index.max_capacity,
        config.chunking.chunk_size,
        config.chunking.overlap_fraction,
        config.chunking.min_chunk_size,
        config.chunking.max_chunk_size,
        config.retrieval.rrf_k,
        config.retrieval.candidate_count,
        config.routing.num_centroids,
        config.routing.lsh_bits,
        config.routing.lsh_num_hashes,
        config.routing.bloom_bits_per_item,
        config.routing.candidate_nodes,
    );

    std::fs::write(&config_path, toml_content)?;
    println!("Created configuration file: {}", config_path.display());

    // Create data directory
    let data_dir = path.join(".dindex");
    std::fs::create_dir_all(&data_dir)?;
    println!("Created data directory: {}", data_dir.display());

    Ok(())
}
