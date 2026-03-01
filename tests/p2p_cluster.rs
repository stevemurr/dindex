//! P2P cluster integration tests using Docker containers.
//!
//! Each test stands up a multi-node cluster of dindex containers on an
//! isolated Docker bridge network, then exercises peer discovery, scraping,
//! indexing, and distributed search across the nodes.
//!
//! Prerequisites:
//!   - Docker and Docker Compose installed
//!   - GPU available for embedding server (or DINDEX_EMBEDDING_ENDPOINT set)
//!   - Network access for web scraping from containers
//!
//! Run:
//!   cargo test --test p2p_cluster -- --ignored --nocapture --test-threads=1
//!
//! Or use the convenience script:
//!   ./tests/p2p/run.sh

use std::process::Command;
use std::time::{Duration, Instant};

const COMPOSE_FILE: &str = "tests/p2p/docker-compose.yml";
const PROJECT_NAME: &str = "dindex-p2p-test";

// ──────── Docker helpers ────────

/// Run a docker compose command (without profiles) and return output.
fn compose(args: &[&str]) -> std::process::Output {
    let mut cmd = Command::new("docker");
    cmd.arg("compose")
        .arg("-f")
        .arg(COMPOSE_FILE)
        .arg("-p")
        .arg(PROJECT_NAME);
    for arg in args {
        cmd.arg(arg);
    }
    cmd.output().expect("failed to run docker compose")
}

/// Run a docker compose command with extra profile flags.
fn compose_with_profiles(profiles: &[&str], args: &[&str]) -> std::process::Output {
    let mut cmd = Command::new("docker");
    cmd.arg("compose")
        .arg("-f")
        .arg(COMPOSE_FILE)
        .arg("-p")
        .arg(PROJECT_NAME);
    for profile in profiles {
        cmd.arg("--profile").arg(profile);
    }
    for arg in args {
        cmd.arg(arg);
    }
    cmd.output().expect("failed to run docker compose")
}

/// Run a command inside a running container.
fn docker_exec(container: &str, cmd: &[&str]) -> std::process::Output {
    let mut command = Command::new("docker");
    command.arg("exec").arg(container);
    for arg in cmd {
        command.arg(arg);
    }
    command.output().expect("failed to run docker exec")
}

/// Get container logs as a string.
fn docker_logs(container: &str) -> String {
    let output = Command::new("docker")
        .arg("logs")
        .arg(container)
        .output()
        .expect("failed to get docker logs");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    format!("{}{}", stdout, stderr)
}

/// Wait for a string pattern to appear in container logs.
fn wait_for_log(container: &str, pattern: &str, timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        let logs = docker_logs(container);
        if logs.contains(pattern) {
            return true;
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    false
}

/// Count occurrences of a pattern in container logs.
fn count_log_matches(container: &str, pattern: &str) -> usize {
    let logs = docker_logs(container);
    logs.matches(pattern).count()
}

/// Parse JSON search output from `dindex search --format json`.
fn parse_search_results(output: &str) -> serde_json::Value {
    // The output may contain log lines before the JSON — find the first '{'.
    if let Some(json_start) = output.find('{') {
        let json_str = &output[json_start..];
        if let Ok(val) = serde_json::from_str(json_str) {
            return val;
        }
    }
    eprintln!("[cluster] Warning: could not parse search output as JSON:\n{}", output);
    serde_json::json!({"results": [], "total_documents": 0, "total_chunks": 0})
}

/// Check if Docker is available.
fn require_docker() {
    let output = Command::new("docker")
        .arg("info")
        .output()
        .expect("docker not found — is Docker installed?");
    assert!(
        output.status.success(),
        "Docker daemon is not running: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Build the dindex image if not already built.
fn build_image() {
    eprintln!("[cluster] Building dindex:p2p-test image...");
    let output = compose(&["build", "node-1"]);
    assert!(
        output.status.success(),
        "docker compose build failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    eprintln!("[cluster] Image built successfully");
}

/// Start the embeddings service and wait for it to be healthy.
fn start_embeddings() {
    eprintln!("[cluster] Starting embeddings service...");
    let output = compose_with_profiles(
        &["with-embeddings"],
        &["up", "-d", "embeddings"],
    );
    if !output.status.success() {
        eprintln!(
            "[cluster] Warning: Failed to start embeddings service:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    // Wait for the embedding server to be healthy (vLLM can take a while to load)
    eprintln!("[cluster] Waiting for embeddings server to be healthy (this may take a few minutes)...");
    let start = Instant::now();
    let timeout = Duration::from_secs(300); // 5 minutes
    loop {
        if start.elapsed() > timeout {
            eprintln!("[cluster] Warning: Embeddings server health check timed out after {:?}", timeout);
            break;
        }

        let health = Command::new("docker")
            .arg("exec")
            .arg("p2p-test-embeddings")
            .arg("curl")
            .arg("-sf")
            .arg("http://localhost:8000/health")
            .output();

        if let Ok(out) = health {
            if out.status.success() {
                eprintln!("[cluster] Embeddings server is healthy ({:.0}s)", start.elapsed().as_secs_f64());
                return;
            }
        }
        std::thread::sleep(Duration::from_secs(5));
    }
}

/// Start a cluster of N nodes (1-indexed).
/// Also starts the embeddings service.
fn start_cluster(node_count: usize) {
    assert!(
        (1..=5).contains(&node_count),
        "node_count must be 1..=5"
    );

    // Start embeddings first
    start_embeddings();

    // Determine which services to start
    let services: Vec<String> = (1..=node_count)
        .map(|n| format!("node-{}", n))
        .collect();

    let service_refs: Vec<&str> = services.iter().map(|s| s.as_str()).collect();

    // Build compose up args
    let mut up_args: Vec<&str> = vec!["up", "-d"];
    up_args.extend_from_slice(&service_refs);

    // Nodes 4-5 require the five-nodes profile
    let profiles: Vec<&str> = if node_count > 3 {
        vec!["with-embeddings", "five-nodes"]
    } else {
        vec!["with-embeddings"]
    };

    let output = compose_with_profiles(&profiles, &up_args);
    assert!(
        output.status.success(),
        "docker compose up failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    eprintln!("[cluster] Started {} node(s)", node_count);
}

/// Wait for all nodes to be healthy (IPC socket exists).
fn wait_for_healthy(node_count: usize, timeout: Duration) {
    eprintln!("[cluster] Waiting for {} node(s) to become healthy...", node_count);
    let start = Instant::now();

    for n in 1..=node_count {
        let container = format!("p2p-test-node-{}", n);
        loop {
            if start.elapsed() > timeout {
                let logs = docker_logs(&container);
                panic!(
                    "Timed out waiting for {} to become healthy. Logs:\n{}",
                    container,
                    logs.chars().take(2000).collect::<String>()
                );
            }

            let output = docker_exec(&container, &[
                "test", "-S", "/run/dindex/dindex/dindex.sock",
            ]);
            if output.status.success() {
                eprintln!("[cluster] {} is healthy", container);
                break;
            }
            std::thread::sleep(Duration::from_secs(3));
        }
    }
}

/// Wait for the embedding engine to initialize on specified nodes.
fn wait_for_embedding_engine(node_count: usize, timeout: Duration) {
    for n in 1..=node_count {
        let container = format!("p2p-test-node-{}", n);
        eprintln!("[cluster] Waiting for embedding engine on {}...", container);
        let ready = wait_for_log(&container, "Embedding engine initialized", timeout);
        if !ready {
            // Check if there's a config error
            let logs = docker_logs(&container);
            if logs.contains("Failed to initialize embedding engine") {
                eprintln!("[cluster] WARNING: {} failed to init embedding engine", container);
                eprintln!("[cluster] Relevant log: {}",
                    logs.lines()
                        .find(|l| l.contains("Failed to initialize"))
                        .unwrap_or("(not found)")
                );
            } else {
                eprintln!("[cluster] WARNING: Timed out waiting for embedding engine on {}", container);
            }
        } else {
            eprintln!("[cluster] {} embedding engine ready", container);
        }
    }
}

/// Wait for peer connections to establish between nodes.
fn wait_for_peers(node_count: usize, timeout: Duration) {
    let expected_peers = node_count - 1;
    eprintln!(
        "[cluster] Waiting for each node to discover {} peer(s)...",
        expected_peers
    );

    let start = Instant::now();
    for n in 1..=node_count {
        let container = format!("p2p-test-node-{}", n);
        loop {
            if start.elapsed() > timeout {
                let logs = docker_logs(&container);
                let found = count_log_matches(&container, "Peer connected");
                panic!(
                    "{} found only {}/{} peers. Logs:\n{}",
                    container,
                    found,
                    expected_peers,
                    logs.chars().take(2000).collect::<String>()
                );
            }

            let found = count_log_matches(&container, "Peer connected");
            if found >= expected_peers {
                eprintln!("[cluster] {} connected to {} peer(s)", container, found);
                break;
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    }
}

/// Tear down the cluster, removing volumes.
fn teardown_cluster() {
    eprintln!("[cluster] Tearing down cluster...");
    // Use all profiles to catch all containers
    let _ = compose_with_profiles(
        &["with-embeddings", "five-nodes"],
        &["down", "-v", "--remove-orphans"],
    );
    eprintln!("[cluster] Cluster torn down");
}

/// Wait for indexed chunks to appear on a node by polling `dindex stats`.
/// Returns the total chunk count, or 0 if timed out.
fn wait_for_chunks(container: &str, min_chunks: usize, timeout: Duration) -> usize {
    eprintln!("[cluster] Waiting for at least {} chunks on {}...", min_chunks, container);
    let start = Instant::now();
    loop {
        if start.elapsed() > timeout {
            eprintln!("[cluster] Timed out waiting for chunks on {}", container);
            return 0;
        }

        let output = docker_exec(
            container,
            &["dindex", "--config", "/etc/dindex/config.toml", "stats"],
        );
        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse "Total chunks: N" from stats output
        for line in stdout.lines() {
            if let Some(rest) = line.strip_prefix("Total chunks:") {
                if let Ok(n) = rest.trim().parse::<usize>() {
                    if n >= min_chunks {
                        eprintln!("[cluster] {} has {} chunks", container, n);
                        return n;
                    }
                }
            }
            // Also try looking in daemon stats output (different format)
            if line.contains("chunks") {
                if let Some(num) = line.split_whitespace()
                    .find_map(|w| w.parse::<usize>().ok())
                {
                    if num >= min_chunks {
                        eprintln!("[cluster] {} has {} chunks", container, num);
                        return num;
                    }
                }
            }
        }

        std::thread::sleep(Duration::from_secs(5));
    }
}

/// Run a scrape command on a container and wait for completion.
fn scrape_on_node(container: &str, url: &str, max_pages: usize, depth: u8) {
    eprintln!("[cluster] Scraping {} on {} (max_pages={}, depth={})", url, container, max_pages, depth);

    let max_pages_str = max_pages.to_string();
    let depth_str = depth.to_string();
    let output = docker_exec(
        container,
        &[
            "dindex",
            "--config", "/etc/dindex/config.toml",
            "scrape", url,
            "--max-pages", &max_pages_str,
            "--depth", &depth_str,
            "--stay-on-domain",
            "--delay-ms", "500",
        ],
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!(
        "[cluster] Scrape on {} finished (status={}):\nstdout: {}\nstderr: {}",
        container, output.status, stdout, stderr
    );

    if stderr.contains("Failed to load config") || stderr.contains("Failed to parse config") {
        panic!(
            "Config parse error on {}! Fix the TOML config. stderr:\n{}",
            container, stderr
        );
    }
}

/// Run a search on a container and return the JSON results.
fn search_on_node(container: &str, query: &str, top_k: usize) -> serde_json::Value {
    eprintln!("[cluster] Searching on {} for '{}'", container, query);

    let top_k_str = top_k.to_string();
    let output = docker_exec(
        container,
        &[
            "dindex",
            "--config", "/etc/dindex/config.toml",
            "search", query,
            "--top-k", &top_k_str,
            "--format", "json",
        ],
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if stderr.contains("Failed to load config") || stderr.contains("Failed to parse config") {
        panic!(
            "Config parse error on {} during search! stderr:\n{}",
            container, stderr
        );
    }

    if !output.status.success() {
        eprintln!("[cluster] Search failed on {}:\nstdout: {}\nstderr: {}", container, stdout, stderr);
        panic!("Search command failed on {}", container);
    }

    eprintln!("[cluster] Search output from {}:\n{}", container, stdout);
    parse_search_results(&stdout)
}

/// Check if search results contain content matching a pattern (case-insensitive).
fn results_contain(results: &serde_json::Value, pattern: &str) -> bool {
    let pattern_lower = pattern.to_lowercase();
    let json_str = serde_json::to_string(results).unwrap_or_default().to_lowercase();
    json_str.contains(&pattern_lower)
}

/// Get total number of result documents from JSON output.
fn result_count(results: &serde_json::Value) -> usize {
    results["total_documents"]
        .as_u64()
        .unwrap_or(0) as usize
}

// ──────── Tests ────────

/// Test 1: Three nodes discover each other via mDNS on the Docker bridge network.
#[test]
#[ignore]
fn test_three_node_peer_discovery() {
    require_docker();
    build_image();

    // Clean up from any previous run
    teardown_cluster();

    start_cluster(3);
    wait_for_healthy(3, Duration::from_secs(120));
    wait_for_peers(3, Duration::from_secs(60));

    // Verify each node logged at least 2 peer connections
    for n in 1..=3 {
        let container = format!("p2p-test-node-{}", n);
        let peer_count = count_log_matches(&container, "Peer connected");
        assert!(
            peer_count >= 2,
            "{} should have at least 2 connected peers, found {}",
            container,
            peer_count
        );
    }

    teardown_cluster();
}

/// Test 2: Single node scrapes a website and performs local search.
#[test]
#[ignore]
fn test_scrape_and_local_search() {
    require_docker();
    build_image();
    teardown_cluster();

    start_cluster(1);
    wait_for_healthy(1, Duration::from_secs(120));
    wait_for_embedding_engine(1, Duration::from_secs(180));

    // Scrape a small Wikipedia page
    scrape_on_node(
        "p2p-test-node-1",
        "https://en.wikipedia.org/wiki/Rust_(programming_language)",
        3,
        1,
    );

    // Wait for the write pipeline to commit (polls dindex stats)
    let chunks = wait_for_chunks("p2p-test-node-1", 1, Duration::from_secs(90));
    assert!(chunks > 0, "Expected at least 1 chunk after scraping");

    // Search locally
    let results = search_on_node("p2p-test-node-1", "Rust programming language", 5);

    eprintln!("[cluster] Search results: {}", serde_json::to_string_pretty(&results).unwrap());

    assert!(
        result_count(&results) > 0,
        "Expected non-empty search results for 'Rust programming language'"
    );
    assert!(
        results_contain(&results, "rust") || results_contain(&results, "wikipedia"),
        "Expected results to reference Rust or Wikipedia"
    );

    teardown_cluster();
}

/// Test 3: Three nodes scrape different topics, then search across the cluster.
/// Each node should find content indexed on other nodes via distributed search.
#[test]
#[ignore]
fn test_cross_node_distributed_search() {
    require_docker();
    build_image();
    teardown_cluster();

    start_cluster(3);
    wait_for_healthy(3, Duration::from_secs(120));
    wait_for_peers(3, Duration::from_secs(60));
    wait_for_embedding_engine(3, Duration::from_secs(180));

    // Each node scrapes a different programming language
    scrape_on_node(
        "p2p-test-node-1",
        "https://en.wikipedia.org/wiki/Rust_(programming_language)",
        3, 1,
    );
    scrape_on_node(
        "p2p-test-node-2",
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        3, 1,
    );
    scrape_on_node(
        "p2p-test-node-3",
        "https://en.wikipedia.org/wiki/Go_(programming_language)",
        3, 1,
    );

    // Wait for all nodes to have indexed chunks
    for n in 1..=3 {
        let container = format!("p2p-test-node-{}", n);
        wait_for_chunks(&container, 1, Duration::from_secs(90));
    }

    // Allow extra time for advertisement broadcasting
    eprintln!("[cluster] Waiting for advertisement broadcasts...");
    std::thread::sleep(Duration::from_secs(10));

    // Cross-node search: from Node 1, search for Python (indexed on Node 2)
    let results = search_on_node("p2p-test-node-1", "Python programming language", 5);
    assert!(
        result_count(&results) > 0,
        "Node 1 should find results for 'Python' (indexed on Node 2)"
    );
    assert!(
        results_contain(&results, "python"),
        "Results from Node 1 should reference Python content from Node 2"
    );

    // Cross-node search: from Node 2, search for Rust (indexed on Node 1)
    let results = search_on_node("p2p-test-node-2", "Rust systems programming", 5);
    assert!(
        result_count(&results) > 0,
        "Node 2 should find results for 'Rust' (indexed on Node 1)"
    );
    assert!(
        results_contain(&results, "rust"),
        "Results from Node 2 should reference Rust content from Node 1"
    );

    // Cross-node search: from Node 3, search for Python (indexed on Node 2)
    let results = search_on_node("p2p-test-node-3", "Python dynamic typing", 5);
    assert!(
        result_count(&results) > 0,
        "Node 3 should find results for 'Python' (indexed on Node 2)"
    );

    teardown_cluster();
}

/// Test 4: Five-node full pipeline — each node scrapes a unique topic,
/// then searches for content that only exists on other nodes.
#[test]
#[ignore]
fn test_five_node_full_pipeline() {
    require_docker();
    build_image();
    teardown_cluster();

    start_cluster(5);
    wait_for_healthy(5, Duration::from_secs(180));
    wait_for_peers(5, Duration::from_secs(90));
    wait_for_embedding_engine(5, Duration::from_secs(180));

    // Each node scrapes a different topic
    let topics = [
        ("p2p-test-node-1", "https://en.wikipedia.org/wiki/Machine_learning"),
        ("p2p-test-node-2", "https://en.wikipedia.org/wiki/Quantum_computing"),
        ("p2p-test-node-3", "https://en.wikipedia.org/wiki/Climate_change"),
        ("p2p-test-node-4", "https://en.wikipedia.org/wiki/Artificial_intelligence"),
        ("p2p-test-node-5", "https://en.wikipedia.org/wiki/Blockchain"),
    ];

    for (container, url) in &topics {
        scrape_on_node(container, url, 3, 1);
    }

    // Wait for all nodes to have indexed chunks
    for n in 1..=5 {
        let container = format!("p2p-test-node-{}", n);
        wait_for_chunks(&container, 1, Duration::from_secs(90));
    }

    // Allow extra time for advertisement broadcasting
    eprintln!("[cluster] Waiting for advertisement broadcasts...");
    std::thread::sleep(Duration::from_secs(15));

    // Cross-node searches: each node searches for a topic on a DIFFERENT node
    let cross_searches = [
        // (search from, query, expected keyword, indexed on)
        ("p2p-test-node-1", "quantum computing qubits", "quantum", "node-2"),
        ("p2p-test-node-2", "climate change global warming", "climate", "node-3"),
        ("p2p-test-node-3", "artificial intelligence neural networks", "artificial", "node-4"),
        ("p2p-test-node-4", "blockchain cryptocurrency", "blockchain", "node-5"),
        ("p2p-test-node-5", "machine learning algorithms", "machine", "node-1"),
    ];

    for (container, query, keyword, source) in &cross_searches {
        let results = search_on_node(container, query, 5);
        assert!(
            result_count(&results) > 0,
            "{} should find results for '{}' (indexed on {})",
            container, query, source
        );
        assert!(
            results_contain(&results, keyword),
            "Results from {} should contain '{}' (content from {})",
            container, keyword, source
        );
    }

    teardown_cluster();
}
