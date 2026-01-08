//! Domain assignment via consistent hashing
//!
//! Assigns domains to nodes without centralized coordination using consistent hashing
//! on hostnames. This approach ensures:
//! - Politeness is local: All crawl-delay and rate-limiting decisions stay on one node
//! - robots.txt caching: Each domain's robots.txt is cached exactly once
//! - Reduced cross-node chatter: Most extracted URLs stay local (same-site links)
//! - Natural load distribution: Domains are roughly uniformly distributed by hash

use blake3::hash;
use std::collections::BTreeMap;

/// Peer identifier type (matches libp2p PeerId string representation)
pub type PeerId = String;

/// Manages domain-to-node assignment using consistent hashing with virtual nodes
#[derive(Debug, Clone)]
pub struct DomainAssignment {
    /// Ring of (hash_position, peer_id) for consistent hashing
    ring: BTreeMap<u64, PeerId>,
    /// Number of virtual nodes per physical node (recommended: 150)
    virtual_nodes: usize,
    /// Local peer ID for quick checks
    local_peer_id: Option<PeerId>,
}

impl DomainAssignment {
    /// Create a new domain assignment with the specified number of virtual nodes
    pub fn new(virtual_nodes: usize) -> Self {
        Self {
            ring: BTreeMap::new(),
            virtual_nodes,
            local_peer_id: None,
        }
    }

    /// Create with default virtual nodes (150)
    pub fn with_defaults() -> Self {
        Self::new(150)
    }

    /// Set the local peer ID for quick ownership checks
    pub fn set_local_peer(&mut self, peer_id: PeerId) {
        self.local_peer_id = Some(peer_id);
    }

    /// Get the local peer ID
    pub fn local_peer_id(&self) -> Option<&PeerId> {
        self.local_peer_id.as_ref()
    }

    /// Check if we own a domain (are responsible for crawling it)
    pub fn is_local_domain(&self, hostname: &str) -> bool {
        if let Some(local_id) = &self.local_peer_id {
            if let Some(assigned) = self.assign_domain(hostname) {
                return &assigned == local_id;
            }
        }
        false
    }

    /// Assign a domain (hostname) to a node
    pub fn assign_domain(&self, hostname: &str) -> Option<PeerId> {
        if self.ring.is_empty() {
            return None;
        }

        let key = Self::hash_hostname(hostname);

        // Find first node >= key on the ring
        self.ring
            .range(key..)
            .next()
            .or_else(|| self.ring.iter().next()) // Wrap around
            .map(|(_, peer)| peer.clone())
    }

    /// Hash a hostname to a u64 position on the ring
    fn hash_hostname(hostname: &str) -> u64 {
        let h = hash(hostname.as_bytes());
        u64::from_be_bytes(h.as_bytes()[..8].try_into().unwrap())
    }

    /// Hash a virtual node key to a ring position
    fn hash_vnode(peer_id: &str, vnode_index: usize) -> u64 {
        let key = format!("{}:{}", peer_id, vnode_index);
        let h = hash(key.as_bytes());
        u64::from_be_bytes(h.as_bytes()[..8].try_into().unwrap())
    }

    /// Handle a node joining the network
    pub fn on_node_join(&mut self, peer_id: PeerId) {
        // Add virtual nodes to ring
        for i in 0..self.virtual_nodes {
            let key = Self::hash_vnode(&peer_id, i);
            self.ring.insert(key, peer_id.clone());
        }
    }

    /// Handle a node leaving the network
    pub fn on_node_leave(&mut self, peer_id: &PeerId) {
        self.ring.retain(|_, p| p != peer_id);
    }

    /// Get all domains that would move from old_owner to new_owner if new_owner joins
    /// This is useful for understanding migration impact
    pub fn domains_affected_by_join(
        &self,
        new_peer: &PeerId,
        sample_domains: &[String],
    ) -> Vec<(String, PeerId)> {
        let mut affected = Vec::new();

        // Create a temporary ring with the new peer
        let mut temp_ring = self.clone();
        temp_ring.on_node_join(new_peer.clone());

        for domain in sample_domains {
            let old_owner = self.assign_domain(domain);
            let new_owner = temp_ring.assign_domain(domain);

            if old_owner != new_owner {
                if let Some(old) = old_owner {
                    affected.push((domain.clone(), old));
                }
            }
        }

        affected
    }

    /// Get the number of nodes in the ring
    pub fn node_count(&self) -> usize {
        let unique_peers: std::collections::HashSet<_> = self.ring.values().collect();
        unique_peers.len()
    }

    /// Get all peer IDs in the ring
    pub fn all_peers(&self) -> Vec<PeerId> {
        let unique_peers: std::collections::HashSet<_> = self.ring.values().cloned().collect();
        unique_peers.into_iter().collect()
    }

    /// Clear the ring
    pub fn clear(&mut self) {
        self.ring.clear();
    }
}

impl Default for DomainAssignment {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_hashing() {
        let mut assignment = DomainAssignment::new(10);

        // Add some nodes
        assignment.on_node_join("peer1".to_string());
        assignment.on_node_join("peer2".to_string());
        assignment.on_node_join("peer3".to_string());

        // Same domain should always map to same peer
        let domain = "example.com";
        let peer1 = assignment.assign_domain(domain);
        let peer2 = assignment.assign_domain(domain);
        assert_eq!(peer1, peer2);

        // Different domains might map to different peers
        let domains = vec![
            "google.com",
            "github.com",
            "rust-lang.org",
            "mozilla.org",
            "example.net",
        ];

        let mut assignments: std::collections::HashMap<PeerId, usize> =
            std::collections::HashMap::new();
        for d in &domains {
            if let Some(peer) = assignment.assign_domain(d) {
                *assignments.entry(peer).or_insert(0) += 1;
            }
        }

        // All domains should be assigned
        let total: usize = assignments.values().sum();
        assert_eq!(total, domains.len());
    }

    #[test]
    fn test_node_leave_reassigns() {
        let mut assignment = DomainAssignment::new(10);

        assignment.on_node_join("peer1".to_string());
        assignment.on_node_join("peer2".to_string());

        let domain = "test.com";
        let original_peer = assignment.assign_domain(domain).unwrap();

        // Remove the peer that owns the domain
        assignment.on_node_leave(&original_peer);

        // Domain should now be assigned to remaining peer
        let new_peer = assignment.assign_domain(domain);
        assert!(new_peer.is_some());
        assert_ne!(new_peer.as_ref(), Some(&original_peer));
    }

    #[test]
    fn test_is_local_domain() {
        let mut assignment = DomainAssignment::new(10);
        assignment.set_local_peer("local_peer".to_string());
        assignment.on_node_join("local_peer".to_string());
        assignment.on_node_join("other_peer".to_string());

        // With only these two peers, some domains should be local
        let domains = vec![
            "a.com", "b.com", "c.com", "d.com", "e.com", "f.com", "g.com", "h.com",
        ];

        let local_count = domains.iter().filter(|d| assignment.is_local_domain(d)).count();

        // Should have some local domains (roughly half with 2 peers)
        assert!(local_count > 0);
        assert!(local_count < domains.len());
    }
}
