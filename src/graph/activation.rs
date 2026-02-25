use std::collections::HashMap;
use rusqlite::Connection;
use crate::error::Result;
use crate::types::*;
use crate::graph::links;

/// Spread activation from seed nodes through the graph.
///
/// Models the Collins & Loftus (1975) spreading activation theory:
/// activation propagates from seed nodes through weighted edges,
/// decaying at each hop, and splitting proportionally at branching points.
pub fn spread_activation(
    conn: &Connection,
    seeds: &[NodeRef],
    max_depth: u32,
    threshold: f32,
    decay_per_hop: f32,
) -> Result<HashMap<NodeRef, f32>> {
    let mut activation: HashMap<NodeRef, f32> = HashMap::new();

    // Seed initial activation
    for seed in seeds {
        *activation.entry(*seed).or_default() += 1.0;
    }

    // Spread for max_depth hops
    for _ in 0..max_depth {
        let mut delta: HashMap<NodeRef, f32> = HashMap::new();

        for (node, &act) in &activation {
            if act < threshold {
                continue;
            }

            let outgoing = links::get_links_from(conn, *node)?;
            if outgoing.is_empty() {
                continue;
            }

            let total_weight: f32 = outgoing.iter().map(|l| l.forward_weight).sum();
            if total_weight <= 0.0 {
                continue;
            }

            for link in &outgoing {
                // Use absolute weight (not proportion) so weak links carry weak signal
                // regardless of how many other links exist. This matches neuroscience:
                // synaptic strength is absolute, not relative to other synapses.
                let spread = act * link.forward_weight * decay_per_hop;
                if spread >= threshold * 0.1 {
                    *delta.entry(link.target).or_default() += spread;
                }
            }
        }

        // Merge deltas
        for (node, extra) in delta {
            let entry = activation.entry(node).or_default();
            *entry = (*entry + extra).min(2.0); // Cap to prevent runaway
        }
    }

    // Filter below threshold
    activation.retain(|_, v| *v >= threshold);
    Ok(activation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;
    use crate::graph::links::create_link;

    #[test]
    fn test_single_hop_spread() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));
        let c = NodeRef::Episode(EpisodeId(3));

        create_link(&conn, a, b, LinkType::Topical, 0.8).unwrap();
        create_link(&conn, a, c, LinkType::Topical, 0.2).unwrap();

        let result = spread_activation(&conn, &[a], 1, 0.05, 0.7).unwrap();

        assert!(result.contains_key(&a));
        assert!(result.contains_key(&b));
        // b should have more activation than c (higher weight)
        assert!(result.get(&b).unwrap_or(&0.0) > result.get(&c).unwrap_or(&0.0));
    }

    #[test]
    fn test_multi_hop_decay() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));
        let c = NodeRef::Episode(EpisodeId(3));

        create_link(&conn, a, b, LinkType::Temporal, 0.9).unwrap();
        create_link(&conn, b, c, LinkType::Temporal, 0.9).unwrap();

        let result = spread_activation(&conn, &[a], 2, 0.05, 0.6).unwrap();

        let act_b = result.get(&b).unwrap_or(&0.0);
        let act_c = result.get(&c).unwrap_or(&0.0);

        // Activation decays with graph distance: b (1 hop) > c (2 hops)
        // Note: b can exceed a's activation because it receives spread
        // from a over multiple iterations. This is correct behavior.
        assert!(act_b > act_c, "b ({act_b}) should be > c ({act_c})");
        assert!(*act_c > 0.0, "c should have nonzero activation from 2-hop spread");
    }

    #[test]
    fn test_threshold_cutoff() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));

        create_link(&conn, a, b, LinkType::Topical, 0.01).unwrap();

        // With a high threshold, b should not appear
        let result = spread_activation(&conn, &[a], 1, 0.5, 0.6).unwrap();
        assert!(!result.contains_key(&b));
    }
}
