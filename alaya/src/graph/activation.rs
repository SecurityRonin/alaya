use crate::error::Result;
use crate::graph::links;
use crate::types::*;
use rusqlite::Connection;
use std::collections::HashMap;

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
    use crate::graph::links::create_link;
    use crate::schema::open_memory_db;

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
        assert!(
            *act_c > 0.0,
            "c should have nonzero activation from 2-hop spread"
        );
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

    #[test]
    fn test_spread_activation_zero_weight_links() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));

        // Create a link with weight 0.0 — total_weight will be 0.0
        create_link(&conn, a, b, LinkType::Topical, 0.0).unwrap();
        // Manually set forward_weight to 0.0 since create_link may enforce minimum
        conn.execute(
            "UPDATE links SET forward_weight = 0.0, backward_weight = 0.0",
            [],
        )
        .unwrap();

        let result = spread_activation(&conn, &[a], 1, 0.05, 0.6).unwrap();
        // b should NOT receive activation because total_weight is 0
        assert!(
            !result.contains_key(&b),
            "zero-weight link should not spread activation"
        );
    }

    #[test]
    fn test_spread_activation_no_outgoing_links() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));

        // No links from a — the outgoing.is_empty() branch
        let result = spread_activation(&conn, &[a], 1, 0.05, 0.6).unwrap();
        assert!(
            result.contains_key(&a),
            "seed should still be in activation"
        );
        assert_eq!(result.len(), 1, "should only contain the seed");
    }

    #[test]
    fn test_spread_activation_empty_seeds() {
        let conn = open_memory_db().unwrap();
        let result = spread_activation(&conn, &[], 2, 0.05, 0.6).unwrap();
        assert!(
            result.is_empty(),
            "empty seeds should produce empty activation"
        );
    }

    #[test]
    fn test_spread_activation_multiple_seeds() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));

        // Both a and b are seeds — each gets initial activation of 1.0
        let result = spread_activation(&conn, &[a, b], 1, 0.05, 0.6).unwrap();
        assert!(result.contains_key(&a), "seed a should be in activation");
        assert!(result.contains_key(&b), "seed b should be in activation");
        assert_eq!(
            *result.get(&a).unwrap(),
            1.0,
            "seed a should start with activation 1.0"
        );
        assert_eq!(
            *result.get(&b).unwrap(),
            1.0,
            "seed b should start with activation 1.0"
        );
    }

    #[test]
    fn test_spread_activation_duplicate_seeds_sum() {
        // Two seeds pointing to the same node — activation should be 2.0
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));

        // Passing the same seed twice sums activation
        let result = spread_activation(&conn, &[a, a], 0, 0.05, 0.6).unwrap();
        // With max_depth=0 no spreading occurs, but both seeds add to a
        assert!(result.contains_key(&a));
        // Activation was 2.0 before cap; cap is 2.0, so value stays 2.0
        assert!(
            *result.get(&a).unwrap() <= 2.0,
            "activation should be capped at 2.0"
        );
    }

    #[test]
    fn test_spread_activation_below_threshold_not_included() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));

        // Create a very weak link
        create_link(&conn, a, b, LinkType::Topical, 0.001).unwrap();

        // With high threshold, b should NOT appear even if linked
        let result = spread_activation(&conn, &[a], 2, 0.5, 0.5).unwrap();
        // a has activation 1.0 which is >= threshold 0.5 → included
        assert!(result.contains_key(&a));
        // b receives 1.0 * 0.001 * 0.5 = 0.0005 which is way below threshold*0.1 = 0.05
        // so spread doesn't propagate, and b is filtered out
        assert!(
            !result.contains_key(&b),
            "very weak link should not spread above threshold"
        );
    }

    #[test]
    fn test_below_threshold_node_skipped_in_spread() {
        // Covers line 32: continue when act < threshold during iteration
        // Node c gets very low activation from hop 1, then on hop 2
        // its activation is below threshold so it is skipped.
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));
        let c = NodeRef::Episode(EpisodeId(3));
        let d = NodeRef::Episode(EpisodeId(4));

        // a -> b (strong), a -> c (very weak), c -> d (strong)
        create_link(&conn, a, b, LinkType::Topical, 0.9).unwrap();
        create_link(&conn, a, c, LinkType::Topical, 0.01).unwrap();
        // Manually make the link extremely weak
        conn.execute(
            "UPDATE links SET forward_weight = 0.001 WHERE target_id = 3",
            [],
        )
        .unwrap();
        create_link(&conn, c, d, LinkType::Topical, 0.9).unwrap();

        // With threshold 0.1 and 2 hops, c gets ~0.001*0.5 = 0.0005 activation.
        // On hop 2, c's activation (0.0005) < threshold (0.1), so it skips spreading to d.
        let result = spread_activation(&conn, &[a], 2, 0.1, 0.5).unwrap();

        // d should NOT be reached because c was below threshold
        assert!(
            !result.contains_key(&d),
            "d should not be reached through below-threshold node c"
        );
    }

    #[test]
    fn test_spread_activation_zero_depth() {
        // max_depth=0 means no spreading at all — only seeds returned
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));
        create_link(&conn, a, b, LinkType::Topical, 0.9).unwrap();

        let result = spread_activation(&conn, &[a], 0, 0.05, 0.6).unwrap();
        assert!(result.contains_key(&a));
        assert!(
            !result.contains_key(&b),
            "zero depth: b should not receive activation"
        );
    }
}
