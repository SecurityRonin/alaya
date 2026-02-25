use rusqlite::Connection;
use crate::error::Result;
use crate::store::{implicit, embeddings};
use crate::graph::links;
use crate::types::*;

/// Default max age for impressions: 90 days in seconds
const MAX_IMPRESSION_AGE_SECS: i64 = 90 * 24 * 3600;

/// Default preference decay half-life: 30 days in seconds
const PREFERENCE_HALF_LIFE_SECS: i64 = 30 * 24 * 3600;

/// Default link pruning threshold
const LINK_PRUNE_THRESHOLD: f32 = 0.02;

/// Default minimum preference confidence
const MIN_PREFERENCE_CONFIDENCE: f32 = 0.05;

/// Default similarity threshold for duplicate detection
const DEDUP_SIMILARITY_THRESHOLD: f32 = 0.95;

/// Run a transformation cycle (asraya-paravrtti).
///
/// Periodic refinement toward clarity: dedup, contradiction resolution,
/// pruning, and decay. Each cycle moves the memory store closer to the
/// "Great Mirror" state — reflecting the user accurately with minimal distortion.
pub fn transform(conn: &Connection) -> Result<TransformationReport> {
    let mut report = TransformationReport::default();

    // 1. Deduplicate semantic nodes with near-identical embeddings
    report.duplicates_merged = dedup_semantic_nodes(conn)?;

    // 2. Prune weak graph links
    report.links_pruned = links::prune_weak_links(conn, LINK_PRUNE_THRESHOLD)? as u32;

    // 3. Decay un-reinforced preferences
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    report.preferences_decayed = implicit::decay_preferences(conn, now, PREFERENCE_HALF_LIFE_SECS)? as u32;

    // 4. Prune weak preferences
    report.preferences_decayed += implicit::prune_weak_preferences(conn, MIN_PREFERENCE_CONFIDENCE)? as u32;

    // 5. Prune old impressions
    report.impressions_pruned = implicit::prune_old_impressions(conn, MAX_IMPRESSION_AGE_SECS)? as u32;

    Ok(report)
}

/// Find and merge semantic nodes with nearly identical embeddings.
fn dedup_semantic_nodes(conn: &Connection) -> Result<u32> {
    // Get all semantic node embeddings
    let mut stmt = conn.prepare(
        "SELECT node_id, embedding FROM embeddings WHERE node_type = 'semantic'"
    )?;
    let nodes: Vec<(i64, Vec<f32>)> = stmt
        .query_map([], |row| {
            let id: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok((id, embeddings::deserialize_embedding(&blob)))
        })?
        .filter_map(|r| r.ok())
        .collect();

    let mut merged = 0u32;
    let mut deleted_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();

    for i in 0..nodes.len() {
        if deleted_ids.contains(&nodes[i].0) {
            continue;
        }
        for j in (i + 1)..nodes.len() {
            if deleted_ids.contains(&nodes[j].0) {
                continue;
            }
            let sim = embeddings::cosine_similarity(&nodes[i].1, &nodes[j].1);
            if sim >= DEDUP_SIMILARITY_THRESHOLD {
                // Keep the first (older), delete the second
                // Transfer any unique links from j to i
                conn.execute(
                    "UPDATE links SET source_id = ?1 WHERE source_type = 'semantic' AND source_id = ?2",
                    [nodes[i].0, nodes[j].0],
                )?;
                conn.execute(
                    "UPDATE links SET target_id = ?1 WHERE target_type = 'semantic' AND target_id = ?2",
                    [nodes[i].0, nodes[j].0],
                )?;
                // Increment corroboration of the kept node
                conn.execute(
                    "UPDATE semantic_nodes SET corroboration_count = corroboration_count + 1 WHERE id = ?1",
                    [nodes[i].0],
                )?;
                // Delete the duplicate
                crate::store::semantic::delete_node(conn, NodeId(nodes[j].0))?;
                deleted_ids.insert(nodes[j].0);
                merged += 1;
            }
        }
    }

    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;

    #[test]
    fn test_transform_empty_db() {
        let conn = open_memory_db().unwrap();
        let report = transform(&conn).unwrap();
        assert_eq!(report.duplicates_merged, 0);
        assert_eq!(report.links_pruned, 0);
    }

    #[test]
    fn test_transform_prunes_weak_links() {
        let conn = open_memory_db().unwrap();
        // Create a weak link
        links::create_link(
            &conn,
            NodeRef::Episode(EpisodeId(1)),
            NodeRef::Episode(EpisodeId(2)),
            LinkType::Temporal,
            0.01,
        ).unwrap();

        let report = transform(&conn).unwrap();
        assert_eq!(report.links_pruned, 1);
    }
}
