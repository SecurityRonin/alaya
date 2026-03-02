use crate::error::Result;
use crate::graph::links;
use crate::store::{categories, embeddings, implicit};
use crate::types::*;
use rusqlite::Connection;
use std::collections::HashMap;

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

/// Minimum cosine similarity for two uncategorized nodes to cluster together
const CATEGORY_CLUSTER_THRESHOLD: f32 = 0.7;

/// Minimum cluster size to form a new category
const MIN_CLUSTER_SIZE: usize = 3;

/// Cosine similarity above which two existing categories should be merged
const CATEGORY_MERGE_THRESHOLD: f32 = 0.85;

/// Categories with stability below this are dissolved
const CATEGORY_DISSOLVE_THRESHOLD: f32 = 0.1;

/// Run a transformation cycle (asraya-paravrtti).
///
/// Periodic refinement toward clarity: dedup, contradiction resolution,
/// pruning, and decay. Each cycle moves the memory store closer to the
/// "Great Mirror" state — reflecting the user accurately with minimal distortion.
pub fn transform(conn: &Connection) -> Result<TransformationReport> {
    let mut report = TransformationReport {
        duplicates_merged: dedup_semantic_nodes(conn)?,
        links_pruned: links::prune_weak_links(conn, LINK_PRUNE_THRESHOLD)? as u32,
        ..Default::default()
    };

    // 3. Decay un-reinforced preferences
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    report.preferences_decayed =
        implicit::decay_preferences(conn, now, PREFERENCE_HALF_LIFE_SECS)? as u32;

    // 4. Prune weak preferences
    report.preferences_decayed +=
        implicit::prune_weak_preferences(conn, MIN_PREFERENCE_CONFIDENCE)? as u32;

    // 5. Prune old impressions
    report.impressions_pruned =
        implicit::prune_old_impressions(conn, MAX_IMPRESSION_AGE_SECS)? as u32;

    // 6. Discover new categories from uncategorized nodes
    report.categories_discovered = discover_categories(conn)?;

    // 7. Maintain existing categories
    let (merged, dissolved) = maintain_categories(conn)?;
    report.categories_merged = merged;
    report.categories_dissolved = dissolved;

    Ok(report)
}

/// Find and merge semantic nodes with nearly identical embeddings.
fn dedup_semantic_nodes(conn: &Connection) -> Result<u32> {
    // Get all semantic node embeddings
    let mut stmt =
        conn.prepare("SELECT node_id, embedding FROM embeddings WHERE node_type = 'semantic'")?;
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

/// Discover new categories from uncategorized semantic nodes via
/// agglomerative clustering on embedding similarity.
fn discover_categories(conn: &Connection) -> Result<u32> {
    let uncategorized = categories::get_uncategorized_node_ids(conn)?;
    if uncategorized.len() < MIN_CLUSTER_SIZE {
        return Ok(0);
    }

    // Collect embeddings for uncategorized nodes
    let mut nodes_with_emb: Vec<(NodeId, Vec<f32>)> = Vec::new();
    for node_id in &uncategorized {
        if let Some(emb) = embeddings::get_embedding(conn, "semantic", node_id.0)? {
            nodes_with_emb.push((*node_id, emb));
        }
    }
    if nodes_with_emb.len() < MIN_CLUSTER_SIZE {
        return Ok(0);
    }

    // Union-Find based agglomerative clustering (single-linkage)
    let n = nodes_with_emb.len();
    let mut parent: Vec<usize> = (0..n).collect();

    // Find with path compression
    fn find(parent: &mut Vec<usize>, i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    // Union
    fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[rb] = ra;
        }
    }

    // Pairwise cosine similarity — merge pairs above threshold
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = embeddings::cosine_similarity(&nodes_with_emb[i].1, &nodes_with_emb[j].1);
            if sim >= CATEGORY_CLUSTER_THRESHOLD {
                union(&mut parent, i, j);
            }
        }
    }

    // Group nodes by cluster root
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        clusters.entry(root).or_default().push(i);
    }

    let mut categories_created = 0u32;

    for members in clusters.values() {
        if members.len() < MIN_CLUSTER_SIZE {
            continue;
        }

        // Pick prototype: node with highest corroboration_count
        let mut best_idx = members[0];
        let mut best_corr: i64 = 0;
        for &idx in members {
            let corr: i64 = conn
                .query_row(
                    "SELECT COALESCE(corroboration_count, 0) FROM semantic_nodes WHERE id = ?1",
                    [nodes_with_emb[idx].0 .0],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            if corr > best_corr {
                best_corr = corr;
                best_idx = idx;
            }
        }
        let prototype_id = nodes_with_emb[best_idx].0;

        // Compute centroid: mean of all member embeddings
        let dim = nodes_with_emb[members[0]].1.len();
        let mut centroid = vec![0.0f32; dim];
        for &idx in members {
            for (d, val) in nodes_with_emb[idx].1.iter().enumerate() {
                centroid[d] += val;
            }
        }
        let count = members.len() as f32;
        for val in &mut centroid {
            *val /= count;
        }

        // Generate placeholder label: first 3 words of prototype content
        let label: String = conn
            .query_row(
                "SELECT content FROM semantic_nodes WHERE id = ?1",
                [prototype_id.0],
                |row| row.get::<_, String>(0),
            )
            .unwrap_or_default()
            .split_whitespace()
            .take(3)
            .collect::<Vec<&str>>()
            .join(" ");
        let label = if label.is_empty() {
            format!("cluster-{categories_created}")
        } else {
            label
        };

        // Store category
        let cat_id = categories::store_category(conn, &label, prototype_id, Some(&centroid))?;

        // Assign each member and create MemberOf link
        for &idx in members {
            let member_id = nodes_with_emb[idx].0;
            categories::assign_node_to_category(conn, member_id, cat_id)?;
            links::create_link(
                conn,
                NodeRef::Semantic(member_id),
                NodeRef::Category(cat_id),
                LinkType::MemberOf,
                0.8,
            )?;
        }

        categories_created += 1;
    }

    Ok(categories_created)
}

/// Maintain existing categories: stability tracking, merge converging,
/// dissolve unstable, garbage-collect empty.
/// Returns (merged_count, dissolved_count).
fn maintain_categories(conn: &Connection) -> Result<(u32, u32)> {
    let mut merged_count = 0u32;
    let mut dissolved_count = 0u32;

    // 1. Increment stability for all non-empty categories
    let all_cats = categories::list_categories(conn, None)?;
    for cat in &all_cats {
        if cat.member_count > 0 {
            categories::increment_stability(conn, cat.id)?;
        }
    }

    // 2. Garbage-collect empty categories (member_count == 0)
    let all_cats = categories::list_categories(conn, None)?;
    for cat in &all_cats {
        if cat.member_count == 0 {
            categories::delete_category(conn, cat.id)?;
            dissolved_count += 1;
        }
    }

    // 3. Merge converging categories (cosine similarity of centroids > threshold)
    // Re-fetch after GC
    let cats = categories::list_categories(conn, None)?;
    let mut deleted: std::collections::HashSet<i64> = std::collections::HashSet::new();

    // Iterate pairs; merge lower-stability into higher-stability
    let len = cats.len();
    for i in 0..len {
        if deleted.contains(&cats[i].id.0) {
            continue;
        }
        for j in (i + 1)..len {
            if deleted.contains(&cats[j].id.0) {
                continue;
            }
            if let (Some(ref ci), Some(ref cj)) =
                (&cats[i].centroid_embedding, &cats[j].centroid_embedding)
            {
                let sim = embeddings::cosine_similarity(ci, cj);
                if sim > CATEGORY_MERGE_THRESHOLD {
                    // Keep the one with higher stability (cats are sorted by stability desc)
                    let (keep_idx, lose_idx) = if cats[i].stability >= cats[j].stability {
                        (i, j)
                    } else {
                        (j, i)
                    };
                    let keep_id = cats[keep_idx].id;
                    let lose_id = cats[lose_idx].id;

                    // Reassign all members of loser to winner
                    conn.execute(
                        "UPDATE semantic_nodes SET category_id = ?1 WHERE category_id = ?2",
                        [keep_id.0, lose_id.0],
                    )?;

                    // Update member count
                    let total: i64 = conn.query_row(
                        "SELECT COUNT(*) FROM semantic_nodes WHERE category_id = ?1",
                        [keep_id.0],
                        |row| row.get(0),
                    )?;
                    conn.execute(
                        "UPDATE categories SET member_count = ?1 WHERE id = ?2",
                        rusqlite::params![total, keep_id.0],
                    )?;

                    // Recompute centroid for merged category
                    let mut stmt = conn.prepare(
                        "SELECT e.embedding FROM embeddings e
                         INNER JOIN semantic_nodes sn ON sn.id = e.node_id AND e.node_type = 'semantic'
                         WHERE sn.category_id = ?1",
                    )?;
                    let embs: Vec<Vec<f32>> = stmt
                        .query_map([keep_id.0], |row| {
                            let blob: Vec<u8> = row.get(0)?;
                            Ok(embeddings::deserialize_embedding(&blob))
                        })?
                        .filter_map(|r| r.ok())
                        .collect();

                    if !embs.is_empty() {
                        let dim = embs[0].len();
                        let mut new_centroid = vec![0.0f32; dim];
                        for emb in &embs {
                            for (d, val) in emb.iter().enumerate() {
                                new_centroid[d] += val;
                            }
                        }
                        let c = embs.len() as f32;
                        for val in &mut new_centroid {
                            *val /= c;
                        }
                        categories::update_centroid(conn, keep_id, &new_centroid)?;
                    }

                    // Update MemberOf links from loser to winner
                    conn.execute(
                        "UPDATE links SET target_id = ?1 WHERE target_type = 'category' AND target_id = ?2 AND link_type = 'member_of'",
                        [keep_id.0, lose_id.0],
                    )?;

                    // Delete the loser category
                    conn.execute("DELETE FROM categories WHERE id = ?1", [lose_id.0])?;
                    deleted.insert(lose_id.0);
                    merged_count += 1;
                }
            }
        }
    }

    // 4. Dissolve unstable categories (stability < threshold)
    // Re-fetch after merges
    let cats = categories::list_categories(conn, None)?;
    for cat in &cats {
        if deleted.contains(&cat.id.0) {
            continue;
        }
        if cat.stability < CATEGORY_DISSOLVE_THRESHOLD && cat.stability > 0.0 {
            // Only dissolve if it has had a chance to stabilize (stability > 0 means
            // it survived at least one cycle)
            categories::delete_category(conn, cat.id)?;
            dissolved_count += 1;
        }
    }

    Ok((merged_count, dissolved_count))
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
        )
        .unwrap();

        let report = transform(&conn).unwrap();
        assert_eq!(report.links_pruned, 1);
    }

    #[test]
    fn test_transform_discovers_categories() {
        let conn = open_memory_db().unwrap();

        // Embeddings: cosine sim between 0.7 and 0.95 (cluster but don't dedup)
        let test_embs: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.8, 0.5, 0.0, 0.0],
            vec![0.7, 0.3, 0.5, 0.0],
            vec![0.9, 0.2, 0.1, 0.3],
        ];

        // Create 4 semantic nodes with similar embeddings (uncategorized)
        for (i, emb) in test_embs.iter().enumerate() {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                [format!("cooking recipe {i}")],
            ).unwrap();
            let node_id: i64 = conn
                .query_row("SELECT last_insert_rowid()", [], |r| r.get(0))
                .unwrap();
            embeddings::store_embedding(&conn, "semantic", node_id, emb, "").unwrap();
        }

        let report = transform(&conn).unwrap();
        assert!(
            report.categories_discovered >= 1,
            "should discover at least 1 category from 4 similar nodes, got {}",
            report.categories_discovered
        );

        let cats = categories::list_categories(&conn, None).unwrap();
        assert!(!cats.is_empty(), "should have created categories");
        // Verify the category has members
        assert!(
            cats[0].member_count >= 3,
            "category should have at least 3 members, got {}",
            cats[0].member_count
        );
    }

    #[test]
    fn test_transform_no_categories_with_few_nodes() {
        let conn = open_memory_db().unwrap();

        // Only 2 semantic nodes — below cluster minimum of 3
        for i in 0..2 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated)
                 VALUES (?1, 'fact', 0.8, 1000, 1000)",
                [format!("node {i}")],
            ).unwrap();
            let node_id: i64 = conn
                .query_row("SELECT last_insert_rowid()", [], |r| r.get(0))
                .unwrap();
            embeddings::store_embedding(&conn, "semantic", node_id, &[0.9, 0.1, 0.0], "").unwrap();
        }

        let report = transform(&conn).unwrap();
        assert_eq!(report.categories_discovered, 0);
    }

    #[test]
    fn test_transform_gc_empty_categories() {
        let conn = open_memory_db().unwrap();

        // Create a dummy semantic node so store_category has a valid prototype
        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated)
             VALUES ('dummy', 'fact', 0.5, 1000, 1000)",
            [],
        ).unwrap();

        // Create a category with 0 members
        categories::store_category(&conn, "empty-cat", NodeId(1), None).unwrap();
        assert_eq!(categories::count_categories(&conn).unwrap(), 1);

        let report = transform(&conn).unwrap();
        // Empty category should be garbage-collected
        assert_eq!(
            categories::count_categories(&conn).unwrap(),
            0,
            "empty category should have been garbage-collected"
        );
        assert!(
            report.categories_dissolved >= 1,
            "should report at least 1 dissolved category"
        );
    }

    #[test]
    fn test_discover_categories_creates_member_of_links() {
        let conn = open_memory_db().unwrap();

        // Create 3 nodes with identical embeddings
        for i in 0..3 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                [format!("topic alpha {i}")],
            ).unwrap();
            let node_id: i64 = conn
                .query_row("SELECT last_insert_rowid()", [], |r| r.get(0))
                .unwrap();
            embeddings::store_embedding(&conn, "semantic", node_id, &[1.0, 0.0, 0.0], "").unwrap();
        }

        let created = discover_categories(&conn).unwrap();
        assert_eq!(created, 1);

        // Should have MemberOf links
        let link_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM links WHERE link_type = 'member_of'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(link_count, 3, "should have 3 MemberOf links");
    }

    #[test]
    fn test_maintain_categories_merges_converging() {
        let conn = open_memory_db().unwrap();

        // Create two categories with very similar centroids
        for i in 0..2 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated)
                 VALUES (?1, 'fact', 0.8, 1000, 1000)",
                [format!("merge-node {i}")],
            ).unwrap();
        }

        let c1 =
            categories::store_category(&conn, "cat-a", NodeId(1), Some(&[1.0, 0.0, 0.0])).unwrap();
        let c2 = categories::store_category(&conn, "cat-b", NodeId(2), Some(&[0.99, 0.01, 0.0]))
            .unwrap();

        // Assign one member to each so they're non-empty and don't get GC'd
        categories::assign_node_to_category(&conn, NodeId(1), c1).unwrap();
        categories::assign_node_to_category(&conn, NodeId(2), c2).unwrap();

        // Store embeddings for the member nodes so centroid recompute works
        embeddings::store_embedding(&conn, "semantic", 1, &[1.0, 0.0, 0.0], "").unwrap();
        embeddings::store_embedding(&conn, "semantic", 2, &[0.99, 0.01, 0.0], "").unwrap();

        let (merged, _dissolved) = maintain_categories(&conn).unwrap();
        assert!(
            merged >= 1,
            "should have merged converging categories, got {merged}",
        );
        assert_eq!(
            categories::count_categories(&conn).unwrap(),
            1,
            "should have 1 category after merge"
        );
    }
}
