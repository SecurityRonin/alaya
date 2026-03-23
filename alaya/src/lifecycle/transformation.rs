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

/// Minimum member count before considering a split
const SPLIT_MIN_MEMBERS: usize = 8;

/// Average cosine similarity to centroid below which a category should split
const SPLIT_COHERENCE_THRESHOLD: f32 = 0.6;

/// Run a transformation cycle (asraya-paravrtti).
///
/// Periodic refinement toward clarity: dedup, contradiction resolution,
/// pruning, and decay. Each cycle moves the memory store closer to the
/// "Great Mirror" state — reflecting the user accurately with minimal distortion.
pub fn transform(conn: &Connection) -> Result<TransformationReport> {
    let mut report = TransformationReport {
        duplicates_merged: dedup_semantic_nodes(conn)?,
        links_decayed: links::decay_links(conn, 0.95)? as u32,
        links_pruned: links::prune_weak_links(conn, LINK_PRUNE_THRESHOLD)? as u32,
        ..Default::default()
    };

    // 3. Decay un-reinforced preferences
    let now = crate::db::now();
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

    // 8. Split large incoherent categories
    report.categories_split = split_large_categories(conn)?;

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
                // Transfer any unique links from j to i (ignore duplicates)
                conn.execute(
                    "UPDATE OR IGNORE links SET source_id = ?1 WHERE source_type = 'semantic' AND source_id = ?2",
                    [nodes[i].0, nodes[j].0],
                )?;
                conn.execute(
                    "UPDATE OR IGNORE links SET target_id = ?1 WHERE target_type = 'semantic' AND target_id = ?2",
                    [nodes[i].0, nodes[j].0],
                )?;
                // Clean up any orphaned links that couldn't be transferred due to duplicates
                conn.execute(
                    "DELETE FROM links WHERE source_type = 'semantic' AND source_id = ?1",
                    [nodes[j].0],
                )?;
                conn.execute(
                    "DELETE FROM links WHERE target_type = 'semantic' AND target_id = ?1",
                    [nodes[j].0],
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
        let cat_id = categories::store_category(conn, &label, prototype_id, Some(&centroid), None)?;

        // Assign each member and create MemberOf link
        for &idx in members {
            let member_id = nodes_with_emb[idx].0;
            categories::assign_node_to_category(conn, member_id, cat_id)?;
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
            if !deleted.contains(&cats[j].id.0) {
                if let (Some(ref ci), Some(ref cj)) =
                    (&cats[i].centroid_embedding, &cats[j].centroid_embedding)
                {
                    let sim = embeddings::cosine_similarity(ci, cj);
                    if sim > CATEGORY_MERGE_THRESHOLD {
                        // Keep i (higher stability — cats sorted by stability DESC)
                        let keep_id = cats[i].id;
                        let lose_id = cats[j].id;

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

                        // Update MemberOf links from loser to winner (ignore duplicates)
                        conn.execute(
                            "UPDATE OR IGNORE links SET target_id = ?1 WHERE target_type = 'category' AND target_id = ?2 AND link_type = 'member_of'",
                            [keep_id.0, lose_id.0],
                        )?;
                        conn.execute(
                            "UPDATE OR IGNORE links SET source_id = ?1 WHERE source_type = 'category' AND source_id = ?2 AND link_type = 'member_of'",
                            [keep_id.0, lose_id.0],
                        )?;
                        // Clean up any orphaned links that couldn't be transferred
                        conn.execute(
                            "DELETE FROM links WHERE link_type = 'member_of' AND ((target_type = 'category' AND target_id = ?1) OR (source_type = 'category' AND source_id = ?1))",
                            [lose_id.0],
                        )?;

                        // Delete the loser category
                        conn.execute("DELETE FROM categories WHERE id = ?1", [lose_id.0])?;
                        deleted.insert(lose_id.0);
                        merged_count += 1;
                    }
                }
            }
        }
    }

    // 4. Dissolve unstable categories (stability < threshold)
    // Re-fetch after merges
    let cats = categories::list_categories(conn, None)?;
    for cat in &cats {
        // Only dissolve if it has had a chance to stabilize (stability > 0 means
        // it survived at least one cycle) and wasn't already deleted by merge
        if !deleted.contains(&cat.id.0)
            && cat.stability < CATEGORY_DISSOLVE_THRESHOLD
            && cat.stability > 0.0
        {
            categories::delete_category(conn, cat.id)?;
            dissolved_count += 1;
        }
    }

    Ok((merged_count, dissolved_count))
}

/// Split categories that are too large and have low internal coherence.
/// Creates sub-categories under the original parent.
fn split_large_categories(conn: &Connection) -> Result<u32> {
    let all_cats = categories::list_categories(conn, None)?;
    let mut splits = 0u32;

    for cat in &all_cats {
        if (cat.member_count as usize) < SPLIT_MIN_MEMBERS {
            continue;
        }

        // Need centroid to compute coherence
        let centroid = match &cat.centroid_embedding {
            Some(c) => c,
            None => continue,
        };

        // Get member node IDs
        let member_ids: Vec<NodeId> = conn
            .prepare("SELECT id FROM semantic_nodes WHERE category_id = ?1")?
            .query_map([cat.id.0], |row| Ok(NodeId(row.get(0)?)))?
            .filter_map(|r| r.ok())
            .collect();

        // Get embeddings for members
        let mut members_with_emb: Vec<(NodeId, Vec<f32>)> = Vec::new();
        for nid in &member_ids {
            if let Some(emb) = embeddings::get_embedding(conn, "semantic", nid.0)? {
                members_with_emb.push((*nid, emb));
            }
        }

        if members_with_emb.len() < SPLIT_MIN_MEMBERS {
            continue;
        }

        // Compute average cosine similarity to centroid
        let total_sim: f32 = members_with_emb
            .iter()
            .map(|(_, emb)| embeddings::cosine_similarity(emb, centroid))
            .sum();
        let coherence = total_sim / members_with_emb.len() as f32;

        if coherence >= SPLIT_COHERENCE_THRESHOLD {
            continue; // Still coherent, don't split
        }

        // Re-cluster members using same union-find algorithm as discover_categories
        let n = members_with_emb.len();
        let mut parent_uf: Vec<usize> = (0..n).collect();

        fn find(parent: &mut Vec<usize>, i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }
        fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[rb] = ra;
            }
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let sim =
                    embeddings::cosine_similarity(&members_with_emb[i].1, &members_with_emb[j].1);
                if sim >= CATEGORY_CLUSTER_THRESHOLD {
                    union(&mut parent_uf, i, j);
                }
            }
        }

        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent_uf, i);
            clusters.entry(root).or_default().push(i);
        }

        // Only split if we get 2+ clusters of MIN_CLUSTER_SIZE
        let valid_clusters: Vec<&Vec<usize>> = clusters
            .values()
            .filter(|c| c.len() >= MIN_CLUSTER_SIZE)
            .collect();

        if valid_clusters.len() < 2 {
            continue; // No meaningful sub-clusters found
        }

        // Create sub-categories under this parent
        for cluster in valid_clusters {
            let proto_idx = cluster[0];
            let proto_node = members_with_emb[proto_idx].0;

            // Label from prototype content (first 40 chars)
            let label_content: String = conn
                .query_row(
                    "SELECT content FROM semantic_nodes WHERE id = ?1",
                    [proto_node.0],
                    |row| row.get(0),
                )
                .unwrap_or_else(|_| "sub-category".to_string());
            let sub_label = if label_content.len() > 40 {
                &label_content[..40]
            } else {
                &label_content
            };

            // Compute sub-cluster centroid
            let dim = members_with_emb[0].1.len();
            let mut sub_centroid = vec![0.0f32; dim];
            for &idx in cluster {
                for (d, val) in members_with_emb[idx].1.iter().enumerate() {
                    sub_centroid[d] += val;
                }
            }
            for val in sub_centroid.iter_mut().take(dim) {
                *val /= cluster.len() as f32;
            }

            let sub_id = categories::store_category(
                conn,
                sub_label,
                proto_node,
                Some(&sub_centroid),
                Some(cat.id),
            )?;

            // Reassign members to sub-category
            for &idx in cluster {
                let nid = members_with_emb[idx].0;
                categories::assign_node_to_category(conn, nid, sub_id)?;
            }
        }

        splits += 1;
    }

    Ok(splits)
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
    fn test_transform_decays_link_weights() {
        let conn = open_memory_db().unwrap();
        // Create a link with moderate weight
        links::create_link(
            &conn,
            NodeRef::Episode(EpisodeId(1)),
            NodeRef::Episode(EpisodeId(2)),
            LinkType::CoRetrieval,
            0.5,
        )
        .unwrap();

        let report = transform(&conn).unwrap();
        assert!(report.links_decayed > 0, "should report decayed links");

        // Verify the weight actually decreased
        let remaining = links::get_links_from(&conn, NodeRef::Episode(EpisodeId(1))).unwrap();
        assert_eq!(remaining.len(), 1);
        assert!(
            remaining[0].forward_weight < 0.5,
            "weight should have decreased from 0.5, got {}",
            remaining[0].forward_weight
        );
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
        categories::store_category(&conn, "empty-cat", NodeId(1), None, None).unwrap();
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
        assert_eq!(
            link_count, 6,
            "should have 6 bidirectional MemberOf links (2 per node)"
        );
    }

    #[test]
    fn test_split_triggers_when_large_and_incoherent() {
        let conn = open_memory_db().unwrap();

        // Create 10 semantic nodes in 2 distinct sub-clusters using 16D embeddings.
        // Sub-cluster A (indices 0-4): live in dims 0-7
        // Sub-cluster B (indices 5-9): live in dims 8-15 (orthogonal to A)
        // Within each cluster: cosine sim ~0.8 (above 0.7 cluster threshold, below 0.95 dedup)
        // Cross-cluster: cosine sim = 0 (orthogonal subspaces)
        let test_embs: Vec<Vec<f32>> = vec![
            vec![
                0.45, 0.45, 0.45, 0.45, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.45, 0.45, 0.45, 0.45, 0.0, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.45, 0.45, 0.45, 0.45, 0.0, 0.0, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.45, 0.45, 0.45, 0.45, 0.0, 0.0, 0.0, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.45, 0.45, 0.45, 0.45, 0.31, 0.31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45, 0.45, 0.45, 0.45, 0.45, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45, 0.45, 0.45, 0.45, 0.0, 0.45, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45, 0.45, 0.45, 0.45, 0.0, 0.0, 0.45, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45, 0.45, 0.45, 0.45, 0.0, 0.0, 0.0, 0.45,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45, 0.45, 0.45, 0.45, 0.31, 0.31, 0.0,
                0.0,
            ],
        ];
        let mut node_ids = Vec::new();
        for (i, emb) in test_embs.iter().enumerate() {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                rusqlite::params![format!("split node {i}")],
            )
            .unwrap();
            let nid = NodeId(conn.last_insert_rowid());
            embeddings::store_embedding(&conn, "semantic", nid.0, emb, "").unwrap();
            node_ids.push(nid);
        }

        // Create a parent category containing all 10
        let cat_id = categories::store_category(&conn, "broad", node_ids[0], None, None).unwrap();
        // Set centroid orthogonal to both clusters — forces low coherence
        let centroid = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.7,
        ];
        categories::update_centroid(&conn, cat_id, &centroid).unwrap();
        for &nid in &node_ids {
            categories::assign_node_to_category(&conn, nid, cat_id).unwrap();
        }

        let report = transform(&conn).unwrap();
        assert!(
            report.categories_split > 0,
            "should have split the broad category"
        );

        // Should have sub-categories with parent_id = cat_id
        let subs = categories::get_subcategories(&conn, cat_id).unwrap();
        assert!(
            subs.len() >= 2,
            "should have at least 2 sub-categories, got {}",
            subs.len()
        );
    }

    #[test]
    fn test_no_split_when_coherent() {
        let conn = open_memory_db().unwrap();

        // 8 nodes with similar embeddings — coherent cluster (varied enough to avoid dedup)
        // All point roughly in the same direction but with enough variation
        let coherent_embs: Vec<Vec<f32>> = vec![
            vec![0.9, 0.3, 0.1, 0.0],
            vec![0.85, 0.35, 0.15, 0.0],
            vec![0.88, 0.28, 0.12, 0.05],
            vec![0.92, 0.25, 0.08, 0.0],
            vec![0.87, 0.32, 0.14, 0.02],
            vec![0.91, 0.27, 0.11, 0.03],
            vec![0.86, 0.34, 0.13, 0.01],
            vec![0.89, 0.30, 0.10, 0.04],
        ];
        let mut node_ids = Vec::new();
        for (i, emb) in coherent_embs.iter().enumerate() {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                rusqlite::params![format!("coherent node {i}")],
            )
            .unwrap();
            let nid = NodeId(conn.last_insert_rowid());
            embeddings::store_embedding(&conn, "semantic", nid.0, emb, "").unwrap();
            node_ids.push(nid);
        }

        let cat_id = categories::store_category(&conn, "tight", node_ids[0], None, None).unwrap();
        let centroid = vec![0.885, 0.305, 0.116, 0.019]; // mean of the 8 embeddings
        categories::update_centroid(&conn, cat_id, &centroid).unwrap();
        for &nid in &node_ids {
            categories::assign_node_to_category(&conn, nid, cat_id).unwrap();
        }

        let report = transform(&conn).unwrap();
        assert_eq!(
            report.categories_split, 0,
            "should NOT split coherent category"
        );
    }

    #[test]
    fn test_no_split_when_small() {
        let conn = open_memory_db().unwrap();

        // Only 5 nodes — below SPLIT_MIN_MEMBERS of 8
        let mut node_ids = Vec::new();
        for i in 0..5 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                rusqlite::params![format!("small node {}", i)],
            )
            .unwrap();
            let nid = NodeId(conn.last_insert_rowid());
            let emb = if i < 3 {
                vec![1.0, 0.0, 0.0, 0.0]
            } else {
                vec![0.0, 0.0, 1.0, 0.0]
            };
            embeddings::store_embedding(&conn, "semantic", nid.0, &emb, "").unwrap();
            node_ids.push(nid);
        }

        let cat_id = categories::store_category(&conn, "small", node_ids[0], None, None).unwrap();
        let centroid = vec![0.5, 0.0, 0.5, 0.0];
        categories::update_centroid(&conn, cat_id, &centroid).unwrap();
        for &nid in &node_ids {
            categories::assign_node_to_category(&conn, nid, cat_id).unwrap();
        }

        let report = transform(&conn).unwrap();
        assert_eq!(
            report.categories_split, 0,
            "should NOT split small category"
        );
    }

    #[test]
    fn test_dedup_merges_near_identical_nodes() {
        let conn = open_memory_db().unwrap();

        // Create two semantic nodes with nearly identical embeddings (sim > 0.95)
        for i in 0..2 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                [format!("duplicate node {i}")],
            )
            .unwrap();
            let node_id: i64 = conn
                .query_row("SELECT last_insert_rowid()", [], |r| r.get(0))
                .unwrap();
            // Both have nearly identical embeddings
            embeddings::store_embedding(&conn, "semantic", node_id, &[1.0, 0.0, 0.0], "").unwrap();
        }

        let merged = dedup_semantic_nodes(&conn).unwrap();
        assert_eq!(merged, 1, "should have merged the duplicate");

        // Verify only 1 semantic node remains
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM semantic_nodes", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1, "should have 1 node after dedup");

        // Verify corroboration_count was incremented on the kept node
        let corr: i64 = conn
            .query_row(
                "SELECT corroboration_count FROM semantic_nodes WHERE id = 1",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(corr, 2, "kept node should have corroboration_count = 2");
    }

    #[test]
    fn test_dedup_skips_already_deleted() {
        let conn = open_memory_db().unwrap();

        // Create 3 semantic nodes with identical embeddings — tests the
        // deleted_ids.contains check in the inner dedup loop
        for i in 0..3 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                [format!("triple dup {i}")],
            )
            .unwrap();
            let node_id: i64 = conn
                .query_row("SELECT last_insert_rowid()", [], |r| r.get(0))
                .unwrap();
            embeddings::store_embedding(&conn, "semantic", node_id, &[1.0, 0.0, 0.0], "").unwrap();
        }

        let merged = dedup_semantic_nodes(&conn).unwrap();
        // Node 2 and 3 are both duplicates of node 1
        // After node 2 is merged into 1, node 3 should also be merged into 1 (not skipped)
        assert_eq!(merged, 2, "should have merged both duplicates");

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM semantic_nodes", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            count, 1,
            "should have 1 node after dedup of 3 identical nodes"
        );
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
            categories::store_category(&conn, "cat-a", NodeId(1), Some(&[1.0, 0.0, 0.0]), None)
                .unwrap();
        let c2 =
            categories::store_category(&conn, "cat-b", NodeId(2), Some(&[0.99, 0.01, 0.0]), None)
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

    #[test]
    fn test_discover_categories_few_embeddings() {
        // Covers line 162: return Ok(0) when uncategorized nodes exist
        // but fewer than MIN_CLUSTER_SIZE (3) have embeddings
        let conn = open_memory_db().unwrap();

        // Create 5 uncategorized nodes but only 2 with embeddings
        for i in 0..5 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                [format!("sparse embed node {i}")],
            ).unwrap();
        }
        // Only give embeddings to 2 nodes (below MIN_CLUSTER_SIZE = 3)
        embeddings::store_embedding(&conn, "semantic", 1, &[1.0, 0.0, 0.0], "").unwrap();
        embeddings::store_embedding(&conn, "semantic", 2, &[0.9, 0.1, 0.0], "").unwrap();

        let discovered = discover_categories(&conn).unwrap();
        assert_eq!(
            discovered, 0,
            "should not discover categories with < 3 embedded nodes"
        );
    }

    #[test]
    fn test_discover_categories_small_cluster_skipped() {
        // Covers line 207: continue when cluster has < MIN_CLUSTER_SIZE members
        // Create nodes where some cluster together and some don't
        let conn = open_memory_db().unwrap();

        // 3 nodes in one tight cluster (will form a category)
        for i in 0..3 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                [format!("cluster node {i}")],
            ).unwrap();
            let nid = conn.last_insert_rowid();
            embeddings::store_embedding(
                &conn,
                "semantic",
                nid,
                &[1.0, 0.0 + (i as f32) * 0.01, 0.0],
                "",
            )
            .unwrap();
        }

        // 2 outlier nodes far away (won't cluster with above or each other → too small)
        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
             VALUES ('outlier a', 'fact', 0.8, 1000, 1000, 1)",
            [],
        ).unwrap();
        embeddings::store_embedding(
            &conn,
            "semantic",
            conn.last_insert_rowid(),
            &[0.0, 1.0, 0.0],
            "",
        )
        .unwrap();

        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
             VALUES ('outlier b', 'fact', 0.8, 1000, 1000, 1)",
            [],
        ).unwrap();
        embeddings::store_embedding(
            &conn,
            "semantic",
            conn.last_insert_rowid(),
            &[0.0, 0.0, 1.0],
            "",
        )
        .unwrap();

        let discovered = discover_categories(&conn).unwrap();
        assert_eq!(
            discovered, 1,
            "should create 1 category from tight cluster, skip outliers"
        );
    }

    #[test]
    fn test_discover_categories_empty_label_fallback() {
        // Covers line 254: format!("cluster-{categories_created}") when prototype content is empty
        let conn = open_memory_db().unwrap();

        // Create 3 nodes with empty content
        for i in 0..3 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES ('', 'fact', 0.8, 1000, 1000, 1)",
                [],
            ).unwrap();
            let nid = conn.last_insert_rowid();
            embeddings::store_embedding(
                &conn,
                "semantic",
                nid,
                &[1.0, 0.0 + (i as f32) * 0.01, 0.0],
                "",
            )
            .unwrap();
        }

        let discovered = discover_categories(&conn).unwrap();
        assert_eq!(discovered, 1);

        // Check the label is "cluster-0" (fallback for empty content)
        let cats = categories::list_categories(&conn, None).unwrap();
        assert_eq!(cats.len(), 1);
        assert!(
            cats[0].label.starts_with("cluster-"),
            "empty content should produce fallback label, got: {}",
            cats[0].label
        );
    }

    #[test]
    fn test_maintain_categories_merge_lower_stability_wins() {
        // Covers line 322: (j, i) branch when cats[j].stability > cats[i].stability
        let conn = open_memory_db().unwrap();

        // Create two categories with near-identical centroids
        // but j (second) has higher stability
        for i in 0..2 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated)
                 VALUES (?1, 'fact', 0.8, 1000, 1000)",
                [format!("stability-node {i}")],
            ).unwrap();
        }

        let c1 = categories::store_category(
            &conn,
            "low-stability",
            NodeId(1),
            Some(&[1.0, 0.0, 0.0]),
            None,
        )
        .unwrap();
        let c2 = categories::store_category(
            &conn,
            "high-stability",
            NodeId(2),
            Some(&[0.99, 0.01, 0.0]),
            None,
        )
        .unwrap();

        // Set different stabilities: c1 low, c2 high
        conn.execute(
            "UPDATE categories SET stability = 0.3 WHERE id = ?1",
            [c1.0],
        )
        .unwrap();
        conn.execute(
            "UPDATE categories SET stability = 0.9 WHERE id = ?1",
            [c2.0],
        )
        .unwrap();

        categories::assign_node_to_category(&conn, NodeId(1), c1).unwrap();
        categories::assign_node_to_category(&conn, NodeId(2), c2).unwrap();
        embeddings::store_embedding(&conn, "semantic", 1, &[1.0, 0.0, 0.0], "").unwrap();
        embeddings::store_embedding(&conn, "semantic", 2, &[0.99, 0.01, 0.0], "").unwrap();

        let (merged, _dissolved) = maintain_categories(&conn).unwrap();
        assert!(merged >= 1, "should merge converging categories");

        // The surviving category should be the one with higher stability (c2)
        let remaining = categories::list_categories(&conn, None).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(
            remaining[0].label, "high-stability",
            "should keep the higher-stability category"
        );
    }

    #[test]
    fn test_split_no_centroid_skips() {
        // Covers line 429: None => continue when category has no centroid
        let conn = open_memory_db().unwrap();

        // Create a large category without a centroid
        for i in 0..10 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                [format!("no-centroid node {i}")],
            ).unwrap();
        }

        // Create category WITHOUT centroid but with many members
        let cat_id =
            categories::store_category(&conn, "no-centroid-cat", NodeId(1), None, None).unwrap();

        for i in 1..=10 {
            categories::assign_node_to_category(&conn, NodeId(i), cat_id).unwrap();
            embeddings::store_embedding(&conn, "semantic", i, &[1.0, 0.0, 0.0], "").unwrap();
        }

        let splits = split_large_categories(&conn).unwrap();
        assert_eq!(splits, 0, "should skip category without centroid");
    }

    #[test]
    fn test_dissolve_unstable_category_negative_initial_stability() {
        // Covers dissolve path: category with stability in (0, 0.1) after increment.
        // increment_stability: new = s + 0.1*(1-s). With s = -0.1:
        //   new = -0.1 + 0.1*1.1 = 0.01, which satisfies 0 < 0.01 < 0.1 → dissolved.
        let conn = open_memory_db().unwrap();

        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
             VALUES ('node', 'fact', 0.8, 1000, 1000, 1)",
            [],
        )
        .unwrap();

        let cat_id =
            categories::store_category(&conn, "will-dissolve", NodeId(1), None, None).unwrap();
        // Set stability to -0.1 so after increment it becomes 0.01 (between 0 and 0.1)
        conn.execute(
            "UPDATE categories SET stability = -0.1 WHERE id = ?1",
            [cat_id.0],
        )
        .unwrap();
        categories::assign_node_to_category(&conn, NodeId(1), cat_id).unwrap();

        let (_, dissolved) = maintain_categories(&conn).unwrap();
        assert_eq!(
            dissolved, 1,
            "category with post-increment stability 0.01 should be dissolved"
        );

        let remaining = categories::list_categories(&conn, None).unwrap();
        assert!(remaining.is_empty(), "dissolved category should be removed");
    }

    #[test]
    fn test_merge_skips_already_deleted_inner() {
        // Covers line 311: inner loop skips category already deleted by an earlier merge
        // Need 3 categories with very similar centroids so A merges B, then loop hits C
        // which was also merged, and should skip it.
        let conn = open_memory_db().unwrap();

        // Create 3 semantic nodes
        for i in 1..=3 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                [format!("triple-merge node {i}")],
            )
            .unwrap();
        }

        // Create 3 categories with nearly identical centroids
        let emb = vec![1.0f32, 0.0, 0.0];
        let c1 = categories::store_category(&conn, "cat-1", NodeId(1), Some(&emb), None).unwrap();
        let c2 = categories::store_category(&conn, "cat-2", NodeId(2), Some(&emb), None).unwrap();
        let c3 = categories::store_category(&conn, "cat-3", NodeId(3), Some(&emb), None).unwrap();

        // All same stability so sorted order is predictable (by id)
        conn.execute(
            "UPDATE categories SET stability = 0.5 WHERE id IN (?1, ?2, ?3)",
            rusqlite::params![c1.0, c2.0, c3.0],
        )
        .unwrap();

        categories::assign_node_to_category(&conn, NodeId(1), c1).unwrap();
        categories::assign_node_to_category(&conn, NodeId(2), c2).unwrap();
        categories::assign_node_to_category(&conn, NodeId(3), c3).unwrap();
        embeddings::store_embedding(&conn, "semantic", 1, &[1.0, 0.0, 0.0], "").unwrap();
        embeddings::store_embedding(&conn, "semantic", 2, &[1.0, 0.0, 0.0], "").unwrap();
        embeddings::store_embedding(&conn, "semantic", 3, &[1.0, 0.0, 0.0], "").unwrap();

        let (merged, _dissolved) = maintain_categories(&conn).unwrap();
        assert!(
            merged >= 2,
            "should merge at least 2 pairs: merged={merged}"
        );

        let remaining = categories::list_categories(&conn, None).unwrap();
        assert_eq!(
            remaining.len(),
            1,
            "only 1 category should survive after triple merge"
        );
    }

    #[test]
    fn test_dissolve_skips_merged_category() {
        // Covers line 397: dissolve loop skips a category that was deleted by merge
        // Need cat A (unstable, high sim to B) and cat B (stable, high sim to A).
        // Merge deletes A, then dissolve should skip A.
        let conn = open_memory_db().unwrap();

        for i in 1..=2 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                [format!("merge-dissolve node {i}")],
            )
            .unwrap();
        }

        let emb = vec![1.0f32, 0.0, 0.0];
        // c1: low stability (would be dissolved) - but also similar to c2 (will be merged first)
        let c1 =
            categories::store_category(&conn, "will-merge-low-stab", NodeId(1), Some(&emb), None)
                .unwrap();
        // c2: high stability
        let c2 = categories::store_category(&conn, "keep-high-stab", NodeId(2), Some(&emb), None)
            .unwrap();

        // c1 has very low stability (below dissolve threshold) AND is similar to c2
        conn.execute(
            "UPDATE categories SET stability = 0.05 WHERE id = ?1",
            [c1.0],
        )
        .unwrap();
        conn.execute(
            "UPDATE categories SET stability = 0.9 WHERE id = ?1",
            [c2.0],
        )
        .unwrap();

        categories::assign_node_to_category(&conn, NodeId(1), c1).unwrap();
        categories::assign_node_to_category(&conn, NodeId(2), c2).unwrap();
        embeddings::store_embedding(&conn, "semantic", 1, &[1.0, 0.0, 0.0], "").unwrap();
        embeddings::store_embedding(&conn, "semantic", 2, &[1.0, 0.0, 0.0], "").unwrap();

        let (merged, _dissolved) = maintain_categories(&conn).unwrap();
        // c1 merged into c2; dissolve should skip c1 since it's already gone
        assert!(merged >= 1, "should merge the similar categories");

        let remaining = categories::list_categories(&conn, None).unwrap();
        assert_eq!(remaining.len(), 1);
    }

    #[test]
    fn test_split_coherent_category_skipped() {
        // Covers line 454: category is coherent (avg sim >= SPLIT_COHERENCE_THRESHOLD), skip split
        let conn = open_memory_db().unwrap();

        // Create 10 nodes with very similar embeddings (coherent)
        for i in 1..=10 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                [format!("coherent node {i}")],
            )
            .unwrap();
        }

        let centroid = vec![1.0f32, 0.0, 0.0];
        let cat_id =
            categories::store_category(&conn, "coherent-cat", NodeId(1), Some(&centroid), None)
                .unwrap();

        // Assign all nodes with very similar embeddings → high coherence
        for i in 1..=10 {
            categories::assign_node_to_category(&conn, NodeId(i), cat_id).unwrap();
            // All embeddings nearly identical to centroid
            embeddings::store_embedding(&conn, "semantic", i, &[1.0, 0.0, 0.0], "").unwrap();
        }

        let splits = split_large_categories(&conn).unwrap();
        assert_eq!(splits, 0, "coherent category should not be split");
    }

    #[test]
    fn test_split_no_meaningful_subclusters() {
        // Covers line 498: members cluster into only 1 valid cluster (or clusters too small)
        let conn = open_memory_db().unwrap();

        // Create 10 nodes: 8 identical + 2 outliers (too few for a second cluster of MIN_CLUSTER_SIZE=3)
        for i in 1..=10 {
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                [format!("cluster node {i}")],
            )
            .unwrap();
        }

        // Centroid that's somewhat different from most members so coherence is low
        let centroid = vec![0.5f32, 0.5, 0.5];
        let cat_id =
            categories::store_category(&conn, "no-split-cat", NodeId(1), Some(&centroid), None)
                .unwrap();

        for i in 1..=8 {
            categories::assign_node_to_category(&conn, NodeId(i), cat_id).unwrap();
            // Main cluster: all similar to each other
            embeddings::store_embedding(&conn, "semantic", i, &[1.0, 0.0, 0.0], "").unwrap();
        }
        // 2 outliers (not enough for MIN_CLUSTER_SIZE=3)
        for i in 9..=10 {
            categories::assign_node_to_category(&conn, NodeId(i), cat_id).unwrap();
            embeddings::store_embedding(&conn, "semantic", i, &[0.0, 0.0, 1.0], "").unwrap();
        }

        let splits = split_large_categories(&conn).unwrap();
        assert_eq!(
            splits, 0,
            "should not split when only 1 valid cluster exists"
        );
    }

    #[test]
    fn test_split_label_truncation() {
        // Covers line 515: label_content.len() > 40 truncation
        // Also covers lines 454 (coherence check) and 498 (valid clusters check)
        let conn = open_memory_db().unwrap();

        // Create 16 nodes with LONG content (> 40 chars) in two distinct clusters
        for i in 1..=16 {
            let long_content = format!(
                "This is a very long content string that exceeds forty characters for node number {i}"
            );
            conn.execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
                 VALUES (?1, 'fact', 0.8, 1000, 1000, 1)",
                [long_content],
            )
            .unwrap();
        }

        // Use a centroid that is distant from BOTH clusters so avg coherence < 0.6
        // cos([1,0,0,0], [0,0,1,0]) = 0 → avg coherence ≈ 0
        // We need centroid such that avg sim to it is < 0.6
        // Use a 4D centroid orthogonal to both clusters
        let centroid = vec![0.0f32, 0.0, 0.0, 1.0];
        let cat_id =
            categories::store_category(&conn, "split-me", NodeId(1), Some(&centroid), None)
                .unwrap();

        // Cluster 1: 8 nodes in one direction (4D)
        for i in 1..=8 {
            categories::assign_node_to_category(&conn, NodeId(i), cat_id).unwrap();
            embeddings::store_embedding(&conn, "semantic", i, &[0.95, 0.05, 0.0, 0.0], "").unwrap();
        }
        // Cluster 2: 8 nodes in orthogonal direction (4D)
        for i in 9..=16 {
            categories::assign_node_to_category(&conn, NodeId(i), cat_id).unwrap();
            embeddings::store_embedding(&conn, "semantic", i, &[0.0, 0.0, 0.95, 0.05], "").unwrap();
        }

        let splits = split_large_categories(&conn).unwrap();
        assert!(
            splits >= 1,
            "should split incoherent category: splits={splits}"
        );

        // Verify sub-categories were created with truncated labels
        let all_cats = categories::list_categories(&conn, None).unwrap();
        // Original category + sub-categories
        let sub_cats: Vec<_> = all_cats.iter().filter(|c| c.id != cat_id).collect();
        assert!(!sub_cats.is_empty(), "should have created sub-categories");
        for cat in &sub_cats {
            assert!(
                cat.label.len() <= 40,
                "sub-category label should be <= 40 chars: '{}'",
                cat.label
            );
        }
    }
}
