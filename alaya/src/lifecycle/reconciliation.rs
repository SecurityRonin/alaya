use crate::error::Result;
use crate::graph::links;
use crate::store::{conflicts, embeddings, semantic};
use crate::types::*;
use rusqlite::Connection;

/// Minimum cosine similarity for two nodes to be flagged as potential conflicts.
const CONFLICT_SIMILARITY_THRESHOLD: f32 = 0.85;

/// Maximum conflicts to send for LLM verification in one batch.
#[allow(dead_code)]
const LLM_VERIFICATION_BATCH_SIZE: usize = 20;

/// Run the full reconciliation lifecycle: detect conflicts, then resolve them.
pub fn reconcile(conn: &Connection, strategy: ConflictStrategy) -> Result<ReconcileReport> {
    let mut report = ReconcileReport::default();

    // Phase 1: Detect
    detect_conflicts(conn, &mut report)?;

    // Phase 2: Resolve (skip for Manual strategy)
    if strategy != ConflictStrategy::Manual {
        resolve_conflicts(conn, strategy, &mut report)?;
    } else {
        // Count pending for report
        let unresolved = conflicts::get_unresolved_conflicts(conn)?;
        report.conflicts_pending = unresolved.len() as u32;
    }

    Ok(report)
}

/// Phase 1: Heuristic conflict detection via cosine similarity.
///
/// Compares semantic nodes within the same category (or both uncategorized)
/// to avoid O(n^2) over the full graph.
fn detect_conflicts(conn: &Connection, report: &mut ReconcileReport) -> Result<()> {
    // Get all non-superseded semantic nodes with embeddings
    let mut stmt = conn.prepare(
        "SELECT sn.id, sn.category_id, e.embedding
         FROM semantic_nodes sn
         JOIN embeddings e ON e.node_type = 'semantic' AND e.node_id = sn.id
         WHERE sn.superseded_by IS NULL",
    )?;
    let nodes: Vec<(NodeId, Option<i64>, Vec<f32>)> = stmt
        .query_map([], |row| {
            let id: i64 = row.get(0)?;
            let cat_id: Option<i64> = row.get(1)?;
            let emb_blob: Vec<u8> = row.get(2)?;
            let embedding = embeddings::deserialize_embedding(&emb_blob);
            Ok((NodeId(id), cat_id, embedding))
        })?
        .filter_map(|r| r.ok())
        .collect();

    let now = crate::db::now();

    // Compare within same category (or both uncategorized = None)
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            // Skip cross-category pairs
            if nodes[i].1 != nodes[j].1 {
                continue;
            }

            let sim = embeddings::cosine_similarity(&nodes[i].2, &nodes[j].2);
            if sim >= CONFLICT_SIMILARITY_THRESHOLD {
                if let Some(_id) =
                    conflicts::insert_conflict(conn, nodes[i].0, nodes[j].0, sim, now)?
                {
                    report.conflicts_detected += 1;
                }
            }
        }
    }

    Ok(())
}

/// Phase 2: Resolve detected/verified conflicts using the given strategy.
fn resolve_conflicts(
    conn: &Connection,
    strategy: ConflictStrategy,
    report: &mut ReconcileReport,
) -> Result<()> {
    let unresolved = conflicts::get_unresolved_conflicts(conn)?;

    let now = crate::db::now();

    for conflict in &unresolved {
        let node_a = semantic::get_semantic_node(conn, conflict.node_a)?;
        let node_b = semantic::get_semantic_node(conn, conflict.node_b)?;

        let (winner, loser) = match strategy {
            ConflictStrategy::Recency => {
                // Most recent source episode wins; fall back to created_at
                if node_a.created_at >= node_b.created_at {
                    (conflict.node_a, conflict.node_b)
                } else {
                    (conflict.node_b, conflict.node_a)
                }
            }
            ConflictStrategy::Confidence => {
                let diff = (node_a.confidence - node_b.confidence).abs();
                if diff < 0.01 {
                    // Tie — fall back to recency
                    if node_a.created_at >= node_b.created_at {
                        (conflict.node_a, conflict.node_b)
                    } else {
                        (conflict.node_b, conflict.node_a)
                    }
                } else if node_a.confidence >= node_b.confidence {
                    (conflict.node_a, conflict.node_b)
                } else {
                    (conflict.node_b, conflict.node_a)
                }
            }
            ConflictStrategy::Corroboration => {
                if node_a.corroboration_count == node_b.corroboration_count {
                    // Tie — fall back to recency
                    if node_a.created_at >= node_b.created_at {
                        (conflict.node_a, conflict.node_b)
                    } else {
                        (conflict.node_b, conflict.node_a)
                    }
                } else if node_a.corroboration_count >= node_b.corroboration_count {
                    (conflict.node_a, conflict.node_b)
                } else {
                    (conflict.node_b, conflict.node_a)
                }
            }
            ConflictStrategy::Manual => unreachable!(), // filtered above
        };

        let resolution_str = match strategy {
            ConflictStrategy::Recency => "recency",
            ConflictStrategy::Confidence => "confidence",
            ConflictStrategy::Corroboration => "corroboration",
            ConflictStrategy::Manual => unreachable!(),
        };

        // Execute resolution
        conflicts::resolve_conflict(conn, conflict.id, winner, resolution_str, now)?;
        conflicts::supersede_node(conn, loser, winner)?;
        links::create_link(
            conn,
            NodeRef::Semantic(winner),
            NodeRef::Semantic(loser),
            LinkType::Supersedes,
            1.0,
        )?;

        report.conflicts_resolved += 1;
        report.nodes_superseded += 1;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;
    use crate::store::semantic::{get_semantic_node, store_semantic_node};

    fn make_contradictory_nodes(conn: &Connection) -> (NodeId, NodeId) {
        // Two nodes with near-identical embeddings (sim > 0.85)
        let a = store_semantic_node(
            conn,
            &NewSemanticNode {
                content: "user prefers dark mode".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: Some(vec![0.9, 0.1, 0.0]),
            },
        )
        .unwrap();
        let b = store_semantic_node(
            conn,
            &NewSemanticNode {
                content: "user prefers light mode".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: vec![],
                embedding: Some(vec![0.85, 0.15, 0.0]),
            },
        )
        .unwrap();

        // Ensure a is strictly older than b so recency tests are deterministic
        conn.execute(
            "UPDATE semantic_nodes SET created_at = 1000 WHERE id = ?1",
            [a.0],
        )
        .unwrap();
        conn.execute(
            "UPDATE semantic_nodes SET created_at = 2000 WHERE id = ?1",
            [b.0],
        )
        .unwrap();

        (a, b)
    }

    #[test]
    fn detection_finds_similar_nodes() {
        let conn = open_memory_db().unwrap();
        let (_a, _b) = make_contradictory_nodes(&conn);

        let report = reconcile(&conn, ConflictStrategy::Manual).unwrap();
        assert_eq!(report.conflicts_detected, 1);
        assert_eq!(report.conflicts_pending, 1);
        assert_eq!(report.conflicts_resolved, 0);
    }

    #[test]
    fn detection_skips_existing_pairs() {
        let conn = open_memory_db().unwrap();
        let (_a, _b) = make_contradictory_nodes(&conn);

        reconcile(&conn, ConflictStrategy::Manual).unwrap();
        // Second run: no new detections
        let report = reconcile(&conn, ConflictStrategy::Manual).unwrap();
        assert_eq!(report.conflicts_detected, 0);
        assert_eq!(report.conflicts_pending, 1);
    }

    #[test]
    fn detection_ignores_cross_category_pairs() {
        let conn = open_memory_db().unwrap();
        // Node A: category 1
        let a = store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "fact in category A".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: Some(vec![0.9, 0.1, 0.0]),
            },
        )
        .unwrap();
        // Node B: same embedding but different category
        let b = store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "fact in category B".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: vec![],
                embedding: Some(vec![0.85, 0.15, 0.0]),
            },
        )
        .unwrap();

        // Insert real category rows to satisfy foreign key
        conn.execute(
            "INSERT INTO categories (id, label, prototype_node_id, member_count, created_at, last_updated, stability)
             VALUES (1, 'cat-a', ?1, 1, 0, 0, 0.0)",
            [a.0],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO categories (id, label, prototype_node_id, member_count, created_at, last_updated, stability)
             VALUES (2, 'cat-b', ?1, 1, 0, 0, 0.0)",
            [b.0],
        )
        .unwrap();

        // Assign different categories
        conn.execute(
            "UPDATE semantic_nodes SET category_id = 1 WHERE id = ?1",
            [a.0],
        )
        .unwrap();
        conn.execute(
            "UPDATE semantic_nodes SET category_id = 2 WHERE id = ?1",
            [b.0],
        )
        .unwrap();

        let report = reconcile(&conn, ConflictStrategy::Manual).unwrap();
        assert_eq!(report.conflicts_detected, 0);
    }

    #[test]
    fn recency_strategy_picks_newer() {
        let conn = open_memory_db().unwrap();
        let (a, _b) = make_contradictory_nodes(&conn);
        // a was created first, b second → b is newer → b wins

        let report = reconcile(&conn, ConflictStrategy::Recency).unwrap();
        assert_eq!(report.conflicts_resolved, 1);
        assert_eq!(report.nodes_superseded, 1);

        // a (older) should be superseded
        let node_a = get_semantic_node(&conn, a).unwrap();
        assert_eq!(node_a.confidence, 0.0);
    }

    #[test]
    fn confidence_strategy_picks_higher() {
        let conn = open_memory_db().unwrap();
        let (_a, b) = make_contradictory_nodes(&conn);
        // a has confidence 0.9, b has 0.8 → a wins

        let report = reconcile(&conn, ConflictStrategy::Confidence).unwrap();
        assert_eq!(report.conflicts_resolved, 1);

        // b (lower confidence) should be superseded
        let node_b = get_semantic_node(&conn, b).unwrap();
        assert_eq!(node_b.confidence, 0.0);
    }

    #[test]
    fn confidence_tie_falls_back_to_recency() {
        let conn = open_memory_db().unwrap();
        // Two nodes with equal confidence
        let a = store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "tied fact A".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.85,
                source_episodes: vec![],
                embedding: Some(vec![0.9, 0.1, 0.0]),
            },
        )
        .unwrap();
        let b = store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "tied fact B".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.85,
                source_episodes: vec![],
                embedding: Some(vec![0.85, 0.15, 0.0]),
            },
        )
        .unwrap();

        // Ensure a is strictly older than b
        conn.execute(
            "UPDATE semantic_nodes SET created_at = 1000 WHERE id = ?1",
            [a.0],
        )
        .unwrap();
        conn.execute(
            "UPDATE semantic_nodes SET created_at = 2000 WHERE id = ?1",
            [b.0],
        )
        .unwrap();

        let report = reconcile(&conn, ConflictStrategy::Confidence).unwrap();
        assert_eq!(report.conflicts_resolved, 1);

        // b is newer (created_at=2000) → b wins, a is superseded
        let node_a = get_semantic_node(&conn, a).unwrap();
        assert_eq!(node_a.confidence, 0.0);
    }

    #[test]
    fn manual_strategy_leaves_unresolved() {
        let conn = open_memory_db().unwrap();
        let (_a, _b) = make_contradictory_nodes(&conn);

        let report = reconcile(&conn, ConflictStrategy::Manual).unwrap();
        assert_eq!(report.conflicts_detected, 1);
        assert_eq!(report.conflicts_resolved, 0);
        assert_eq!(report.conflicts_pending, 1);
    }

    #[test]
    fn resolution_creates_supersedes_link() {
        let conn = open_memory_db().unwrap();
        let (_a, b) = make_contradictory_nodes(&conn);
        // b wins (newer), a loses

        reconcile(&conn, ConflictStrategy::Recency).unwrap();

        // Check that a Supersedes link exists: winner -> loser
        let links_from_b = links::get_links_from(&conn, NodeRef::Semantic(b)).unwrap();
        let supersedes = links_from_b
            .iter()
            .find(|l| l.link_type == LinkType::Supersedes);
        assert!(supersedes.is_some(), "should have a Supersedes link");
    }

    #[test]
    fn superseded_node_excluded_from_queries() {
        let conn = open_memory_db().unwrap();
        let (_a, _b) = make_contradictory_nodes(&conn);

        reconcile(&conn, ConflictStrategy::Recency).unwrap();

        // Only 1 fact should be visible
        let facts = semantic::find_by_type(&conn, SemanticType::Fact, 10).unwrap();
        assert_eq!(facts.len(), 1);
    }

    #[test]
    fn empty_store_produces_empty_report() {
        let conn = open_memory_db().unwrap();
        let report = reconcile(&conn, ConflictStrategy::Recency).unwrap();
        assert_eq!(report.conflicts_detected, 0);
        assert_eq!(report.conflicts_resolved, 0);
        assert_eq!(report.conflicts_pending, 0);
        assert_eq!(report.nodes_superseded, 0);
    }

    #[test]
    fn no_conflicts_when_nodes_dissimilar() {
        let conn = open_memory_db().unwrap();
        // Orthogonal embeddings → similarity ~0
        store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "fact about cooking".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: Some(vec![1.0, 0.0, 0.0]),
            },
        )
        .unwrap();
        store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "fact about programming".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: Some(vec![0.0, 1.0, 0.0]),
            },
        )
        .unwrap();

        let report = reconcile(&conn, ConflictStrategy::Recency).unwrap();
        assert_eq!(report.conflicts_detected, 0);
    }

    #[test]
    fn confidence_tie_node_a_newer_wins() {
        let conn = open_memory_db().unwrap();
        // Two nodes with identical confidence
        let a = store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "tied newer fact".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.85,
                source_episodes: vec![],
                embedding: Some(vec![0.9, 0.1, 0.0]),
            },
        )
        .unwrap();
        let b = store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "tied older fact".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.85,
                source_episodes: vec![],
                embedding: Some(vec![0.85, 0.15, 0.0]),
            },
        )
        .unwrap();

        // Make a NEWER than b (opposite of the existing tie test)
        conn.execute(
            "UPDATE semantic_nodes SET created_at = 2000 WHERE id = ?1",
            [a.0],
        )
        .unwrap();
        conn.execute(
            "UPDATE semantic_nodes SET created_at = 1000 WHERE id = ?1",
            [b.0],
        )
        .unwrap();

        let report = reconcile(&conn, ConflictStrategy::Confidence).unwrap();
        assert_eq!(report.conflicts_resolved, 1);

        // a is newer so a wins; b is superseded
        let node_b = get_semantic_node(&conn, b).unwrap();
        assert_eq!(node_b.confidence, 0.0);
        let node_a = get_semantic_node(&conn, a).unwrap();
        assert!(node_a.confidence > 0.0);
    }

    #[test]
    fn confidence_strategy_node_b_higher() {
        let conn = open_memory_db().unwrap();
        // a has LOW confidence, b has HIGH confidence
        let a = store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "low confidence fact".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.5,
                source_episodes: vec![],
                embedding: Some(vec![0.9, 0.1, 0.0]),
            },
        )
        .unwrap();
        let b = store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "high confidence fact".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.95,
                source_episodes: vec![],
                embedding: Some(vec![0.85, 0.15, 0.0]),
            },
        )
        .unwrap();

        let report = reconcile(&conn, ConflictStrategy::Confidence).unwrap();
        assert_eq!(report.conflicts_resolved, 1);

        // a (lower confidence) should be superseded
        let node_a = get_semantic_node(&conn, a).unwrap();
        assert_eq!(node_a.confidence, 0.0);
        let node_b = get_semantic_node(&conn, b).unwrap();
        assert!(node_b.confidence > 0.0);
    }

    #[test]
    fn corroboration_strategy_picks_higher() {
        let conn = open_memory_db().unwrap();
        let (_a, b) = make_contradictory_nodes(&conn);
        // a.corroboration_count=5, b.corroboration_count=2 → a wins
        conn.execute(
            "UPDATE semantic_nodes SET corroboration_count = 5 WHERE id = ?1",
            [_a.0],
        )
        .unwrap();
        conn.execute(
            "UPDATE semantic_nodes SET corroboration_count = 2 WHERE id = ?1",
            [b.0],
        )
        .unwrap();

        let report = reconcile(&conn, ConflictStrategy::Corroboration).unwrap();
        assert_eq!(report.conflicts_resolved, 1);

        // b (lower corroboration) should be superseded
        let node_b = get_semantic_node(&conn, b).unwrap();
        assert_eq!(node_b.confidence, 0.0);
    }

    #[test]
    fn corroboration_strategy_node_b_higher() {
        let conn = open_memory_db().unwrap();
        let (a, _b) = make_contradictory_nodes(&conn);
        // a.corroboration_count=1, b.corroboration_count=10 → b wins
        conn.execute(
            "UPDATE semantic_nodes SET corroboration_count = 1 WHERE id = ?1",
            [a.0],
        )
        .unwrap();
        conn.execute(
            "UPDATE semantic_nodes SET corroboration_count = 10 WHERE id = ?1",
            [_b.0],
        )
        .unwrap();

        let report = reconcile(&conn, ConflictStrategy::Corroboration).unwrap();
        assert_eq!(report.conflicts_resolved, 1);

        // a (lower corroboration) should be superseded
        let node_a = get_semantic_node(&conn, a).unwrap();
        assert_eq!(node_a.confidence, 0.0);
    }

    #[test]
    fn corroboration_tie_falls_back_to_recency() {
        let conn = open_memory_db().unwrap();
        let (a, _b) = make_contradictory_nodes(&conn);
        // Both have equal corroboration_count → fall back to recency
        // a.created_at=1000 (older), b.created_at=2000 (newer) → b wins
        conn.execute(
            "UPDATE semantic_nodes SET corroboration_count = 3 WHERE id = ?1",
            [a.0],
        )
        .unwrap();
        conn.execute(
            "UPDATE semantic_nodes SET corroboration_count = 3 WHERE id = ?1",
            [_b.0],
        )
        .unwrap();

        let report = reconcile(&conn, ConflictStrategy::Corroboration).unwrap();
        assert_eq!(report.conflicts_resolved, 1);

        // a (older, created_at=1000) should be superseded
        let node_a = get_semantic_node(&conn, a).unwrap();
        assert_eq!(node_a.confidence, 0.0);
    }

    #[test]
    fn idempotent_second_reconcile_no_new_detections() {
        let conn = open_memory_db().unwrap();
        let (_a, _b) = make_contradictory_nodes(&conn);

        let report1 = reconcile(&conn, ConflictStrategy::Recency).unwrap();
        assert_eq!(report1.conflicts_detected, 1);
        assert_eq!(report1.conflicts_resolved, 1);

        let report2 = reconcile(&conn, ConflictStrategy::Recency).unwrap();
        assert_eq!(report2.conflicts_detected, 0);
        assert_eq!(report2.conflicts_resolved, 0);
    }
}
