use crate::error::Result;
use crate::store::strengths;
use crate::types::*;
use rusqlite::Connection;

/// Default decay factor per sweep (applied to retrieval strength).
const DEFAULT_DECAY_FACTOR: f32 = 0.95;

/// Thresholds for archiving nodes.
const ARCHIVE_STORAGE_THRESHOLD: f32 = 0.1;
const ARCHIVE_RETRIEVAL_THRESHOLD: f32 = 0.05;

/// Run a forgetting sweep.
///
/// Models the Bjork & Bjork (1992) "New Theory of Disuse":
/// - Storage strength (how well-learned) monotonically increases with access
/// - Retrieval strength (how accessible now) decays over time
///
/// Nodes with low storage AND low retrieval are archived (deleted).
/// Nodes with high storage but low retrieval are "latent" — they exist
/// but are hard to find without a strong cue.
pub fn forget(conn: &Connection) -> Result<ForgettingReport> {
    let mut report = ForgettingReport {
        nodes_decayed: strengths::decay_all_retrieval(conn, DEFAULT_DECAY_FACTOR)? as u32,
        ..Default::default()
    };

    // Find and archive nodes below both thresholds
    let archivable =
        strengths::find_archivable(conn, ARCHIVE_STORAGE_THRESHOLD, ARCHIVE_RETRIEVAL_THRESHOLD)?;

    for node in &archivable {
        match node {
            NodeRef::Episode(id) => {
                crate::store::episodic::delete_episodes(conn, &[*id])?;
            }
            NodeRef::Semantic(id) => {
                crate::store::semantic::delete_node(conn, *id)?;
            }
            NodeRef::Preference(_) => {
                // Preferences are handled by transformation/decay, not forgetting
                continue;
            }
            NodeRef::Category(_) => {
                // Categories are managed by transformation (merge/dissolve), not forgetting
                continue;
            }
        }
        // Clean up the strength record
        conn.execute(
            "DELETE FROM node_strengths WHERE node_type = ?1 AND node_id = ?2",
            rusqlite::params![node.type_str(), node.id()],
        )?;
        report.nodes_archived += 1;
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;
    use crate::store::{episodic, strengths};

    #[test]
    fn test_forget_empty_db() {
        let conn = open_memory_db().unwrap();
        let report = forget(&conn).unwrap();
        assert_eq!(report.nodes_decayed, 0);
        assert_eq!(report.nodes_archived, 0);
    }

    #[test]
    fn test_decay_reduces_retrieval_strength() {
        let conn = open_memory_db().unwrap();
        let node = NodeRef::Episode(EpisodeId(1));

        // Create episode and init strength
        episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "test".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();
        strengths::init_strength(&conn, node).unwrap();

        let before = strengths::get_strength(&conn, node).unwrap();
        forget(&conn).unwrap();
        let after = strengths::get_strength(&conn, node).unwrap();

        assert!(after.retrieval_strength < before.retrieval_strength);
    }

    #[test]
    fn test_archive_semantic_node() {
        use crate::store::semantic;

        let conn = open_memory_db().unwrap();

        // Store a semantic node
        let node_id = semantic::store_semantic_node(
            &conn,
            &crate::types::NewSemanticNode {
                content: "archivable knowledge".to_string(),
                node_type: crate::types::SemanticType::Fact,
                confidence: 0.5,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();

        let node = NodeRef::Semantic(node_id);
        strengths::init_strength(&conn, node).unwrap();

        // Manually set both strengths very low (below thresholds)
        conn.execute(
            "UPDATE node_strengths SET storage_strength = 0.05, retrieval_strength = 0.01
             WHERE node_type = 'semantic' AND node_id = ?1",
            [node_id.0],
        )
        .unwrap();

        // Verify the node exists before forget
        let before = semantic::get_semantic_node(&conn, node_id);
        assert!(before.is_ok(), "semantic node should exist before forget");

        let report = forget(&conn).unwrap();
        assert_eq!(report.nodes_archived, 1);

        // Verify the semantic node was deleted
        let after = semantic::get_semantic_node(&conn, node_id);
        assert!(
            after.is_err(),
            "semantic node should be deleted after archive"
        );
    }

    #[test]
    fn test_forget_skips_preferences_and_categories() {
        let conn = open_memory_db().unwrap();

        // Insert a preference node_strengths record with very low strengths
        conn.execute(
            "INSERT INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength, access_count, last_accessed)
             VALUES ('preference', 1, 0.01, 0.001, 1, 1000)",
            [],
        )
        .unwrap();

        // Insert a category node_strengths record with very low strengths
        conn.execute(
            "INSERT INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength, access_count, last_accessed)
             VALUES ('category', 1, 0.01, 0.001, 1, 1000)",
            [],
        )
        .unwrap();

        let report = forget(&conn).unwrap();
        // Preferences and categories should be skipped (continue), not archived
        assert_eq!(report.nodes_archived, 0);

        // Verify the strength records still exist (they were not cleaned up)
        let pref_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM node_strengths WHERE node_type = 'preference'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            pref_count, 1,
            "preference strength record should still exist"
        );

        let cat_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM node_strengths WHERE node_type = 'category'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(cat_count, 1, "category strength record should still exist");
    }

    #[test]
    fn test_archive_episode_node() {
        let conn = open_memory_db().unwrap();

        // Store an episode
        episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "soon to be forgotten".to_string(),
                role: crate::types::Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: crate::types::EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();
        let ep_id = EpisodeId(conn.last_insert_rowid());
        let node = NodeRef::Episode(ep_id);

        // Init strength and immediately set both to below archival thresholds
        strengths::init_strength(&conn, node).unwrap();
        conn.execute(
            "UPDATE node_strengths SET storage_strength = 0.05, retrieval_strength = 0.01
             WHERE node_type = 'episode' AND node_id = ?1",
            [ep_id.0],
        )
        .unwrap();

        // Verify the episode exists before forget
        let count_before: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM episodes WHERE id = ?1",
                [ep_id.0],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count_before, 1);

        let report = forget(&conn).unwrap();
        assert_eq!(report.nodes_archived, 1, "episode should be archived");

        // Verify the episode was deleted
        let count_after: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM episodes WHERE id = ?1",
                [ep_id.0],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count_after, 0, "episode should be deleted after archiving");

        // Verify strength record was cleaned up
        let strength_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM node_strengths WHERE node_type = 'episode' AND node_id = ?1",
                [ep_id.0],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            strength_count, 0,
            "strength record should be removed after archive"
        );
    }

    #[test]
    fn test_forget_nodes_decayed_count() {
        let conn = open_memory_db().unwrap();
        let node = NodeRef::Episode(EpisodeId(1));

        episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "content".to_string(),
                role: crate::types::Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: crate::types::EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();
        strengths::init_strength(&conn, node).unwrap();

        let report = forget(&conn).unwrap();
        // nodes_decayed = rows updated by decay_all_retrieval
        assert_eq!(
            report.nodes_decayed, 1,
            "one strength row should have been decayed"
        );
    }
}
