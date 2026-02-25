use rusqlite::Connection;
use crate::error::Result;
use crate::store::strengths;
use crate::types::*;

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
    let mut report = ForgettingReport::default();

    // Decay retrieval strength across all nodes
    report.nodes_decayed = strengths::decay_all_retrieval(conn, DEFAULT_DECAY_FACTOR)? as u32;

    // Find and archive nodes below both thresholds
    let archivable = strengths::find_archivable(
        conn,
        ARCHIVE_STORAGE_THRESHOLD,
        ARCHIVE_RETRIEVAL_THRESHOLD,
    )?;

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
        episodic::store_episode(&conn, &NewEpisode {
            content: "test".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 1000,
            context: EpisodeContext::default(),
            embedding: None,
        }).unwrap();
        strengths::init_strength(&conn, node).unwrap();

        let before = strengths::get_strength(&conn, node).unwrap();
        forget(&conn).unwrap();
        let after = strengths::get_strength(&conn, node).unwrap();

        assert!(after.retrieval_strength < before.retrieval_strength);
    }
}
