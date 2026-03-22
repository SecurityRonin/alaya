//! Shared test utilities — fixtures, factories, builders.
//! Only compiled in test builds.

#[cfg(test)]
pub(crate) mod fixtures {
    use crate::types::*;
    use rusqlite::Connection;

    /// Open an in-memory DB with schema initialized.
    pub fn test_db() -> Connection {
        crate::schema::open_memory_db().unwrap()
    }

    /// Episode factory with sensible defaults.
    pub fn episode(content: &str) -> NewEpisode {
        NewEpisode {
            content: content.to_string(),
            role: Role::User,
            session_id: "test-session".to_string(),
            timestamp: 1000,
            context: EpisodeContext::default(),
            embedding: None,
        }
    }

    /// Episode with custom role and timestamp.
    pub fn episode_at(content: &str, role: Role, ts: i64) -> NewEpisode {
        NewEpisode {
            timestamp: ts,
            role,
            ..episode(content)
        }
    }

    /// Insert a semantic node directly in the DB, returning its NodeId.
    pub fn insert_semantic_node(conn: &Connection, content: &str, confidence: f32) -> NodeId {
        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
             VALUES (?1, 'fact', ?2, 1000, 1000, 1)",
            rusqlite::params![content, confidence],
        )
        .unwrap();
        NodeId(conn.last_insert_rowid())
    }

    /// Insert a semantic node with a specific type.
    pub fn insert_typed_node(
        conn: &Connection,
        content: &str,
        node_type: &str,
        confidence: f32,
    ) -> NodeId {
        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
             VALUES (?1, ?2, ?3, 1000, 1000, 1)",
            rusqlite::params![content, node_type, confidence],
        )
        .unwrap();
        NodeId(conn.last_insert_rowid())
    }

    /// Store an episode via the store module, returning its EpisodeId.
    pub fn store_test_episode(conn: &Connection, content: &str) -> EpisodeId {
        crate::store::episodic::store_episode(conn, &episode(content)).unwrap()
    }
}
