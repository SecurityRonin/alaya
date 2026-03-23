//! Shared test utilities — fixtures, factories, builders.
//! Only compiled in test builds.

#[cfg(test)]
pub(crate) mod fixtures {
    use crate::types::*;
    use rusqlite::Connection;

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

}
