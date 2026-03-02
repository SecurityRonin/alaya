use crate::error::{AlayaError, Result};
use crate::types::*;
use rusqlite::{params, Connection};

pub fn store_semantic_node(conn: &Connection, node: &NewSemanticNode) -> Result<NodeId> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let sources_json = serde_json::to_string(&node.source_episodes)?;
    conn.execute(
        "INSERT INTO semantic_nodes (content, node_type, confidence, source_episodes_json, created_at, last_corroborated, corroboration_count)
         VALUES (?1, ?2, ?3, ?4, ?5, ?5, 1)",
        params![node.content, node.node_type.as_str(), node.confidence, sources_json, now],
    )?;
    let id = NodeId(conn.last_insert_rowid());

    if let Some(ref emb) = node.embedding {
        crate::store::embeddings::store_embedding(conn, "semantic", id.0, emb, "")?;
    }

    Ok(id)
}

#[allow(dead_code)]
pub fn get_semantic_node(conn: &Connection, id: NodeId) -> Result<SemanticNode> {
    conn.query_row(
        "SELECT id, content, node_type, confidence, source_episodes_json,
                created_at, last_corroborated, corroboration_count
         FROM semantic_nodes WHERE id = ?1",
        [id.0],
        |row| {
            let sources_str: String = row.get(4)?;
            Ok(SemanticNode {
                id: NodeId(row.get(0)?),
                content: row.get(1)?,
                node_type: SemanticType::from_str(&row.get::<_, String>(2)?)
                    .unwrap_or(SemanticType::Fact),
                confidence: row.get(3)?,
                source_episodes: serde_json::from_str(&sources_str).unwrap_or_default(),
                created_at: row.get(5)?,
                last_corroborated: row.get(6)?,
                corroboration_count: row.get(7)?,
            })
        },
    )
    .map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => {
            AlayaError::NotFound(format!("semantic node {}", id.0))
        }
        other => AlayaError::Db(other),
    })
}

#[allow(dead_code)]
pub fn update_corroboration(conn: &Connection, id: NodeId) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let changed = conn.execute(
        "UPDATE semantic_nodes SET corroboration_count = corroboration_count + 1,
                last_corroborated = ?2 WHERE id = ?1",
        params![id.0, now],
    )?;
    if changed == 0 {
        return Err(AlayaError::NotFound(format!("semantic node {}", id.0)));
    }
    Ok(())
}

pub fn find_by_type(
    conn: &Connection,
    node_type: SemanticType,
    limit: u32,
) -> Result<Vec<SemanticNode>> {
    let mut stmt = conn.prepare(
        "SELECT id, content, node_type, confidence, source_episodes_json,
                created_at, last_corroborated, corroboration_count
         FROM semantic_nodes WHERE node_type = ?1
         ORDER BY confidence DESC LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![node_type.as_str(), limit], |row| {
        let sources_str: String = row.get(4)?;
        Ok(SemanticNode {
            id: NodeId(row.get(0)?),
            content: row.get(1)?,
            node_type: SemanticType::from_str(&row.get::<_, String>(2)?)
                .unwrap_or(SemanticType::Fact),
            confidence: row.get(3)?,
            source_episodes: serde_json::from_str(&sources_str).unwrap_or_default(),
            created_at: row.get(5)?,
            last_corroborated: row.get(6)?,
            corroboration_count: row.get(7)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn delete_node(conn: &Connection, id: NodeId) -> Result<()> {
    conn.execute("DELETE FROM semantic_nodes WHERE id = ?1", [id.0])?;
    // Also clean up embedding and links
    conn.execute(
        "DELETE FROM embeddings WHERE node_type = 'semantic' AND node_id = ?1",
        [id.0],
    )?;
    conn.execute("DELETE FROM links WHERE (source_type = 'semantic' AND source_id = ?1) OR (target_type = 'semantic' AND target_id = ?1)", [id.0])?;
    conn.execute(
        "DELETE FROM node_strengths WHERE node_type = 'semantic' AND node_id = ?1",
        [id.0],
    )?;
    Ok(())
}

pub fn count_nodes(conn: &Connection) -> Result<u64> {
    let count: i64 = conn.query_row("SELECT count(*) FROM semantic_nodes", [], |row| row.get(0))?;
    Ok(count as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;

    #[test]
    fn test_store_and_get() {
        let conn = open_memory_db().unwrap();
        let id = store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "User is a Rust developer".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: vec![EpisodeId(1), EpisodeId(2)],
                embedding: None,
            },
        )
        .unwrap();
        let node = get_semantic_node(&conn, id).unwrap();
        assert_eq!(node.content, "User is a Rust developer");
        assert_eq!(node.confidence, 0.8);
        assert_eq!(node.source_episodes.len(), 2);
    }

    #[test]
    fn test_corroboration() {
        let conn = open_memory_db().unwrap();
        let id = store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "fact".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.5,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        update_corroboration(&conn, id).unwrap();
        let node = get_semantic_node(&conn, id).unwrap();
        assert_eq!(node.corroboration_count, 2);
    }

    #[test]
    fn test_find_by_type() {
        let conn = open_memory_db().unwrap();
        store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "high confidence fact".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "low confidence fact".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.3,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "a relationship".to_string(),
                node_type: SemanticType::Relationship,
                confidence: 0.7,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();

        // Filter by Fact type
        let facts = find_by_type(&conn, SemanticType::Fact, 10).unwrap();
        assert_eq!(facts.len(), 2);
        // Should be ordered by confidence DESC
        assert!(facts[0].confidence >= facts[1].confidence);

        // Filter by Relationship type
        let rels = find_by_type(&conn, SemanticType::Relationship, 10).unwrap();
        assert_eq!(rels.len(), 1);

        // No events stored
        let events = find_by_type(&conn, SemanticType::Event, 10).unwrap();
        assert!(events.is_empty());

        // Test limit
        let limited = find_by_type(&conn, SemanticType::Fact, 1).unwrap();
        assert_eq!(limited.len(), 1);
        assert_eq!(limited[0].content, "high confidence fact");
    }

    #[test]
    fn test_delete_node_cascades() {
        let conn = open_memory_db().unwrap();
        let id = store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "to delete".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.5,
                source_episodes: vec![],
                embedding: Some(vec![1.0, 0.0, 0.0]),
            },
        )
        .unwrap();

        // Create a link referencing this node
        use crate::graph::links;
        use crate::types::{EpisodeId, LinkType, NodeRef};
        links::create_link(
            &conn,
            NodeRef::Semantic(id),
            NodeRef::Episode(EpisodeId(1)),
            LinkType::Causal,
            0.7,
        )
        .unwrap();

        // Init strength
        crate::store::strengths::init_strength(&conn, NodeRef::Semantic(id)).unwrap();

        // Verify everything exists
        assert_eq!(count_nodes(&conn).unwrap(), 1);
        assert_eq!(
            crate::store::embeddings::count_embeddings(&conn).unwrap(),
            1
        );
        assert_eq!(crate::graph::links::count_links(&conn).unwrap(), 1);

        // Delete
        delete_node(&conn, id).unwrap();

        // Verify cascade
        assert_eq!(count_nodes(&conn).unwrap(), 0);
        assert_eq!(
            crate::store::embeddings::count_embeddings(&conn).unwrap(),
            0
        );
        assert_eq!(crate::graph::links::count_links(&conn).unwrap(), 0);
    }

    #[test]
    fn test_get_semantic_node_not_found() {
        let conn = open_memory_db().unwrap();
        let result = get_semantic_node(&conn, NodeId(999));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::AlayaError::NotFound(_)
        ));
    }

    #[test]
    fn test_count_nodes() {
        let conn = open_memory_db().unwrap();
        assert_eq!(count_nodes(&conn).unwrap(), 0);

        let id = store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "a fact".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.5,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        assert_eq!(count_nodes(&conn).unwrap(), 1);

        delete_node(&conn, id).unwrap();
        assert_eq!(count_nodes(&conn).unwrap(), 0);
    }

    #[test]
    fn test_update_corroboration_not_found() {
        let conn = open_memory_db().unwrap();
        let result = update_corroboration(&conn, NodeId(999));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::AlayaError::NotFound(_)
        ));
    }
}
