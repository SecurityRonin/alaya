use rusqlite::{params, Connection};
use crate::error::{AlayaError, Result};
use crate::types::*;

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
                node_type: SemanticType::from_str(&row.get::<_, String>(2)?).unwrap_or(SemanticType::Fact),
                confidence: row.get(3)?,
                source_episodes: serde_json::from_str(&sources_str).unwrap_or_default(),
                created_at: row.get(5)?,
                last_corroborated: row.get(6)?,
                corroboration_count: row.get(7)?,
            })
        },
    )
    .map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => AlayaError::NotFound(format!("semantic node {}", id.0)),
        other => AlayaError::Db(other),
    })
}

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

pub fn find_by_type(conn: &Connection, node_type: SemanticType, limit: u32) -> Result<Vec<SemanticNode>> {
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
            node_type: SemanticType::from_str(&row.get::<_, String>(2)?).unwrap_or(SemanticType::Fact),
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
    conn.execute("DELETE FROM embeddings WHERE node_type = 'semantic' AND node_id = ?1", [id.0])?;
    conn.execute("DELETE FROM links WHERE (source_type = 'semantic' AND source_id = ?1) OR (target_type = 'semantic' AND target_id = ?1)", [id.0])?;
    conn.execute("DELETE FROM node_strengths WHERE node_type = 'semantic' AND node_id = ?1", [id.0])?;
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
        let id = store_semantic_node(&conn, &NewSemanticNode {
            content: "User is a Rust developer".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.8,
            source_episodes: vec![EpisodeId(1), EpisodeId(2)],
            embedding: None,
        }).unwrap();
        let node = get_semantic_node(&conn, id).unwrap();
        assert_eq!(node.content, "User is a Rust developer");
        assert_eq!(node.confidence, 0.8);
        assert_eq!(node.source_episodes.len(), 2);
    }

    #[test]
    fn test_corroboration() {
        let conn = open_memory_db().unwrap();
        let id = store_semantic_node(&conn, &NewSemanticNode {
            content: "fact".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.5,
            source_episodes: vec![],
            embedding: None,
        }).unwrap();
        update_corroboration(&conn, id).unwrap();
        let node = get_semantic_node(&conn, id).unwrap();
        assert_eq!(node.corroboration_count, 2);
    }
}
