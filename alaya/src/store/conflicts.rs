#![allow(dead_code)]

use crate::error::Result;
use crate::types::*;
use rusqlite::{params, Connection, OptionalExtension};

/// Insert a new detected conflict. Returns None if the pair already exists.
pub fn insert_conflict(
    conn: &Connection,
    node_a: NodeId,
    node_b: NodeId,
    similarity: f32,
    detected_at: i64,
) -> Result<Option<ConflictId>> {
    let (a, b) = if node_a.0 <= node_b.0 {
        (node_a, node_b)
    } else {
        (node_b, node_a)
    };
    let changed = conn.execute(
        "INSERT OR IGNORE INTO conflicts (node_a_id, node_b_id, similarity, status, detected_at)
         VALUES (?1, ?2, ?3, 'detected', ?4)",
        params![a.0, b.0, similarity, detected_at],
    )?;
    if changed == 0 {
        return Ok(None);
    }
    Ok(Some(ConflictId(conn.last_insert_rowid())))
}

/// Get all conflicts with the given status.
pub fn get_conflicts_by_status(conn: &Connection, status: ConflictStatus) -> Result<Vec<Conflict>> {
    let mut stmt = conn.prepare(
        "SELECT id, node_a_id, node_b_id, similarity, status, detected_at
         FROM conflicts WHERE status = ?1
         ORDER BY detected_at DESC",
    )?;
    let rows = stmt.query_map(params![status.as_str()], |row| {
        let status_str: String = row.get(4)?;
        Ok(Conflict {
            id: ConflictId(row.get(0)?),
            node_a: NodeId(row.get(1)?),
            node_b: NodeId(row.get(2)?),
            similarity: row.get(3)?,
            status: ConflictStatus::from_str(&status_str).unwrap_or(ConflictStatus::Detected),
            detected_at: row.get(5)?,
        })
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

/// Get unresolved conflicts (detected + verified).
pub fn get_unresolved_conflicts(conn: &Connection) -> Result<Vec<Conflict>> {
    let mut stmt = conn.prepare(
        "SELECT id, node_a_id, node_b_id, similarity, status, detected_at
         FROM conflicts WHERE status IN ('detected', 'verified')
         ORDER BY detected_at DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        let status_str: String = row.get(4)?;
        Ok(Conflict {
            id: ConflictId(row.get(0)?),
            node_a: NodeId(row.get(1)?),
            node_b: NodeId(row.get(2)?),
            similarity: row.get(3)?,
            status: ConflictStatus::from_str(&status_str).unwrap_or(ConflictStatus::Detected),
            detected_at: row.get(5)?,
        })
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

/// Update a conflict's status.
pub fn update_conflict_status(
    conn: &Connection,
    id: ConflictId,
    status: ConflictStatus,
) -> Result<()> {
    conn.execute(
        "UPDATE conflicts SET status = ?1 WHERE id = ?2",
        params![status.as_str(), id.0],
    )?;
    Ok(())
}

/// Resolve a conflict: set winner, resolution strategy, timestamp.
pub fn resolve_conflict(
    conn: &Connection,
    id: ConflictId,
    winner_id: NodeId,
    resolution: &str,
    resolved_at: i64,
) -> Result<()> {
    conn.execute(
        "UPDATE conflicts SET status = 'resolved', winner_id = ?1, resolution = ?2, resolved_at = ?3
         WHERE id = ?4",
        params![winner_id.0, resolution, resolved_at, id.0],
    )?;
    Ok(())
}

/// Set superseded_by on a semantic node and zero its confidence.
pub fn supersede_node(conn: &Connection, loser: NodeId, winner: NodeId) -> Result<()> {
    conn.execute(
        "UPDATE semantic_nodes SET superseded_by = ?1, confidence = 0.0 WHERE id = ?2",
        params![winner.0, loser.0],
    )?;
    Ok(())
}

/// Check if a pair already exists in the conflicts table.
pub fn conflict_exists(conn: &Connection, node_a: NodeId, node_b: NodeId) -> Result<bool> {
    let (a, b) = if node_a.0 <= node_b.0 {
        (node_a, node_b)
    } else {
        (node_b, node_a)
    };
    let exists: bool = conn
        .query_row(
            "SELECT 1 FROM conflicts WHERE node_a_id = ?1 AND node_b_id = ?2",
            params![a.0, b.0],
            |_| Ok(true),
        )
        .optional()?
        .unwrap_or(false);
    Ok(exists)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;
    use crate::store::semantic::store_semantic_node;

    fn setup() -> (Connection, NodeId, NodeId) {
        let conn = open_memory_db().unwrap();
        let a = store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "user prefers dark mode".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        let b = store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "user prefers light mode".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        (conn, a, b)
    }

    #[test]
    fn insert_and_get_conflict() {
        let (conn, a, b) = setup();
        let id = insert_conflict(&conn, a, b, 0.92, 1000)
            .unwrap()
            .expect("should insert");
        let conflicts = get_conflicts_by_status(&conn, ConflictStatus::Detected).unwrap();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].id, id);
        assert!((conflicts[0].similarity - 0.92).abs() < 1e-6);
    }

    #[test]
    fn insert_duplicate_returns_none() {
        let (conn, a, b) = setup();
        insert_conflict(&conn, a, b, 0.92, 1000).unwrap();
        let dup = insert_conflict(&conn, a, b, 0.95, 2000).unwrap();
        assert!(dup.is_none());
    }

    #[test]
    fn insert_normalizes_order() {
        let (conn, a, b) = setup();
        insert_conflict(&conn, b, a, 0.92, 1000).unwrap();
        let dup = insert_conflict(&conn, a, b, 0.95, 2000).unwrap();
        assert!(dup.is_none());
    }

    #[test]
    fn get_unresolved_includes_detected_and_verified() {
        let (conn, a, b) = setup();
        let id = insert_conflict(&conn, a, b, 0.92, 1000).unwrap().unwrap();

        let unresolved = get_unresolved_conflicts(&conn).unwrap();
        assert_eq!(unresolved.len(), 1);

        update_conflict_status(&conn, id, ConflictStatus::Verified).unwrap();
        let unresolved = get_unresolved_conflicts(&conn).unwrap();
        assert_eq!(unresolved.len(), 1);

        resolve_conflict(&conn, id, a, "recency", 2000).unwrap();
        let unresolved = get_unresolved_conflicts(&conn).unwrap();
        assert_eq!(unresolved.len(), 0);
    }

    #[test]
    fn resolve_conflict_sets_fields() {
        let (conn, a, b) = setup();
        let id = insert_conflict(&conn, a, b, 0.92, 1000).unwrap().unwrap();
        resolve_conflict(&conn, id, a, "confidence", 2000).unwrap();

        let resolved = get_conflicts_by_status(&conn, ConflictStatus::Resolved).unwrap();
        assert_eq!(resolved.len(), 1);
    }

    #[test]
    fn supersede_node_zeros_confidence() {
        let (conn, a, b) = setup();
        supersede_node(&conn, b, a).unwrap();

        let node_b = crate::store::semantic::get_semantic_node(&conn, b).unwrap();
        assert_eq!(node_b.confidence, 0.0);
    }

    #[test]
    fn conflict_exists_check() {
        let (conn, a, b) = setup();
        assert!(!conflict_exists(&conn, a, b).unwrap());
        insert_conflict(&conn, a, b, 0.92, 1000).unwrap();
        assert!(conflict_exists(&conn, a, b).unwrap());
        assert!(conflict_exists(&conn, b, a).unwrap());
    }

    #[test]
    fn dismissed_not_in_unresolved() {
        let (conn, a, b) = setup();
        let id = insert_conflict(&conn, a, b, 0.92, 1000).unwrap().unwrap();
        update_conflict_status(&conn, id, ConflictStatus::Dismissed).unwrap();
        let unresolved = get_unresolved_conflicts(&conn).unwrap();
        assert_eq!(unresolved.len(), 0);
    }
}
