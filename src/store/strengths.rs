use rusqlite::{params, Connection};
use crate::error::Result;
use crate::types::*;

pub fn init_strength(conn: &Connection, node: NodeRef) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    conn.execute(
        "INSERT OR IGNORE INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength, access_count, last_accessed)
         VALUES (?1, ?2, 0.5, 1.0, 1, ?3)",
        params![node.type_str(), node.id(), now],
    )?;
    Ok(())
}

pub fn get_strength(conn: &Connection, node: NodeRef) -> Result<NodeStrength> {
    let result = conn.query_row(
        "SELECT storage_strength, retrieval_strength, access_count, last_accessed
         FROM node_strengths WHERE node_type = ?1 AND node_id = ?2",
        params![node.type_str(), node.id()],
        |row| {
            Ok(NodeStrength {
                node: node,
                storage_strength: row.get(0)?,
                retrieval_strength: row.get(1)?,
                access_count: row.get(2)?,
                last_accessed: row.get(3)?,
            })
        },
    );
    match result {
        Ok(s) => Ok(s),
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            // Return default strength if not tracked yet
            Ok(NodeStrength {
                node,
                storage_strength: 0.5,
                retrieval_strength: 0.5,
                access_count: 0,
                last_accessed: 0,
            })
        }
        Err(e) => Err(e.into()),
    }
}

pub fn on_access(conn: &Connection, node: NodeRef) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    // Upsert: if exists, update; if not, create
    conn.execute(
        "INSERT INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength, access_count, last_accessed)
         VALUES (?1, ?2, 0.6, 1.0, 1, ?3)
         ON CONFLICT(node_type, node_id) DO UPDATE SET
             storage_strength = MIN(1.0, storage_strength + 0.05 * (1.0 - storage_strength)),
             retrieval_strength = 1.0,
             access_count = access_count + 1,
             last_accessed = ?3",
        params![node.type_str(), node.id(), now],
    )?;
    Ok(())
}

pub fn boost_retrieval(conn: &Connection, node: NodeRef, factor: f32) -> Result<()> {
    conn.execute(
        "UPDATE node_strengths SET retrieval_strength = MIN(1.0, retrieval_strength * ?3)
         WHERE node_type = ?1 AND node_id = ?2",
        params![node.type_str(), node.id(), factor],
    )?;
    Ok(())
}

pub fn suppress_retrieval(conn: &Connection, node: NodeRef, factor: f32) -> Result<()> {
    conn.execute(
        "UPDATE node_strengths SET retrieval_strength = retrieval_strength * ?3
         WHERE node_type = ?1 AND node_id = ?2",
        params![node.type_str(), node.id(), factor],
    )?;
    Ok(())
}

pub fn decay_all_retrieval(conn: &Connection, decay_factor: f32) -> Result<u64> {
    let changed = conn.execute(
        "UPDATE node_strengths SET retrieval_strength = retrieval_strength * ?1
         WHERE retrieval_strength > 0.01",
        [decay_factor],
    )?;
    Ok(changed as u64)
}

pub fn find_archivable(conn: &Connection, storage_thresh: f32, retrieval_thresh: f32) -> Result<Vec<NodeRef>> {
    let mut stmt = conn.prepare(
        "SELECT node_type, node_id FROM node_strengths
         WHERE storage_strength < ?1 AND retrieval_strength < ?2",
    )?;
    let rows = stmt.query_map(params![storage_thresh, retrieval_thresh], |row| {
        let ntype: String = row.get(0)?;
        let nid: i64 = row.get(1)?;
        Ok((ntype, nid))
    })?;
    Ok(rows
        .filter_map(|r| r.ok())
        .filter_map(|(t, id)| NodeRef::from_parts(&t, id))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;

    #[test]
    fn test_init_and_access() {
        let conn = open_memory_db().unwrap();
        let node = NodeRef::Episode(EpisodeId(1));
        init_strength(&conn, node).unwrap();
        let s = get_strength(&conn, node).unwrap();
        assert_eq!(s.access_count, 1);
        assert!((s.retrieval_strength - 1.0).abs() < 0.01);

        on_access(&conn, node).unwrap();
        let s = get_strength(&conn, node).unwrap();
        assert_eq!(s.access_count, 2);
        assert!(s.storage_strength > 0.5);
    }

    #[test]
    fn test_suppress_and_decay() {
        let conn = open_memory_db().unwrap();
        let node = NodeRef::Episode(EpisodeId(1));
        init_strength(&conn, node).unwrap();

        suppress_retrieval(&conn, node, 0.5).unwrap();
        let s = get_strength(&conn, node).unwrap();
        assert!((s.retrieval_strength - 0.5).abs() < 0.01);

        decay_all_retrieval(&conn, 0.9).unwrap();
        let s = get_strength(&conn, node).unwrap();
        assert!(s.retrieval_strength < 0.5);
    }
}
