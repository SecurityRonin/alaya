use crate::error::Result;
use crate::types::*;
use rusqlite::{params, Connection};

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

#[allow(dead_code)]
pub fn get_strength(conn: &Connection, node: NodeRef) -> Result<NodeStrength> {
    let result = conn.query_row(
        "SELECT storage_strength, retrieval_strength, access_count, last_accessed
         FROM node_strengths WHERE node_type = ?1 AND node_id = ?2",
        params![node.type_str(), node.id()],
        |row| {
            Ok(NodeStrength {
                node,
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

#[allow(dead_code)]
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

pub fn find_archivable(
    conn: &Connection,
    storage_thresh: f32,
    retrieval_thresh: f32,
) -> Result<Vec<NodeRef>> {
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
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_on_access_ss_bounded(access_count in 1u32..100) {
            let conn = open_memory_db().unwrap();
            let node = NodeRef::Episode(EpisodeId(1));
            init_strength(&conn, node).unwrap();

            for _ in 0..access_count {
                on_access(&conn, node).unwrap();
            }

            let s = get_strength(&conn, node).unwrap();
            prop_assert!(s.storage_strength >= 0.0, "SS below 0: {}", s.storage_strength);
            prop_assert!(s.storage_strength <= 1.0, "SS above 1: {}", s.storage_strength);
            prop_assert!(s.retrieval_strength >= 0.0, "RS below 0: {}", s.retrieval_strength);
            prop_assert!(s.retrieval_strength <= 1.0, "RS above 1: {}", s.retrieval_strength);
        }

        #[test]
        fn prop_suppress_keeps_rs_non_negative(factor in 0.0f32..1.0f32) {
            let conn = open_memory_db().unwrap();
            let node = NodeRef::Episode(EpisodeId(1));
            init_strength(&conn, node).unwrap();

            suppress_retrieval(&conn, node, factor).unwrap();
            let s = get_strength(&conn, node).unwrap();
            prop_assert!(s.retrieval_strength >= 0.0, "RS should be >= 0, got {}", s.retrieval_strength);
        }

        #[test]
        fn prop_decay_all_keeps_rs_non_negative(factor in 0.0f32..1.0f32) {
            let conn = open_memory_db().unwrap();
            let node = NodeRef::Episode(EpisodeId(1));
            init_strength(&conn, node).unwrap();

            decay_all_retrieval(&conn, factor).unwrap();
            let s = get_strength(&conn, node).unwrap();
            prop_assert!(s.retrieval_strength >= 0.0, "RS should be >= 0, got {}", s.retrieval_strength);
        }
    }

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

    #[test]
    fn test_boost_retrieval() {
        let conn = open_memory_db().unwrap();
        let node = NodeRef::Episode(EpisodeId(1));
        init_strength(&conn, node).unwrap();

        // Suppress first to get below 1.0
        suppress_retrieval(&conn, node, 0.5).unwrap();
        let before = get_strength(&conn, node).unwrap();
        assert!((before.retrieval_strength - 0.5).abs() < 0.01);

        // Boost
        boost_retrieval(&conn, node, 1.5).unwrap();
        let after = get_strength(&conn, node).unwrap();
        assert!(after.retrieval_strength > before.retrieval_strength);
        // Should be clamped at 1.0 (MIN(1.0, 0.5 * 1.5) = 0.75)
        assert!((after.retrieval_strength - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_boost_retrieval_clamps_at_one() {
        let conn = open_memory_db().unwrap();
        let node = NodeRef::Episode(EpisodeId(1));
        init_strength(&conn, node).unwrap();

        // Retrieval starts at 1.0, boosting by 2.0 should still clamp to 1.0
        boost_retrieval(&conn, node, 2.0).unwrap();
        let s = get_strength(&conn, node).unwrap();
        assert!((s.retrieval_strength - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_find_archivable() {
        let conn = open_memory_db().unwrap();
        let node1 = NodeRef::Episode(EpisodeId(1));
        let node2 = NodeRef::Episode(EpisodeId(2));

        init_strength(&conn, node1).unwrap();
        init_strength(&conn, node2).unwrap();

        // Suppress node1's retrieval strength dramatically
        suppress_retrieval(&conn, node1, 0.01).unwrap();
        // Also reduce storage strength by direct SQL
        conn.execute(
            "UPDATE node_strengths SET storage_strength = 0.05 WHERE node_id = 1",
            [],
        )
        .unwrap();

        let archivable = find_archivable(&conn, 0.1, 0.05).unwrap();
        // node1 has storage=0.05 < 0.1 AND retrieval=0.01 < 0.05 => archivable
        assert_eq!(archivable.len(), 1);
        assert_eq!(archivable[0], node1);
    }

    #[test]
    fn test_find_archivable_empty() {
        let conn = open_memory_db().unwrap();
        let archivable = find_archivable(&conn, 0.1, 0.05).unwrap();
        assert!(archivable.is_empty());
    }

    #[test]
    fn test_get_strength_default_for_untracked() {
        let conn = open_memory_db().unwrap();
        let node = NodeRef::Episode(EpisodeId(999));
        let s = get_strength(&conn, node).unwrap();
        // Default strength for untracked node
        assert_eq!(s.access_count, 0);
        assert!((s.storage_strength - 0.5).abs() < 0.01);
        assert!((s.retrieval_strength - 0.5).abs() < 0.01);
    }
}
