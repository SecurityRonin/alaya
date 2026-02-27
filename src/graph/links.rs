use rusqlite::{params, Connection};
use crate::error::Result;
use crate::types::*;

pub fn create_link(conn: &Connection, source: NodeRef, target: NodeRef, link_type: LinkType, weight: f32) -> Result<LinkId> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    conn.execute(
        "INSERT OR IGNORE INTO links (source_type, source_id, target_type, target_id, forward_weight, backward_weight, link_type, created_at, last_activated, activation_count)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8, 1)",
        params![
            source.type_str(), source.id(),
            target.type_str(), target.id(),
            weight, weight * 0.5,
            link_type.as_str(), now
        ],
    )?;
    // If it already existed (IGNORE), get its id
    let id: i64 = conn.query_row(
        "SELECT id FROM links WHERE source_type = ?1 AND source_id = ?2 AND target_type = ?3 AND target_id = ?4 AND link_type = ?5",
        params![source.type_str(), source.id(), target.type_str(), target.id(), link_type.as_str()],
        |row| row.get(0),
    )?;
    Ok(LinkId(id))
}

pub fn get_links_from(conn: &Connection, node: NodeRef) -> Result<Vec<Link>> {
    let mut stmt = conn.prepare(
        "SELECT id, source_type, source_id, target_type, target_id,
                forward_weight, backward_weight, link_type,
                created_at, last_activated, activation_count
         FROM links WHERE source_type = ?1 AND source_id = ?2
         ORDER BY forward_weight DESC",
    )?;
    let rows = stmt.query_map(params![node.type_str(), node.id()], map_link)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn get_links_to(conn: &Connection, node: NodeRef) -> Result<Vec<Link>> {
    let mut stmt = conn.prepare(
        "SELECT id, source_type, source_id, target_type, target_id,
                forward_weight, backward_weight, link_type,
                created_at, last_activated, activation_count
         FROM links WHERE target_type = ?1 AND target_id = ?2
         ORDER BY backward_weight DESC",
    )?;
    let rows = stmt.query_map(params![node.type_str(), node.id()], map_link)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Hebbian co-retrieval: strengthen the forward weight when source and target
/// are retrieved together. Asymptotic approach to 1.0.
pub fn on_co_retrieval(conn: &Connection, source: NodeRef, target: NodeRef) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let learning_rate = 0.1;
    // Try to update existing link
    let updated = conn.execute(
        "UPDATE links SET
            forward_weight = forward_weight + ?6 * (1.0 - forward_weight),
            last_activated = ?5,
            activation_count = activation_count + 1
         WHERE source_type = ?1 AND source_id = ?2
           AND target_type = ?3 AND target_id = ?4",
        params![source.type_str(), source.id(), target.type_str(), target.id(), now, learning_rate],
    )?;
    if updated == 0 {
        // Create a new co-retrieval link
        create_link(conn, source, target, LinkType::CoRetrieval, 0.3)?;
    }
    Ok(())
}

pub fn decay_links(conn: &Connection, decay_factor: f32) -> Result<u64> {
    let changed = conn.execute(
        "UPDATE links SET
            forward_weight = forward_weight * ?1,
            backward_weight = backward_weight * ?1
         WHERE forward_weight > 0.01 OR backward_weight > 0.01",
        [decay_factor],
    )?;
    Ok(changed as u64)
}

pub fn prune_weak_links(conn: &Connection, threshold: f32) -> Result<u64> {
    let deleted = conn.execute(
        "DELETE FROM links WHERE forward_weight < ?1 AND backward_weight < ?1",
        [threshold],
    )?;
    Ok(deleted as u64)
}

pub fn count_links(conn: &Connection) -> Result<u64> {
    let count: i64 = conn.query_row("SELECT count(*) FROM links", [], |row| row.get(0))?;
    Ok(count as u64)
}

fn map_link(row: &rusqlite::Row<'_>) -> rusqlite::Result<Link> {
    let source_type: String = row.get(1)?;
    let source_id: i64 = row.get(2)?;
    let target_type: String = row.get(3)?;
    let target_id: i64 = row.get(4)?;
    let link_type_str: String = row.get(7)?;
    Ok(Link {
        id: LinkId(row.get(0)?),
        source: NodeRef::from_parts(&source_type, source_id).unwrap_or(NodeRef::Episode(EpisodeId(0))),
        target: NodeRef::from_parts(&target_type, target_id).unwrap_or(NodeRef::Episode(EpisodeId(0))),
        forward_weight: row.get(5)?,
        backward_weight: row.get(6)?,
        link_type: LinkType::from_str(&link_type_str).unwrap_or(LinkType::CoRetrieval),
        created_at: row.get(8)?,
        last_activated: row.get(9)?,
        activation_count: row.get(10)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;

    #[test]
    fn test_create_and_query_links() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));
        create_link(&conn, a, b, LinkType::Temporal, 0.5).unwrap();

        let from_a = get_links_from(&conn, a).unwrap();
        assert_eq!(from_a.len(), 1);
        assert_eq!(from_a[0].target, b);

        let to_b = get_links_to(&conn, b).unwrap();
        assert_eq!(to_b.len(), 1);
    }

    #[test]
    fn test_co_retrieval_strengthening() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Semantic(NodeId(1));
        create_link(&conn, a, b, LinkType::Topical, 0.3).unwrap();
        let initial = get_links_from(&conn, a).unwrap()[0].forward_weight;

        on_co_retrieval(&conn, a, b).unwrap();
        let after = get_links_from(&conn, a).unwrap()[0].forward_weight;
        assert!(after > initial, "weight should increase after co-retrieval");
    }

    #[test]
    fn test_prune_weak() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));
        create_link(&conn, a, b, LinkType::Temporal, 0.01).unwrap();
        assert_eq!(count_links(&conn).unwrap(), 1);
        prune_weak_links(&conn, 0.05).unwrap();
        assert_eq!(count_links(&conn).unwrap(), 0);
    }

    #[test]
    fn test_decay_links() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));
        create_link(&conn, a, b, LinkType::Temporal, 0.5).unwrap();

        let before = get_links_from(&conn, a).unwrap()[0].forward_weight;
        decay_links(&conn, 0.9).unwrap();
        let after = get_links_from(&conn, a).unwrap()[0].forward_weight;

        assert!(after < before, "weight should decrease after decay");
        assert!((after - before * 0.9).abs() < 0.01);
    }

    #[test]
    fn test_decay_links_skips_very_weak() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));
        create_link(&conn, a, b, LinkType::Temporal, 0.005).unwrap();

        // Link weight is 0.005, which is below 0.01 threshold
        let decayed = decay_links(&conn, 0.9).unwrap();
        // The link has forward=0.005 and backward=0.0025
        // Both are below 0.01, so the WHERE clause (forward > 0.01 OR backward > 0.01) should exclude it
        assert_eq!(decayed, 0);
    }

    #[test]
    fn test_co_retrieval_creates_new_link() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));

        // No existing link between a and b
        assert_eq!(count_links(&conn).unwrap(), 0);

        // Co-retrieval should create a new CoRetrieval link
        on_co_retrieval(&conn, a, b).unwrap();
        assert_eq!(count_links(&conn).unwrap(), 1);

        let links = get_links_from(&conn, a).unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].link_type, LinkType::CoRetrieval);
    }

    #[test]
    fn test_count_links() {
        let conn = open_memory_db().unwrap();
        assert_eq!(count_links(&conn).unwrap(), 0);

        create_link(&conn, NodeRef::Episode(EpisodeId(1)), NodeRef::Episode(EpisodeId(2)), LinkType::Temporal, 0.5).unwrap();
        assert_eq!(count_links(&conn).unwrap(), 1);

        create_link(&conn, NodeRef::Episode(EpisodeId(2)), NodeRef::Episode(EpisodeId(3)), LinkType::Temporal, 0.5).unwrap();
        assert_eq!(count_links(&conn).unwrap(), 2);
    }

    #[test]
    fn test_create_link_duplicate_is_ignored() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));

        let id1 = create_link(&conn, a, b, LinkType::Temporal, 0.5).unwrap();
        let id2 = create_link(&conn, a, b, LinkType::Temporal, 0.8).unwrap();

        // Same link ID (INSERT OR IGNORE)
        assert_eq!(id1, id2);
        assert_eq!(count_links(&conn).unwrap(), 1);

        // Weight should remain original (0.5), not updated to 0.8
        let links = get_links_from(&conn, a).unwrap();
        assert!((links[0].forward_weight - 0.5).abs() < 0.01);
    }
}
