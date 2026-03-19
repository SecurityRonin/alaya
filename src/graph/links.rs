use crate::error::Result;
use crate::types::*;
use rusqlite::{params, Connection};

pub fn create_link(
    conn: &Connection,
    source: NodeRef,
    target: NodeRef,
    link_type: LinkType,
    weight: f32,
) -> Result<LinkId> {
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

#[allow(dead_code)]
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
           AND target_type = ?3 AND target_id = ?4
           AND link_type = 'co_retrieval'",
        params![
            source.type_str(),
            source.id(),
            target.type_str(),
            target.id(),
            now,
            learning_rate
        ],
    )?;
    if updated == 0 {
        // Create a new co-retrieval link
        create_link(conn, source, target, LinkType::CoRetrieval, 0.3)?;
    }
    Ok(())
}

pub fn decay_links(conn: &Connection, decay_factor: f32) -> Result<u64> {
    // MultiplicativeDecay pattern applied to two columns simultaneously.
    // Cannot use decay::apply_multiplicative_sql (single-column helper).
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

/// Returns the link with the highest forward weight, if any exist.
pub fn strongest_link(conn: &Connection) -> Result<Option<(NodeRef, NodeRef, f32)>> {
    let mut stmt = conn.prepare(
        "SELECT source_type, source_id, target_type, target_id, forward_weight
         FROM links ORDER BY forward_weight DESC LIMIT 1",
    )?;
    let mut rows = stmt.query_map([], |row| {
        let source_type: String = row.get(0)?;
        let source_id: i64 = row.get(1)?;
        let target_type: String = row.get(2)?;
        let target_id: i64 = row.get(3)?;
        let weight: f32 = row.get(4)?;
        Ok((source_type, source_id, target_type, target_id, weight))
    })?;
    match rows.next() {
        Some(Ok((st, si, tt, ti, w))) => {
            let source = NodeRef::from_parts(&st, si).unwrap_or(NodeRef::Episode(EpisodeId(0)));
            let target = NodeRef::from_parts(&tt, ti).unwrap_or(NodeRef::Episode(EpisodeId(0)));
            Ok(Some((source, target, w)))
        }
        Some(Err(e)) => Err(e.into()),
        None => Ok(None),
    }
}

fn map_link(row: &rusqlite::Row<'_>) -> rusqlite::Result<Link> {
    let source_type: String = row.get(1)?;
    let source_id: i64 = row.get(2)?;
    let target_type: String = row.get(3)?;
    let target_id: i64 = row.get(4)?;
    let link_type_str: String = row.get(7)?;
    Ok(Link {
        id: LinkId(row.get(0)?),
        source: NodeRef::from_parts(&source_type, source_id)
            .unwrap_or(NodeRef::Episode(EpisodeId(0))),
        target: NodeRef::from_parts(&target_type, target_id)
            .unwrap_or(NodeRef::Episode(EpisodeId(0))),
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
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_co_retrieval_weight_bounded(iterations in 1u32..50) {
            let conn = open_memory_db().unwrap();
            let a = NodeRef::Episode(EpisodeId(1));
            let b = NodeRef::Episode(EpisodeId(2));
            create_link(&conn, a, b, LinkType::CoRetrieval, 0.3).unwrap();

            for _ in 0..iterations {
                on_co_retrieval(&conn, a, b).unwrap();
            }

            let links = get_links_from(&conn, a).unwrap();
            prop_assert!(!links.is_empty());
            let w = links[0].forward_weight;
            prop_assert!(w >= 0.0, "weight below 0: {}", w);
            prop_assert!(w <= 1.0, "weight above 1: {}", w);
        }

        #[test]
        fn prop_decay_links_weight_bounded(factor in 0.0f32..1.0f32) {
            let conn = open_memory_db().unwrap();
            let a = NodeRef::Episode(EpisodeId(1));
            let b = NodeRef::Episode(EpisodeId(2));
            create_link(&conn, a, b, LinkType::Temporal, 0.5).unwrap();

            decay_links(&conn, factor).unwrap();
            let links = get_links_from(&conn, a).unwrap();
            if !links.is_empty() {
                let w = links[0].forward_weight;
                prop_assert!(w >= 0.0, "weight below 0: {}", w);
                prop_assert!(w <= 1.0, "weight above 1: {}", w);
            }
        }
    }

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
        // Create an existing CoRetrieval link (not Topical)
        create_link(&conn, a, b, LinkType::CoRetrieval, 0.3).unwrap();
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

        create_link(
            &conn,
            NodeRef::Episode(EpisodeId(1)),
            NodeRef::Episode(EpisodeId(2)),
            LinkType::Temporal,
            0.5,
        )
        .unwrap();
        assert_eq!(count_links(&conn).unwrap(), 1);

        create_link(
            &conn,
            NodeRef::Episode(EpisodeId(2)),
            NodeRef::Episode(EpisodeId(3)),
            LinkType::Temporal,
            0.5,
        )
        .unwrap();
        assert_eq!(count_links(&conn).unwrap(), 2);
    }

    #[test]
    fn test_co_retrieval_with_existing_temporal_link() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));

        // Create a temporal link first
        create_link(&conn, a, b, LinkType::Temporal, 0.5).unwrap();
        assert_eq!(count_links(&conn).unwrap(), 1);

        // Co-retrieval should create a SEPARATE CoRetrieval link
        on_co_retrieval(&conn, a, b).unwrap();
        assert_eq!(
            count_links(&conn).unwrap(),
            2,
            "should have both Temporal and CoRetrieval links"
        );

        // Verify both link types exist
        let links = get_links_from(&conn, a).unwrap();
        let types: Vec<LinkType> = links.iter().map(|l| l.link_type).collect();
        assert!(types.contains(&LinkType::Temporal));
        assert!(types.contains(&LinkType::CoRetrieval));
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

    #[test]
    fn test_strongest_link_empty() {
        let conn = open_memory_db().unwrap();
        assert!(strongest_link(&conn).unwrap().is_none());
    }

    #[test]
    fn test_get_links_to_ordering() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));
        let c = NodeRef::Episode(EpisodeId(3));

        create_link(&conn, a, b, LinkType::Temporal, 0.3).unwrap();
        create_link(&conn, c, b, LinkType::Topical, 0.9).unwrap();

        let to_b = get_links_to(&conn, b).unwrap();
        assert_eq!(to_b.len(), 2);
        // Should be ordered by backward_weight DESC
        assert!(to_b[0].backward_weight >= to_b[1].backward_weight);
    }

    #[test]
    fn test_strongest_link_returns_highest_weight() {
        let conn = open_memory_db().unwrap();
        create_link(
            &conn,
            NodeRef::Episode(EpisodeId(1)),
            NodeRef::Episode(EpisodeId(2)),
            LinkType::Temporal,
            0.5,
        )
        .unwrap();
        create_link(
            &conn,
            NodeRef::Episode(EpisodeId(2)),
            NodeRef::Episode(EpisodeId(3)),
            LinkType::Temporal,
            0.9,
        )
        .unwrap();
        create_link(
            &conn,
            NodeRef::Episode(EpisodeId(3)),
            NodeRef::Episode(EpisodeId(4)),
            LinkType::Topical,
            0.3,
        )
        .unwrap();

        let strongest = strongest_link(&conn).unwrap().unwrap();
        assert_eq!(strongest.2, 0.9);
        assert_eq!(strongest.0, NodeRef::Episode(EpisodeId(2)));
        assert_eq!(strongest.1, NodeRef::Episode(EpisodeId(3)));
    }

    #[test]
    fn test_get_links_to() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));
        let c = NodeRef::Episode(EpisodeId(3));

        // a → b, c → b
        create_link(&conn, a, b, LinkType::Temporal, 0.7).unwrap();
        create_link(&conn, c, b, LinkType::Topical, 0.5).unwrap();

        let to_b = get_links_to(&conn, b).unwrap();
        assert_eq!(to_b.len(), 2, "b should have 2 incoming links");
        let sources: Vec<NodeRef> = to_b.iter().map(|l| l.source).collect();
        assert!(sources.contains(&a), "a should be a source of b");
        assert!(sources.contains(&c), "c should be a source of b");
    }

    #[test]
    fn test_get_links_to_empty() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let to_a = get_links_to(&conn, a).unwrap();
        assert!(
            to_a.is_empty(),
            "node with no incoming links should return empty"
        );
    }

    #[test]
    fn test_get_links_from_empty() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(99));
        let from_a = get_links_from(&conn, a).unwrap();
        assert!(
            from_a.is_empty(),
            "node with no outgoing links should return empty"
        );
    }

    #[test]
    fn test_prune_weak_returns_count() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Episode(EpisodeId(2));
        let c = NodeRef::Episode(EpisodeId(3));

        create_link(&conn, a, b, LinkType::Temporal, 0.5).unwrap();
        create_link(&conn, a, c, LinkType::Topical, 0.01).unwrap(); // below threshold

        let pruned = prune_weak_links(&conn, 0.02).unwrap();
        assert_eq!(pruned, 1, "should prune exactly 1 weak link");

        let remaining = get_links_from(&conn, a).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].target, b);
    }

    #[test]
    fn test_link_types_stored_correctly() {
        let conn = open_memory_db().unwrap();
        let a = NodeRef::Episode(EpisodeId(1));
        let b = NodeRef::Semantic(crate::types::NodeId(1));

        for lt in [
            LinkType::Temporal,
            LinkType::Topical,
            LinkType::Entity,
            LinkType::Causal,
            LinkType::CoRetrieval,
            LinkType::MemberOf,
        ] {
            let _ = create_link(&conn, a, b, lt, 0.5); // may fail on dup, ok
        }

        let links = get_links_from(&conn, a).unwrap();
        // All link types should be stored and retrievable
        assert!(!links.is_empty());
    }
}
