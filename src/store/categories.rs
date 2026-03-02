use crate::error::{AlayaError, Result};
use crate::store::embeddings::{deserialize_embedding, serialize_embedding};
use crate::types::*;
use rusqlite::{params, Connection};

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

pub fn store_category(
    conn: &Connection,
    label: &str,
    prototype_node: NodeId,
    centroid: Option<&[f32]>,
) -> Result<CategoryId> {
    let ts = now();
    let blob = centroid.map(serialize_embedding);
    conn.execute(
        "INSERT INTO categories (label, prototype_node_id, centroid_embedding, created_at, last_updated)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![label, prototype_node.0, blob, ts, ts],
    )?;
    Ok(CategoryId(conn.last_insert_rowid()))
}

pub fn get_category(conn: &Connection, id: CategoryId) -> Result<Category> {
    conn.query_row(
        "SELECT id, label, prototype_node_id, member_count, centroid_embedding,
                created_at, last_updated, stability
         FROM categories WHERE id = ?1",
        [id.0],
        |row| {
            let blob: Option<Vec<u8>> = row.get(4)?;
            Ok(Category {
                id: CategoryId(row.get(0)?),
                label: row.get(1)?,
                prototype_node: NodeId(row.get(2)?),
                member_count: row.get(3)?,
                centroid_embedding: blob.map(|b| deserialize_embedding(&b)),
                created_at: row.get(5)?,
                last_updated: row.get(6)?,
                stability: row.get(7)?,
            })
        },
    )
    .map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => AlayaError::NotFound(format!("category {}", id.0)),
        other => AlayaError::Db(other),
    })
}

pub fn list_categories(conn: &Connection, min_stability: Option<f32>) -> Result<Vec<Category>> {
    let (sql, has_filter) = match min_stability {
        Some(_) => (
            "SELECT id, label, prototype_node_id, member_count, centroid_embedding,
                    created_at, last_updated, stability
             FROM categories WHERE stability >= ?1
             ORDER BY stability DESC, member_count DESC",
            true,
        ),
        None => (
            "SELECT id, label, prototype_node_id, member_count, centroid_embedding,
                    created_at, last_updated, stability
             FROM categories
             ORDER BY stability DESC, member_count DESC",
            false,
        ),
    };

    let mut stmt = conn.prepare(sql)?;

    let row_mapper = |row: &rusqlite::Row<'_>| {
        let blob: Option<Vec<u8>> = row.get(4)?;
        Ok(Category {
            id: CategoryId(row.get(0)?),
            label: row.get(1)?,
            prototype_node: NodeId(row.get(2)?),
            member_count: row.get(3)?,
            centroid_embedding: blob.map(|b| deserialize_embedding(&b)),
            created_at: row.get(5)?,
            last_updated: row.get(6)?,
            stability: row.get(7)?,
        })
    };

    let rows: Vec<Category> = if has_filter {
        stmt.query_map(params![min_stability.unwrap()], row_mapper)?
            .filter_map(|r| r.ok())
            .collect()
    } else {
        stmt.query_map([], row_mapper)?
            .filter_map(|r| r.ok())
            .collect()
    };

    Ok(rows)
}

pub fn assign_node_to_category(
    conn: &Connection,
    node_id: NodeId,
    category_id: CategoryId,
) -> Result<()> {
    let ts = now();
    conn.execute(
        "UPDATE semantic_nodes SET category_id = ?1 WHERE id = ?2",
        params![category_id.0, node_id.0],
    )?;
    conn.execute(
        "UPDATE categories SET member_count = member_count + 1, last_updated = ?2 WHERE id = ?1",
        params![category_id.0, ts],
    )?;
    Ok(())
}

pub fn get_node_category(conn: &Connection, node_id: NodeId) -> Result<Option<Category>> {
    let cat_id: Option<i64> = conn
        .query_row(
            "SELECT category_id FROM semantic_nodes WHERE id = ?1",
            [node_id.0],
            |row| row.get(0),
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => {
                AlayaError::NotFound(format!("semantic node {}", node_id.0))
            }
            other => AlayaError::Db(other),
        })?;

    match cat_id {
        Some(id) => Ok(Some(get_category(conn, CategoryId(id))?)),
        None => Ok(None),
    }
}

pub fn update_centroid(conn: &Connection, category_id: CategoryId, centroid: &[f32]) -> Result<()> {
    let ts = now();
    let blob = serialize_embedding(centroid);
    conn.execute(
        "UPDATE categories SET centroid_embedding = ?1, last_updated = ?2 WHERE id = ?3",
        params![blob, ts, category_id.0],
    )?;
    Ok(())
}

pub fn increment_stability(conn: &Connection, category_id: CategoryId) -> Result<()> {
    let ts = now();
    conn.execute(
        "UPDATE categories SET stability = stability + 0.1 * (1.0 - stability), last_updated = ?2 WHERE id = ?1",
        params![category_id.0, ts],
    )?;
    Ok(())
}

pub fn delete_category(conn: &Connection, category_id: CategoryId) -> Result<()> {
    conn.execute(
        "UPDATE semantic_nodes SET category_id = NULL WHERE category_id = ?1",
        [category_id.0],
    )?;
    conn.execute("DELETE FROM categories WHERE id = ?1", [category_id.0])?;
    Ok(())
}

pub fn get_uncategorized_node_ids(conn: &Connection) -> Result<Vec<NodeId>> {
    let mut stmt = conn.prepare("SELECT id FROM semantic_nodes WHERE category_id IS NULL")?;
    let rows = stmt.query_map([], |row| Ok(NodeId(row.get(0)?)))?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

#[allow(dead_code)]
pub fn count_categories(conn: &Connection) -> Result<u64> {
    let count: i64 = conn.query_row("SELECT count(*) FROM categories", [], |row| row.get(0))?;
    Ok(count as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;

    /// Helper: insert a bare semantic node and return its rowid.
    fn insert_semantic_node(conn: &Connection) -> NodeId {
        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated)
             VALUES ('test node', 'fact', 0.8, 1000, 1000)",
            [],
        )
        .unwrap();
        NodeId(conn.last_insert_rowid())
    }

    #[test]
    fn test_store_and_get_category() {
        let conn = open_memory_db().unwrap();
        let proto = insert_semantic_node(&conn);

        let centroid = vec![1.0f32, 2.0, 3.0];
        let id = store_category(&conn, "animals", proto, Some(&centroid)).unwrap();

        let cat = get_category(&conn, id).unwrap();
        assert_eq!(cat.label, "animals");
        assert_eq!(cat.prototype_node, proto);
        assert_eq!(cat.member_count, 0);
        assert_eq!(cat.stability, 0.0);
        assert_eq!(cat.centroid_embedding.as_deref(), Some(centroid.as_slice()));
        assert!(cat.created_at > 0);
        assert_eq!(cat.created_at, cat.last_updated);
    }

    #[test]
    fn test_list_categories() {
        let conn = open_memory_db().unwrap();
        let p1 = insert_semantic_node(&conn);
        let p2 = insert_semantic_node(&conn);

        let _id1 = store_category(&conn, "alpha", p1, None).unwrap();
        let id2 = store_category(&conn, "beta", p2, None).unwrap();

        // Bump stability of id2 so it sorts first
        increment_stability(&conn, id2).unwrap();

        let all = list_categories(&conn, None).unwrap();
        assert_eq!(all.len(), 2);
        // beta should come first (higher stability)
        assert_eq!(all[0].label, "beta");

        // With min_stability filter — only beta has stability > 0
        let filtered = list_categories(&conn, Some(0.05)).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, id2);

        // High threshold — nothing
        let empty = list_categories(&conn, Some(0.99)).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_assign_node_to_category() {
        let conn = open_memory_db().unwrap();
        let proto = insert_semantic_node(&conn);
        let node = insert_semantic_node(&conn);
        let cat_id = store_category(&conn, "tools", proto, None).unwrap();

        assign_node_to_category(&conn, node, cat_id).unwrap();

        let cat = get_category(&conn, cat_id).unwrap();
        assert_eq!(cat.member_count, 1);
    }

    #[test]
    fn test_get_node_category() {
        let conn = open_memory_db().unwrap();
        let proto = insert_semantic_node(&conn);
        let node = insert_semantic_node(&conn);
        let cat_id = store_category(&conn, "colors", proto, None).unwrap();

        assign_node_to_category(&conn, node, cat_id).unwrap();

        let result = get_node_category(&conn, node).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, cat_id);
    }

    #[test]
    fn test_get_node_category_none() {
        let conn = open_memory_db().unwrap();
        let node = insert_semantic_node(&conn);

        let result = get_node_category(&conn, node).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_update_centroid() {
        let conn = open_memory_db().unwrap();
        let proto = insert_semantic_node(&conn);
        let id = store_category(&conn, "shapes", proto, None).unwrap();

        // Initially no centroid
        let cat = get_category(&conn, id).unwrap();
        assert!(cat.centroid_embedding.is_none());

        // Update centroid
        let new_centroid = vec![0.5f32, 0.6, 0.7];
        update_centroid(&conn, id, &new_centroid).unwrap();

        let cat = get_category(&conn, id).unwrap();
        assert_eq!(
            cat.centroid_embedding.as_deref(),
            Some(new_centroid.as_slice())
        );
    }

    #[test]
    fn test_increment_stability() {
        let conn = open_memory_db().unwrap();
        let proto = insert_semantic_node(&conn);
        let id = store_category(&conn, "stable", proto, None).unwrap();

        assert_eq!(get_category(&conn, id).unwrap().stability, 0.0);

        increment_stability(&conn, id).unwrap();
        let s = get_category(&conn, id).unwrap().stability;
        assert!(s > 0.0, "stability should have increased");
        // 0.0 + 0.1 * (1.0 - 0.0) = 0.1
        assert!((s - 0.1).abs() < 1e-6);

        increment_stability(&conn, id).unwrap();
        let s2 = get_category(&conn, id).unwrap().stability;
        // 0.1 + 0.1 * (1.0 - 0.1) = 0.1 + 0.09 = 0.19
        assert!((s2 - 0.19).abs() < 1e-5);
    }

    #[test]
    fn test_delete_category() {
        let conn = open_memory_db().unwrap();
        let proto = insert_semantic_node(&conn);
        let node = insert_semantic_node(&conn);
        let cat_id = store_category(&conn, "temp", proto, None).unwrap();
        assign_node_to_category(&conn, node, cat_id).unwrap();

        delete_category(&conn, cat_id).unwrap();

        let all = list_categories(&conn, None).unwrap();
        assert!(all.is_empty());

        // Node should be uncategorized now
        let result = get_node_category(&conn, node).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_uncategorized_nodes() {
        let conn = open_memory_db().unwrap();
        let n1 = insert_semantic_node(&conn);
        let n2 = insert_semantic_node(&conn);

        let uncategorized = get_uncategorized_node_ids(&conn).unwrap();
        assert_eq!(uncategorized.len(), 2);

        // Assign one
        let proto = insert_semantic_node(&conn);
        let cat_id = store_category(&conn, "misc", proto, None).unwrap();
        assign_node_to_category(&conn, n1, cat_id).unwrap();

        let uncategorized = get_uncategorized_node_ids(&conn).unwrap();
        // n2 and proto are uncategorized (proto was also inserted as semantic node)
        assert!(uncategorized.contains(&n2));
        assert!(!uncategorized.contains(&n1));
    }

    #[test]
    fn test_count_categories() {
        let conn = open_memory_db().unwrap();
        assert_eq!(count_categories(&conn).unwrap(), 0);

        let p1 = insert_semantic_node(&conn);
        let p2 = insert_semantic_node(&conn);
        store_category(&conn, "cat1", p1, None).unwrap();
        store_category(&conn, "cat2", p2, None).unwrap();

        assert_eq!(count_categories(&conn).unwrap(), 2);
    }
}
