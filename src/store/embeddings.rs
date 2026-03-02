use crate::error::Result;
use crate::types::*;
use rusqlite::{params, Connection};

pub fn serialize_embedding(vec: &[f32]) -> Vec<u8> {
    vec.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn deserialize_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    (dot / denom) as f32
}

pub fn store_embedding(
    conn: &Connection,
    node_type: &str,
    node_id: i64,
    embedding: &[f32],
    model: &str,
) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let blob = serialize_embedding(embedding);
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (node_type, node_id, embedding, model, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![node_type, node_id, blob, model, now],
    )?;
    Ok(())
}

pub fn get_embedding(conn: &Connection, node_type: &str, node_id: i64) -> Result<Option<Vec<f32>>> {
    let result = conn.query_row(
        "SELECT embedding FROM embeddings WHERE node_type = ?1 AND node_id = ?2",
        params![node_type, node_id],
        |row| {
            let blob: Vec<u8> = row.get(0)?;
            Ok(deserialize_embedding(&blob))
        },
    );
    match result {
        Ok(vec) => Ok(Some(vec)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

#[allow(dead_code)]
pub fn get_unembedded_episodes(conn: &Connection, limit: u32) -> Result<Vec<EpisodeId>> {
    let mut stmt = conn.prepare(
        "SELECT e.id FROM episodes e
         LEFT JOIN embeddings em ON em.node_type = 'episode' AND em.node_id = e.id
         WHERE em.id IS NULL
         ORDER BY e.timestamp ASC LIMIT ?1",
    )?;
    let rows = stmt.query_map([limit], |row| Ok(EpisodeId(row.get(0)?)))?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn search_by_vector(
    conn: &Connection,
    query_vec: &[f32],
    node_type_filter: Option<&str>,
    limit: usize,
) -> Result<Vec<(NodeRef, f32)>> {
    // Collect all candidate embeddings
    let candidates: Vec<(String, i64, Vec<u8>)> = if let Some(t) = node_type_filter {
        let mut stmt = conn
            .prepare("SELECT node_type, node_id, embedding FROM embeddings WHERE node_type = ?1")?;
        let rows = stmt.query_map([t], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Vec<u8>>(2)?,
            ))
        })?;
        rows.filter_map(|r| r.ok()).collect()
    } else {
        let mut stmt = conn.prepare("SELECT node_type, node_id, embedding FROM embeddings")?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Vec<u8>>(2)?,
            ))
        })?;
        rows.filter_map(|r| r.ok()).collect()
    };

    let mut results: Vec<(NodeRef, f32)> = candidates
        .into_iter()
        .filter_map(|(ntype, nid, blob)| {
            let node_ref = NodeRef::from_parts(&ntype, nid)?;
            let emb = deserialize_embedding(&blob);
            let sim = cosine_similarity(query_vec, &emb);
            if sim > 0.0 {
                Some((node_ref, sim))
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    Ok(results)
}

pub fn count_embeddings(conn: &Connection) -> Result<u64> {
    let count: i64 = conn.query_row("SELECT count(*) FROM embeddings", [], |row| row.get(0))?;
    Ok(count as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;

    #[test]
    fn test_serialize_roundtrip() {
        let vec = vec![1.0f32, 2.0, 3.0, -1.5];
        let blob = serialize_embedding(&vec);
        let restored = deserialize_embedding(&blob);
        assert_eq!(vec, restored);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_store_and_search() {
        let conn = open_memory_db().unwrap();
        // Store an episode first so foreign-key-like semantics work
        conn.execute(
            "INSERT INTO episodes (content, role, session_id, timestamp) VALUES ('test', 'user', 's1', 1000)",
            [],
        ).unwrap();

        store_embedding(&conn, "episode", 1, &[1.0, 0.0, 0.0], "test").unwrap();
        store_embedding(&conn, "episode", 2, &[0.9, 0.1, 0.0], "test").unwrap();
        store_embedding(&conn, "episode", 3, &[0.0, 0.0, 1.0], "test").unwrap();

        let results = search_by_vector(&conn, &[1.0, 0.0, 0.0], None, 10).unwrap();
        assert!(results.len() >= 2);
        // First result should be the most similar
        assert_eq!(results[0].0, NodeRef::Episode(EpisodeId(1)));
    }

    #[test]
    fn test_get_embedding_found() {
        let conn = open_memory_db().unwrap();
        store_embedding(&conn, "episode", 1, &[1.0, 2.0, 3.0], "test").unwrap();

        let result = get_embedding(&conn, "episode", 1).unwrap();
        assert!(result.is_some());
        let emb = result.unwrap();
        assert_eq!(emb, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_get_embedding_not_found() {
        let conn = open_memory_db().unwrap();
        let result = get_embedding(&conn, "episode", 999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_unembedded_episodes() {
        let conn = open_memory_db().unwrap();
        // Store 3 episodes
        use crate::store::episodic;
        use crate::types::{EpisodeContext, NewEpisode, Role};
        for i in 1..=3 {
            episodic::store_episode(
                &conn,
                &NewEpisode {
                    content: format!("ep {i}"),
                    role: Role::User,
                    session_id: "s1".to_string(),
                    timestamp: 1000 * i,
                    context: EpisodeContext::default(),
                    embedding: None,
                },
            )
            .unwrap();
        }

        // All 3 should be unembedded
        let unembedded = get_unembedded_episodes(&conn, 10).unwrap();
        assert_eq!(unembedded.len(), 3);

        // Embed episode 1
        store_embedding(&conn, "episode", 1, &[1.0, 0.0], "test").unwrap();

        // Now only 2 should be unembedded
        let unembedded = get_unembedded_episodes(&conn, 10).unwrap();
        assert_eq!(unembedded.len(), 2);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_count_embeddings() {
        let conn = open_memory_db().unwrap();
        assert_eq!(count_embeddings(&conn).unwrap(), 0);
        store_embedding(&conn, "episode", 1, &[1.0], "test").unwrap();
        assert_eq!(count_embeddings(&conn).unwrap(), 1);
    }
}
