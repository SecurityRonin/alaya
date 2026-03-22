use crate::error::Result;
use crate::types::*;
use rusqlite::{params, Connection, OptionalExtension};

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
    conn.query_row(
        "SELECT embedding FROM embeddings WHERE node_type = ?1 AND node_id = ?2",
        params![node_type, node_id],
        |row| {
            let blob: Vec<u8> = row.get(0)?;
            Ok(deserialize_embedding(&blob))
        },
    )
    .optional()
    .map_err(|e| e.into())
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
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_cosine_similarity_bounded(
            a in proptest::collection::vec(-10.0f32..10.0f32, 3..8),
            b in proptest::collection::vec(-10.0f32..10.0f32, 3..8),
        ) {
            if a.len() == b.len() {
                let sim = cosine_similarity(&a, &b);
                prop_assert!(sim >= -1.0 - f32::EPSILON, "cosine sim {} below -1.0", sim);
                prop_assert!(sim <= 1.0 + f32::EPSILON, "cosine sim {} above 1.0", sim);
            }
        }

        #[test]
        fn prop_cosine_self_similarity_is_one(
            a in proptest::collection::vec(0.1f32..10.0f32, 3..8),
        ) {
            let sim = cosine_similarity(&a, &a);
            prop_assert!((sim - 1.0).abs() < 0.001, "self-similarity should be ~1.0, got {}", sim);
        }

        #[test]
        fn prop_cosine_different_lengths_returns_zero(
            a in proptest::collection::vec(-10.0f32..10.0f32, 3..5),
            b in proptest::collection::vec(-10.0f32..10.0f32, 6..8),
        ) {
            let sim = cosine_similarity(&a, &b);
            prop_assert!((sim - 0.0).abs() < f32::EPSILON, "different lengths should return 0.0");
        }

        #[test]
        fn prop_cosine_zero_vector_returns_zero(
            a in proptest::collection::vec(-10.0f32..10.0f32, 3..8),
        ) {
            let zeros = vec![0.0f32; a.len()];
            let sim = cosine_similarity(&a, &zeros);
            prop_assert!((sim - 0.0).abs() < f32::EPSILON, "zero vector should return 0.0");
        }
    }

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

    #[test]
    fn test_search_by_vector_with_type_filter() {
        let conn = open_memory_db().unwrap();
        store_embedding(&conn, "episode", 1, &[1.0, 0.0, 0.0], "test").unwrap();
        store_embedding(&conn, "semantic", 1, &[0.9, 0.1, 0.0], "test").unwrap();

        // Filter by "episode" — should only return the episode embedding
        let results = search_by_vector(&conn, &[1.0, 0.0, 0.0], Some("episode"), 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0].0, NodeRef::Episode(_)));

        // Filter by "semantic" — should only return the semantic embedding
        let results = search_by_vector(&conn, &[1.0, 0.0, 0.0], Some("semantic"), 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0].0, NodeRef::Semantic(_)));
    }

    #[test]
    fn test_search_by_vector_filters_non_positive_similarity() {
        let conn = open_memory_db().unwrap();
        // Store an embedding orthogonal to our query
        store_embedding(&conn, "episode", 1, &[0.0, 1.0, 0.0], "test").unwrap();
        // Store one opposite to our query
        store_embedding(&conn, "episode", 2, &[-1.0, 0.0, 0.0], "test").unwrap();

        // Query with [1, 0, 0] — orthogonal gives sim=0, opposite gives sim<0
        let results = search_by_vector(&conn, &[1.0, 0.0, 0.0], None, 10).unwrap();
        // Neither should appear (sim <= 0 filtered out)
        assert!(
            results.is_empty(),
            "non-positive similarities should be filtered out"
        );
    }

    #[test]
    fn test_search_by_vector_truncates_to_limit() {
        let conn = open_memory_db().unwrap();
        for i in 1..=5 {
            store_embedding(&conn, "episode", i, &[1.0, 0.0, (i as f32) * 0.01], "test").unwrap();
        }
        let results = search_by_vector(&conn, &[1.0, 0.0, 0.0], None, 2).unwrap();
        assert_eq!(results.len(), 2, "should truncate to limit");
    }

    // --- Additional coverage tests ---

    #[test]
    fn test_serialize_empty_vector() {
        let blob = serialize_embedding(&[]);
        assert!(
            blob.is_empty(),
            "serializing empty slice produces empty bytes"
        );
        let restored = deserialize_embedding(&blob);
        assert!(restored.is_empty());
    }

    #[test]
    fn test_serialize_single_element() {
        let vec = vec![std::f32::consts::PI];
        let blob = serialize_embedding(&vec);
        assert_eq!(blob.len(), 4, "single f32 should be 4 bytes");
        let restored = deserialize_embedding(&blob);
        assert_eq!(restored.len(), 1);
        assert!((restored[0] - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn test_serialize_large_vector_roundtrip() {
        let vec: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let blob = serialize_embedding(&vec);
        assert_eq!(blob.len(), 1024 * 4);
        let restored = deserialize_embedding(&blob);
        assert_eq!(restored.len(), vec.len());
        for (a, b) in vec.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-7, "value mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_deserialize_truncated_bytes_drops_partial_chunk() {
        // 9 bytes: 2 complete f32s (8 bytes) + 1 trailing byte (dropped by chunks_exact)
        let mut blob = serialize_embedding(&[1.0f32, 2.0f32]);
        blob.push(0xFF); // trailing garbage byte
        let restored = deserialize_embedding(&blob);
        assert_eq!(
            restored.len(),
            2,
            "partial trailing chunk should be silently dropped"
        );
        assert!((restored[0] - 1.0f32).abs() < 1e-7);
        assert!((restored[1] - 2.0f32).abs() < 1e-7);
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        // Opposite vectors → cosine similarity of -1.0
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![-1.0f32, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - (-1.0f32)).abs() < 1e-6,
            "opposite vectors should give -1.0, got {sim}"
        );
    }

    #[test]
    fn test_store_embedding_overwrite_keeps_count_at_one() {
        // INSERT OR REPLACE: storing same (node_type, node_id) twice must not duplicate the row
        let conn = open_memory_db().unwrap();
        store_embedding(&conn, "episode", 42, &[1.0, 0.0], "modelA").unwrap();
        assert_eq!(count_embeddings(&conn).unwrap(), 1);

        store_embedding(&conn, "episode", 42, &[0.0, 1.0], "modelB").unwrap();
        assert_eq!(
            count_embeddings(&conn).unwrap(),
            1,
            "overwrite should not duplicate the row"
        );

        // Value should be updated to the new embedding
        let emb = get_embedding(&conn, "episode", 42).unwrap().unwrap();
        assert_eq!(
            emb,
            vec![0.0f32, 1.0f32],
            "stored value should reflect the latest write"
        );
    }

    #[test]
    fn test_search_by_vector_empty_store_no_filter() {
        // None filter path with zero rows in the table
        let conn = open_memory_db().unwrap();
        let results = search_by_vector(&conn, &[1.0, 0.0, 0.0], None, 10).unwrap();
        assert!(results.is_empty(), "empty store should return no results");
    }

    #[test]
    fn test_search_by_vector_unknown_node_type_filtered_out() {
        // Exercises the filter_map None branch in NodeRef::from_parts.
        // Bypass store_embedding to insert a row with an unrecognised node_type directly.
        let conn = open_memory_db().unwrap();
        let blob = serialize_embedding(&[1.0f32, 0.0, 0.0]);
        conn.execute(
            "INSERT INTO embeddings (node_type, node_id, embedding, model, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params!["unknown_type", 1i64, blob, "test", 0i64],
        )
        .unwrap();

        // search_by_vector should silently skip rows where NodeRef::from_parts returns None
        let results = search_by_vector(&conn, &[1.0, 0.0, 0.0], None, 10).unwrap();
        assert!(
            results.is_empty(),
            "rows with unrecognised node_type should be filtered out by NodeRef::from_parts"
        );
    }

    #[test]
    fn test_get_unembedded_episodes_respects_limit() {
        use crate::store::episodic;
        use crate::types::{EpisodeContext, NewEpisode, Role};

        let conn = open_memory_db().unwrap();
        // Insert 5 unembedded episodes
        for i in 1..=5 {
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

        // Request only 3 — should honour the limit
        let unembedded = get_unembedded_episodes(&conn, 3).unwrap();
        assert_eq!(
            unembedded.len(),
            3,
            "get_unembedded_episodes should respect the limit parameter"
        );
    }

    #[test]
    fn test_search_by_vector_results_sorted_descending() {
        // Verify that results are returned in descending similarity order
        let conn = open_memory_db().unwrap();
        // Episode 1: perfectly aligned with query → sim ≈ 1.0
        store_embedding(&conn, "episode", 1, &[1.0, 0.0, 0.0], "test").unwrap();
        // Episode 2: slightly off → lower sim
        store_embedding(&conn, "episode", 2, &[0.7, 0.7, 0.0], "test").unwrap();
        // Episode 3: even less aligned
        store_embedding(&conn, "episode", 3, &[0.1, 0.99, 0.0], "test").unwrap();

        let results = search_by_vector(&conn, &[1.0, 0.0, 0.0], None, 10).unwrap();
        assert!(results.len() >= 2, "expected at least 2 results");
        for i in 0..results.len() - 1 {
            assert!(
                results[i].1 >= results[i + 1].1,
                "results should be sorted descending by similarity: {} < {}",
                results[i].1,
                results[i + 1].1
            );
        }
    }
}
