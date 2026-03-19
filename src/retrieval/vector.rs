use crate::error::Result;
use crate::store::embeddings;
use crate::types::*;
use rusqlite::Connection;

/// Search all embeddings by vector similarity.
pub fn search_vector(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> Result<Vec<(NodeRef, f64)>> {
    let results = embeddings::search_by_vector(conn, query_embedding, None, limit)?;
    Ok(results
        .into_iter()
        .map(|(nr, sim)| (nr, sim as f64))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;
    use crate::store::embeddings::store_embedding;

    #[test]
    fn test_vector_search_empty() {
        let conn = open_memory_db().unwrap();
        let results = search_vector(&conn, &[1.0, 0.0, 0.0], 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_vector_search_with_results() {
        let conn = open_memory_db().unwrap();
        // Insert an episode row so the embedding has a valid parent node
        conn.execute(
            "INSERT INTO episodes (content, role, session_id, timestamp) VALUES ('hello', 'user', 's1', 1)",
            [],
        )
        .unwrap();
        store_embedding(&conn, "episode", 1, &[1.0, 0.0, 0.0], "test").unwrap();

        let results = search_vector(&conn, &[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeRef::Episode(EpisodeId(1)));
    }

    #[test]
    fn test_vector_search_f32_to_f64_cast() {
        let conn = open_memory_db().unwrap();
        conn.execute(
            "INSERT INTO episodes (content, role, session_id, timestamp) VALUES ('hello', 'user', 's1', 1)",
            [],
        )
        .unwrap();
        // Identical unit vector → cosine similarity == 1.0
        store_embedding(&conn, "episode", 1, &[1.0, 0.0, 0.0], "test").unwrap();

        let results = search_vector(&conn, &[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1);
        let sim: f64 = results[0].1;
        assert!((sim - 1.0_f64).abs() < 1e-6, "expected ~1.0, got {sim}");
    }

    #[test]
    fn test_vector_search_limit_zero() {
        let conn = open_memory_db().unwrap();
        conn.execute(
            "INSERT INTO episodes (content, role, session_id, timestamp) VALUES ('hello', 'user', 's1', 1)",
            [],
        )
        .unwrap();
        store_embedding(&conn, "episode", 1, &[1.0, 0.0, 0.0], "test").unwrap();

        // limit=0 should return an empty vec
        let results = search_vector(&conn, &[1.0, 0.0, 0.0], 0).unwrap();
        assert!(results.is_empty());
    }
}
