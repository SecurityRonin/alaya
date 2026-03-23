use crate::error::Result;
use crate::store::embeddings;
use crate::types::*;
use rusqlite::Connection;

/// Search all embeddings by vector similarity.
///
/// When the `vec-sqlite` feature is enabled, episode embeddings are searched
/// using KNN via the vec0 virtual table (much faster than brute-force at scale).
/// Non-episode node types still fall back to brute-force cosine scan.
pub fn search_vector(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> Result<Vec<(NodeRef, f64)>> {
    #[cfg(feature = "vec-sqlite")]
    {
        // Try KNN for episodes via sqlite-vec. If the vec_episodes table
        // hasn't been created (extension not initialised), fall back to
        // the brute-force path for all node types.
        match crate::store::vec_search::knn_search(conn, query_embedding, limit) {
            Ok(vec_results) => {
                let mut results: Vec<(NodeRef, f64)> = vec_results
                    .into_iter()
                    .map(|(id, sim)| (NodeRef::Episode(EpisodeId(id)), sim as f64))
                    .collect();

                // Also search non-episode embeddings via brute-force and merge
                let brute_results =
                    embeddings::search_by_vector(conn, query_embedding, Some("semantic"), limit)?;
                results.extend(
                    brute_results
                        .into_iter()
                        .map(|(nr, sim)| (nr, sim as f64)),
                );

                // Re-sort by similarity descending and truncate
                results
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                results.truncate(limit);
                return Ok(results);
            }
            Err(_) => {
                // vec_episodes table not available — fall through to brute-force
            }
        }
    }

    {
        let results = embeddings::search_by_vector(conn, query_embedding, None, limit)?;
        Ok(results
            .into_iter()
            .map(|(nr, sim)| (nr, sim as f64))
            .collect())
    }
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
