use rusqlite::Connection;
use crate::error::Result;
use crate::types::*;
use crate::store::embeddings;

/// Search all embeddings by vector similarity.
pub fn search_vector(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> Result<Vec<(NodeRef, f64)>> {
    let results = embeddings::search_by_vector(conn, query_embedding, None, limit)?;
    Ok(results.into_iter().map(|(nr, sim)| (nr, sim as f64)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;

    #[test]
    fn test_vector_search_empty() {
        let conn = open_memory_db().unwrap();
        let results = search_vector(&conn, &[1.0, 0.0, 0.0], 10).unwrap();
        assert!(results.is_empty());
    }
}
