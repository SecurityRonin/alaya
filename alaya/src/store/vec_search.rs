//! KNN vector search via the sqlite-vec extension (vec0 virtual table).
//!
//! This module is only compiled when the `vec-sqlite` feature is enabled.

use crate::error::Result;
use rusqlite::Connection;

/// Convert an `&[f32]` slice to its raw byte representation for sqlite-vec.
///
/// sqlite-vec expects embedding vectors as raw little-endian f32 bytes.
fn f32_slice_to_bytes(vec: &[f32]) -> &[u8] {
    // SAFETY: f32 has no alignment/padding concerns when reinterpreted as bytes,
    // and the lifetime of the returned slice is tied to the input slice.
    unsafe { std::slice::from_raw_parts(vec.as_ptr() as *const u8, vec.len() * 4) }
}

/// Register the sqlite-vec extension via `sqlite3_auto_extension`.
///
/// **Must be called once before any `Connection` is opened.** Calling it
/// multiple times is harmless (SQLite deduplicates auto-extensions).
#[allow(dead_code)]
pub fn init_vec_extension() {
    unsafe {
        #[allow(clippy::missing_transmute_annotations)]
        let func = std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ());
        rusqlite::ffi::sqlite3_auto_extension(Some(func));
    }
}

/// Create the `vec_episodes` virtual table with the given embedding dimensions.
///
/// Uses `CREATE VIRTUAL TABLE IF NOT EXISTS` so it is safe to call repeatedly.
#[allow(dead_code)]
pub fn create_vec_table(conn: &Connection, dimensions: usize) -> Result<()> {
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodes USING vec0(\
            episode_id INTEGER PRIMARY KEY, \
            embedding float[{dimensions}] distance_metric=cosine\
        )"
    ))?;
    Ok(())
}

/// Insert or replace a vector in the `vec_episodes` table.
pub fn upsert_vec(conn: &Connection, node_id: i64, embedding: &[f32]) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO vec_episodes(episode_id, embedding) VALUES (?, ?)",
        rusqlite::params![node_id, f32_slice_to_bytes(embedding)],
    )?;
    Ok(())
}

/// Perform a KNN search returning `(episode_id, similarity_score)` pairs.
///
/// The vec0 table returns cosine *distance* (1 - similarity), so we convert
/// back to similarity for the caller.
pub fn knn_search(conn: &Connection, query: &[f32], limit: usize) -> Result<Vec<(i64, f32)>> {
    let mut stmt = conn.prepare(
        "SELECT episode_id, distance FROM vec_episodes \
         WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
    )?;

    let rows = stmt.query_map(
        rusqlite::params![f32_slice_to_bytes(query), limit as i64],
        |row| {
            let id: i64 = row.get(0)?;
            let distance: f64 = row.get(1)?;
            Ok((id, 1.0f32 - distance as f32))
        },
    )?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

/// Delete a vector from the `vec_episodes` table.
#[allow(dead_code)]
pub fn delete_vec(conn: &Connection, node_id: i64) -> Result<()> {
    conn.execute(
        "DELETE FROM vec_episodes WHERE episode_id = ?",
        rusqlite::params![node_id],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: initialise extension, open an in-memory DB, and create the vec table.
    fn setup_vec_db(dimensions: usize) -> Connection {
        init_vec_extension();
        let conn = Connection::open_in_memory().unwrap();
        create_vec_table(&conn, dimensions).unwrap();
        conn
    }

    #[test]
    fn test_vec_table_creation() {
        let _conn = setup_vec_db(3);
        // If we get here without error the table was created successfully.
    }

    #[test]
    fn test_vec_upsert_and_search() {
        let conn = setup_vec_db(3);

        // Three vectors — v1 is closest to the query, v3 is furthest.
        let v1 = [1.0f32, 0.0, 0.0];
        let v2 = [0.7f32, 0.7, 0.0];
        let v3 = [0.0f32, 0.0, 1.0];

        upsert_vec(&conn, 1, &v1).unwrap();
        upsert_vec(&conn, 2, &v2).unwrap();
        upsert_vec(&conn, 3, &v3).unwrap();

        let query = [1.0f32, 0.0, 0.0];
        let results = knn_search(&conn, &query, 10).unwrap();

        assert_eq!(results.len(), 3, "should return all 3 vectors");
        // Nearest first
        assert_eq!(results[0].0, 1, "nearest should be episode 1");
        assert_eq!(results[1].0, 2, "second nearest should be episode 2");
        // Similarities should be descending
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    #[test]
    fn test_vec_search_respects_limit() {
        let conn = setup_vec_db(3);

        for i in 1..=5 {
            upsert_vec(&conn, i, &[1.0, 0.0, (i as f32) * 0.01]).unwrap();
        }

        let results = knn_search(&conn, &[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2, "should respect the limit of 2");
    }

    #[test]
    fn test_vec_delete() {
        let conn = setup_vec_db(3);

        upsert_vec(&conn, 1, &[1.0, 0.0, 0.0]).unwrap();
        upsert_vec(&conn, 2, &[0.0, 1.0, 0.0]).unwrap();

        delete_vec(&conn, 1).unwrap();

        let results = knn_search(&conn, &[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1, "deleted vector should not appear");
        assert_eq!(results[0].0, 2, "only episode 2 should remain");
    }

    #[test]
    fn test_vec_similarity_score() {
        let conn = setup_vec_db(3);

        let v = [1.0f32, 0.0, 0.0];
        upsert_vec(&conn, 1, &v).unwrap();

        let results = knn_search(&conn, &v, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            (results[0].1 - 1.0f32).abs() < 1e-5,
            "identical vector should have similarity ~1.0, got {}",
            results[0].1
        );
    }
}
