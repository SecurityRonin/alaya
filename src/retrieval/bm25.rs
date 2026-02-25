use rusqlite::{Connection, params};
use crate::error::Result;
use crate::types::*;

/// Search episodes via FTS5 BM25 ranking.
/// Returns (EpisodeId, normalized_score) where score is in [0.0, 1.0].
pub fn search_bm25(conn: &Connection, query: &str, limit: usize) -> Result<Vec<(EpisodeId, f64)>> {
    if query.trim().is_empty() {
        return Ok(vec![]);
    }

    // Sanitize query for FTS5: remove special characters that FTS5 interprets
    let sanitized: String = query
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect();

    if sanitized.trim().is_empty() {
        return Ok(vec![]);
    }

    let fetch_limit = (limit * 3) as u32;
    let mut stmt = conn.prepare(
        "SELECT e.id, rank
         FROM episodes_fts fts
         JOIN episodes e ON e.id = fts.rowid
         WHERE episodes_fts MATCH ?1
         ORDER BY rank
         LIMIT ?2",
    )?;

    let rows: Vec<(i64, f64)> = stmt
        .query_map(params![sanitized.trim(), fetch_limit], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    if rows.is_empty() {
        return Ok(vec![]);
    }

    // Normalize FTS5 ranks (negative values, lower = better) to [0, 1]
    let min_rank = rows.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
    let max_rank = rows.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max);
    let range = max_rank - min_rank;

    let mut results: Vec<(EpisodeId, f64)> = rows
        .into_iter()
        .map(|(id, rank)| {
            let normalized = if range.abs() < 1e-10 {
                1.0
            } else {
                1.0 - ((rank - min_rank) / range)
            };
            (EpisodeId(id), normalized)
        })
        .collect();

    results.truncate(limit);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;
    use crate::store::episodic;
    use crate::types::*;

    #[test]
    fn test_bm25_search() {
        let conn = open_memory_db().unwrap();

        episodic::store_episode(&conn, &NewEpisode {
            content: "I love programming in Rust".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 1000,
            context: EpisodeContext::default(),
            embedding: None,
        }).unwrap();

        episodic::store_episode(&conn, &NewEpisode {
            content: "Python is great for data science".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 2000,
            context: EpisodeContext::default(),
            embedding: None,
        }).unwrap();

        let results = search_bm25(&conn, "Rust programming", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, EpisodeId(1));
    }

    #[test]
    fn test_empty_query() {
        let conn = open_memory_db().unwrap();
        let results = search_bm25(&conn, "", 10).unwrap();
        assert!(results.is_empty());
    }
}
