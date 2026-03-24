use crate::error::Result;
use crate::types::*;
use rusqlite::Connection;

/// Search episodes via FTS5 BM25 ranking.
/// Returns (EpisodeId, normalized_score) where score is in [0.0, 1.0].
///
/// Applies temporal filters (`after_timestamp`, `before_timestamp`) and
/// session scoping (`session_filter`) from the [`QueryContext`] when present.
pub fn search_bm25(
    conn: &Connection,
    query: &str,
    limit: usize,
    context: &QueryContext,
) -> Result<Vec<(EpisodeId, f64)>> {
    if query.trim().is_empty() {
        return Ok(vec![]);
    }

    // Sanitize query for FTS5: remove special characters that FTS5 interprets
    let sanitized: String = query
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();

    if sanitized.trim().is_empty() {
        return Ok(vec![]);
    }

    let fetch_limit = (limit * 3) as u32;

    // Build dynamic WHERE clause for temporal/session filters
    let mut extra_clauses = String::new();
    let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    // ?1 = sanitized query, ?2 = fetch_limit
    param_values.push(Box::new(sanitized.trim().to_string()));
    param_values.push(Box::new(fetch_limit));

    let mut param_idx = 3;

    if let Some(after) = context.after_timestamp {
        extra_clauses.push_str(&format!(" AND e.timestamp >= ?{param_idx}"));
        param_values.push(Box::new(after));
        param_idx += 1;
    }

    if let Some(before) = context.before_timestamp {
        extra_clauses.push_str(&format!(" AND e.timestamp <= ?{param_idx}"));
        param_values.push(Box::new(before));
        param_idx += 1;
    }

    if let Some(ref session) = context.session_filter {
        extra_clauses.push_str(&format!(" AND e.session_id = ?{param_idx}"));
        param_values.push(Box::new(session.clone()));
        // param_idx += 1; // last param, no need to increment
        let _ = param_idx; // suppress unused warning
    }

    let sql = format!(
        "SELECT e.id, rank
         FROM episodes_fts fts
         JOIN episodes e ON e.id = fts.rowid
         WHERE episodes_fts MATCH ?1{extra_clauses}
         ORDER BY rank
         LIMIT ?2"
    );

    let mut stmt = conn.prepare(&sql)?;

    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        param_values.iter().map(|p| p.as_ref()).collect();

    let rows: Vec<(i64, f64)> = stmt
        .query_map(param_refs.as_slice(), |row| Ok((row.get(0)?, row.get(1)?)))?
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
    #[test]
    fn test_bm25_search() {
        let conn = open_memory_db().unwrap();

        episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "I love programming in Rust".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();

        episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "Python is great for data science".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 2000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();

        let results = search_bm25(&conn, "Rust programming", 10, &QueryContext::default()).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, EpisodeId(1));
    }

    #[test]
    fn test_empty_query() {
        let conn = open_memory_db().unwrap();
        let results = search_bm25(&conn, "", 10, &QueryContext::default()).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_special_chars_only_query() {
        let conn = open_memory_db().unwrap();
        // A query of only special characters should sanitize to empty and return empty
        let results = search_bm25(&conn, "!@#$%^&*()", 10, &QueryContext::default()).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_single_result_normalization() {
        let conn = open_memory_db().unwrap();
        episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "unique frobnicator keyword".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();

        // Single result means min_rank == max_rank, range == 0 => score = 1.0
        let results = search_bm25(&conn, "frobnicator", 10, &QueryContext::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            (results[0].1 - 1.0).abs() < 0.01,
            "single result should have normalized score of 1.0, got {}",
            results[0].1
        );
    }

    #[test]
    fn test_bm25_limit_truncates_results() {
        let conn = open_memory_db().unwrap();
        // Store 5 episodes all containing "widget"
        for i in 0..5 {
            episodic::store_episode(
                &conn,
                &NewEpisode {
                    content: format!("widget number {i} description"),
                    role: Role::User,
                    session_id: "s1".to_string(),
                    timestamp: 1000 + i * 100,
                    context: EpisodeContext::default(),
                    embedding: None,
                },
            )
            .unwrap();
        }
        // Request only 2 results
        let results = search_bm25(&conn, "widget", 2, &QueryContext::default()).unwrap();
        assert!(
            results.len() <= 2,
            "should respect limit of 2, got {}",
            results.len()
        );
    }

    #[test]
    fn test_bm25_whitespace_only_query() {
        let conn = open_memory_db().unwrap();
        // A query that is only whitespace sanitizes to empty after trim
        let results = search_bm25(&conn, "   ", 10, &QueryContext::default()).unwrap();
        assert!(
            results.is_empty(),
            "whitespace-only query should return empty"
        );
    }

    #[test]
    fn test_bm25_multiple_results_scores_in_range() {
        let conn = open_memory_db().unwrap();
        episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "programming Rust systems".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();
        episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "Rust ownership and borrowing".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 2000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();
        let results = search_bm25(&conn, "Rust", 10, &QueryContext::default()).unwrap();
        assert!(!results.is_empty());
        for (_, score) in &results {
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "score out of [0,1]: {score}"
            );
        }
    }
}
