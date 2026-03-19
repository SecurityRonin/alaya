//! Unified decay strategies for temporal value reduction.
//!
//! The codebase has multiple decay patterns (links, preferences, retrieval
//! strengths, recency scoring). This module provides a shared trait so the
//! math is defined once and tested thoroughly.

use crate::error::Result;
use rusqlite::Connection;

/// Strategy for computing temporal decay factors.
pub trait Decay {
    /// Compute a multiplicative decay factor (0.0..=1.0) for the given elapsed time.
    fn factor(&self, elapsed_secs: i64) -> f64;
}

/// Multiplicative decay: multiplies by a fixed factor each sweep.
/// Used by link decay and retrieval strength decay.
#[allow(dead_code)]
pub struct MultiplicativeDecay {
    pub factor: f64,
}

/// Exponential decay: exp(-0.693 * elapsed / half_life).
/// For Rust contexts, uses exact f64 exponential.
/// Note: decay_preferences uses its own SQL with a time-conditional WHERE clause
/// that cannot be expressed generically, so it calls Decay::factor() only.
pub struct ExponentialDecay {
    pub half_life_secs: i64,
}

impl Decay for MultiplicativeDecay {
    fn factor(&self, _elapsed_secs: i64) -> f64 {
        self.factor
    }
}

impl Decay for ExponentialDecay {
    fn factor(&self, elapsed_secs: i64) -> f64 {
        let elapsed = elapsed_secs.max(0) as f64;
        (-0.693 * elapsed / self.half_life_secs as f64).exp()
    }
}

/// Apply multiplicative decay to a single column via SQL UPDATE.
/// Only used where the SQL pattern is simple: `SET col = col * factor WHERE col > 0.01`.
/// decay_preferences has a complex WHERE clause and is NOT a candidate for this helper.
pub fn apply_multiplicative_sql(
    conn: &Connection,
    table: &str,
    column: &str,
    factor: f64,
) -> Result<u64> {
    let sql = format!("UPDATE {table} SET {column} = {column} * ?1 WHERE {column} > 0.01");
    let changed = conn.execute(&sql, [factor])?;
    Ok(changed as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;

    // --- MultiplicativeDecay ---

    #[test]
    fn multiplicative_factor_returns_stored_factor() {
        let d = MultiplicativeDecay { factor: 0.9 };
        assert!((d.factor(1000) - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn multiplicative_factor_ignores_elapsed() {
        let d = MultiplicativeDecay { factor: 0.85 };
        assert_eq!(d.factor(0), d.factor(999_999));
    }

    // --- ExponentialDecay ---

    #[test]
    fn exponential_factor_zero_elapsed() {
        let d = ExponentialDecay {
            half_life_secs: 86400,
        };
        assert!((d.factor(0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn exponential_factor_at_half_life() {
        let d = ExponentialDecay {
            half_life_secs: 86400,
        };
        let f = d.factor(86400);
        assert!((f - 0.5).abs() < 0.01);
    }

    #[test]
    fn exponential_factor_large_elapsed() {
        let d = ExponentialDecay {
            half_life_secs: 86400,
        };
        let f = d.factor(86400 * 100);
        assert!(f < 0.001);
        assert!(f >= 0.0);
    }

    #[test]
    fn exponential_factor_negative_elapsed() {
        let d = ExponentialDecay {
            half_life_secs: 86400,
        };
        let f = d.factor(-1000);
        assert!((f - 1.0).abs() < f64::EPSILON);
    }

    // --- apply_multiplicative_sql ---

    #[test]
    fn apply_multiplicative_sql_updates_rows() {
        let conn = open_memory_db().unwrap();
        conn.execute(
            "INSERT INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength, access_count, last_accessed)
             VALUES ('semantic', 1, 1.0, 0.8, 0, 0)",
            [],
        ).unwrap();
        let changed =
            apply_multiplicative_sql(&conn, "node_strengths", "retrieval_strength", 0.9).unwrap();
        assert!(changed > 0);
    }

    #[test]
    fn apply_multiplicative_sql_skips_below_threshold() {
        let conn = open_memory_db().unwrap();
        conn.execute(
            "INSERT INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength, access_count, last_accessed)
             VALUES ('semantic', 1, 1.0, 0.005, 0, 0)",
            [],
        ).unwrap();
        let changed =
            apply_multiplicative_sql(&conn, "node_strengths", "retrieval_strength", 0.9).unwrap();
        assert_eq!(changed, 0);
    }

    #[test]
    fn apply_multiplicative_sql_empty_table() {
        let conn = open_memory_db().unwrap();
        let changed =
            apply_multiplicative_sql(&conn, "node_strengths", "retrieval_strength", 0.9).unwrap();
        assert_eq!(changed, 0);
    }

    #[test]
    fn apply_multiplicative_sql_multiple_rows() {
        let conn = open_memory_db().unwrap();
        conn.execute(
            "INSERT INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength, access_count, last_accessed)
             VALUES ('semantic', 1, 1.0, 0.8, 0, 0)",
            [],
        ).unwrap();
        conn.execute(
            "INSERT INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength, access_count, last_accessed)
             VALUES ('semantic', 2, 1.0, 0.6, 0, 0)",
            [],
        ).unwrap();
        let changed =
            apply_multiplicative_sql(&conn, "node_strengths", "retrieval_strength", 0.9).unwrap();
        assert_eq!(changed, 2);
    }

    #[test]
    fn apply_multiplicative_sql_mixed_rows() {
        let conn = open_memory_db().unwrap();
        conn.execute(
            "INSERT INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength, access_count, last_accessed)
             VALUES ('semantic', 1, 1.0, 0.8, 0, 0)",
            [],
        ).unwrap();
        conn.execute(
            "INSERT INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength, access_count, last_accessed)
             VALUES ('semantic', 2, 1.0, 0.005, 0, 0)",
            [],
        ).unwrap();
        let changed =
            apply_multiplicative_sql(&conn, "node_strengths", "retrieval_strength", 0.9).unwrap();
        assert_eq!(changed, 1);
    }

    #[test]
    fn apply_multiplicative_sql_factor_zero() {
        let conn = open_memory_db().unwrap();
        conn.execute(
            "INSERT INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength, access_count, last_accessed)
             VALUES ('semantic', 1, 1.0, 0.8, 0, 0)",
            [],
        ).unwrap();
        let changed =
            apply_multiplicative_sql(&conn, "node_strengths", "retrieval_strength", 0.0).unwrap();
        assert!(changed > 0);
    }

    #[test]
    fn apply_multiplicative_sql_factor_one() {
        let conn = open_memory_db().unwrap();
        conn.execute(
            "INSERT INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength, access_count, last_accessed)
             VALUES ('semantic', 1, 1.0, 0.8, 0, 0)",
            [],
        ).unwrap();
        let changed =
            apply_multiplicative_sql(&conn, "node_strengths", "retrieval_strength", 1.0).unwrap();
        assert!(changed > 0);
    }

    #[test]
    fn apply_multiplicative_sql_factor_greater_than_one() {
        let conn = open_memory_db().unwrap();
        conn.execute(
            "INSERT INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength, access_count, last_accessed)
             VALUES ('semantic', 1, 1.0, 0.8, 0, 0)",
            [],
        ).unwrap();
        let changed =
            apply_multiplicative_sql(&conn, "node_strengths", "retrieval_strength", 1.5).unwrap();
        assert!(changed > 0);
    }

    #[test]
    fn exponential_factor_zero_half_life() {
        let d = ExponentialDecay { half_life_secs: 0 };
        let f = d.factor(100);
        // Division by zero yields infinity, exp(-infinity) = 0.0
        assert_eq!(f, 0.0);
    }
}
