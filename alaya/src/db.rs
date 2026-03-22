//! Shared database helpers — timestamps, transactions, JSON, error context.

use crate::error::{AlayaError, Result};
use crate::schema;
use rusqlite::Connection;
use std::time::{SystemTime, UNIX_EPOCH};

/// Current Unix timestamp in seconds.
pub(crate) fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Run `f` inside a BEGIN IMMEDIATE transaction, committing on success.
pub(crate) fn transact<F, T>(conn: &Connection, f: F) -> Result<T>
where
    F: FnOnce(&rusqlite::Transaction) -> Result<T>,
{
    let tx = schema::begin_immediate(conn)?;
    let result = f(&tx)?;
    tx.commit().with_context("commit")?;
    Ok(result)
}

/// Serialize a value to JSON string.
pub(crate) fn to_json<T: serde::Serialize>(value: &T) -> Result<String> {
    serde_json::to_string(value).map_err(Into::into)
}

/// Deserialize JSON, returning `T::default()` on parse failure.
/// Logs a warning via tracing when the tracing feature is enabled.
pub(crate) fn from_json_or_default<T: serde::de::DeserializeOwned + Default>(s: &str) -> T {
    match serde_json::from_str(s) {
        Ok(v) => v,
        Err(_e) => {
            #[cfg(feature = "tracing")]
            tracing::warn!(input = s, error = %_e, "JSON parse failed, using default");
            T::default()
        }
    }
}

/// Extension trait for adding operation context to rusqlite errors.
pub(crate) trait ResultExt<T> {
    fn with_context(self, ctx: &str) -> Result<T>;
}

impl<T> ResultExt<T> for std::result::Result<T, rusqlite::Error> {
    fn with_context(self, ctx: &str) -> Result<T> {
        self.map_err(|e| AlayaError::Db {
            source: e,
            context: ctx.to_string(),
        })
    }
}

/// Context is only applied to [`AlayaError::Db`] errors. For all other
/// `AlayaError` variants the error is returned unchanged and `ctx` is ignored.
impl<T> ResultExt<T> for Result<T> {
    fn with_context(self, ctx: &str) -> Result<T> {
        self.map_err(|e| match e {
            AlayaError::Db { source, .. } => AlayaError::Db {
                source,
                context: ctx.to_string(),
            },
            other => other,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_returns_reasonable_timestamp() {
        let ts = now();
        assert!(ts > 1_704_067_200, "timestamp too old: {ts}");
        assert!(ts < 4_102_444_800, "timestamp too far in future: {ts}");
    }

    #[test]
    fn transact_commits_on_success() {
        let conn = crate::schema::open_memory_db().unwrap();
        let result = transact(&conn, |tx| {
            tx.execute("INSERT INTO episodes (content, role, session_id, timestamp) VALUES ('test', 'user', 's1', 1000)", [])?;
            Ok(42)
        });
        assert_eq!(result.unwrap(), 42);
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM episodes", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn transact_rolls_back_on_error() {
        let conn = crate::schema::open_memory_db().unwrap();
        let result: crate::Result<()> = transact(&conn, |tx| {
            tx.execute("INSERT INTO episodes (content, role, session_id, timestamp) VALUES ('test', 'user', 's1', 1000)", [])?;
            Err(crate::AlayaError::InvalidInput("intentional".into()))
        });
        assert!(result.is_err());
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM episodes", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn to_json_serializes() {
        let v = vec!["a", "b"];
        let json = to_json(&v).unwrap();
        assert_eq!(json, r#"["a","b"]"#);
    }

    #[test]
    fn from_json_or_default_parses() {
        let v: Vec<String> = from_json_or_default(r#"["a","b"]"#);
        assert_eq!(v, vec!["a", "b"]);
    }

    #[test]
    fn from_json_or_default_returns_default_on_bad_input() {
        let v: Vec<String> = from_json_or_default("not json");
        assert!(v.is_empty());
    }

    #[test]
    fn result_ext_adds_context_to_rusqlite_error() {
        let err: std::result::Result<(), rusqlite::Error> =
            Err(rusqlite::Error::QueryReturnedNoRows);
        let result: crate::Result<()> = err.with_context("test_operation");
        match result.unwrap_err() {
            crate::AlayaError::Db { context, .. } => assert_eq!(context, "test_operation"),
            other => panic!("expected Db error, got: {other}"),
        }
    }

    #[test]
    fn result_ext_passes_through_ok() {
        let ok: std::result::Result<i32, rusqlite::Error> = Ok(42);
        assert_eq!(ok.with_context("ctx").unwrap(), 42);
    }
}
