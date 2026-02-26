use rusqlite::Connection;

use crate::error::Result;

/// Open (or create) an alaya database at the given path.
/// Initializes WAL mode, foreign keys, and all tables.
pub fn open_db(path: &str) -> Result<Connection> {
    let conn = Connection::open(path)?;
    init_db(&conn)?;
    Ok(conn)
}

/// Open an in-memory database for testing.
pub fn open_memory_db() -> Result<Connection> {
    let conn = Connection::open_in_memory()?;
    init_db(&conn)?;
    Ok(conn)
}

/// Start a write transaction with IMMEDIATE locking.
/// This prevents SQLITE_BUSY errors under concurrent readers by acquiring
/// the write lock at BEGIN rather than at first write statement.
///
/// Uses `new_unchecked` because `AlayaStore` methods take `&self`, not `&mut self`.
/// Safety from overlapping transactions is guaranteed at the application level:
/// each write method opens, uses, and commits a single transaction.
pub(crate) fn begin_immediate(conn: &Connection) -> Result<rusqlite::Transaction<'_>> {
    Ok(rusqlite::Transaction::new_unchecked(
        conn,
        rusqlite::TransactionBehavior::Immediate,
    )?)
}

fn init_db(conn: &Connection) -> Result<()> {
    conn.execute_batch("PRAGMA journal_mode = WAL;")?;
    conn.execute_batch("PRAGMA foreign_keys = ON;")?;
    conn.execute_batch("PRAGMA synchronous = NORMAL;")?;
    conn.execute_batch("PRAGMA user_version = 1;")?;

    conn.execute_batch(
        "
        -- =================================================================
        -- Episodic store (hippocampus)
        -- =================================================================
        CREATE TABLE IF NOT EXISTS episodes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            content      TEXT    NOT NULL,
            role         TEXT    NOT NULL,
            session_id   TEXT    NOT NULL,
            timestamp    INTEGER NOT NULL,
            context_json TEXT    NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_episodes_session
            ON episodes(session_id);
        CREATE INDEX IF NOT EXISTS idx_episodes_timestamp
            ON episodes(timestamp);

        -- FTS5 full-text index on episode content
        CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
            USING fts5(content, content=episodes, content_rowid=id);

        -- Keep FTS5 in sync via triggers
        CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes
        BEGIN
            INSERT INTO episodes_fts(rowid, content) VALUES (new.id, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes
        BEGIN
            INSERT INTO episodes_fts(episodes_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
        END;

        CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE OF content ON episodes
        BEGIN
            INSERT INTO episodes_fts(episodes_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            INSERT INTO episodes_fts(rowid, content) VALUES (new.id, new.content);
        END;

        -- =================================================================
        -- Semantic store (neocortex)
        -- =================================================================
        CREATE TABLE IF NOT EXISTS semantic_nodes (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            content             TEXT    NOT NULL,
            node_type           TEXT    NOT NULL,
            confidence          REAL    NOT NULL DEFAULT 0.5,
            source_episodes_json TEXT   NOT NULL DEFAULT '[]',
            created_at          INTEGER NOT NULL,
            last_corroborated   INTEGER NOT NULL,
            corroboration_count INTEGER NOT NULL DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_semantic_type
            ON semantic_nodes(node_type);

        -- =================================================================
        -- Implicit store — impressions (vasana raw traces)
        -- =================================================================
        CREATE TABLE IF NOT EXISTS impressions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            domain      TEXT    NOT NULL,
            observation TEXT    NOT NULL,
            valence     REAL    NOT NULL DEFAULT 0.0,
            timestamp   INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_impressions_domain
            ON impressions(domain);
        CREATE INDEX IF NOT EXISTS idx_impressions_timestamp
            ON impressions(timestamp);

        -- =================================================================
        -- Implicit store — crystallized preferences
        -- =================================================================
        CREATE TABLE IF NOT EXISTS preferences (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            domain          TEXT    NOT NULL,
            preference      TEXT    NOT NULL,
            confidence      REAL    NOT NULL DEFAULT 0.5,
            evidence_count  INTEGER NOT NULL DEFAULT 1,
            first_observed  INTEGER NOT NULL,
            last_reinforced INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_preferences_domain
            ON preferences(domain);

        -- =================================================================
        -- Embeddings (shared across all stores)
        -- =================================================================
        CREATE TABLE IF NOT EXISTS embeddings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            node_type TEXT    NOT NULL,
            node_id   INTEGER NOT NULL,
            embedding BLOB    NOT NULL,
            model     TEXT    NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_node
            ON embeddings(node_type, node_id);

        -- =================================================================
        -- Graph overlay (Hebbian links)
        -- =================================================================
        CREATE TABLE IF NOT EXISTS links (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type     TEXT    NOT NULL,
            source_id       INTEGER NOT NULL,
            target_type     TEXT    NOT NULL,
            target_id       INTEGER NOT NULL,
            forward_weight  REAL    NOT NULL DEFAULT 0.5,
            backward_weight REAL    NOT NULL DEFAULT 0.5,
            link_type       TEXT    NOT NULL,
            created_at      INTEGER NOT NULL,
            last_activated  INTEGER NOT NULL,
            activation_count INTEGER NOT NULL DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_links_source
            ON links(source_type, source_id);
        CREATE INDEX IF NOT EXISTS idx_links_target
            ON links(target_type, target_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_links_pair
            ON links(source_type, source_id, target_type, target_id, link_type);

        -- =================================================================
        -- Node strengths (Bjork dual-strength model)
        -- =================================================================
        CREATE TABLE IF NOT EXISTS node_strengths (
            node_type          TEXT    NOT NULL,
            node_id            INTEGER NOT NULL,
            storage_strength   REAL    NOT NULL DEFAULT 0.5,
            retrieval_strength REAL    NOT NULL DEFAULT 1.0,
            access_count       INTEGER NOT NULL DEFAULT 1,
            last_accessed      INTEGER NOT NULL,
            PRIMARY KEY (node_type, node_id)
        );
        ",
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_memory_db() {
        let conn = open_memory_db().unwrap();

        // Verify all tables exist
        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();

        assert!(tables.contains(&"episodes".to_string()));
        assert!(tables.contains(&"semantic_nodes".to_string()));
        assert!(tables.contains(&"impressions".to_string()));
        assert!(tables.contains(&"preferences".to_string()));
        assert!(tables.contains(&"embeddings".to_string()));
        assert!(tables.contains(&"links".to_string()));
        assert!(tables.contains(&"node_strengths".to_string()));
    }

    #[test]
    fn test_fts5_trigger_sync() {
        let conn = open_memory_db().unwrap();

        conn.execute(
            "INSERT INTO episodes (content, role, session_id, timestamp) VALUES (?1, ?2, ?3, ?4)",
            ("hello world", "user", "s1", 1000),
        )
        .unwrap();

        // FTS5 should find it
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM episodes_fts WHERE episodes_fts MATCH 'hello'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);

        // Delete and verify FTS5 is cleaned up
        conn.execute("DELETE FROM episodes WHERE id = 1", [])
            .unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM episodes_fts WHERE episodes_fts MATCH 'hello'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_idempotent_init() {
        let conn = open_memory_db().unwrap();
        // Second init should not fail
        init_db(&conn).unwrap();
    }

    #[test]
    fn test_schema_version_is_set() {
        let conn = open_memory_db().unwrap();
        let version: i64 = conn
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .unwrap();
        assert_eq!(version, 1, "schema version should be 1 after init");
    }

    #[test]
    fn test_begin_immediate_transaction() {
        let conn = open_memory_db().unwrap();
        let tx = begin_immediate(&conn).unwrap();
        tx.execute(
            "INSERT INTO episodes (content, role, session_id, timestamp) VALUES (?1, ?2, ?3, ?4)",
            ("test", "user", "s1", &1000i64),
        )
        .unwrap();
        tx.commit().unwrap();

        let count: i64 = conn
            .query_row("SELECT count(*) FROM episodes", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_immediate_transaction_rollback_on_drop() {
        let conn = open_memory_db().unwrap();
        {
            let tx = begin_immediate(&conn).unwrap();
            tx.execute(
                "INSERT INTO episodes (content, role, session_id, timestamp) VALUES (?1, ?2, ?3, ?4)",
                ("test", "user", "s1", &1000i64),
            )
            .unwrap();
            // tx drops here without commit — should rollback
        }

        let count: i64 = conn
            .query_row("SELECT count(*) FROM episodes", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0, "uncommitted transaction should rollback on drop");
    }
}
