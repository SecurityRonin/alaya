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

/// Initialize schema on an already-open connection (used by encrypted open).
#[cfg(feature = "sqlcipher")]
pub(crate) fn initialize(conn: &Connection) -> Result<()> {
    init_db(conn)
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
    conn.execute_batch("PRAGMA user_version = 5;")?;

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

        -- =================================================================
        -- Categories (emergent ontology)
        -- =================================================================
        CREATE TABLE IF NOT EXISTS categories (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            label               TEXT    NOT NULL,
            prototype_node_id   INTEGER REFERENCES semantic_nodes(id),
            member_count        INTEGER NOT NULL DEFAULT 0,
            centroid_embedding  BLOB,
            created_at          INTEGER NOT NULL,
            last_updated        INTEGER NOT NULL,
            stability           REAL    NOT NULL DEFAULT 0.0,
            parent_id           INTEGER REFERENCES categories(id)
        );

        CREATE INDEX IF NOT EXISTS idx_categories_stability
            ON categories(stability);

        -- =================================================================
        -- Tombstones: track deleted nodes for cascade auditing
        -- =================================================================
        CREATE TABLE IF NOT EXISTS tombstones (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            node_type   TEXT NOT NULL,
            node_id     INTEGER NOT NULL,
            deleted_at  INTEGER NOT NULL,
            reason      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_tombstones_type_id ON tombstones(node_type, node_id);

        -- =================================================================
        -- Conflicts (reconciliation)
        -- =================================================================
        CREATE TABLE IF NOT EXISTS conflicts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            node_a_id       INTEGER NOT NULL REFERENCES semantic_nodes(id),
            node_b_id       INTEGER NOT NULL REFERENCES semantic_nodes(id),
            similarity      REAL    NOT NULL,
            status          TEXT    NOT NULL DEFAULT 'detected',
            resolution      TEXT,
            winner_id       INTEGER REFERENCES semantic_nodes(id),
            detected_at     INTEGER NOT NULL,
            resolved_at     INTEGER,
            UNIQUE(node_a_id, node_b_id)
        );
        CREATE INDEX IF NOT EXISTS idx_conflicts_status ON conflicts(status);
        ",
    )?;

    // Migration v1->v2: add category_id to semantic_nodes
    let has_category: bool = conn
        .prepare("SELECT category_id FROM semantic_nodes LIMIT 0")
        .is_ok();
    if !has_category {
        conn.execute_batch(
            "ALTER TABLE semantic_nodes ADD COLUMN category_id INTEGER REFERENCES categories(id);
             CREATE INDEX IF NOT EXISTS idx_semantic_category ON semantic_nodes(category_id);",
        )?;
    }

    // Migration v4->v5: add superseded_by to semantic_nodes
    let has_superseded: bool = conn
        .prepare("SELECT superseded_by FROM semantic_nodes LIMIT 0")
        .is_ok();
    if !has_superseded {
        conn.execute_batch(
            "ALTER TABLE semantic_nodes ADD COLUMN superseded_by INTEGER REFERENCES semantic_nodes(id);",
        )?;
    }

    Ok(())
}

/// Record a tombstone for a deleted node.
pub(crate) fn record_tombstone(
    conn: &Connection,
    node_type: &str,
    node_id: i64,
    reason: Option<&str>,
) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    conn.execute(
        "INSERT INTO tombstones (node_type, node_id, deleted_at, reason) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![node_type, node_id, now, reason],
    )?;
    Ok(())
}

#[cfg(test)]
/// Count tombstones (for testing).
pub(crate) fn count_tombstones(conn: &Connection) -> Result<u64> {
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM tombstones", [], |row| row.get(0))?;
    Ok(count as u64)
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
        assert!(tables.contains(&"categories".to_string()));
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
        assert_eq!(
            version, 5,
            "schema version should be 5 after conflicts migration"
        );
    }

    #[test]
    fn test_schema_version_is_5_compat() {
        let conn = open_memory_db().unwrap();
        let version: i64 = conn
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .unwrap();
        assert_eq!(
            version, 5,
            "schema version should be 5 after conflicts migration"
        );
    }

    #[test]
    fn test_tombstones_table_exists() {
        let conn = open_memory_db().unwrap();
        let exists: bool = conn.prepare("SELECT 1 FROM tombstones LIMIT 0").is_ok();
        assert!(exists, "tombstones table should exist");
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
    fn test_categories_table_exists() {
        let conn = open_memory_db().unwrap();
        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert!(tables.contains(&"categories".to_string()));
    }

    #[test]
    fn test_semantic_nodes_has_category_id() {
        let conn = open_memory_db().unwrap();
        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, category_id)
             VALUES ('test', 'fact', 0.5, 1000, 1000, NULL)",
            [],
        ).unwrap();
    }

    #[test]
    fn test_tombstone_recorded_on_episode_delete() {
        use crate::store::episodic;
        use crate::types::*;

        let conn = open_memory_db().unwrap();
        let id = episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "temp data".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();

        episodic::delete_episodes(&conn, &[id]).unwrap();
        assert_eq!(count_tombstones(&conn).unwrap(), 1);
    }

    #[test]
    fn test_tombstone_recorded_on_semantic_delete() {
        use crate::store::semantic;
        use crate::types::*;

        let conn = open_memory_db().unwrap();
        let id = semantic::store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "temp fact".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.5,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();

        semantic::delete_node(&conn, id).unwrap();
        assert_eq!(count_tombstones(&conn).unwrap(), 1);
    }

    #[test]
    fn test_schema_version_is_5() {
        let conn = open_memory_db().unwrap();
        let version: i64 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 5);
    }

    #[test]
    fn test_categories_has_parent_id() {
        let conn = open_memory_db().unwrap();
        // Insert a semantic node so the FK on prototype_node_id is satisfied
        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated)
             VALUES ('proto', 'fact', 0.5, 1000, 1000)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO categories (label, prototype_node_id, created_at, last_updated, parent_id)
             VALUES ('test', 1, 1000, 1000, NULL)",
            [],
        )
        .unwrap();
        let parent_id: Option<i64> = conn
            .query_row("SELECT parent_id FROM categories WHERE id = 1", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert!(parent_id.is_none());
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

    #[test]
    fn test_conflicts_table_exists() {
        let conn = open_memory_db().unwrap();
        let exists: bool = conn.prepare("SELECT 1 FROM conflicts LIMIT 0").is_ok();
        assert!(exists, "conflicts table should exist");
    }

    #[test]
    fn test_semantic_nodes_has_superseded_by() {
        let conn = open_memory_db().unwrap();
        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, superseded_by)
             VALUES ('test', 'fact', 0.5, 1000, 1000, NULL)",
            [],
        )
        .unwrap();
    }
}
