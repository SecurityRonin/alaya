//! JSON export/import for memory backup and portability.
//!
//! Exports all memory data (episodes, semantic nodes, preferences, impressions,
//! categories, and links) to a portable JSON format. Imports restore data into
//! a fresh or existing database, skipping duplicates via INSERT OR IGNORE.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

use crate::error::Result;
use crate::schema;

/// Current export format version. Bump when the export schema changes.
const EXPORT_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Export data types — flat, ID-portable representations of each table
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExportData {
    pub version: u32,
    pub episodes: Vec<ExportEpisode>,
    pub semantic_nodes: Vec<ExportSemanticNode>,
    pub preferences: Vec<ExportPreference>,
    pub impressions: Vec<ExportImpression>,
    pub categories: Vec<ExportCategory>,
    pub links: Vec<ExportLink>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportEpisode {
    pub id: i64,
    pub content: String,
    pub role: String,
    pub session_id: String,
    pub timestamp: i64,
    pub context_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSemanticNode {
    pub id: i64,
    pub content: String,
    pub node_type: String,
    pub confidence: f64,
    pub source_episodes_json: String,
    pub created_at: i64,
    pub last_corroborated: i64,
    pub corroboration_count: i64,
    pub category_id: Option<i64>,
    pub superseded_by: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportPreference {
    pub id: i64,
    pub domain: String,
    pub preference: String,
    pub confidence: f64,
    pub evidence_count: i64,
    pub first_observed: i64,
    pub last_reinforced: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportImpression {
    pub id: i64,
    pub domain: String,
    pub observation: String,
    pub valence: f64,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportCategory {
    pub id: i64,
    pub label: String,
    pub prototype_node_id: i64,
    pub member_count: i64,
    pub centroid_embedding: Option<Vec<u8>>,
    pub created_at: i64,
    pub last_updated: i64,
    pub stability: f64,
    pub parent_id: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportLink {
    pub id: i64,
    pub source_type: String,
    pub source_id: i64,
    pub target_type: String,
    pub target_id: i64,
    pub forward_weight: f64,
    pub backward_weight: f64,
    pub link_type: String,
    pub created_at: i64,
    pub last_activated: i64,
    pub activation_count: i64,
}

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExportReport {
    pub episodes: u32,
    pub semantic_nodes: u32,
    pub preferences: u32,
    pub impressions: u32,
    pub categories: u32,
    pub links: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImportReport {
    pub episodes_imported: u32,
    pub semantic_nodes_imported: u32,
    pub preferences_imported: u32,
    pub impressions_imported: u32,
    pub categories_imported: u32,
    pub links_imported: u32,
    pub skipped: u32,
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

/// Export all memory data to JSON.
///
/// Writes human-readable (pretty-printed) JSON to `writer`.
/// Returns an [`ExportReport`] with counts of exported records.
pub fn export_json(conn: &Connection, writer: &mut dyn Write) -> Result<ExportReport> {
    let mut data = ExportData {
        version: EXPORT_VERSION,
        ..Default::default()
    };
    let mut report = ExportReport::default();

    // Episodes
    {
        let mut stmt = conn.prepare(
            "SELECT id, content, role, session_id, timestamp, context_json FROM episodes ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ExportEpisode {
                id: row.get(0)?,
                content: row.get(1)?,
                role: row.get(2)?,
                session_id: row.get(3)?,
                timestamp: row.get(4)?,
                context_json: row.get(5)?,
            })
        })?;
        for row in rows {
            data.episodes.push(row?);
            report.episodes += 1;
        }
    }

    // Semantic nodes
    {
        let mut stmt = conn.prepare(
            "SELECT id, content, node_type, confidence, source_episodes_json,
                    created_at, last_corroborated, corroboration_count,
                    category_id, superseded_by
             FROM semantic_nodes ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ExportSemanticNode {
                id: row.get(0)?,
                content: row.get(1)?,
                node_type: row.get(2)?,
                confidence: row.get(3)?,
                source_episodes_json: row.get(4)?,
                created_at: row.get(5)?,
                last_corroborated: row.get(6)?,
                corroboration_count: row.get(7)?,
                category_id: row.get(8)?,
                superseded_by: row.get(9)?,
            })
        })?;
        for row in rows {
            data.semantic_nodes.push(row?);
            report.semantic_nodes += 1;
        }
    }

    // Preferences
    {
        let mut stmt = conn.prepare(
            "SELECT id, domain, preference, confidence, evidence_count,
                    first_observed, last_reinforced
             FROM preferences ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ExportPreference {
                id: row.get(0)?,
                domain: row.get(1)?,
                preference: row.get(2)?,
                confidence: row.get(3)?,
                evidence_count: row.get(4)?,
                first_observed: row.get(5)?,
                last_reinforced: row.get(6)?,
            })
        })?;
        for row in rows {
            data.preferences.push(row?);
            report.preferences += 1;
        }
    }

    // Impressions
    {
        let mut stmt = conn.prepare(
            "SELECT id, domain, observation, valence, timestamp FROM impressions ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ExportImpression {
                id: row.get(0)?,
                domain: row.get(1)?,
                observation: row.get(2)?,
                valence: row.get(3)?,
                timestamp: row.get(4)?,
            })
        })?;
        for row in rows {
            data.impressions.push(row?);
            report.impressions += 1;
        }
    }

    // Categories
    {
        let mut stmt = conn.prepare(
            "SELECT id, label, prototype_node_id, member_count, centroid_embedding,
                    created_at, last_updated, stability, parent_id
             FROM categories ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ExportCategory {
                id: row.get(0)?,
                label: row.get(1)?,
                prototype_node_id: row.get(2)?,
                member_count: row.get(3)?,
                centroid_embedding: row.get(4)?,
                created_at: row.get(5)?,
                last_updated: row.get(6)?,
                stability: row.get(7)?,
                parent_id: row.get(8)?,
            })
        })?;
        for row in rows {
            data.categories.push(row?);
            report.categories += 1;
        }
    }

    // Links
    {
        let mut stmt = conn.prepare(
            "SELECT id, source_type, source_id, target_type, target_id,
                    forward_weight, backward_weight, link_type,
                    created_at, last_activated, activation_count
             FROM links ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ExportLink {
                id: row.get(0)?,
                source_type: row.get(1)?,
                source_id: row.get(2)?,
                target_type: row.get(3)?,
                target_id: row.get(4)?,
                forward_weight: row.get(5)?,
                backward_weight: row.get(6)?,
                link_type: row.get(7)?,
                created_at: row.get(8)?,
                last_activated: row.get(9)?,
                activation_count: row.get(10)?,
            })
        })?;
        for row in rows {
            data.links.push(row?);
            report.links += 1;
        }
    }

    serde_json::to_writer_pretty(writer, &data)?;
    Ok(report)
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

/// Import memory data from JSON.
///
/// Reads JSON from `reader`, inserts records into the database inside a
/// transaction for atomicity. Uses `INSERT OR IGNORE` to skip records that
/// would cause unique-constraint violations (e.g. duplicate primary keys).
///
/// Returns an [`ImportReport`] with counts of imported and skipped records.
pub fn import_json(conn: &Connection, reader: &mut dyn Read) -> Result<ImportReport> {
    let data: ExportData = serde_json::from_reader(reader)?;
    let mut report = ImportReport::default();

    let tx = schema::begin_immediate(conn)?;

    // Episodes
    for ep in &data.episodes {
        let changed = tx.execute(
            "INSERT OR IGNORE INTO episodes (id, content, role, session_id, timestamp, context_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![ep.id, ep.content, ep.role, ep.session_id, ep.timestamp, ep.context_json],
        )?;
        if changed > 0 {
            report.episodes_imported += 1;
        } else {
            report.skipped += 1;
        }
    }

    // Semantic nodes
    for node in &data.semantic_nodes {
        let changed = tx.execute(
            "INSERT OR IGNORE INTO semantic_nodes (id, content, node_type, confidence,
                source_episodes_json, created_at, last_corroborated, corroboration_count,
                category_id, superseded_by)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                node.id, node.content, node.node_type, node.confidence,
                node.source_episodes_json, node.created_at, node.last_corroborated,
                node.corroboration_count, node.category_id, node.superseded_by
            ],
        )?;
        if changed > 0 {
            report.semantic_nodes_imported += 1;
        } else {
            report.skipped += 1;
        }
    }

    // Preferences
    for pref in &data.preferences {
        let changed = tx.execute(
            "INSERT OR IGNORE INTO preferences (id, domain, preference, confidence,
                evidence_count, first_observed, last_reinforced)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                pref.id, pref.domain, pref.preference, pref.confidence,
                pref.evidence_count, pref.first_observed, pref.last_reinforced
            ],
        )?;
        if changed > 0 {
            report.preferences_imported += 1;
        } else {
            report.skipped += 1;
        }
    }

    // Impressions
    for imp in &data.impressions {
        let changed = tx.execute(
            "INSERT OR IGNORE INTO impressions (id, domain, observation, valence, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![imp.id, imp.domain, imp.observation, imp.valence, imp.timestamp],
        )?;
        if changed > 0 {
            report.impressions_imported += 1;
        } else {
            report.skipped += 1;
        }
    }

    // Categories
    for cat in &data.categories {
        let changed = tx.execute(
            "INSERT OR IGNORE INTO categories (id, label, prototype_node_id, member_count,
                centroid_embedding, created_at, last_updated, stability, parent_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                cat.id, cat.label, cat.prototype_node_id, cat.member_count,
                cat.centroid_embedding, cat.created_at, cat.last_updated,
                cat.stability, cat.parent_id
            ],
        )?;
        if changed > 0 {
            report.categories_imported += 1;
        } else {
            report.skipped += 1;
        }
    }

    // Links
    for link in &data.links {
        let changed = tx.execute(
            "INSERT OR IGNORE INTO links (id, source_type, source_id, target_type, target_id,
                forward_weight, backward_weight, link_type,
                created_at, last_activated, activation_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                link.id, link.source_type, link.source_id, link.target_type, link.target_id,
                link.forward_weight, link.backward_weight, link.link_type,
                link.created_at, link.last_activated, link.activation_count
            ],
        )?;
        if changed > 0 {
            report.links_imported += 1;
        } else {
            report.skipped += 1;
        }
    }

    tx.commit()?;
    Ok(report)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;
    use crate::testutil::fixtures::*;
    use crate::store;

    #[test]
    fn test_export_empty_db() {
        let conn = open_memory_db().unwrap();
        let mut buf = Vec::new();
        let report = export_json(&conn, &mut buf).unwrap();

        assert_eq!(report.episodes, 0);
        assert_eq!(report.semantic_nodes, 0);
        assert_eq!(report.preferences, 0);
        assert_eq!(report.impressions, 0);
        assert_eq!(report.categories, 0);
        assert_eq!(report.links, 0);

        // Verify valid JSON with correct structure
        let data: ExportData = serde_json::from_slice(&buf).unwrap();
        assert_eq!(data.version, EXPORT_VERSION);
        assert!(data.episodes.is_empty());
        assert!(data.semantic_nodes.is_empty());
    }

    #[test]
    fn test_export_with_episodes() {
        let conn = open_memory_db().unwrap();

        // Store some episodes
        store::episodic::store_episode(&conn, &episode("Hello world")).unwrap();
        store::episodic::store_episode(&conn, &episode("Rust is great")).unwrap();

        let mut buf = Vec::new();
        let report = export_json(&conn, &mut buf).unwrap();

        assert_eq!(report.episodes, 2);

        // Verify JSON content
        let data: ExportData = serde_json::from_slice(&buf).unwrap();
        assert_eq!(data.episodes.len(), 2);
        assert_eq!(data.episodes[0].content, "Hello world");
        assert_eq!(data.episodes[1].content, "Rust is great");
        assert_eq!(data.episodes[0].role, "user");
    }

    #[test]
    fn test_export_import_roundtrip() {
        // DB1: populate with data
        let conn1 = open_memory_db().unwrap();
        store::episodic::store_episode(&conn1, &episode("Episode one")).unwrap();
        store::episodic::store_episode(&conn1, &episode("Episode two")).unwrap();
        insert_semantic_node(&conn1, "Rust has zero-cost abstractions", 0.9);
        insert_semantic_node(&conn1, "Memory safety without GC", 0.85);

        // Store an impression
        store::implicit::store_impression(
            &conn1,
            &crate::types::NewImpression {
                domain: "programming".to_string(),
                observation: "likes Rust".to_string(),
                valence: 0.8,
            },
        )
        .unwrap();

        // Store a preference
        store::implicit::store_preference(&conn1, "language", "Rust over C++", 0.75).unwrap();

        // Export from DB1
        let mut buf = Vec::new();
        let export_report = export_json(&conn1, &mut buf).unwrap();

        assert_eq!(export_report.episodes, 2);
        assert_eq!(export_report.semantic_nodes, 2);
        assert_eq!(export_report.impressions, 1);
        assert_eq!(export_report.preferences, 1);

        // DB2: import into fresh database
        let conn2 = open_memory_db().unwrap();
        let import_report = import_json(&conn2, &mut buf.as_slice()).unwrap();

        assert_eq!(import_report.episodes_imported, 2);
        assert_eq!(import_report.semantic_nodes_imported, 2);
        assert_eq!(import_report.impressions_imported, 1);
        assert_eq!(import_report.preferences_imported, 1);
        assert_eq!(import_report.skipped, 0);

        // Verify data is in DB2
        let ep = store::episodic::get_episode(&conn2, crate::types::EpisodeId(1)).unwrap();
        assert_eq!(ep.content, "Episode one");

        let ep2 = store::episodic::get_episode(&conn2, crate::types::EpisodeId(2)).unwrap();
        assert_eq!(ep2.content, "Episode two");
    }

    #[test]
    fn test_export_report_counts() {
        let conn = open_memory_db().unwrap();

        // Insert data of each type
        store::episodic::store_episode(&conn, &episode("ep1")).unwrap();
        store::episodic::store_episode(&conn, &episode("ep2")).unwrap();
        store::episodic::store_episode(&conn, &episode("ep3")).unwrap();
        insert_semantic_node(&conn, "node1", 0.5);
        store::implicit::store_impression(
            &conn,
            &crate::types::NewImpression {
                domain: "d".to_string(),
                observation: "o".to_string(),
                valence: 0.0,
            },
        )
        .unwrap();
        store::implicit::store_impression(
            &conn,
            &crate::types::NewImpression {
                domain: "d".to_string(),
                observation: "o2".to_string(),
                valence: 0.1,
            },
        )
        .unwrap();

        let mut buf = Vec::new();
        let report = export_json(&conn, &mut buf).unwrap();

        assert_eq!(report.episodes, 3);
        assert_eq!(report.semantic_nodes, 1);
        assert_eq!(report.impressions, 2);
        assert_eq!(report.preferences, 0);
        assert_eq!(report.categories, 0);
        assert_eq!(report.links, 0);
    }

    #[test]
    fn test_import_into_nonempty_db() {
        // Prepare export data
        let conn1 = open_memory_db().unwrap();
        store::episodic::store_episode(&conn1, &episode("existing ep")).unwrap();
        insert_semantic_node(&conn1, "existing node", 0.7);

        let mut buf = Vec::new();
        export_json(&conn1, &mut buf).unwrap();

        // Prepare target DB that already has data
        let conn2 = open_memory_db().unwrap();
        store::episodic::store_episode(&conn2, &episode("already here")).unwrap();

        // Import — should not crash and should skip duplicates
        let report = import_json(&conn2, &mut buf.as_slice()).unwrap();

        // Episode id=1 already exists in conn2, so it should be skipped.
        // The semantic node (id=1) does not exist in conn2, so it should be imported.
        assert_eq!(report.episodes_imported, 0);
        assert_eq!(report.semantic_nodes_imported, 1);
        assert_eq!(report.skipped, 1);

        // Verify original data is untouched
        let ep = store::episodic::get_episode(&conn2, crate::types::EpisodeId(1)).unwrap();
        assert_eq!(ep.content, "already here");
    }
}
