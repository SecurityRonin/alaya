//! Handler logic for the `visualize` MCP tool.

use serde_json::Value;

use crate::Alaya;

/// Generate a Mermaid diagram of the memory graph.
///
/// Parameters (from MCP tool input):
/// - `max_nodes` (optional, default 50): maximum number of nodes to include
/// - `min_weight` (optional, default 0.1): minimum link weight to include
pub fn visualize(alaya: &Alaya, params: &Value) -> crate::Result<String> {
    let max_nodes = params
        .get("max_nodes")
        .and_then(|v| v.as_u64())
        .unwrap_or(50) as usize;
    let min_weight = params
        .get("min_weight")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.1);

    let conn = alaya.admin().conn;

    let mut lines = vec!["graph TD".to_string()];

    // Track node IDs for class assignments
    let mut episode_ids = Vec::new();
    let mut semantic_ids = Vec::new();
    let mut category_ids = Vec::new();
    let mut preference_ids = Vec::new();

    // 1. Query recent episodes (limited by max_nodes)
    {
        let mut stmt = conn.prepare(
            "SELECT id, content, session_id FROM episodes ORDER BY timestamp DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([max_nodes as i64], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        for row in rows {
            let (id, content, _session_id) = row?;
            let node_id = format!("ep_{id}");
            let label = escape_mermaid(&truncate(&content, 30));
            lines.push(format!("    {node_id}[\"{label}\"]"));
            episode_ids.push(node_id);
        }
    }

    // 2. Query semantic nodes
    {
        let mut stmt = conn.prepare(
            "SELECT id, content, node_type FROM semantic_nodes WHERE superseded_by IS NULL LIMIT ?1",
        )?;
        let rows = stmt.query_map([max_nodes as i64], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        for row in rows {
            let (id, content, _node_type) = row?;
            let node_id = format!("sn_{id}");
            let label = escape_mermaid(&truncate(&content, 30));
            lines.push(format!("    {node_id}[\"{label}\"]"));
            semantic_ids.push(node_id);
        }
    }

    // 3. Query categories
    {
        let mut stmt = conn.prepare("SELECT id, label FROM categories LIMIT ?1")?;
        let rows = stmt.query_map([max_nodes as i64], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;
        for row in rows {
            let (id, label) = row?;
            let node_id = format!("cat_{id}");
            let label = escape_mermaid(&truncate(&label, 30));
            lines.push(format!("    {node_id}[\"{label}\"]"));
            category_ids.push(node_id);
        }
    }

    // 4. Query preferences
    {
        let mut stmt =
            conn.prepare("SELECT id, domain, preference FROM preferences LIMIT ?1")?;
        let rows = stmt.query_map([max_nodes as i64], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        for row in rows {
            let (id, domain, preference) = row?;
            let node_id = format!("pref_{id}");
            let combined = format!("{domain}: {preference}");
            let label = escape_mermaid(&truncate(&combined, 30));
            lines.push(format!("    {node_id}[\"{label}\"]"));
            preference_ids.push(node_id);
        }
    }

    // 5. Query links above min_weight
    {
        let mut stmt = conn.prepare(
            "SELECT source_type, source_id, target_type, target_id, forward_weight, link_type \
             FROM links WHERE forward_weight >= ?1 LIMIT 200",
        )?;
        let rows = stmt.query_map([min_weight], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, f64>(4)?,
                row.get::<_, String>(5)?,
            ))
        })?;
        for row in rows {
            let (source_type, source_id, target_type, target_id, weight, _link_type) = row?;
            let src = node_mermaid_id(&source_type, source_id);
            let tgt = node_mermaid_id(&target_type, target_id);
            lines.push(format!("    {src} --> |\"{weight:.2}\"| {tgt}"));
        }
    }

    // 6. Add styling classes
    lines.push(String::new());
    lines.push(
        "    classDef episode fill:#4A90D9,stroke:#333,color:#fff;".to_string(),
    );
    lines.push(
        "    classDef semantic fill:#7BC67B,stroke:#333,color:#fff;".to_string(),
    );
    lines.push(
        "    classDef category fill:#9B59B6,stroke:#333,color:#fff;".to_string(),
    );
    lines.push(
        "    classDef preference fill:#E67E22,stroke:#333,color:#fff;".to_string(),
    );

    // 7. Assign classes to nodes
    if !episode_ids.is_empty() {
        lines.push(format!(
            "    class {} episode;",
            episode_ids.join(",")
        ));
    }
    if !semantic_ids.is_empty() {
        lines.push(format!(
            "    class {} semantic;",
            semantic_ids.join(",")
        ));
    }
    if !category_ids.is_empty() {
        lines.push(format!(
            "    class {} category;",
            category_ids.join(",")
        ));
    }
    if !preference_ids.is_empty() {
        lines.push(format!(
            "    class {} preference;",
            preference_ids.join(",")
        ));
    }

    Ok(lines.join("\n"))
}

/// Map a DB node type string and id to a Mermaid node identifier.
fn node_mermaid_id(node_type: &str, id: i64) -> String {
    match node_type {
        "episode" => format!("ep_{id}"),
        "semantic" => format!("sn_{id}"),
        "category" => format!("cat_{id}"),
        "preference" => format!("pref_{id}"),
        other => format!("{other}_{id}"),
    }
}

/// Truncate content to `max` characters for node labels.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}...")
    }
}

/// Escape characters that are special in Mermaid labels.
/// Note: `&` must be replaced first to avoid double-escaping.
fn escape_mermaid(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "'")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('[', "(")
        .replace(']', ")")
}

/// MCP handler wrapper for the visualize tool.
pub fn handle_visualize(server: &super::AlayaMcp, params: VisualizeParams) -> String {
    let params_value = serde_json::json!({
        "max_nodes": params.max_nodes,
        "min_weight": params.min_weight,
    });
    match server.with_store(|s| visualize(s, &params_value)) {
        Ok(diagram) => diagram,
        Err(e) => format!("Error: {e}"),
    }
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct VisualizeParams {
    /// Maximum number of nodes to include per type (default: 50)
    #[schemars(description = "Maximum number of nodes to include per type (default: 50)")]
    pub max_nodes: Option<u64>,

    /// Minimum link weight to include (default: 0.1)
    #[schemars(description = "Minimum link weight to include edges (default: 0.1)")]
    pub min_weight: Option<f64>,
}

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_alaya() -> Alaya {
        Alaya::open_in_memory().unwrap()
    }

    #[test]
    fn test_visualize_empty_db() {
        let alaya = make_alaya();
        let result = visualize(&alaya, &json!({})).unwrap();
        assert!(result.starts_with("graph TD"));
        // Should still be valid Mermaid even with no data
        assert!(result.contains("classDef episode"));
        assert!(result.contains("classDef semantic"));
    }

    #[test]
    fn test_visualize_with_episodes() {
        let alaya = make_alaya();
        alaya
            .admin()
            .conn
            .execute(
                "INSERT INTO episodes (content, role, session_id, timestamp) VALUES (?1, ?2, ?3, ?4)",
                ("Hello world", "user", "s1", 1000),
            )
            .unwrap();
        let result = visualize(&alaya, &json!({})).unwrap();
        assert!(result.contains("ep_1"));
        assert!(result.contains("Hello world"));
    }

    #[test]
    fn test_visualize_with_links() {
        let alaya = make_alaya();
        let conn = alaya.admin().conn;
        conn.execute(
            "INSERT INTO episodes (content, role, session_id, timestamp) VALUES ('ep1', 'user', 's1', 1000)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO episodes (content, role, session_id, timestamp) VALUES ('ep2', 'user', 's1', 2000)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO links (source_type, source_id, target_type, target_id, forward_weight, backward_weight, link_type, created_at, last_activated)
             VALUES ('episode', 1, 'episode', 2, 0.8, 0.5, 'temporal', 1000, 1000)",
            [],
        )
        .unwrap();
        let result = visualize(&alaya, &json!({})).unwrap();
        assert!(result.contains("-->"));
    }

    #[test]
    fn test_visualize_respects_max_nodes() {
        let alaya = make_alaya();
        let conn = alaya.admin().conn;
        for i in 1..=20 {
            conn.execute(
                "INSERT INTO episodes (content, role, session_id, timestamp) VALUES (?1, ?2, ?3, ?4)",
                (format!("episode {i}"), "user", "s1", i * 1000),
            )
            .unwrap();
        }
        let result = visualize(&alaya, &json!({"max_nodes": 5})).unwrap();
        // Count actual node definitions (lines with ep_N[")
        let node_count = result.lines().filter(|l| l.contains("ep_") && l.contains("[\"")).count();
        assert!(
            node_count <= 5,
            "should have at most 5 episode nodes, got {node_count}"
        );
    }

    #[test]
    fn test_visualize_min_weight_filter() {
        let alaya = make_alaya();
        let conn = alaya.admin().conn;
        conn.execute(
            "INSERT INTO episodes (content, role, session_id, timestamp) VALUES ('a', 'user', 's1', 1000)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO episodes (content, role, session_id, timestamp) VALUES ('b', 'user', 's1', 2000)",
            [],
        )
        .unwrap();
        // Weak link (below threshold)
        conn.execute(
            "INSERT INTO links (source_type, source_id, target_type, target_id, forward_weight, backward_weight, link_type, created_at, last_activated)
             VALUES ('episode', 1, 'episode', 2, 0.05, 0.05, 'temporal', 1000, 1000)",
            [],
        )
        .unwrap();
        let result = visualize(&alaya, &json!({"min_weight": 0.1})).unwrap();
        // Link should NOT appear because weight (0.05) < min_weight (0.1)
        assert!(
            !result.contains("-->"),
            "weak link should be filtered out"
        );
    }

    #[test]
    fn test_visualize_with_semantic_nodes() {
        let alaya = make_alaya();
        let conn = alaya.admin().conn;
        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated)
             VALUES ('Rust is fast', 'fact', 0.9, 1000, 1000)",
            [],
        )
        .unwrap();
        let result = visualize(&alaya, &json!({})).unwrap();
        assert!(result.contains("sn_1"));
        assert!(result.contains("Rust is fast"));
        assert!(result.contains("class sn_1 semantic"));
    }

    #[test]
    fn test_visualize_with_categories() {
        let alaya = make_alaya();
        let conn = alaya.admin().conn;
        // Need a semantic node for the foreign key
        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated)
             VALUES ('proto', 'fact', 0.9, 1000, 1000)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO categories (label, prototype_node_id, created_at, last_updated)
             VALUES ('Programming', 1, 1000, 1000)",
            [],
        )
        .unwrap();
        let result = visualize(&alaya, &json!({})).unwrap();
        assert!(result.contains("cat_1"));
        assert!(result.contains("Programming"));
        assert!(result.contains("class cat_1 category"));
    }

    #[test]
    fn test_visualize_with_preferences() {
        let alaya = make_alaya();
        let conn = alaya.admin().conn;
        conn.execute(
            "INSERT INTO preferences (domain, preference, confidence, first_observed, last_reinforced)
             VALUES ('style', 'concise answers', 0.8, 1000, 2000)",
            [],
        )
        .unwrap();
        let result = visualize(&alaya, &json!({})).unwrap();
        assert!(result.contains("pref_1"));
        assert!(result.contains("style: concise answers"));
        assert!(result.contains("class pref_1 preference"));
    }

    #[test]
    fn test_truncate_short() {
        assert_eq!(truncate("hello", 30), "hello");
    }

    #[test]
    fn test_truncate_long() {
        let long = "a".repeat(50);
        let result = truncate(&long, 30);
        assert!(result.ends_with("..."));
        assert_eq!(result.len(), 33); // 30 + "..."
    }

    #[test]
    fn test_escape_mermaid() {
        assert_eq!(escape_mermaid("hello \"world\""), "hello 'world'");
        assert_eq!(escape_mermaid("a<b>c"), "a&lt;b&gt;c");
        assert_eq!(escape_mermaid("[test]"), "(test)");
    }

    #[test]
    fn test_node_mermaid_id() {
        assert_eq!(node_mermaid_id("episode", 1), "ep_1");
        assert_eq!(node_mermaid_id("semantic", 2), "sn_2");
        assert_eq!(node_mermaid_id("category", 3), "cat_3");
        assert_eq!(node_mermaid_id("preference", 4), "pref_4");
        assert_eq!(node_mermaid_id("unknown", 5), "unknown_5");
    }

    #[test]
    fn test_handle_visualize_via_server() {
        use super::super::AlayaMcp;

        let alaya = make_alaya();
        let srv = AlayaMcp::new(alaya);
        let result = handle_visualize(
            &srv,
            VisualizeParams {
                max_nodes: None,
                min_weight: None,
            },
        );
        assert!(result.starts_with("graph TD"));
    }

    #[test]
    fn test_handle_visualize_error() {
        use super::super::AlayaMcp;

        let alaya = make_alaya();
        // Drop a required table to trigger an error
        alaya
            .admin()
            .conn
            .execute_batch("DROP TABLE episodes")
            .unwrap();
        let srv = AlayaMcp::new(alaya);
        let result = handle_visualize(
            &srv,
            VisualizeParams {
                max_nodes: None,
                min_weight: None,
            },
        );
        assert!(result.starts_with("Error:"));
    }

    #[test]
    fn test_visualize_superseded_nodes_excluded() {
        let alaya = make_alaya();
        let conn = alaya.admin().conn;
        // Insert two semantic nodes, one superseded
        conn.execute(
            "INSERT INTO semantic_nodes (id, content, node_type, confidence, created_at, last_corroborated)
             VALUES (1, 'old fact', 'fact', 0.9, 1000, 1000)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO semantic_nodes (id, content, node_type, confidence, created_at, last_corroborated, superseded_by)
             VALUES (2, 'superseded fact', 'fact', 0.9, 1000, 1000, 1)",
            [],
        )
        .unwrap();
        let result = visualize(&alaya, &json!({})).unwrap();
        assert!(result.contains("sn_1"));
        assert!(!result.contains("sn_2"), "superseded nodes should be excluded");
    }
}
