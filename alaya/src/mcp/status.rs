//! Handler logic for the `status` MCP tool.

use std::sync::atomic::Ordering;

pub fn handle_status(server: &super::AlayaMcp) -> String {
    let st = match server.with_store(|s| s.admin().status()) {
        Ok(st) => st,
        Err(e) => return format!("Error: {e}"),
    };

    let session_eps = server.episode_count.load(Ordering::Relaxed);
    let unconsolidated = server.unconsolidated_count.load(Ordering::Relaxed);

    let knowledge_line = match server.with_store(|s| s.knowledge().breakdown()) {
        Ok(breakdown) if !breakdown.is_empty() => {
            super::serialization::format_knowledge_breakdown(&breakdown)
        }
        Ok(_) => "none".to_string(),
        Err(_) => "error".to_string(),
    };

    let cat_line = match server.with_store(|s| s.admin().categories(None)) {
        Ok(cats) if !cats.is_empty() => super::serialization::format_category_line(&cats),
        Ok(_) => "0".to_string(),
        Err(_) => "error".to_string(),
    };

    let strongest_desc = match server.with_store(|s| {
        let link = s.graph().strongest_link()?;
        match link {
            Some((src, tgt, w)) => {
                let src_label = s
                    .admin()
                    .node_content(src)?
                    .unwrap_or_else(|| format!("{}#{}", src.type_str(), src.id()));
                let tgt_label = s
                    .admin()
                    .node_content(tgt)?
                    .unwrap_or_else(|| format!("{}#{}", tgt.type_str(), tgt.id()));
                Ok(Some(format!(
                    " (strongest: \"{src_label}\" <-> \"{tgt_label}\" weight {w:.2})"
                )))
            }
            None => Ok(None),
        }
    }) {
        Ok(Some(desc)) => desc,
        _ => String::new(),
    };

    let total_nodes = st.episode_count + st.semantic_node_count;
    let coverage = if total_nodes > 0 {
        format!(
            "{}/{} nodes ({}%)",
            st.embedding_count,
            total_nodes,
            st.embedding_count * 100 / total_nodes
        )
    } else {
        "0/0 nodes".to_string()
    };

    super::serialization::format_status(
        &st,
        session_eps,
        unconsolidated,
        &knowledge_line,
        &cat_line,
        &strongest_desc,
        &coverage,
    )
}

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use crate::Alaya;

    use super::super::{AlayaMcp, RememberParams};

    fn make_server() -> AlayaMcp {
        let store = Alaya::open_in_memory().unwrap();
        AlayaMcp::new(store)
    }

    fn server_with_episodes(n: u32) -> AlayaMcp {
        let srv = make_server();
        for i in 0..n {
            srv.remember(RememberParams {
                content: format!("Message number {i}"),
                role: "user".into(),
                session_id: "sess-1".into(),
            });
        }
        srv
    }

    #[test]
    fn status_empty_store() {
        let srv = make_server();
        let result = srv.status();
        assert!(result.contains("Memory Status:"));
        assert!(result.contains("Episodes: 0"));
        assert!(result.contains("Knowledge: none"));
    }

    #[test]
    fn status_after_storing_episodes() {
        let srv = server_with_episodes(3);
        let result = srv.status();
        assert!(result.contains("Memory Status:"));
        assert!(result.contains("Episodes: 3"));
        assert!(result.contains("3 this session"));
        assert!(result.contains("3 unconsolidated"));
    }

    #[test]
    fn status_shows_session_and_unconsolidated() {
        let srv = make_server();
        for i in 0..5 {
            srv.remember(RememberParams {
                content: format!("Msg {i}"),
                role: "user".into(),
                session_id: "s1".into(),
            });
        }
        let result = srv.status();
        assert!(result.contains("5 this session"));
        assert!(result.contains("5 unconsolidated"));
    }

    #[test]
    fn status_shows_knowledge_after_learn() {
        use super::super::{LearnFactEntry, LearnParams};
        let srv = make_server();
        srv.learn(LearnParams {
            facts: vec![
                LearnFactEntry {
                    content: "Rust is a systems language".into(),
                    node_type: "fact".into(),
                    confidence: None,
                },
                LearnFactEntry {
                    content: "Alaya is a memory system".into(),
                    node_type: "concept".into(),
                    confidence: None,
                },
            ],
            session_id: None,
        });
        let result = srv.status();
        // knowledge_line should be populated (non-empty path in handle_status)
        assert!(
            !result.contains("Knowledge: none"),
            "Status should show knowledge after learn: {result}"
        );
        assert!(
            result.contains("Knowledge:"),
            "Status should include Knowledge line: {result}"
        );
        // Both node types should appear in the breakdown
        assert!(
            result.contains("facts") || result.contains("concepts"),
            "Knowledge breakdown should mention node types: {result}"
        );
    }

    #[test]
    fn status_coverage_nonzero_when_nodes_exist() {
        use super::super::{LearnFactEntry, LearnParams};
        let srv = make_server();
        // Store an episode (contributes to total_nodes as episode_count)
        srv.remember(RememberParams {
            content: "Episode for coverage".into(),
            role: "user".into(),
            session_id: "s1".into(),
        });
        srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Fact for coverage".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: None,
        });
        let result = srv.status();
        // With nodes present, coverage shows "N/M nodes" (not "0/0 nodes")
        assert!(
            result.contains("Embedding coverage:"),
            "Status should include embedding coverage: {result}"
        );
        assert!(
            result.contains("nodes"),
            "Coverage line should mention nodes: {result}"
        );
    }

    #[test]
    fn status_graph_line_present() {
        let srv = server_with_episodes(2);
        let result = srv.status();
        // Graph line always present with link count
        assert!(
            result.contains("Graph:"),
            "Status should always include Graph line: {result}"
        );
        assert!(
            result.contains("links"),
            "Graph line should mention links: {result}"
        );
    }

    #[test]
    fn status_db_error_episodes_table() {
        let store = Alaya::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE episodes")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.status();
        assert!(
            result.starts_with("Error:"),
            "Should return error when episodes table is missing: {result}"
        );
    }

    #[test]
    fn status_db_error_knowledge_breakdown() {
        // Rename node_type column so knowledge_breakdown() fails (line 19)
        // but status() itself succeeds (only needs COUNT(*))
        let store = Alaya::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("ALTER TABLE semantic_nodes RENAME COLUMN node_type TO broken_col")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.status();
        assert!(
            result.contains("error"),
            "Knowledge line should show error from broken column: {result}"
        );
    }

    #[test]
    fn status_db_error_categories() {
        // Rename a column so list_categories() fails (line 25)
        // but count_categories (COUNT(*)) still works for status()
        let store = Alaya::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("ALTER TABLE categories RENAME COLUMN stability TO broken_stab")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.status();
        assert!(
            result.contains("error"),
            "Categories line should show error from broken column: {result}"
        );
    }

    #[test]
    fn status_preferences_line_present() {
        let srv = make_server();
        let result = srv.status();
        // Preferences line always present
        assert!(
            result.contains("Preferences:"),
            "Status should always include Preferences line: {result}"
        );
        assert!(
            result.contains("crystallized"),
            "Preferences line should mention crystallized: {result}"
        );
        assert!(
            result.contains("impressions"),
            "Preferences line should mention impressions: {result}"
        );
    }
}
