//! Handler logic for the `status` MCP tool.

use std::sync::atomic::Ordering;

pub fn handle_status(server: &super::AlayaMcp) -> String {
    let st = match server.with_store(|s| s.status()) {
        Ok(st) => st,
        Err(e) => return format!("Error: {e}"),
    };

    let session_eps = server.episode_count.load(Ordering::Relaxed);
    let unconsolidated = server.unconsolidated_count.load(Ordering::Relaxed);

    let knowledge_line = match server.with_store(|s| s.knowledge_breakdown()) {
        Ok(breakdown) if !breakdown.is_empty() => {
            super::serialization::format_knowledge_breakdown(&breakdown)
        }
        Ok(_) => "none".to_string(),
        Err(_) => "error".to_string(),
    };

    let cat_line = match server.with_store(|s| s.categories(None)) {
        Ok(cats) if !cats.is_empty() => super::serialization::format_category_line(&cats),
        Ok(_) => "0".to_string(),
        Err(_) => "error".to_string(),
    };

    let strongest_desc = match server.with_store(|s| {
        let link = s.strongest_link()?;
        match link {
            Some((src, tgt, w)) => {
                let src_label = s
                    .node_content(src)?
                    .unwrap_or_else(|| format!("{}#{}", src.type_str(), src.id()));
                let tgt_label = s
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
    use crate::AlayaStore;

    use super::super::{AlayaMcp, RememberParams};

    fn make_server() -> AlayaMcp {
        let store = AlayaStore::open_in_memory().unwrap();
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
}
