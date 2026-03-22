//! Handler logic for the `maintain`, `purge`, `reconcile`, and `conflicts` MCP tools.

use crate::PurgeFilter;

use super::PurgeParams;

pub fn handle_maintain(server: &super::AlayaMcp) -> String {
    let transform = server.with_store(|s| s.transform());
    let forget = server.with_store(|s| s.forget());

    match (transform, forget) {
        (Ok(tr), Ok(fr)) => format!(
            "Maintenance complete:\n  Duplicates merged: {}\n  Links pruned: {}\n  Preferences decayed: {}\n  Nodes decayed: {}\n  Nodes archived: {}",
            tr.duplicates_merged,
            tr.links_pruned,
            tr.preferences_decayed,
            fr.nodes_decayed,
            fr.nodes_archived,
        ),
        (Err(e), _) | (_, Err(e)) => format!("Error: {e}"),
    }
}

pub fn handle_purge(server: &super::AlayaMcp, params: PurgeParams) -> String {
    let filter = match params.scope.as_str() {
        "session" => match params.session_id {
            Some(sid) => PurgeFilter::Session(sid),
            None => return "Error: session_id required for scope 'session'".to_string(),
        },
        "older_than" => match params.before_timestamp {
            Some(ts) => PurgeFilter::OlderThan(ts),
            None => return "Error: before_timestamp required for scope 'older_than'".to_string(),
        },
        "all" => PurgeFilter::All,
        _ => {
            return format!(
                "Error: invalid scope '{}'. Use: session, older_than, all",
                params.scope
            )
        }
    };

    match server.with_store(|s| s.purge(filter)) {
        Ok(report) => format!(
            "Purge complete: {} episodes deleted",
            report.episodes_deleted
        ),
        Err(e) => format!("Error: {e}"),
    }
}

pub fn handle_reconcile(server: &super::AlayaMcp) -> String {
    match server.with_store(|s| s.reconcile()) {
        Ok(report) => format!(
            "Reconciliation complete: detected: {}, resolved: {}, pending: {}, superseded: {}",
            report.conflicts_detected,
            report.conflicts_resolved,
            report.conflicts_pending,
            report.nodes_superseded,
        ),
        Err(e) => format!("Error: {e}"),
    }
}

pub fn handle_conflicts(server: &super::AlayaMcp) -> String {
    match server.with_store(|s| s.conflicts()) {
        Ok(conflicts) if conflicts.is_empty() => "No unresolved conflicts.".to_string(),
        Ok(conflicts) => {
            let mut out = format!("Found {} unresolved conflicts:\n\n", conflicts.len());
            for c in &conflicts {
                out.push_str(&format!(
                    "- Conflict #{}: node {} vs node {} (similarity: {:.2}, status: {})\n",
                    c.id.0,
                    c.node_a.0,
                    c.node_b.0,
                    c.similarity,
                    c.status.as_str(),
                ));
            }
            out
        }
        Err(e) => format!("Error: {e}"),
    }
}

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use crate::AlayaStore;

    use super::super::{AlayaMcp, PurgeParams, RememberParams};

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
    fn maintain_empty_store() {
        let srv = make_server();
        let result = srv.maintain();
        assert!(result.contains("Maintenance complete"));
        assert!(result.contains("Duplicates merged: 0"));
        assert!(result.contains("Links pruned: 0"));
    }

    #[test]
    fn maintain_after_data() {
        let srv = server_with_episodes(5);
        let result = srv.maintain();
        assert!(result.contains("Maintenance complete"));
    }

    #[test]
    fn purge_session() {
        let srv = make_server();
        for i in 0..3 {
            srv.remember(RememberParams {
                content: format!("Sess A msg {i}"),
                role: "user".into(),
                session_id: "sess-a".into(),
            });
        }
        for i in 0..2 {
            srv.remember(RememberParams {
                content: format!("Sess B msg {i}"),
                role: "user".into(),
                session_id: "sess-b".into(),
            });
        }

        let result = srv.purge(PurgeParams {
            scope: "session".into(),
            session_id: Some("sess-a".into()),
            before_timestamp: None,
        });
        assert!(result.contains("Purge complete"));
        assert!(result.contains("3 episodes deleted"));
    }

    #[test]
    fn purge_older_than() {
        let srv = make_server();
        srv.remember(RememberParams {
            content: "Old message".into(),
            role: "user".into(),
            session_id: "s1".into(),
        });

        let result = srv.purge(PurgeParams {
            scope: "older_than".into(),
            session_id: None,
            before_timestamp: Some(i64::MAX),
        });
        assert!(result.contains("Purge complete"));
        assert!(result.contains("episodes deleted"));
    }

    #[test]
    fn purge_all() {
        let srv = server_with_episodes(5);
        let result = srv.purge(PurgeParams {
            scope: "all".into(),
            session_id: None,
            before_timestamp: None,
        });
        assert!(result.contains("Purge complete"));
        let status = srv.status();
        assert!(
            status.contains("Episodes: 0"),
            "All episodes should be gone after purge all: {status}"
        );
    }

    #[test]
    fn purge_invalid_scope() {
        let srv = make_server();
        let result = srv.purge(PurgeParams {
            scope: "invalid".into(),
            session_id: None,
            before_timestamp: None,
        });
        assert!(result.starts_with("Error: invalid scope"));
        assert!(result.contains("invalid"));
    }

    #[test]
    fn purge_session_without_session_id() {
        let srv = make_server();
        let result = srv.purge(PurgeParams {
            scope: "session".into(),
            session_id: None,
            before_timestamp: None,
        });
        assert_eq!(result, "Error: session_id required for scope 'session'");
    }

    #[test]
    fn purge_older_than_without_timestamp() {
        let srv = make_server();
        let result = srv.purge(PurgeParams {
            scope: "older_than".into(),
            session_id: None,
            before_timestamp: None,
        });
        assert_eq!(
            result,
            "Error: before_timestamp required for scope 'older_than'"
        );
    }

    #[test]
    fn maintain_db_error() {
        let store = AlayaStore::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE semantic_nodes")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.maintain();
        assert!(
            result.starts_with("Error:"),
            "Should return error when DB is corrupted: {result}"
        );
    }

    #[test]
    fn purge_db_error() {
        let store = AlayaStore::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE episodes")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.purge(PurgeParams {
            scope: "all".into(),
            session_id: None,
            before_timestamp: None,
        });
        assert!(
            result.starts_with("Error:"),
            "Should return error when DB is corrupted: {result}"
        );
    }

    #[test]
    fn purge_session_deletes_only_that_session() {
        let srv = make_server();
        srv.remember(RememberParams {
            content: "Keep me".into(),
            role: "user".into(),
            session_id: "keep".into(),
        });
        srv.remember(RememberParams {
            content: "Delete me".into(),
            role: "user".into(),
            session_id: "delete".into(),
        });

        srv.purge(PurgeParams {
            scope: "session".into(),
            session_id: Some("delete".into()),
            before_timestamp: None,
        });

        let result = srv.recall(super::super::RecallParams {
            query: "Keep me".into(),
            max_results: None,
            boost_category: None,
        });
        assert!(result.contains("Found"));
        assert!(result.contains("Keep me"));
    }

    #[test]
    fn reconcile_empty_store() {
        let srv = make_server();
        let result = srv.reconcile_memories();
        assert!(result.contains("Reconciliation complete"));
        assert!(result.contains("detected: 0"));
    }

    #[test]
    fn conflicts_empty_store() {
        let srv = make_server();
        let result = srv.list_conflicts();
        assert_eq!(result, "No unresolved conflicts.");
    }

    #[test]
    fn reconcile_db_error() {
        let store = AlayaStore::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE conflicts")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.reconcile_memories();
        assert!(
            result.starts_with("Error:"),
            "Should return error when DB is corrupted: {result}"
        );
    }

    #[test]
    fn conflicts_with_data() {
        let store = AlayaStore::open_in_memory().unwrap();
        // Insert two semantic nodes and a conflict row directly
        store
            .raw_conn()
            .execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated)
                 VALUES ('fact A', 'fact', 0.9, 1000, 1000)",
                [],
            )
            .unwrap();
        store
            .raw_conn()
            .execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated)
                 VALUES ('fact B', 'fact', 0.8, 2000, 2000)",
                [],
            )
            .unwrap();
        store
            .raw_conn()
            .execute(
                "INSERT INTO conflicts (node_a_id, node_b_id, similarity, status, detected_at)
                 VALUES (1, 2, 0.92, 'detected', 1000)",
                [],
            )
            .unwrap();

        let srv = AlayaMcp::new(store);
        let result = srv.list_conflicts();
        assert!(
            result.contains("Found 1 unresolved conflict"),
            "Should show conflicts: {result}"
        );
        assert!(
            result.contains("node 1 vs node 2"),
            "Should show node IDs: {result}"
        );
        assert!(
            result.contains("similarity: 0.92"),
            "Should show similarity: {result}"
        );
    }

    #[test]
    fn conflicts_db_error() {
        let store = AlayaStore::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE conflicts")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.list_conflicts();
        assert!(
            result.starts_with("Error:"),
            "Should return error when DB is corrupted: {result}"
        );
    }
}
