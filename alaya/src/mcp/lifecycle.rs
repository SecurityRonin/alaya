//! Handler logic for the `maintain` and `purge` MCP tools.

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
}
