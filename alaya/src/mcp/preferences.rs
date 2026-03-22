//! Handler logic for the `learn` and `preferences` MCP tools.

use std::sync::atomic::Ordering;

use crate::{NewSemanticNode, SemanticType};

use super::{LearnParams, PreferencesParams};

pub fn handle_learn(server: &super::AlayaMcp, params: LearnParams) -> String {
    // Resolve session_id → source episode IDs
    let source_episodes = match &params.session_id {
        Some(sid) => match server.with_store(|s| s.episodes().by_session(sid)) {
            Ok(eps) => eps.iter().map(|e| e.id).collect::<Vec<_>>(),
            Err(e) => return format!("Error resolving session '{sid}': {e}"),
        },
        None => vec![],
    };

    // Convert LearnFactEntry → NewSemanticNode
    let nodes: Vec<NewSemanticNode> = params
        .facts
        .iter()
        .map(|fact| {
            let node_type = SemanticType::from_str(&fact.node_type).unwrap_or(SemanticType::Fact);
            let confidence = fact.confidence.unwrap_or(0.8).clamp(0.0, 1.0);
            NewSemanticNode {
                content: fact.content.clone(),
                node_type,
                confidence,
                source_episodes: source_episodes.clone(),
                embedding: None,
            }
        })
        .collect();

    let count = nodes.len();
    match server.with_store(|s| s.knowledge().learn(nodes)) {
        Ok(report) => {
            server.unconsolidated_count.store(0, Ordering::Relaxed);
            format!(
                "Learned {} facts: {} nodes created, {} links created, {} categories assigned",
                count, report.nodes_created, report.links_created, report.categories_assigned
            )
        }
        Err(e) => format!("Error: {e}"),
    }
}

pub fn handle_preferences(server: &super::AlayaMcp, params: PreferencesParams) -> String {
    match server.with_store(|s| s.admin().preferences(params.domain.as_deref())) {
        Ok(prefs) if prefs.is_empty() => "No preferences found.".to_string(),
        Ok(prefs) => super::serialization::format_preferences(&prefs),
        Err(e) => format!("Error: {e}"),
    }
}

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use crate::provider::MockProvider;
    use crate::types::{EpisodeContext, Interaction, NewImpression, Role};
    use crate::Alaya;

    use super::super::{AlayaMcp, LearnFactEntry, LearnParams, PreferencesParams, RememberParams};

    fn make_server() -> AlayaMcp {
        let store = Alaya::open_in_memory().unwrap();
        AlayaMcp::new(store)
    }

    #[test]
    fn preferences_empty_store() {
        let srv = make_server();
        let result = srv.preferences(PreferencesParams { domain: None });
        assert_eq!(result, "No preferences found.");
    }

    #[test]
    fn preferences_with_domain_filter_no_crash() {
        let srv = make_server();
        let result = srv.preferences(PreferencesParams {
            domain: Some("style".into()),
        });
        assert_eq!(result, "No preferences found.");
    }

    #[test]
    fn learn_three_facts() {
        let srv = make_server();
        let result = srv.learn(LearnParams {
            facts: vec![
                LearnFactEntry {
                    content: "Rust is fast".into(),
                    node_type: "fact".into(),
                    confidence: None,
                },
                LearnFactEntry {
                    content: "Alaya uses SQLite".into(),
                    node_type: "fact".into(),
                    confidence: Some(0.9),
                },
                LearnFactEntry {
                    content: "MCP is a protocol".into(),
                    node_type: "concept".into(),
                    confidence: None,
                },
            ],
            session_id: None,
        });
        assert!(result.starts_with("Learned 3 facts:"));
        assert!(result.contains("3 nodes created"));
    }

    #[test]
    fn learn_with_session_id_links_episodes() {
        let srv = make_server();
        srv.remember(RememberParams {
            content: "User said something".into(),
            role: "user".into(),
            session_id: "sess-link".into(),
        });
        srv.remember(RememberParams {
            content: "Assistant replied".into(),
            role: "assistant".into(),
            session_id: "sess-link".into(),
        });

        let result = srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Extracted from conversation".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: Some("sess-link".into()),
        });
        assert!(result.starts_with("Learned 1 facts:"));
        assert!(result.contains("1 nodes created"));
        assert!(result.contains("links created"));
    }

    #[test]
    fn learn_resets_unconsolidated_counter() {
        let srv = make_server();
        for i in 0..5 {
            srv.remember(RememberParams {
                content: format!("Ep {i}"),
                role: "user".into(),
                session_id: "s1".into(),
            });
        }

        let status = srv.status();
        assert!(status.contains("5 unconsolidated"));

        srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Learned fact".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: None,
        });

        let status = srv.status();
        assert!(status.contains("0 unconsolidated"));
    }

    #[test]
    fn learn_invalid_node_type_defaults_to_fact() {
        let srv = make_server();
        let result = srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Something interesting".into(),
                node_type: "invalid_type".into(),
                confidence: None,
            }],
            session_id: None,
        });
        assert!(result.starts_with("Learned 1 facts:"));

        let knowledge = srv.knowledge(super::super::KnowledgeParams {
            node_type: Some("fact".into()),
            min_confidence: None,
            limit: None,
            category: None,
        });
        assert!(knowledge.contains("Something interesting"));
    }

    #[test]
    fn learn_with_clamped_confidence() {
        let srv = make_server();
        let result = srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Over-confident fact".into(),
                node_type: "fact".into(),
                confidence: Some(5.0),
            }],
            session_id: None,
        });
        assert!(result.starts_with("Learned 1 facts:"));

        let knowledge = srv.knowledge(super::super::KnowledgeParams {
            node_type: None,
            min_confidence: Some(1.0),
            limit: None,
            category: None,
        });
        assert!(knowledge.contains("Over-confident fact"));
    }

    #[test]
    fn learn_empty_facts_vec() {
        let srv = make_server();
        let result = srv.learn(LearnParams {
            facts: vec![],
            session_id: None,
        });
        assert!(result.starts_with("Learned 0 facts:"));
    }

    #[test]
    fn learn_with_nonexistent_session() {
        let srv = make_server();
        let result = srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "A fact".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: Some("nonexistent-session".into()),
        });
        assert!(result.starts_with("Learned 1 facts:"));
    }

    #[test]
    fn learn_session_resolve_db_error() {
        let store = Alaya::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE episodes")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "A fact".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: Some("nonexistent".into()),
        });
        assert!(
            result.starts_with("Error resolving session"),
            "Should return session resolve error: {result}"
        );
    }

    #[test]
    fn learn_db_error() {
        let store = Alaya::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE semantic_nodes")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "A fact".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: None,
        });
        assert!(
            result.starts_with("Error:"),
            "Should return error: {result}"
        );
    }

    #[test]
    fn preferences_db_error() {
        let store = Alaya::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE preferences")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.preferences(PreferencesParams { domain: None });
        assert!(
            result.starts_with("Error:"),
            "Should return error: {result}"
        );
    }

    #[test]
    fn preferences_with_crystallized_data() {
        // Pre-populate preferences via perfuming, then test the handler (covers line 52)
        let store = Alaya::open_in_memory().unwrap();
        let provider = MockProvider::with_impressions(vec![NewImpression {
            domain: "style".to_string(),
            observation: "prefers dark mode".to_string(),
            valence: 1.0,
        }]);

        // Perfume 6 times to reach crystallization threshold (5)
        for i in 0..6 {
            let interaction = Interaction {
                text: format!("interaction {i}"),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000 + i * 100,
                context: EpisodeContext::default(),
            };
            store.lifecycle().perfume(&interaction, &provider).unwrap();
        }

        let srv = AlayaMcp::new(store);
        let result = srv.preferences(PreferencesParams { domain: None });
        // Should format non-empty preferences (covers line 52)
        assert!(
            !result.contains("No preferences found"),
            "Should have crystallized preferences: {result}"
        );
    }
}
