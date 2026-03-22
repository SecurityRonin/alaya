//! Handler logic for the `knowledge`, `categories`, `neighbors`, and `node_category` MCP tools.

use crate::{CategoryId, EpisodeId, KnowledgeFilter, NodeId, NodeRef, PreferenceId, SemanticType};

use super::{CategoriesParams, KnowledgeParams, NeighborsParams, NodeCategoryParams};

pub fn handle_knowledge(server: &super::AlayaMcp, params: KnowledgeParams) -> String {
    let filter = KnowledgeFilter {
        node_type: params.node_type.as_deref().and_then(SemanticType::from_str),
        min_confidence: params.min_confidence,
        limit: params.limit.or(Some(20)),
        category: params.category,
    };

    match server.with_store(|s| s.knowledge().filter(Some(filter))) {
        Ok(nodes) if nodes.is_empty() => "No knowledge found.".to_string(),
        Ok(nodes) => {
            let mut out = format!("Found {} knowledge nodes:\n\n", nodes.len());
            for n in &nodes {
                out.push_str(&format!(
                    "- [{}] {} (confidence: {:.2})\n",
                    n.node_type.as_str(),
                    n.content,
                    n.confidence
                ));
            }
            out
        }
        Err(e) => format!("Error: {e}"),
    }
}

pub fn handle_categories(server: &super::AlayaMcp, params: CategoriesParams) -> String {
    match server.with_store(|s| s.admin().categories(params.min_stability)) {
        Ok(cats) if cats.is_empty() => "No categories found.".to_string(),
        Ok(cats) => super::serialization::format_categories(&cats),
        Err(e) => format!("Error: {e}"),
    }
}

pub fn handle_neighbors(server: &super::AlayaMcp, params: NeighborsParams) -> String {
    let node_ref = match params.node_type.to_lowercase().as_str() {
        "episode" => NodeRef::Episode(EpisodeId(params.node_id)),
        "semantic" => NodeRef::Semantic(NodeId(params.node_id)),
        "preference" => NodeRef::Preference(PreferenceId(params.node_id)),
        "category" => NodeRef::Category(CategoryId(params.node_id)),
        _ => {
            return format!(
                "Error: invalid node_type '{}'. Use: episode, semantic, preference, category",
                params.node_type
            )
        }
    };
    let depth = params.depth.unwrap_or(1);

    match server.with_store(|s| s.graph().neighbors(node_ref, depth)) {
        Ok(neighbors) if neighbors.is_empty() => "No neighbors found.".to_string(),
        Ok(neighbors) => super::serialization::format_neighbors(&neighbors),
        Err(e) => format!("Error: {e}"),
    }
}

pub fn handle_node_category(server: &super::AlayaMcp, params: NodeCategoryParams) -> String {
    match server.with_store(|s| s.admin().node_category(NodeId(params.node_id))) {
        Ok(Some(cat)) => super::serialization::format_node_category(params.node_id, &cat),
        Ok(None) => format!("Node {} is uncategorized.", params.node_id),
        Err(e) => format!("Error: {e}"),
    }
}

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use crate::{
        Alaya, EpisodeContext, EpisodeId, NewEpisode, NewSemanticNode, Role, SemanticType,
    };

    use super::super::{
        AlayaMcp, CategoriesParams, KnowledgeParams, LearnFactEntry, LearnParams, NeighborsParams,
        NodeCategoryParams, RememberParams,
    };

    fn make_server() -> AlayaMcp {
        let store = Alaya::open_in_memory().unwrap();
        AlayaMcp::new(store)
    }

    #[test]
    fn knowledge_empty_store() {
        let srv = make_server();
        let result = srv.knowledge(KnowledgeParams {
            node_type: None,
            min_confidence: None,
            limit: None,
            category: None,
        });
        assert_eq!(result, "No knowledge found.");
    }

    #[test]
    fn knowledge_after_learn() {
        let srv = make_server();
        srv.learn(LearnParams {
            facts: vec![
                LearnFactEntry {
                    content: "Rust is a systems language".into(),
                    node_type: "fact".into(),
                    confidence: Some(0.9),
                },
                LearnFactEntry {
                    content: "Alaya means storehouse".into(),
                    node_type: "concept".into(),
                    confidence: Some(0.85),
                },
            ],
            session_id: None,
        });

        let result = srv.knowledge(KnowledgeParams {
            node_type: None,
            min_confidence: None,
            limit: None,
            category: None,
        });
        assert!(result.contains("Found 2 knowledge nodes"));
        assert!(result.contains("Rust is a systems language"));
        assert!(result.contains("Alaya means storehouse"));
    }

    #[test]
    fn knowledge_with_node_type_filter() {
        let srv = make_server();
        srv.learn(LearnParams {
            facts: vec![
                LearnFactEntry {
                    content: "Fact one".into(),
                    node_type: "fact".into(),
                    confidence: None,
                },
                LearnFactEntry {
                    content: "Concept one".into(),
                    node_type: "concept".into(),
                    confidence: None,
                },
            ],
            session_id: None,
        });

        let result = srv.knowledge(KnowledgeParams {
            node_type: Some("fact".into()),
            min_confidence: None,
            limit: None,
            category: None,
        });
        assert!(result.contains("Fact one"));
        assert!(!result.contains("Concept one"));
    }

    #[test]
    fn knowledge_with_category_filter_no_crash() {
        let srv = make_server();
        let result = srv.knowledge(KnowledgeParams {
            node_type: None,
            min_confidence: None,
            limit: None,
            category: Some("nonexistent".into()),
        });
        assert!(result == "No knowledge found." || result.contains("Found"));
    }

    #[test]
    fn knowledge_with_min_confidence_filter() {
        let srv = make_server();
        srv.learn(LearnParams {
            facts: vec![
                LearnFactEntry {
                    content: "Low confidence fact".into(),
                    node_type: "fact".into(),
                    confidence: Some(0.3),
                },
                LearnFactEntry {
                    content: "High confidence fact".into(),
                    node_type: "fact".into(),
                    confidence: Some(0.95),
                },
            ],
            session_id: None,
        });

        let result = srv.knowledge(KnowledgeParams {
            node_type: None,
            min_confidence: Some(0.9),
            limit: None,
            category: None,
        });
        assert!(result.contains("High confidence fact"));
        assert!(!result.contains("Low confidence fact"));
    }

    #[test]
    fn knowledge_with_limit() {
        let srv = make_server();
        srv.learn(LearnParams {
            facts: (0..10)
                .map(|i| LearnFactEntry {
                    content: format!("Knowledge item {i}"),
                    node_type: "fact".into(),
                    confidence: None,
                })
                .collect(),
            session_id: None,
        });

        let result = srv.knowledge(KnowledgeParams {
            node_type: None,
            min_confidence: None,
            limit: Some(3),
            category: None,
        });
        assert!(result.contains("Found 3 knowledge nodes"));
    }

    #[test]
    fn categories_empty_store() {
        let srv = make_server();
        let result = srv.categories(CategoriesParams {
            min_stability: None,
        });
        assert_eq!(result, "No categories found.");
    }

    #[test]
    fn categories_with_min_stability_no_crash() {
        let srv = make_server();
        let result = srv.categories(CategoriesParams {
            min_stability: Some(0.5),
        });
        assert_eq!(result, "No categories found.");
    }

    #[test]
    fn neighbors_episode_node() {
        let srv = make_server();
        srv.remember(RememberParams {
            content: "Test episode".into(),
            role: "user".into(),
            session_id: "s1".into(),
        });

        let result = srv.neighbors(NeighborsParams {
            node_type: "episode".into(),
            node_id: 1,
            depth: None,
        });
        assert!(!result.starts_with("Error:"));
    }

    #[test]
    fn neighbors_semantic_node_no_crash() {
        let srv = make_server();
        let result = srv.neighbors(NeighborsParams {
            node_type: "semantic".into(),
            node_id: 1,
            depth: None,
        });
        assert!(!result.starts_with("Error:") || result.contains("No neighbors"));
    }

    #[test]
    fn neighbors_preference_node_no_crash() {
        let srv = make_server();
        let result = srv.neighbors(NeighborsParams {
            node_type: "preference".into(),
            node_id: 1,
            depth: None,
        });
        assert!(!result.starts_with("Error:") || result.contains("No neighbors"));
    }

    #[test]
    fn neighbors_category_node_no_crash() {
        let srv = make_server();
        let result = srv.neighbors(NeighborsParams {
            node_type: "category".into(),
            node_id: 1,
            depth: None,
        });
        assert!(!result.starts_with("Error:") || result.contains("No neighbors"));
    }

    #[test]
    fn neighbors_invalid_node_type() {
        let srv = make_server();
        let result = srv.neighbors(NeighborsParams {
            node_type: "bogus".into(),
            node_id: 1,
            depth: None,
        });
        assert!(result.starts_with("Error: invalid node_type"));
        assert!(result.contains("bogus"));
    }

    #[test]
    fn neighbors_with_depth() {
        let srv = make_server();
        srv.remember(RememberParams {
            content: "Test".into(),
            role: "user".into(),
            session_id: "s1".into(),
        });
        let result = srv.neighbors(NeighborsParams {
            node_type: "episode".into(),
            node_id: 1,
            depth: Some(2),
        });
        assert!(!result.starts_with("Error: invalid"));
    }

    #[test]
    fn node_category_nonexistent() {
        let srv = make_server();
        let result = srv.node_category(NodeCategoryParams { node_id: 9999 });
        assert!(result.contains("uncategorized"));
    }

    #[test]
    fn node_category_after_learn() {
        let srv = make_server();
        srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Test knowledge".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: None,
        });
        let result = srv.node_category(NodeCategoryParams { node_id: 1 });
        assert!(result.contains("Node 1"));
    }

    /// Pre-populate a store with categorized semantic nodes for testing.
    fn make_server_with_categories() -> AlayaMcp {
        let store = Alaya::open_in_memory().unwrap();
        for i in 0..5 {
            store
                .episodes()
                .store(&NewEpisode {
                    content: format!("cooking topic {i}"),
                    role: Role::User,
                    session_id: "s1".to_string(),
                    timestamp: 1000 + i * 100,
                    context: EpisodeContext::default(),
                    embedding: None,
                })
                .unwrap();
        }
        store
            .knowledge()
            .learn(vec![
                NewSemanticNode {
                    content: "User cooks pasta regularly".to_string(),
                    node_type: SemanticType::Fact,
                    confidence: 0.9,
                    source_episodes: vec![EpisodeId(1)],
                    embedding: Some(vec![0.8, 0.3, 0.1]),
                },
                NewSemanticNode {
                    content: "User likes Italian food".to_string(),
                    node_type: SemanticType::Fact,
                    confidence: 0.85,
                    source_episodes: vec![EpisodeId(2)],
                    embedding: Some(vec![0.4, 0.8, 0.2]),
                },
                NewSemanticNode {
                    content: "User knows many recipes".to_string(),
                    node_type: SemanticType::Concept,
                    confidence: 0.8,
                    source_episodes: vec![EpisodeId(3)],
                    embedding: Some(vec![0.6, 0.5, 0.5]),
                },
            ])
            .unwrap();
        store.lifecycle().transform().unwrap();
        AlayaMcp::new(store)
    }

    #[test]
    fn categories_with_data() {
        let srv = make_server_with_categories();
        let result = srv.categories(CategoriesParams {
            min_stability: None,
        });
        // Should format non-empty categories (covers line 36)
        assert!(
            !result.starts_with("No categories") && !result.starts_with("Error"),
            "Should have categories: {result}"
        );
    }

    #[test]
    fn neighbors_with_data() {
        let srv = make_server_with_categories();
        // Semantic nodes have Causal links to episodes — use semantic node to find neighbors
        let result = srv.neighbors(NeighborsParams {
            node_type: "semantic".into(),
            node_id: 1,
            depth: Some(2),
        });
        // Should format non-empty neighbors (covers line 58)
        assert!(
            !result.starts_with("No neighbors") && !result.starts_with("Error"),
            "Semantic node should have neighbors via causal links: {result}"
        );
    }

    #[test]
    fn knowledge_db_error() {
        let store = Alaya::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE semantic_nodes")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.knowledge(KnowledgeParams {
            node_type: None,
            min_confidence: None,
            limit: None,
            category: None,
        });
        assert!(
            result.starts_with("Error:"),
            "Should return error: {result}"
        );
    }

    #[test]
    fn categories_db_error() {
        let store = Alaya::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE categories")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.categories(CategoriesParams {
            min_stability: None,
        });
        assert!(
            result.starts_with("Error:"),
            "Should return error: {result}"
        );
    }

    #[test]
    fn neighbors_db_error() {
        let store = Alaya::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE links")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.neighbors(NeighborsParams {
            node_type: "episode".into(),
            node_id: 1,
            depth: None,
        });
        assert!(
            result.starts_with("Error:"),
            "Should return error: {result}"
        );
    }

    #[test]
    fn node_category_db_error() {
        let store = Alaya::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE semantic_nodes")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.node_category(NodeCategoryParams { node_id: 1 });
        assert!(
            result.starts_with("Error:"),
            "Should return error: {result}"
        );
    }

    #[test]
    fn node_category_with_categorized_node() {
        let srv = make_server_with_categories();
        // After transform, at least some semantic nodes should be categorized.
        // Try node IDs 1-3 (the semantic nodes we created)
        let mut found_categorized = false;
        for id in 1..=3 {
            let result = srv.node_category(NodeCategoryParams { node_id: id });
            if !result.contains("uncategorized") && !result.starts_with("Error") {
                found_categorized = true;
                break;
            }
        }
        assert!(
            found_categorized,
            "At least one semantic node should be categorized after transform"
        );
    }
}
