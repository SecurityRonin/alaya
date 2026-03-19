//! Formatting helpers for MCP tool handler responses.

use crate::types::{Category, MemoryStatus, Preference};
use crate::{NodeRef, SemanticType};

pub fn format_preferences(prefs: &[Preference]) -> String {
    let mut out = format!("Found {} preferences:\n\n", prefs.len());
    for p in prefs {
        out.push_str(&format!(
            "- [{}] {} (confidence: {:.2}, evidence: {})\n",
            p.domain, p.preference, p.confidence, p.evidence_count
        ));
    }
    out
}

pub fn format_categories(cats: &[Category]) -> String {
    let mut out = format!("Found {} categories:\n\n", cats.len());
    for c in cats {
        let parent = c
            .parent_id
            .map(|p| format!(" (parent: {})", p.0))
            .unwrap_or_default();
        out.push_str(&format!(
            "- [{}] {} — {} members, stability: {:.2}{}\n",
            c.id.0, c.label, c.member_count, c.stability, parent
        ));
    }
    out
}

pub fn format_neighbors(neighbors: &[(NodeRef, f32)]) -> String {
    let mut out = format!("Found {} neighbors:\n\n", neighbors.len());
    for (nr, weight) in neighbors {
        let (ntype, nid) = match nr {
            NodeRef::Episode(id) => ("episode", id.0),
            NodeRef::Semantic(id) => ("semantic", id.0),
            NodeRef::Preference(id) => ("preference", id.0),
            NodeRef::Category(id) => ("category", id.0),
        };
        out.push_str(&format!("- {ntype} #{nid} (weight: {weight:.3})\n"));
    }
    out
}

pub fn format_node_category(node_id: i64, cat: &Category) -> String {
    let parent = cat
        .parent_id
        .map(|p| format!(" (parent: {})", p.0))
        .unwrap_or_default();
    format!(
        "Node {} belongs to category [{}] '{}' — {} members, stability: {:.2}{}",
        node_id, cat.id.0, cat.label, cat.member_count, cat.stability, parent
    )
}

pub fn format_knowledge_breakdown(
    breakdown: &std::collections::HashMap<SemanticType, u64>,
) -> String {
    let mut parts = Vec::new();
    for (st, label) in [
        (SemanticType::Fact, "facts"),
        (SemanticType::Relationship, "relationships"),
        (SemanticType::Event, "events"),
        (SemanticType::Concept, "concepts"),
    ] {
        if let Some(&count) = breakdown.get(&st) {
            parts.push(format!("{count} {label}"));
        }
    }
    parts.join(", ")
}

pub fn format_category_line(cats: &[Category]) -> String {
    let labels: Vec<&str> = cats.iter().map(|c| c.label.as_str()).collect();
    format!("{} ({})", cats.len(), labels.join(", "))
}

pub fn format_status(
    st: &MemoryStatus,
    session_eps: u32,
    unconsolidated: u32,
    knowledge_line: &str,
    cat_line: &str,
    strongest_desc: &str,
    coverage: &str,
) -> String {
    let mut out = format!("Memory Status:\n  Episodes: {}", st.episode_count);
    if session_eps > 0 || unconsolidated > 0 {
        out.push_str(&format!(
            " ({session_eps} this session, {unconsolidated} unconsolidated)"
        ));
    }
    out.push_str(&format!("\n  Knowledge: {knowledge_line}"));
    out.push_str(&format!("\n  Categories: {cat_line}"));
    out.push_str(&format!(
        "\n  Preferences: {} crystallized, {} impressions accumulating",
        st.preference_count, st.impression_count
    ));
    out.push_str(&format!(
        "\n  Graph: {} links{strongest_desc}",
        st.link_count
    ));
    out.push_str(&format!("\n  Embedding coverage: {coverage}"));
    out
}

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use super::*;
    use crate::{CategoryId, EpisodeId, NodeId, PreferenceId};

    // -----------------------------------------------------------------------
    // format_preferences
    // -----------------------------------------------------------------------

    #[test]
    fn format_preferences_with_data() {
        let prefs = vec![
            Preference {
                id: PreferenceId(1),
                domain: "style".to_string(),
                preference: "concise code".to_string(),
                confidence: 0.85,
                evidence_count: 3,
                first_observed: 1000,
                last_reinforced: 2000,
            },
            Preference {
                id: PreferenceId(2),
                domain: "tone".to_string(),
                preference: "friendly".to_string(),
                confidence: 0.72,
                evidence_count: 5,
                first_observed: 1100,
                last_reinforced: 2100,
            },
        ];
        let result = format_preferences(&prefs);
        assert!(result.contains("Found 2 preferences:"));
        assert!(result.contains("[style] concise code (confidence: 0.85, evidence: 3)"));
        assert!(result.contains("[tone] friendly (confidence: 0.72, evidence: 5)"));
    }

    #[test]
    fn format_preferences_empty() {
        let result = format_preferences(&[]);
        assert!(result.contains("Found 0 preferences:"));
    }

    #[test]
    fn format_preferences_single_item() {
        let prefs = vec![Preference {
            id: PreferenceId(99),
            domain: "coding".to_string(),
            preference: "use snake_case".to_string(),
            confidence: 1.0,
            evidence_count: 1,
            first_observed: 0,
            last_reinforced: 0,
        }];
        let result = format_preferences(&prefs);
        assert!(result.contains("Found 1 preferences:"));
        assert!(result.contains("[coding] use snake_case (confidence: 1.00, evidence: 1)"));
    }

    #[test]
    fn format_preferences_zero_confidence_and_evidence() {
        let prefs = vec![Preference {
            id: PreferenceId(1),
            domain: "style".to_string(),
            preference: "unknown".to_string(),
            confidence: 0.0,
            evidence_count: 0,
            first_observed: 0,
            last_reinforced: 0,
        }];
        let result = format_preferences(&prefs);
        assert!(result.contains("confidence: 0.00, evidence: 0"));
    }

    #[test]
    fn format_preferences_special_characters_in_fields() {
        let prefs = vec![Preference {
            id: PreferenceId(1),
            domain: "lang/framework".to_string(),
            preference: "tabs > spaces (always!)".to_string(),
            confidence: 0.99,
            evidence_count: 42,
            first_observed: 0,
            last_reinforced: 0,
        }];
        let result = format_preferences(&prefs);
        assert!(result.contains("[lang/framework] tabs > spaces (always!)"));
    }

    // -----------------------------------------------------------------------
    // format_categories
    // -----------------------------------------------------------------------

    #[test]
    fn format_categories_with_data() {
        let cats = vec![
            Category {
                id: CategoryId(1),
                label: "programming".to_string(),
                prototype_node: NodeId(1),
                member_count: 10,
                centroid_embedding: None,
                created_at: 1000,
                last_updated: 2000,
                stability: 0.95,
                parent_id: None,
            },
            Category {
                id: CategoryId(2),
                label: "rust-lang".to_string(),
                prototype_node: NodeId(2),
                member_count: 5,
                centroid_embedding: None,
                created_at: 1100,
                last_updated: 2100,
                stability: 0.80,
                parent_id: Some(CategoryId(1)),
            },
        ];
        let result = format_categories(&cats);
        assert!(result.contains("Found 2 categories:"));
        assert!(result.contains("[1] programming"));
        assert!(result.contains("10 members"));
        assert!(result.contains("stability: 0.95"));
        assert!(result.contains("(parent: 1)"));
    }

    #[test]
    fn format_categories_no_parent() {
        let cats = vec![Category {
            id: CategoryId(1),
            label: "general".to_string(),
            prototype_node: NodeId(1),
            member_count: 3,
            centroid_embedding: None,
            created_at: 1000,
            last_updated: 2000,
            stability: 0.5,
            parent_id: None,
        }];
        let result = format_categories(&cats);
        assert!(!result.contains("parent:"));
    }

    #[test]
    fn format_categories_single_item_with_parent() {
        let cats = vec![Category {
            id: CategoryId(7),
            label: "child".to_string(),
            prototype_node: NodeId(1),
            member_count: 1,
            centroid_embedding: None,
            created_at: 0,
            last_updated: 0,
            stability: 0.33,
            parent_id: Some(CategoryId(3)),
        }];
        let result = format_categories(&cats);
        assert!(result.contains("Found 1 categories:"));
        assert!(result.contains("[7] child"));
        assert!(result.contains("1 members"));
        assert!(result.contains("stability: 0.33"));
        assert!(result.contains("(parent: 3)"));
    }

    #[test]
    fn format_categories_zero_members_zero_stability() {
        let cats = vec![Category {
            id: CategoryId(2),
            label: "empty-cat".to_string(),
            prototype_node: NodeId(1),
            member_count: 0,
            centroid_embedding: None,
            created_at: 0,
            last_updated: 0,
            stability: 0.0,
            parent_id: None,
        }];
        let result = format_categories(&cats);
        assert!(result.contains("0 members, stability: 0.00"));
        assert!(!result.contains("parent:"));
    }

    #[test]
    fn format_categories_empty() {
        let result = format_categories(&[]);
        assert!(result.contains("Found 0 categories:"));
    }

    // -----------------------------------------------------------------------
    // format_neighbors
    // -----------------------------------------------------------------------

    #[test]
    fn format_neighbors_all_node_types() {
        let neighbors = vec![
            (NodeRef::Episode(EpisodeId(1)), 0.9),
            (NodeRef::Semantic(NodeId(2)), 0.75),
            (NodeRef::Preference(PreferenceId(3)), 0.5),
            (NodeRef::Category(CategoryId(4)), 0.3),
        ];
        let result = format_neighbors(&neighbors);
        assert!(result.contains("Found 4 neighbors:"));
        assert!(result.contains("episode #1 (weight: 0.900)"));
        assert!(result.contains("semantic #2 (weight: 0.750)"));
        assert!(result.contains("preference #3 (weight: 0.500)"));
        assert!(result.contains("category #4 (weight: 0.300)"));
    }

    #[test]
    fn format_neighbors_empty() {
        let result = format_neighbors(&[]);
        assert!(result.contains("Found 0 neighbors:"));
    }

    #[test]
    fn format_neighbors_single_episode() {
        let neighbors = vec![(NodeRef::Episode(EpisodeId(5)), 1.0_f32)];
        let result = format_neighbors(&neighbors);
        assert!(result.contains("Found 1 neighbors:"));
        assert!(result.contains("episode #5 (weight: 1.000)"));
    }

    #[test]
    fn format_neighbors_single_semantic() {
        let neighbors = vec![(NodeRef::Semantic(NodeId(10)), 0.0_f32)];
        let result = format_neighbors(&neighbors);
        assert!(result.contains("semantic #10 (weight: 0.000)"));
    }

    #[test]
    fn format_neighbors_single_preference() {
        let neighbors = vec![(NodeRef::Preference(PreferenceId(3)), 0.5_f32)];
        let result = format_neighbors(&neighbors);
        assert!(result.contains("preference #3 (weight: 0.500)"));
    }

    #[test]
    fn format_neighbors_single_category() {
        let neighbors = vec![(NodeRef::Category(CategoryId(8)), 0.123_f32)];
        let result = format_neighbors(&neighbors);
        assert!(result.contains("category #8 (weight: 0.123)"));
    }

    // -----------------------------------------------------------------------
    // format_node_category
    // -----------------------------------------------------------------------

    #[test]
    fn format_node_category_with_parent() {
        let cat = Category {
            id: CategoryId(5),
            label: "algorithms".to_string(),
            prototype_node: NodeId(5),
            member_count: 8,
            centroid_embedding: None,
            created_at: 1000,
            last_updated: 2000,
            stability: 0.88,
            parent_id: Some(CategoryId(1)),
        };
        let result = format_node_category(42, &cat);
        assert!(result.contains("Node 42 belongs to category [5] 'algorithms'"));
        assert!(result.contains("8 members"));
        assert!(result.contains("stability: 0.88"));
        assert!(result.contains("(parent: 1)"));
    }

    #[test]
    fn format_node_category_no_parent() {
        let cat = Category {
            id: CategoryId(1),
            label: "general".to_string(),
            prototype_node: NodeId(1),
            member_count: 2,
            centroid_embedding: None,
            created_at: 1000,
            last_updated: 2000,
            stability: 0.5,
            parent_id: None,
        };
        let result = format_node_category(7, &cat);
        assert!(result.contains("Node 7 belongs to category [1] 'general'"));
        assert!(!result.contains("parent:"));
    }

    #[test]
    fn format_node_category_zero_members_zero_stability() {
        let cat = Category {
            id: CategoryId(1),
            label: "empty".to_string(),
            prototype_node: NodeId(1),
            member_count: 0,
            centroid_embedding: None,
            created_at: 0,
            last_updated: 0,
            stability: 0.0,
            parent_id: None,
        };
        let result = format_node_category(42, &cat);
        assert!(result.contains("Node 42 belongs to category [1] 'empty'"));
        assert!(result.contains("0 members, stability: 0.00"));
        assert!(!result.contains("parent:"));
    }

    #[test]
    fn format_node_category_large_node_id() {
        let cat = Category {
            id: CategoryId(999),
            label: "big".to_string(),
            prototype_node: NodeId(1),
            member_count: 100,
            centroid_embedding: None,
            created_at: 0,
            last_updated: 0,
            stability: 1.0,
            parent_id: None,
        };
        let result = format_node_category(1_000_000, &cat);
        assert!(result.contains("Node 1000000 belongs to category [999] 'big'"));
        assert!(result.contains("stability: 1.00"));
    }

    // -----------------------------------------------------------------------
    // format_knowledge_breakdown
    // -----------------------------------------------------------------------

    #[test]
    fn format_knowledge_breakdown_all_types() {
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert(SemanticType::Fact, 10);
        breakdown.insert(SemanticType::Relationship, 5);
        breakdown.insert(SemanticType::Event, 3);
        breakdown.insert(SemanticType::Concept, 7);
        let result = format_knowledge_breakdown(&breakdown);
        assert!(result.contains("10 facts"));
        assert!(result.contains("5 relationships"));
        assert!(result.contains("3 events"));
        assert!(result.contains("7 concepts"));
    }

    #[test]
    fn format_knowledge_breakdown_partial() {
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert(SemanticType::Fact, 2);
        let result = format_knowledge_breakdown(&breakdown);
        assert_eq!(result, "2 facts");
        assert!(!result.contains("relationships"));
    }

    #[test]
    fn format_knowledge_breakdown_empty() {
        let breakdown = std::collections::HashMap::new();
        let result = format_knowledge_breakdown(&breakdown);
        assert_eq!(result, "");
    }

    #[test]
    fn format_knowledge_breakdown_single_type_relationships() {
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert(SemanticType::Relationship, 7_u64);
        let result = format_knowledge_breakdown(&breakdown);
        assert_eq!(result, "7 relationships");
    }

    #[test]
    fn format_knowledge_breakdown_single_type_events() {
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert(SemanticType::Event, 3_u64);
        let result = format_knowledge_breakdown(&breakdown);
        assert_eq!(result, "3 events");
    }

    #[test]
    fn format_knowledge_breakdown_single_type_concepts() {
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert(SemanticType::Concept, 1_u64);
        let result = format_knowledge_breakdown(&breakdown);
        assert_eq!(result, "1 concepts");
    }

    #[test]
    fn format_knowledge_breakdown_zero_count_not_included() {
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert(SemanticType::Fact, 0_u64);
        let result = format_knowledge_breakdown(&breakdown);
        assert_eq!(result, "0 facts");
    }

    #[test]
    fn format_knowledge_breakdown_ordering_facts_first() {
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert(SemanticType::Concept, 2_u64);
        breakdown.insert(SemanticType::Fact, 5_u64);
        let result = format_knowledge_breakdown(&breakdown);
        assert!(result.starts_with("5 facts"));
        assert!(result.contains("2 concepts"));
    }

    // -----------------------------------------------------------------------
    // format_category_line
    // -----------------------------------------------------------------------

    #[test]
    fn format_category_line_with_labels() {
        let cats = vec![
            Category {
                id: CategoryId(1),
                label: "rust".to_string(),
                prototype_node: NodeId(1),
                member_count: 3,
                centroid_embedding: None,
                created_at: 1000,
                last_updated: 2000,
                stability: 0.5,
                parent_id: None,
            },
            Category {
                id: CategoryId(2),
                label: "python".to_string(),
                prototype_node: NodeId(2),
                member_count: 2,
                centroid_embedding: None,
                created_at: 1100,
                last_updated: 2100,
                stability: 0.4,
                parent_id: None,
            },
        ];
        let result = format_category_line(&cats);
        assert_eq!(result, "2 (rust, python)");
    }

    #[test]
    fn format_category_line_empty_slice() {
        let result = format_category_line(&[]);
        assert_eq!(result, "0 ()");
    }

    #[test]
    fn format_category_line_single_item() {
        let cats = vec![Category {
            id: CategoryId(1),
            label: "solo".to_string(),
            prototype_node: NodeId(1),
            member_count: 1,
            centroid_embedding: None,
            created_at: 0,
            last_updated: 0,
            stability: 1.0,
            parent_id: None,
        }];
        let result = format_category_line(&cats);
        assert_eq!(result, "1 (solo)");
    }

    // -----------------------------------------------------------------------
    // format_status
    // -----------------------------------------------------------------------

    #[test]
    fn format_status_full() {
        let st = MemoryStatus {
            episode_count: 50,
            semantic_node_count: 20,
            preference_count: 3,
            impression_count: 10,
            link_count: 45,
            embedding_count: 30,
            category_count: 2,
        };
        let result = format_status(
            &st,
            5,
            3,
            "10 facts, 5 relationships",
            "2 (rust, python)",
            " (strongest: \"a\" <-> \"b\" weight 0.95)",
            "30/70 nodes (42%)",
        );
        assert!(result.contains("Memory Status:"));
        assert!(result.contains("Episodes: 50"));
        assert!(result.contains("5 this session, 3 unconsolidated"));
        assert!(result.contains("Knowledge: 10 facts, 5 relationships"));
        assert!(result.contains("Categories: 2 (rust, python)"));
        assert!(result.contains("3 crystallized, 10 impressions"));
        assert!(result.contains("45 links"));
        assert!(result.contains("strongest: \"a\" <-> \"b\" weight 0.95"));
        assert!(result.contains("Embedding coverage: 30/70 nodes (42%)"));
    }

    #[test]
    fn format_status_no_session_data() {
        let st = MemoryStatus {
            episode_count: 0,
            semantic_node_count: 0,
            preference_count: 0,
            impression_count: 0,
            link_count: 0,
            embedding_count: 0,
            category_count: 0,
        };
        let result = format_status(&st, 0, 0, "none", "0", "", "0/0 nodes");
        assert!(result.contains("Episodes: 0"));
        assert!(!result.contains("this session"));
    }

    #[test]
    fn format_status_all_zeros() {
        let st = MemoryStatus {
            episode_count: 0,
            semantic_node_count: 0,
            preference_count: 0,
            impression_count: 0,
            link_count: 0,
            embedding_count: 0,
            category_count: 0,
        };
        let result = format_status(&st, 0, 0, "", "0 ()", "", "0/0");
        assert!(result.contains("Episodes: 0"));
        assert!(!result.contains("this session"));
        assert!(result.contains("Preferences: 0 crystallized, 0 impressions"));
        assert!(result.contains("Graph: 0 links"));
    }

    #[test]
    fn format_status_session_eps_nonzero_but_unconsolidated_zero() {
        let st = MemoryStatus {
            episode_count: 5,
            semantic_node_count: 0,
            preference_count: 0,
            impression_count: 0,
            link_count: 0,
            embedding_count: 0,
            category_count: 0,
        };
        let result = format_status(&st, 5, 0, "5 facts", "1 (rust)", "", "5/5");
        assert!(result.contains("Episodes: 5"));
        assert!(result.contains("5 this session, 0 unconsolidated"));
    }

    #[test]
    fn format_status_strongest_desc_included() {
        let st = MemoryStatus {
            episode_count: 10,
            semantic_node_count: 3,
            preference_count: 1,
            impression_count: 0,
            link_count: 8,
            embedding_count: 3,
            category_count: 1,
        };
        let result = format_status(
            &st,
            0,
            0,
            "3 facts",
            "1 (general)",
            " — strongest: Rust",
            "3/3",
        );
        assert!(result.contains("Graph: 8 links — strongest: Rust"));
    }
}
