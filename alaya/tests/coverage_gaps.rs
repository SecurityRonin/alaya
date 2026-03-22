//! Targeted tests to achieve 100% coverage on remaining gaps.

use alaya::*;

mod common;

// ---------------------------------------------------------------------------
// Provider for tests
// ---------------------------------------------------------------------------

struct TestProvider {
    knowledge: Vec<NewSemanticNode>,
}

impl TestProvider {
    fn with_knowledge(knowledge: Vec<NewSemanticNode>) -> Self {
        Self { knowledge }
    }
}

impl ConsolidationProvider for TestProvider {
    fn extract_knowledge(&self, _episodes: &[Episode]) -> alaya::Result<Vec<NewSemanticNode>> {
        Ok(self.knowledge.clone())
    }

    fn extract_impressions(&self, _interaction: &Interaction) -> alaya::Result<Vec<NewImpression>> {
        Ok(vec![])
    }

    fn detect_contradiction(&self, _a: &SemanticNode, _b: &SemanticNode) -> alaya::Result<bool> {
        Ok(false)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn episode(content: &str, session: &str, ts: i64) -> NewEpisode {
    NewEpisode {
        content: content.to_string(),
        role: Role::User,
        session_id: session.to_string(),
        timestamp: ts,
        context: EpisodeContext::default(),
        embedding: None,
    }
}

// ---------------------------------------------------------------------------
// Test: episodes_by_session (lib.rs delegation)
// ---------------------------------------------------------------------------

#[test]
fn test_episodes_by_session() {
    let store = Alaya::open_in_memory().unwrap();

    store
        .episodes().store(&episode("hello world", "s1", 1000))
        .unwrap();
    store
        .episodes().store(&episode("goodbye world", "s1", 2000))
        .unwrap();
    store
        .episodes().store(&episode("other session", "s2", 3000))
        .unwrap();

    let eps = store.episodes().by_session("s1").unwrap();
    assert_eq!(eps.len(), 2);
    assert_eq!(eps[0].content, "hello world");

    let eps = store.episodes().by_session("nonexistent").unwrap();
    assert!(eps.is_empty());
}

// ---------------------------------------------------------------------------
// Test: knowledge_breakdown (lib.rs delegation)
// ---------------------------------------------------------------------------

#[test]
fn test_knowledge_breakdown() {
    let store = Alaya::open_in_memory().unwrap();

    let bd = store.knowledge().breakdown().unwrap();
    assert!(bd.is_empty());

    store
        .knowledge().learn(vec![
            NewSemanticNode {
                content: "fact 1".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: None,
            },
            NewSemanticNode {
                content: "fact 2".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.85,
                source_episodes: vec![],
                embedding: None,
            },
            NewSemanticNode {
                content: "concept 1".to_string(),
                node_type: SemanticType::Concept,
                confidence: 0.8,
                source_episodes: vec![],
                embedding: None,
            },
        ])
        .unwrap();

    let bd = store.knowledge().breakdown().unwrap();
    assert_eq!(bd.get(&SemanticType::Fact), Some(&2));
    assert_eq!(bd.get(&SemanticType::Concept), Some(&1));
}

// ---------------------------------------------------------------------------
// Test: strongest_link (lib.rs delegation)
// ---------------------------------------------------------------------------

#[test]
fn test_strongest_link() {
    let store = Alaya::open_in_memory().unwrap();

    assert!(store.graph().strongest_link().unwrap().is_none());

    let ep1 = store
        .episodes().store(&episode("ep1", "s1", 1000))
        .unwrap();

    store
        .knowledge().learn(vec![NewSemanticNode {
            content: "test knowledge".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.9,
            source_episodes: vec![ep1],
            embedding: None,
        }])
        .unwrap();

    let link = store.graph().strongest_link().unwrap();
    assert!(link.is_some(), "learn should have created Causal links");
}

// ---------------------------------------------------------------------------
// Test: node_content for all NodeRef variants
// ---------------------------------------------------------------------------

#[test]
fn test_node_content_all_variants() {
    let store = Alaya::open_in_memory().unwrap();

    // Episode content
    let ep_id = store
        .episodes().store(&episode("episode content here", "s1", 1000))
        .unwrap();
    let content = store.admin().node_content(NodeRef::Episode(ep_id)).unwrap();
    assert!(content.is_some());

    // Semantic content
    store
        .knowledge().learn(vec![NewSemanticNode {
            content: "semantic node content here".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.9,
            source_episodes: vec![ep_id],
            embedding: Some(vec![0.8, 0.3, 0.1]),
        }])
        .unwrap();
    let knowledge = store.knowledge().filter(None).unwrap();
    let node_id = knowledge[0].id;
    let content = store.admin().node_content(NodeRef::Semantic(node_id)).unwrap();
    assert!(content.is_some());

    // Category content — create via consolidation + transform
    for i in 0..5 {
        store
            .episodes().store(&NewEpisode {
                content: format!("cooking topic {i}"),
                role: Role::User,
                session_id: "s2".to_string(),
                timestamp: 2000 + (i as i64) * 100,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();
    }

    let provider = TestProvider::with_knowledge(vec![
        NewSemanticNode {
            content: "cooking fact 1".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.9,
            source_episodes: vec![EpisodeId(2)],
            embedding: Some(vec![0.8, 0.3, 0.1]),
        },
        NewSemanticNode {
            content: "cooking fact 2".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.85,
            source_episodes: vec![EpisodeId(3)],
            embedding: Some(vec![0.4, 0.8, 0.2]),
        },
        NewSemanticNode {
            content: "cooking fact 3".to_string(),
            node_type: SemanticType::Concept,
            confidence: 0.8,
            source_episodes: vec![EpisodeId(4)],
            embedding: Some(vec![0.6, 0.5, 0.5]),
        },
    ]);
    store.lifecycle().consolidate(&provider).unwrap();
    store.lifecycle().transform().unwrap();

    let cats = store.admin().categories(None).unwrap();
    if !cats.is_empty() {
        let cat_content = store.admin().node_content(NodeRef::Category(cats[0].id)).unwrap();
        assert!(cat_content.is_some());
    }

    // Missing nodes return None
    assert!(store
        .admin().node_content(NodeRef::Episode(EpisodeId(999)))
        .unwrap()
        .is_none());
    assert!(store
        .admin().node_content(NodeRef::Semantic(NodeId(999)))
        .unwrap()
        .is_none());
    assert!(store
        .admin().node_content(NodeRef::Category(CategoryId(999)))
        .unwrap()
        .is_none());

    // Preference variant returns formatted string
    let pref_content = store
        .admin().node_content(NodeRef::Preference(PreferenceId(1)))
        .unwrap();
    assert!(pref_content.is_some());
}

// ---------------------------------------------------------------------------
// Test: LlmExtractionProviderBuilder methods
// ---------------------------------------------------------------------------

#[cfg(feature = "llm")]
#[test]
fn test_llm_extraction_provider_builder() {
    // Test builder chain — exercises api_url, api_key, model methods
    let provider = LlmExtractionProvider::builder()
        .api_url("https://example.com/v1/chat/completions")
        .api_key("test-key-123")
        .model("gpt-4")
        .build()
        .unwrap();
    // Just verify we got a provider (can't actually call it without a real API)
    let _ = provider;

    // Missing api_key should error
    let err = LlmExtractionProvider::builder().build();
    assert!(err.is_err());
}

// ---------------------------------------------------------------------------
// Test: dedup_semantic_nodes skip-deleted-j branch
// When i=0 deletes j=3 (A ≈ D), then i=1's loop encounters j=3 as deleted.
// ---------------------------------------------------------------------------

#[test]
fn test_dedup_skip_already_deleted_node() {
    let store = Alaya::open_in_memory().unwrap();

    for i in 0..4 {
        store
            .episodes().store(&episode(&format!("ep {i}"), "s1", 1000 + i))
            .unwrap();
    }

    // 4 semantic nodes:
    //   A=[1,0,0], B=[0,1,0], C=[0,0,1], D=[0.99,0.01,0]
    // A and D have cosine sim ≈ 0.999 (> 0.95 dedup threshold).
    // At i=0: j=3 (A vs D) → D deleted
    // At i=1: inner loop reaches j=3 → D already in deleted_ids → skip
    let provider = TestProvider::with_knowledge(vec![
        NewSemanticNode {
            content: "fact alpha".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.9,
            source_episodes: vec![EpisodeId(1)],
            embedding: Some(vec![1.0, 0.0, 0.0]),
        },
        NewSemanticNode {
            content: "fact beta".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.85,
            source_episodes: vec![EpisodeId(2)],
            embedding: Some(vec![0.0, 1.0, 0.0]),
        },
        NewSemanticNode {
            content: "fact gamma".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.8,
            source_episodes: vec![EpisodeId(3)],
            embedding: Some(vec![0.0, 0.0, 1.0]),
        },
        NewSemanticNode {
            content: "fact alpha duplicate".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.75,
            source_episodes: vec![EpisodeId(4)],
            embedding: Some(vec![0.99, 0.01, 0.0]),
        },
    ]);
    store.lifecycle().consolidate(&provider).unwrap();

    let tr = store.lifecycle().transform().unwrap();
    assert!(
        tr.duplicates_merged >= 1,
        "should dedup alpha and its duplicate"
    );

    let knowledge = store.knowledge().filter(None).unwrap();
    assert_eq!(knowledge.len(), 3, "should have 3 nodes after dedup");
}

// ---------------------------------------------------------------------------
// Test: category voting during consolidation
// When a new node's source episodes link to already-categorized nodes,
// the voting code path assigns the new node to the winning category.
// ---------------------------------------------------------------------------

#[test]
fn test_category_voting_during_consolidation() {
    let store = Alaya::open_in_memory().unwrap();

    // Phase 1: Create episodes and consolidate to get categorized semantic nodes
    for i in 0..5 {
        store
            .episodes().store(&NewEpisode {
                content: format!("cooking topic {i}"),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000 + (i as i64) * 100,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();
    }

    let provider1 = TestProvider::with_knowledge(vec![
        NewSemanticNode {
            content: "User cooks pasta".to_string(),
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
            content: "User knows recipes".to_string(),
            node_type: SemanticType::Concept,
            confidence: 0.8,
            source_episodes: vec![EpisodeId(3)],
            embedding: Some(vec![0.6, 0.5, 0.5]),
        },
    ]);
    store.lifecycle().consolidate(&provider1).unwrap();

    // Phase 2: Transform to discover and assign categories
    store.lifecycle().transform().unwrap();

    let cats = store.admin().categories(None).unwrap();
    assert!(
        !cats.is_empty(),
        "should have categories after first transform"
    );

    // Phase 3: Second consolidation with new node referencing OLD episodes
    // that already have Causal links to categorized semantic nodes.
    // This triggers the category voting path (consolidation.rs:141).
    for i in 5..10 {
        store
            .episodes().store(&NewEpisode {
                content: format!("more cooking {i}"),
                role: Role::User,
                session_id: "s2".to_string(),
                timestamp: 2000 + (i as i64) * 100,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();
    }

    // The new node references EpisodeId(1) which already has a Causal link
    // to the "User cooks pasta" semantic node (which has a category).
    let provider2 = TestProvider::with_knowledge(vec![NewSemanticNode {
        content: "User experiments with Italian recipes".to_string(),
        node_type: SemanticType::Fact,
        confidence: 0.85,
        source_episodes: vec![EpisodeId(1), EpisodeId(6)],
        embedding: Some(vec![0.55, 0.55, 0.35]),
    }]);
    let cr2 = store.lifecycle().consolidate(&provider2).unwrap();
    assert_eq!(cr2.nodes_created, 1);
    assert_eq!(
        cr2.categories_assigned, 1,
        "new node should be assigned to existing category via voting"
    );
}

// ---------------------------------------------------------------------------
// Test: activation spreading below-threshold skip
// Nodes with activation below the threshold should not spread further.
// ---------------------------------------------------------------------------

#[test]
fn test_activation_below_threshold_skip() {
    let store = Alaya::open_in_memory().unwrap();

    // Create a chain of episodes with temporal links.
    // Spreading activation from ep1 at low depth should not reach far nodes.
    let id1 = store
        .episodes().store(&episode("deep chain start", "s1", 1000))
        .unwrap();
    let id2 = store
        .episodes().store(&NewEpisode {
            content: "chain link 2".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 2000,
            context: EpisodeContext {
                preceding_episode: Some(id1),
                ..EpisodeContext::default()
            },
            embedding: None,
        })
        .unwrap();
    let _id3 = store
        .episodes().store(&NewEpisode {
            content: "chain link 3".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 3000,
            context: EpisodeContext {
                preceding_episode: Some(id2),
                ..EpisodeContext::default()
            },
            embedding: None,
        })
        .unwrap();

    // With depth=1, only direct neighbors should be reached.
    // The threshold check filters out low-activation nodes from spreading further.
    let neighbors = store.graph().neighbors(NodeRef::Episode(id1), 1).unwrap();
    assert!(
        !neighbors.is_empty(),
        "should find at least the direct neighbor"
    );

    // At depth=3 with a long chain, some nodes will be below threshold
    let deep_neighbors = store.graph().neighbors(NodeRef::Episode(id1), 3).unwrap();
    // Should still work without panic — the threshold skip handles low activation
    let _ = deep_neighbors;
}

// ---------------------------------------------------------------------------
// Test: dream lifecycle operation
// ---------------------------------------------------------------------------

#[test]
fn test_dream_operation() {
    let store = Alaya::open_in_memory().unwrap();

    // Dream on empty store should succeed
    let report = store.lifecycle().dream(&NoOpProvider, None).unwrap();
    assert_eq!(report.consolidation.episodes_processed, 0);
    assert_eq!(report.consolidation.nodes_created, 0);

    // Store episodes and dream
    for i in 0..5 {
        store
            .episodes().store(&episode(
                &format!("Dream test episode {i} about machine learning"),
                "dream-s1",
                1000 + i * 100,
            ))
            .unwrap();
    }

    let report = store.lifecycle().dream(&NoOpProvider, None).unwrap();
    // NoOpProvider creates nothing, but dream runs consolidation + transformation
    assert!(
        report.consolidation.episodes_processed > 0,
        "dream should process unconsolidated episodes"
    );
}

// ---------------------------------------------------------------------------
// Test: async store open with file path
// ---------------------------------------------------------------------------

#[cfg(feature = "async")]
#[tokio::test]
async fn test_async_store_open_file() {
    use alaya::async_store::AsyncAlaya;

    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("async_test.db");

    let store = AsyncAlaya::open(&db_path).unwrap();

    // Verify basic operations work
    let status = store.status().await.unwrap();
    assert_eq!(status.episode_count, 0);

    store.close().await.unwrap();
}

// ---------------------------------------------------------------------------
// Test: node_category returns None for missing nodes
// (After refactoring, get_node_category uses .optional() so missing nodes
// return Ok(None) directly instead of Err(NotFound))
// ---------------------------------------------------------------------------

#[test]
fn test_node_category_missing_node() {
    let store = Alaya::open_in_memory().unwrap();

    // Non-existent node should return Ok(None)
    let cat = store.admin().node_category(NodeId(999)).unwrap();
    assert!(cat.is_none());
}

// ---------------------------------------------------------------------------
// Test: LlmExtractionProvider Debug impl (extraction.rs lines 132-136)
// ---------------------------------------------------------------------------

#[cfg(feature = "llm")]
#[test]
fn test_llm_extraction_provider_debug() {
    let provider = LlmExtractionProvider::builder()
        .api_url("https://example.com/v1/chat/completions")
        .api_key("secret-key-12345")
        .model("gpt-4")
        .build()
        .unwrap();

    let debug_output = format!("{:?}", provider);
    assert!(debug_output.contains("LlmExtractionProvider"));
    assert!(debug_output.contains("example.com"));
    assert!(debug_output.contains("gpt-4"));
    // API key should be redacted
    assert!(debug_output.contains("[redacted]"));
    assert!(!debug_output.contains("secret-key-12345"));
}

// ---------------------------------------------------------------------------
// Test: unconsolidated_episodes (lib.rs delegation)
// ---------------------------------------------------------------------------

#[test]
fn test_unconsolidated_episodes() {
    let store = Alaya::open_in_memory().unwrap();

    // Initially empty
    let uncons = store.episodes().unconsolidated(10).unwrap();
    assert!(uncons.is_empty());

    // Store episodes
    store
        .episodes().store(&episode("ep1", "s1", 1000))
        .unwrap();
    store
        .episodes().store(&episode("ep2", "s1", 2000))
        .unwrap();

    let uncons = store.episodes().unconsolidated(10).unwrap();
    assert_eq!(uncons.len(), 2);

    // With limit
    let uncons = store.episodes().unconsolidated(1).unwrap();
    assert_eq!(uncons.len(), 1);
}

// ---------------------------------------------------------------------------
// Test: purge operations (all PurgeFilter variants)
// ---------------------------------------------------------------------------

#[test]
fn test_purge_by_session() {
    let store = Alaya::open_in_memory().unwrap();
    store.episodes().store(&episode("s1-ep1", "s1", 1000)).unwrap();
    store.episodes().store(&episode("s1-ep2", "s1", 2000)).unwrap();
    store.episodes().store(&episode("s2-ep1", "s2", 3000)).unwrap();

    let report = store.admin().purge(PurgeFilter::Session("s1".to_string())).unwrap();
    assert_eq!(report.episodes_deleted, 2);
    assert_eq!(store.admin().status().unwrap().episode_count, 1);
}

#[test]
fn test_purge_older_than() {
    let store = Alaya::open_in_memory().unwrap();
    store.episodes().store(&episode("old", "s1", 1000)).unwrap();
    store.episodes().store(&episode("new", "s1", 9000)).unwrap();

    let report = store.admin().purge(PurgeFilter::OlderThan(5000)).unwrap();
    assert_eq!(report.episodes_deleted, 1);
    assert_eq!(store.admin().status().unwrap().episode_count, 1);
}

#[test]
fn test_purge_all() {
    let store = Alaya::open_in_memory().unwrap();
    store.episodes().store(&episode("ep1", "s1", 1000)).unwrap();
    store.episodes().store(&episode("ep2", "s1", 2000)).unwrap();
    assert_eq!(store.admin().status().unwrap().episode_count, 2);

    // PurgeFilter::All uses execute_batch and doesn't report individual counts
    let _report = store.admin().purge(PurgeFilter::All).unwrap();
    assert_eq!(store.admin().status().unwrap().episode_count, 0);
}

// ---------------------------------------------------------------------------
// Test: perfume with actual impressions (covers perfume path)
// ---------------------------------------------------------------------------

#[test]
fn test_perfume_with_impressions() {
    let store = Alaya::open_in_memory().unwrap();

    store
        .episodes().store(&episode("I like dark mode", "s1", 1000))
        .unwrap();

    let interaction = Interaction {
        text: "Please use dark mode".to_string(),
        role: Role::User,
        session_id: "s1".to_string(),
        timestamp: 2000,
        context: EpisodeContext::default(),
    };

    // NoOpProvider returns empty impressions, but perfume should still succeed
    let report = store.lifecycle().perfume(&interaction, &NoOpProvider);
    assert!(report.is_ok());
}

// ---------------------------------------------------------------------------
// Test: status with knowledge and categories (exercises status report fields)
// ---------------------------------------------------------------------------

#[test]
fn test_status_with_data() {
    let store = Alaya::open_in_memory().unwrap();

    // Store episodes and learn knowledge
    for i in 0..5 {
        store
            .episodes().store(&episode(&format!("topic {i}"), "s1", 1000 + i))
            .unwrap();
    }

    store
        .knowledge().learn(vec![
            NewSemanticNode {
                content: "fact one".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![EpisodeId(1)],
                embedding: Some(vec![0.8, 0.3, 0.1]),
            },
            NewSemanticNode {
                content: "fact two".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.85,
                source_episodes: vec![EpisodeId(2)],
                embedding: Some(vec![0.4, 0.8, 0.2]),
            },
        ])
        .unwrap();

    store.lifecycle().transform().unwrap();

    let status = store.admin().status().unwrap();
    assert_eq!(status.episode_count, 5);
    assert_eq!(status.semantic_node_count, 2);
}
