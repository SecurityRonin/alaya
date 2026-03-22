use alaya::*;

mod common;

// ---------------------------------------------------------------------------
// Test provider (implements ConsolidationProvider for E2E tests)
// ---------------------------------------------------------------------------

/// A simple rule-based provider for integration tests.
/// MockProvider is #[cfg(test)] inside the crate, so we implement
/// ConsolidationProvider directly here for E2E coverage.
struct TestProvider {
    /// If set, extract_knowledge returns this node for any batch.
    knowledge: Vec<NewSemanticNode>,
    /// If set, extract_impressions returns these for any interaction.
    impressions: Vec<NewImpression>,
}

impl TestProvider {
    fn with_knowledge(knowledge: Vec<NewSemanticNode>) -> Self {
        Self {
            knowledge,
            impressions: vec![],
        }
    }

    #[allow(dead_code)]
    fn with_impressions(impressions: Vec<NewImpression>) -> Self {
        Self {
            knowledge: vec![],
            impressions,
        }
    }
}

impl ConsolidationProvider for TestProvider {
    fn extract_knowledge(&self, _episodes: &[Episode]) -> alaya::Result<Vec<NewSemanticNode>> {
        Ok(self.knowledge.clone())
    }

    fn extract_impressions(&self, _interaction: &Interaction) -> alaya::Result<Vec<NewImpression>> {
        Ok(self.impressions.clone())
    }

    fn detect_contradiction(&self, _a: &SemanticNode, _b: &SemanticNode) -> alaya::Result<bool> {
        Ok(false)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a `NewEpisode` with sensible defaults.
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

/// Build a `NewEpisode` that is chained to a preceding episode (creates a temporal link).
fn chained_episode(content: &str, session: &str, ts: i64, preceding: EpisodeId) -> NewEpisode {
    NewEpisode {
        content: content.to_string(),
        role: Role::User,
        session_id: session.to_string(),
        timestamp: ts,
        context: EpisodeContext {
            preceding_episode: Some(preceding),
            ..EpisodeContext::default()
        },
        embedding: None,
    }
}

// ---------------------------------------------------------------------------
// Test 3: Full retrieval pipeline with temporal links
// ---------------------------------------------------------------------------

#[test]
fn test_full_retrieval_pipeline_with_temporal_links() {
    let store = Alaya::open_in_memory().unwrap();

    // Store 5 episodes chained via preceding_episode to create temporal links.
    let id1 = store
        .episodes().store(&episode("Rust memory safety guarantees", "chain-s1", 1000))
        .unwrap();
    let id2 = store
        .episodes().store(&chained_episode(
            "The borrow checker enforces ownership rules",
            "chain-s1",
            2000,
            id1,
        ))
        .unwrap();
    let id3 = store
        .episodes().store(&chained_episode(
            "Lifetimes prevent dangling references",
            "chain-s1",
            3000,
            id2,
        ))
        .unwrap();
    let id4 = store
        .episodes().store(&chained_episode(
            "Smart pointers like Box and Rc manage heap allocation",
            "chain-s1",
            4000,
            id3,
        ))
        .unwrap();
    let _id5 = store
        .episodes().store(&chained_episode(
            "Unsafe blocks opt out of the borrow checker",
            "chain-s1",
            5000,
            id4,
        ))
        .unwrap();

    // Verify links were created (4 temporal links for the chain of 5)
    let status = store.admin().status().unwrap();
    assert_eq!(status.episode_count, 5);
    assert_eq!(
        status.link_count, 4,
        "should have 4 temporal links for a chain of 5 episodes"
    );

    // Verify Hebbian co-retrieval: querying creates co-retrieval links between
    // results. Count links before and after the FIRST query that returns multiple results.
    let links_before = store.admin().status().unwrap().link_count;
    let results = store.knowledge().query(&Query::simple("borrow checker")).unwrap();
    assert!(
        !results.is_empty(),
        "query should return episodes about 'borrow checker'"
    );
    if results.len() >= 2 {
        let links_after = store.admin().status().unwrap().link_count;
        assert!(
            links_after > links_before,
            "co-retrieval should create Hebbian links between co-retrieved episodes ({links_before} -> {links_after})",
        );
    }

    // Spreading activation from the first episode should find temporal neighbors
    let neighbors = store.graph().neighbors(NodeRef::Episode(id1), 2).unwrap();
    assert!(
        !neighbors.is_empty(),
        "spreading activation from episode 1 should find temporal neighbors"
    );

    // The direct neighbor of id1 should be id2 (the immediately chained episode)
    let neighbor_refs: Vec<NodeRef> = neighbors.iter().map(|(nr, _)| *nr).collect();
    assert!(
        neighbor_refs.contains(&NodeRef::Episode(id2)),
        "episode 2 should be a neighbor of episode 1 via temporal link"
    );
}

// ---------------------------------------------------------------------------
// Test 10: Cross-domain bridging via category MemberOf links
// ---------------------------------------------------------------------------

#[test]
fn test_cross_domain_bridging_via_categories() {
    let store = Alaya::open_in_memory().unwrap();

    // Store episodes about two related subtopics
    for i in 0..5 {
        store
            .episodes().store(&NewEpisode {
                content: format!("Rust async programming with tokio topic {i}"),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000 + (i as i64) * 100,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();
    }

    // Consolidate with semantic nodes that will cluster together
    // Embeddings chosen so all 4 nodes cluster (cosine sim >= 0.7)
    let provider = TestProvider::with_knowledge(vec![
        NewSemanticNode {
            content: "User programs in Rust async".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.9,
            source_episodes: vec![EpisodeId(1)],
            embedding: Some(vec![0.8, 0.3, 0.1]),
        },
        NewSemanticNode {
            content: "User uses tokio runtime".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.85,
            source_episodes: vec![EpisodeId(2)],
            embedding: Some(vec![0.4, 0.8, 0.2]),
        },
        NewSemanticNode {
            content: "User builds async web services".to_string(),
            node_type: SemanticType::Concept,
            confidence: 0.8,
            source_episodes: vec![EpisodeId(3)],
            embedding: Some(vec![0.6, 0.5, 0.5]),
        },
    ]);

    let cr = store.lifecycle().consolidate(&provider).unwrap();
    assert_eq!(cr.nodes_created, 3, "should create 3 semantic nodes");

    // Transform: discovers categories and creates MemberOf links
    let tr = store.lifecycle().transform().unwrap();
    assert!(
        tr.categories_discovered >= 1,
        "should discover at least 1 category from 3 similar nodes"
    );

    // Verify MemberOf links exist in the graph
    let status = store.admin().status().unwrap();
    assert!(
        status.link_count > 0,
        "should have links including MemberOf links"
    );

    // Verify categories were created with members
    let cats = store.admin().categories(None).unwrap();
    assert!(!cats.is_empty(), "should have categories");
    assert!(
        cats[0].member_count >= 3,
        "category should have at least 3 members"
    );

    // Query for one subtopic — spreading activation should follow:
    // Semantic node → (MemberOf) → Category → (MemberOf) → other Semantic nodes
    let results = store.knowledge().query(&Query::simple("tokio async")).unwrap();
    assert!(!results.is_empty(), "query should return results");

    // Verify neighbors() can traverse through category
    // Get the first semantic node
    let knowledge = store.knowledge().filter(None).unwrap();
    assert!(knowledge.len() >= 3);

    // Use neighbors() from one semantic node — should find others through category
    let node_ref = NodeRef::Semantic(knowledge[0].id);
    let neighbors = store.graph().neighbors(node_ref, 2).unwrap();
    assert!(
        !neighbors.is_empty(),
        "semantic node should have neighbors (at least Category via MemberOf)"
    );

    // At depth 2, neighbors should include the category AND other semantic nodes
    // that share the category through MemberOf links
    let has_category = neighbors
        .iter()
        .any(|(nr, _)| matches!(nr, NodeRef::Category(_)));
    assert!(
        has_category,
        "neighbors at depth 2 should include the category node via MemberOf"
    );
}
