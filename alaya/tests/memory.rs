#![allow(deprecated)]

use alaya::*;

mod common;

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

/// Store `count` episodes in a session, returning all created IDs.
fn store_n_episodes(
    store: &AlayaStore,
    session: &str,
    count: usize,
    base_ts: i64,
) -> Vec<EpisodeId> {
    (0..count)
        .map(|i| {
            store
                .store_episode(&episode(
                    &format!("Episode {i} in session {session} about Rust programming"),
                    session,
                    base_ts + (i as i64) * 100,
                ))
                .unwrap()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test 2: Persistence across open/close
// ---------------------------------------------------------------------------

#[test]
fn test_persistence_across_open_close() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("persistence_test.db");

    // First session: open, store episodes, drop
    {
        let store = AlayaStore::open(&db_path).unwrap();
        store
            .store_episode(&episode(
                "Rust has zero-cost abstractions",
                "persist-s1",
                1000,
            ))
            .unwrap();
        store
            .store_episode(&episode(
                "Ownership prevents data races",
                "persist-s1",
                2000,
            ))
            .unwrap();
        store
            .store_episode(&episode(
                "The borrow checker catches bugs at compile time",
                "persist-s1",
                3000,
            ))
            .unwrap();

        let status = store.status().unwrap();
        assert_eq!(status.episode_count, 3);
        // store is dropped here
    }

    // Second session: reopen the same file and verify data survived
    {
        let store = AlayaStore::open(&db_path).unwrap();

        let status = store.status().unwrap();
        assert_eq!(
            status.episode_count, 3,
            "episodes should persist across open/close"
        );

        // Query should still find results
        let results = store.query(&Query::simple("Rust")).unwrap();
        assert!(
            !results.is_empty(),
            "query should return persisted episodes after reopen"
        );

        // Verify content is intact by checking that "zero-cost" appears in results
        let has_zero_cost = results.iter().any(|m| m.content.contains("zero-cost"));
        assert!(has_zero_cost, "persisted content should be retrievable");
    }
}

// ---------------------------------------------------------------------------
// Test: learn() creates knowledge (provider-less consolidation)
// ---------------------------------------------------------------------------

#[test]
fn test_learn_creates_knowledge() {
    let store = AlayaStore::open_in_memory().unwrap();

    // Store 5 episodes to serve as source references
    let ep_ids = store_n_episodes(&store, "learn-s1", 5, 1_000);

    // Learn 2 facts referencing those episodes
    let nodes = vec![
        NewSemanticNode {
            content: "User programs in Rust".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.9,
            source_episodes: vec![ep_ids[0], ep_ids[1], ep_ids[2]],
            embedding: None,
        },
        NewSemanticNode {
            content: "User prefers functional style".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.85,
            source_episodes: vec![ep_ids[3], ep_ids[4]],
            embedding: None,
        },
    ];

    let report = store.learn(nodes).unwrap();
    assert_eq!(report.nodes_created, 2, "should create 2 semantic nodes");
    assert_eq!(
        report.links_created, 5,
        "should create 5 Causal links (3 + 2 source episodes)"
    );

    // Verify knowledge() returns the learned facts
    let knowledge = store.knowledge_nodes(None).unwrap();
    assert_eq!(knowledge.len(), 2, "should have 2 semantic nodes");
    let contents: Vec<&str> = knowledge.iter().map(|n| n.content.as_str()).collect();
    assert!(contents.contains(&"User programs in Rust"));
    assert!(contents.contains(&"User prefers functional style"));

    // Verify status counts increase
    let status = store.status().unwrap();
    assert_eq!(status.semantic_node_count, 2);
    assert_eq!(status.episode_count, 5);
}

// ---------------------------------------------------------------------------
// Test: learn() creates Causal links between facts and source episodes
// ---------------------------------------------------------------------------

#[test]
fn test_learn_creates_causal_links() {
    let store = AlayaStore::open_in_memory().unwrap();

    let ep_ids = store_n_episodes(&store, "causal-s1", 3, 1_000);

    let nodes = vec![NewSemanticNode {
        content: "User discusses Rust ownership".to_string(),
        node_type: SemanticType::Fact,
        confidence: 0.8,
        source_episodes: ep_ids.clone(),
        embedding: None,
    }];

    let report = store.learn(nodes).unwrap();
    assert_eq!(report.nodes_created, 1);
    assert_eq!(report.links_created, 3);

    // Get the created semantic node
    let knowledge = store.knowledge_nodes(None).unwrap();
    assert_eq!(knowledge.len(), 1);
    let node_id = knowledge[0].id;

    // Verify neighbors() finds Causal links to source episodes
    let neighbors = store.neighbors(NodeRef::Semantic(node_id), 1).unwrap();
    assert!(
        !neighbors.is_empty(),
        "semantic node should have episode neighbors via Causal links"
    );

    // All 3 source episodes should appear as neighbors
    for ep_id in &ep_ids {
        let has_ep = neighbors
            .iter()
            .any(|(nr, _)| *nr == NodeRef::Episode(*ep_id));
        assert!(
            has_ep,
            "episode {ep_id:?} should be a neighbor of the semantic node",
        );
    }
}

// ---------------------------------------------------------------------------
// Test: learn() marks source episodes as consolidated
// ---------------------------------------------------------------------------

#[test]
fn test_learn_marks_episodes_consolidated() {
    let store = AlayaStore::open_in_memory().unwrap();

    // Store 5 episodes
    let ep_ids = store_n_episodes(&store, "consol-s1", 5, 1_000);

    // All 5 should initially be unconsolidated
    let uncons = store.unconsolidated_episodes(100).unwrap();
    assert_eq!(
        uncons.len(),
        5,
        "all 5 episodes should be unconsolidated before learn()"
    );

    // Learn a fact referencing episodes 0, 1, 2
    let nodes = vec![NewSemanticNode {
        content: "Fact from first three episodes".to_string(),
        node_type: SemanticType::Fact,
        confidence: 0.8,
        source_episodes: vec![ep_ids[0], ep_ids[1], ep_ids[2]],
        embedding: None,
    }];

    store.learn(nodes).unwrap();

    // Episodes 0, 1, 2 should now be consolidated (linked to a semantic node)
    let uncons = store.unconsolidated_episodes(100).unwrap();
    assert_eq!(
        uncons.len(),
        2,
        "only 2 episodes (3 and 4) should remain unconsolidated"
    );

    // The unconsolidated ones should be episodes 3 and 4
    let uncons_ids: Vec<EpisodeId> = uncons.iter().map(|e| e.id).collect();
    assert!(uncons_ids.contains(&ep_ids[3]));
    assert!(uncons_ids.contains(&ep_ids[4]));
}

// ---------------------------------------------------------------------------
// Test: learn() with empty vec produces zero report
// ---------------------------------------------------------------------------

#[test]
fn test_learn_empty_vec() {
    let store = AlayaStore::open_in_memory().unwrap();

    let report = store.learn(vec![]).unwrap();
    assert_eq!(report.nodes_created, 0);
    assert_eq!(report.links_created, 0);
    assert_eq!(report.categories_assigned, 0);
    assert_eq!(report.episodes_processed, 0);
}
