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

/// Store `count` episodes in a session, returning all created IDs.
fn store_n_episodes(
    store: &Alaya,
    session: &str,
    count: usize,
    base_ts: i64,
) -> Vec<EpisodeId> {
    (0..count)
        .map(|i| {
            store
                .episodes().store(&episode(
                    &format!("Episode {i} in session {session} about Rust programming"),
                    session,
                    base_ts + (i as i64) * 100,
                ))
                .unwrap()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test 1: Multi-session lifecycle
// ---------------------------------------------------------------------------

#[test]
fn test_multi_session_lifecycle() {
    let store = Alaya::open_in_memory().unwrap();

    // Store episodes across 3 sessions (4 each = 12 total)
    let _s1_ids = store_n_episodes(&store, "session-1", 4, 1_000);
    let _s2_ids = store_n_episodes(&store, "session-2", 4, 2_000);
    let _s3_ids = store_n_episodes(&store, "session-3", 4, 3_000);

    let status = store.admin().status().unwrap();
    assert_eq!(
        status.episode_count, 12,
        "should have 12 episodes across 3 sessions"
    );

    // Query -- BM25 should find episodes mentioning "Rust"
    let results = store.knowledge().query(&Query::simple("Rust programming")).unwrap();
    assert!(!results.is_empty(), "query should return matching episodes");

    // Consolidate with NoOpProvider -- won't create semantic nodes but should
    // process episodes (>= 3 unconsolidated) and report them.
    let cr = store.lifecycle().consolidate(&NoOpProvider).unwrap();
    // NoOp returns empty knowledge, so no nodes created, but episodes_processed
    // should reflect the batch that was read.
    assert_eq!(
        cr.nodes_created, 0,
        "NoOp provider creates no semantic nodes"
    );
    // With NoOp, episodes_processed should be > 0 because we have >= 3
    // unconsolidated episodes and the batch is read before the provider returns
    // an empty vec.
    assert!(
        cr.episodes_processed > 0,
        "consolidation should process the unconsolidated batch (got {})",
        cr.episodes_processed
    );

    // Perfume -- extract impressions from an interaction (part of the lifecycle)
    let perfume_provider = TestProvider::with_impressions(vec![NewImpression {
        domain: "lifecycle".to_string(),
        observation: "user tests full lifecycle".to_string(),
        valence: 0.8,
    }]);
    let interaction = Interaction {
        text: "lifecycle test interaction".to_string(),
        role: Role::User,
        session_id: "session-1".to_string(),
        timestamp: 5000,
        context: EpisodeContext::default(),
    };
    let pr = store.lifecycle().perfume(&interaction, &perfume_provider).unwrap();
    assert_eq!(
        pr.impressions_stored, 1,
        "perfume should store 1 impression"
    );

    // Transform -- dedup/prune/decay pass (no duplicates expected)
    let tr = store.lifecycle().transform().unwrap();
    // With only episodes and no semantic nodes or preferences, everything
    // should be zero (no duplicates to merge, no links to prune, etc.)
    assert_eq!(tr.duplicates_merged, 0);

    // After storing 12 episodes, each has an initialized strength record.
    // Forget decays retrieval strength on all strength records.
    let fr = store.lifecycle().forget().unwrap();
    assert!(
        fr.nodes_decayed > 0,
        "forget should decay the 12 episode strength records (got {})",
        fr.nodes_decayed
    );

    // Status should still reflect 12 episodes (forget only archives nodes
    // with very low strength, and a single decay pass won't drop them that far).
    let final_status = store.admin().status().unwrap();
    assert_eq!(
        final_status.episode_count, 12,
        "episodes should survive a single forget pass"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Multi-session purge isolation
// ---------------------------------------------------------------------------

#[test]
fn test_multi_session_purge_isolation() {
    let store = Alaya::open_in_memory().unwrap();

    // Store episodes in three sessions
    let _alpha_ids = store_n_episodes(&store, "alpha", 3, 1_000);
    let _beta_ids = store_n_episodes(&store, "beta", 3, 2_000);
    let _gamma_ids = store_n_episodes(&store, "gamma", 3, 3_000);

    assert_eq!(store.admin().status().unwrap().episode_count, 9);

    // Purge session "beta"
    let purge_report = store
        .admin().purge(PurgeFilter::Session("beta".to_string()))
        .unwrap();
    assert_eq!(
        purge_report.episodes_deleted, 3,
        "purging 'beta' should delete its 3 episodes"
    );

    let status = store.admin().status().unwrap();
    assert_eq!(
        status.episode_count, 6,
        "6 episodes should remain after purging 'beta'"
    );

    // Verify "alpha" episodes survive by querying
    let alpha_results = store.knowledge().query(&Query::simple("session alpha")).unwrap();
    assert!(
        !alpha_results.is_empty(),
        "'alpha' episodes should survive the purge of 'beta'"
    );

    // Purge with OlderThan: alpha timestamps are 1000-1200, gamma are 3000-3200.
    // Use a cutoff of 2500 to remove alpha but keep gamma.
    let purge_report2 = store.admin().purge(PurgeFilter::OlderThan(2500)).unwrap();
    assert_eq!(
        purge_report2.episodes_deleted, 3,
        "OlderThan(2500) should remove the 3 'alpha' episodes"
    );

    let final_status = store.admin().status().unwrap();
    assert_eq!(
        final_status.episode_count, 3,
        "only 'gamma' episodes should remain"
    );

    // Verify the remaining episodes are from gamma
    let gamma_results = store.knowledge().query(&Query::simple("session gamma")).unwrap();
    assert!(
        !gamma_results.is_empty(),
        "'gamma' episodes should still be queryable"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Lifecycle idempotence
// ---------------------------------------------------------------------------

#[test]
fn test_lifecycle_idempotence() {
    // Part A: Run lifecycle operations twice on an empty DB -- should produce
    // consistent zero reports and no errors.
    let store = Alaya::open_in_memory().unwrap();

    for pass in 0..2 {
        let cr = store.lifecycle().consolidate(&NoOpProvider).unwrap();
        assert_eq!(cr.episodes_processed, 0, "empty consolidate pass {pass}");
        assert_eq!(cr.nodes_created, 0, "empty consolidate pass {pass}");

        let tr = store.lifecycle().transform().unwrap();
        assert_eq!(tr.duplicates_merged, 0, "empty transform pass {pass}");
        assert_eq!(tr.links_pruned, 0, "empty transform pass {pass}");

        let fr = store.lifecycle().forget().unwrap();
        assert_eq!(fr.nodes_decayed, 0, "empty forget pass {pass}");
        assert_eq!(fr.nodes_archived, 0, "empty forget pass {pass}");
    }

    // Part B: Store episodes and run the lifecycle twice. The second pass
    // should not panic and should produce a consistent state.
    let _ids = store_n_episodes(&store, "idempotence", 6, 1_000);

    let status_before = store.admin().status().unwrap();
    assert_eq!(status_before.episode_count, 6);

    // First lifecycle pass
    let cr1 = store.lifecycle().consolidate(&NoOpProvider).unwrap();
    let _tr1 = store.lifecycle().transform().unwrap();
    let fr1 = store.lifecycle().forget().unwrap();

    // Consolidation should process the batch (>= 3 episodes)
    assert!(
        cr1.episodes_processed > 0,
        "first consolidate should process episodes"
    );
    // Forget should decay the 6 strength records
    assert!(fr1.nodes_decayed > 0, "first forget should decay nodes");

    // Second lifecycle pass -- should not panic
    let cr2 = store.lifecycle().consolidate(&NoOpProvider).unwrap();
    let _tr2 = store.lifecycle().transform().unwrap();
    let fr2 = store.lifecycle().forget().unwrap();

    // NoOp never creates semantic links, so episodes remain "unconsolidated"
    // and the second consolidation pass should still process them.
    assert!(
        cr2.episodes_processed > 0,
        "second consolidate should re-process (NoOp leaves no links)"
    );

    // Forget should still decay whatever strength records exist
    assert!(
        fr2.nodes_decayed > 0,
        "second forget should still decay nodes"
    );

    // Status should be consistent -- episodes should still exist (strength
    // hasn't dropped below archive thresholds after only 2 decay passes).
    let status_after = store.admin().status().unwrap();
    assert_eq!(
        status_after.episode_count, 6,
        "episodes should survive two lifecycle passes"
    );

    // Sanity: transform twice in a row is also fine
    let tr_a = store.lifecycle().transform().unwrap();
    let tr_b = store.lifecycle().transform().unwrap();
    assert_eq!(tr_a.duplicates_merged, tr_b.duplicates_merged);
}

// ---------------------------------------------------------------------------
// Test 6: Preference crystallization end-to-end
// ---------------------------------------------------------------------------

#[test]
fn test_preference_crystallization_e2e() {
    let store = Alaya::open_in_memory().unwrap();

    // TestProvider returns one impression in "code_style" domain per perfume call
    let provider = TestProvider::with_impressions(vec![NewImpression {
        domain: "code_style".to_string(),
        observation: "prefers functional style".to_string(),
        valence: 0.9,
    }]);

    // Perfume 4 times -- below CRYSTALLIZATION_THRESHOLD (5)
    for i in 0..4 {
        let interaction = Interaction {
            text: format!("I like map/filter/fold {i}"),
            role: Role::User,
            session_id: "crystal-s1".to_string(),
            timestamp: 1000 + i * 100,
            context: EpisodeContext::default(),
        };
        let report = store.lifecycle().perfume(&interaction, &provider).unwrap();
        assert_eq!(
            report.preferences_crystallized, 0,
            "pass {i}: should not crystallize below threshold"
        );
    }

    // No preferences yet
    let prefs = store.admin().preferences(Some("code_style")).unwrap();
    assert!(
        prefs.is_empty(),
        "no preference should exist before threshold"
    );

    // 5th perfume crosses the threshold -- crystallization happens
    let interaction = Interaction {
        text: "I like map/filter/fold 4".to_string(),
        role: Role::User,
        session_id: "crystal-s1".to_string(),
        timestamp: 1400,
        context: EpisodeContext::default(),
    };
    let report = store.lifecycle().perfume(&interaction, &provider).unwrap();
    assert_eq!(
        report.preferences_crystallized, 1,
        "5th impression should trigger crystallization"
    );

    // Verify the crystallized preference
    let prefs = store.admin().preferences(Some("code_style")).unwrap();
    assert_eq!(prefs.len(), 1);
    assert_eq!(prefs[0].domain, "code_style");

    // Unrelated domain should be empty
    let other = store.admin().preferences(Some("other_domain")).unwrap();
    assert!(other.is_empty());

    // All preferences (no filter) should include the one we created
    let all = store.admin().preferences(None).unwrap();
    assert!(!all.is_empty());

    // 6th perfume should reinforce the existing preference, not create a new one
    let interaction = Interaction {
        text: "functional style is great".to_string(),
        role: Role::User,
        session_id: "crystal-s1".to_string(),
        timestamp: 1500,
        context: EpisodeContext::default(),
    };
    let report = store.lifecycle().perfume(&interaction, &provider).unwrap();
    assert_eq!(
        report.preferences_crystallized, 0,
        "should reinforce, not re-crystallize"
    );
    assert_eq!(
        report.preferences_reinforced, 1,
        "should reinforce existing preference"
    );

    // Still exactly one preference
    let prefs = store.admin().preferences(Some("code_style")).unwrap();
    assert_eq!(prefs.len(), 1);
}

// ---------------------------------------------------------------------------
// Test 7: Memory decay and revival (Bjork desirable difficulty)
// ---------------------------------------------------------------------------

#[test]
fn test_memory_decay_and_revival() {
    let store = Alaya::open_in_memory().unwrap();

    // Store episodes that will be our "memories"
    store
        .episodes().store(&episode("Rust async runtime uses tokio", "decay-s1", 1000))
        .unwrap();
    store
        .episodes().store(&episode(
            "Tokio has a multi-threaded scheduler",
            "decay-s1",
            2000,
        ))
        .unwrap();
    store
        .episodes().store(&episode("Async functions return futures", "decay-s1", 3000))
        .unwrap();

    let status = store.admin().status().unwrap();
    assert_eq!(status.episode_count, 3);

    // Run forget 10 times to decay retrieval strength significantly.
    // Each pass multiplies retrieval by 0.95. After 10 passes: 1.0 * 0.95^10 ≈ 0.60
    for _ in 0..10 {
        let report = store.lifecycle().forget().unwrap();
        assert!(
            report.nodes_decayed > 0,
            "should decay strength records each pass"
        );
    }

    // Episodes should still exist (storage strength stays at 0.5, well above
    // the archive threshold of 0.1, so no archival happens)
    assert_eq!(
        store.admin().status().unwrap().episode_count,
        3,
        "episodes should survive 10 decay passes"
    );

    // Now REVIVE the memory by querying it. The retrieval pipeline calls
    // on_access() for each returned result, which resets retrieval_strength to 1.0.
    let results = store.knowledge().query(&Query::simple("Rust async tokio")).unwrap();
    assert!(
        !results.is_empty(),
        "decayed memories should still be retrievable (they're latent, not gone)"
    );

    // After querying, run more forget passes. The revived memories should
    // have retrieval_strength reset to 1.0, so they survive more decay.
    for _ in 0..5 {
        store.lifecycle().forget().unwrap();
    }

    // Still alive after 15 total decay passes (10 before revival + 5 after)
    // because the query revived retrieval strength to 1.0 midway
    assert_eq!(
        store.admin().status().unwrap().episode_count,
        3,
        "revived memories should survive additional decay passes"
    );

    // The memories should still be queryable after all this
    let results = store.knowledge().query(&Query::simple("tokio")).unwrap();
    assert!(
        !results.is_empty(),
        "revived memories should still be queryable after further decay"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Emergent category lifecycle
// ---------------------------------------------------------------------------

/// Test the full emergent category lifecycle:
/// 1. Store episodes with embeddings
/// 2. Consolidate -> creates semantic nodes (no categories yet)
/// 3. Transform -> discovers categories from clustered nodes
/// 4. Second consolidation -> assigns new nodes to existing categories
#[test]
fn test_emergent_category_lifecycle() {
    let store = Alaya::open_in_memory().unwrap();

    // Phase 1: Store episodes about cooking
    for i in 0..5 {
        store
            .episodes().store(&NewEpisode {
                content: format!(
                    "I made {} for dinner",
                    ["pasta", "risotto", "gnocchi", "lasagna", "ravioli"][i]
                ),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000 + (i as i64) * 100,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();
    }

    // Phase 2: Consolidate with provider that returns semantic nodes with embeddings.
    // Embeddings are chosen so that:
    //   - pairwise cosine similarity is in [0.7, 0.95) -> clusters together
    //   - no pair exceeds 0.95 -> avoids dedup merging
    //   - source_episodes are non-overlapping -> avoids link conflicts
    let provider = TestProvider::with_knowledge(vec![
        NewSemanticNode {
            content: "User cooks pasta dishes".to_string(),
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
            content: "User knows many Italian recipes".to_string(),
            node_type: SemanticType::Concept,
            confidence: 0.8,
            source_episodes: vec![EpisodeId(3)],
            embedding: Some(vec![0.6, 0.5, 0.5]),
        },
        NewSemanticNode {
            content: "User enjoys cooking dinner".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.75,
            source_episodes: vec![EpisodeId(4)],
            embedding: Some(vec![0.3, 0.6, 0.7]),
        },
    ]);
    let cr = store.lifecycle().consolidate(&provider).unwrap();
    assert_eq!(cr.nodes_created, 4);
    // No categories exist yet -- consolidation doesn't create them
    assert_eq!(cr.categories_assigned, 0);
    assert!(
        store.admin().categories(None).unwrap().is_empty(),
        "no categories should exist before transform"
    );

    // Phase 3: Transform discovers categories from clustered semantic nodes
    let tr = store.lifecycle().transform().unwrap();
    assert!(
        tr.categories_discovered >= 1,
        "transform should discover at least 1 category from 4 similar nodes"
    );

    let cats = store.admin().categories(None).unwrap();
    assert!(!cats.is_empty(), "should have categories after transform");
    let cat = &cats[0];
    assert!(
        cat.member_count >= 3,
        "category should have at least 3 members (cluster minimum)"
    );

    // Phase 4: Second consolidation -- new nodes should be assigned to existing category
    // Store more episodes for the second batch
    for i in 5..10 {
        store
            .episodes().store(&NewEpisode {
                content: format!("cooking episode {i}"),
                role: Role::User,
                session_id: "s2".to_string(),
                timestamp: 2000 + (i as i64) * 100,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();
    }

    let provider2 = TestProvider::with_knowledge(vec![NewSemanticNode {
        content: "User experiments with Italian recipes".to_string(),
        node_type: SemanticType::Fact,
        confidence: 0.85,
        source_episodes: vec![EpisodeId(6), EpisodeId(7)],
        embedding: Some(vec![0.55, 0.55, 0.35]), // similar to cooking cluster centroid
    }]);
    let cr2 = store.lifecycle().consolidate(&provider2).unwrap();
    assert_eq!(cr2.nodes_created, 1);
    assert_eq!(
        cr2.categories_assigned, 1,
        "new node with similar embedding should be assigned to existing cooking category"
    );

    // Verify the node was actually assigned
    let knowledge = store.knowledge().filter(None).unwrap();
    let new_node = knowledge
        .iter()
        .find(|n| n.content == "User experiments with Italian recipes")
        .unwrap();
    let node_cat = store.admin().node_category(new_node.id).unwrap();
    assert!(node_cat.is_some(), "new node should have a category");
}

// ---------------------------------------------------------------------------
// Test 9: Category survives transform cycles (stability tracking)
// ---------------------------------------------------------------------------

#[test]
fn test_category_survives_transform_cycles() {
    let store = Alaya::open_in_memory().unwrap();

    // Create a batch of episodes and consolidate
    for i in 0..5 {
        store
            .episodes().store(&NewEpisode {
                content: format!("Rust memory management topic {i}"),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000 + (i as i64) * 100,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();
    }

    // Embeddings chosen so pairwise cosine sim is in [0.7, 0.95) -- clusters but no dedup
    let provider = TestProvider::with_knowledge(vec![
        NewSemanticNode {
            content: "Rust ownership".to_string(),
            node_type: SemanticType::Concept,
            confidence: 0.9,
            source_episodes: vec![EpisodeId(1)],
            embedding: Some(vec![0.3, 0.8, 0.1]),
        },
        NewSemanticNode {
            content: "Rust borrowing".to_string(),
            node_type: SemanticType::Concept,
            confidence: 0.85,
            source_episodes: vec![EpisodeId(2)],
            embedding: Some(vec![0.1, 0.7, 0.6]),
        },
        NewSemanticNode {
            content: "Rust lifetimes".to_string(),
            node_type: SemanticType::Concept,
            confidence: 0.8,
            source_episodes: vec![EpisodeId(3)],
            embedding: Some(vec![0.4, 0.6, 0.5]),
        },
    ]);
    store.lifecycle().consolidate(&provider).unwrap();

    // First transform: discovers category
    store.lifecycle().transform().unwrap();
    let cats = store.admin().categories(None).unwrap();
    assert!(!cats.is_empty());
    let initial_stability = cats[0].stability;

    // Second transform: should increment stability
    store.lifecycle().transform().unwrap();
    let cats = store.admin().categories(None).unwrap();
    assert!(!cats.is_empty());
    assert!(
        cats[0].stability > initial_stability,
        "stability should increase after surviving a transform cycle"
    );
}
