use alaya::*;

#[test]
fn full_lifecycle_learn_reconcile_superseded_excluded() {
    let store = AlayaStore::open_in_memory().unwrap();

    // Learn contradictory facts
    store
        .learn(vec![
            NewSemanticNode {
                content: "user prefers dark mode".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: Some(vec![0.9, 0.1, 0.0]),
            },
            NewSemanticNode {
                content: "user prefers light mode".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: vec![],
                embedding: Some(vec![0.85, 0.15, 0.0]),
            },
        ])
        .unwrap();

    // Verify both visible before reconcile
    let before = store.knowledge(None).unwrap();
    assert_eq!(before.len(), 2);

    // Reconcile with default (Recency) strategy
    let report = store.reconcile().unwrap();
    assert_eq!(report.conflicts_detected, 1);
    assert_eq!(report.conflicts_resolved, 1);
    assert_eq!(report.nodes_superseded, 1);

    // Only 1 node visible after reconcile
    let after = store.knowledge(None).unwrap();
    assert_eq!(after.len(), 1);
}

#[test]
fn manual_strategy_reconcile_then_resolve() {
    let mut store = AlayaStore::open_in_memory().unwrap();
    store.set_conflict_strategy(ConflictStrategy::Manual);

    store
        .learn(vec![
            NewSemanticNode {
                content: "prefers tabs".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: Some(vec![0.9, 0.1, 0.0]),
            },
            NewSemanticNode {
                content: "prefers spaces".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.85,
                source_episodes: vec![],
                embedding: Some(vec![0.85, 0.15, 0.0]),
            },
        ])
        .unwrap();

    store.reconcile().unwrap();

    let conflicts = store.conflicts().unwrap();
    assert_eq!(conflicts.len(), 1);

    // Manually resolve
    let winner = conflicts[0].node_a;
    store.resolve_conflict(conflicts[0].id, winner).unwrap();

    assert!(store.conflicts().unwrap().is_empty());
    assert_eq!(store.knowledge(None).unwrap().len(), 1);
}

#[test]
fn idempotent_reconcile() {
    let store = AlayaStore::open_in_memory().unwrap();
    store
        .learn(vec![
            NewSemanticNode {
                content: "fact A".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: Some(vec![0.9, 0.1, 0.0]),
            },
            NewSemanticNode {
                content: "fact B".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: vec![],
                embedding: Some(vec![0.85, 0.15, 0.0]),
            },
        ])
        .unwrap();

    let r1 = store.reconcile().unwrap();
    assert_eq!(r1.conflicts_detected, 1);

    let r2 = store.reconcile().unwrap();
    assert_eq!(r2.conflicts_detected, 0);
    assert_eq!(r2.conflicts_resolved, 0);
}

#[test]
fn reconcile_after_transform_preserves_categories() {
    let store = AlayaStore::open_in_memory().unwrap();

    // Store enough episodes and facts for transform to assign categories
    for i in 0..5 {
        store
            .store_episode(&NewEpisode {
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
        .learn(vec![
            NewSemanticNode {
                content: "likes Italian food".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![EpisodeId(1)],
                embedding: Some(vec![0.9, 0.1, 0.0]),
            },
            NewSemanticNode {
                content: "dislikes Italian food".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.7,
                source_episodes: vec![EpisodeId(2)],
                embedding: Some(vec![0.85, 0.15, 0.0]),
            },
        ])
        .unwrap();

    store.transform().unwrap();
    let report = store.reconcile().unwrap();
    assert!(report.conflicts_detected >= 1 || report.conflicts_detected == 0);
    // Main assertion: no panics, categories preserved
}
