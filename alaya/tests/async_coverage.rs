//! Comprehensive tests for AsyncAlaya to cover all async method bodies
//! and actor match arms.

#![cfg(feature = "async")]

use alaya::async_store::AsyncAlaya;
use alaya::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn new_episode(content: &str, session: &str, ts: i64) -> NewEpisode {
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
// Test: exercises every async read method
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_async_read_methods() {
    let store = AsyncAlaya::open_in_memory().unwrap();

    // Store some episodes
    let ep1 = store.store_episode(new_episode("hello world", "s1", 1000)).await.unwrap();
    let ep2 = store.store_episode(new_episode("goodbye world", "s1", 2000)).await.unwrap();
    store.store_episode(new_episode("other session", "s2", 3000)).await.unwrap();

    // status
    let status = store.status().await.unwrap();
    assert_eq!(status.episode_count, 3);

    // query
    let results = store
        .query(Query {
            text: "hello".to_string(),
            embedding: None,
            context: QueryContext::default(),
            max_results: 5,
            boost_categories: None,
        })
        .await
        .unwrap();
    assert!(!results.is_empty());

    // preferences (empty)
    let prefs = store.preferences(None).await.unwrap();
    assert!(prefs.is_empty());

    // preferences with domain filter
    let prefs = store.preferences(Some("style".to_string())).await.unwrap();
    assert!(prefs.is_empty());

    // knowledge (empty)
    let knowledge = store.knowledge(None).await.unwrap();
    assert!(knowledge.is_empty());

    // knowledge with filter
    let knowledge = store
        .knowledge(Some(KnowledgeFilter {
            node_type: Some(SemanticType::Fact),
            min_confidence: None,
            category: None,
            limit: None,
        }))
        .await
        .unwrap();
    assert!(knowledge.is_empty());

    // categories
    let cats = store.categories(None).await.unwrap();
    assert!(cats.is_empty());

    let cats = store.categories(Some(0.5)).await.unwrap();
    assert!(cats.is_empty());

    // subcategories (no categories exist, so use a dummy id)
    let subcats = store.subcategories(CategoryId(1)).await.unwrap();
    assert!(subcats.is_empty());

    // node_category
    let cat = store.node_category(NodeId(999)).await.unwrap();
    assert!(cat.is_none());

    // neighbors
    let neighbors = store.neighbors(NodeRef::Episode(ep1), 1).await.unwrap();
    // ep1 should have temporal link to ep2 (same session)
    let _ = neighbors;

    // strongest_link
    let link = store.strongest_link().await.unwrap();
    // May or may not have a link depending on temporal links
    let _ = link;

    // node_content
    let content = store.node_content(NodeRef::Episode(ep1)).await.unwrap();
    assert!(content.is_some());
    let missing = store.node_content(NodeRef::Episode(EpisodeId(999))).await.unwrap();
    assert!(missing.is_none());

    // knowledge_breakdown
    let breakdown = store.knowledge_breakdown().await.unwrap();
    assert!(breakdown.is_empty());

    // episodes_by_session
    let eps = store.episodes_by_session("s1".to_string()).await.unwrap();
    assert_eq!(eps.len(), 2);

    let eps = store.episodes_by_session("nonexistent".to_string()).await.unwrap();
    assert!(eps.is_empty());

    // unconsolidated_episodes
    let uncons = store.unconsolidated_episodes(10).await.unwrap();
    assert_eq!(uncons.len(), 3);

    store.close().await.unwrap();
}

// ---------------------------------------------------------------------------
// Test: exercises all lifecycle async methods
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_async_lifecycle_methods() {
    let store = AsyncAlaya::open_in_memory().unwrap();

    // Store episodes for lifecycle operations
    let ep1 = store.store_episode(new_episode("I love Rust programming", "s1", 1000)).await.unwrap();
    store.store_episode(new_episode("Tokio is great for async", "s1", 2000)).await.unwrap();
    store.store_episode(new_episode("SQLite is reliable", "s1", 3000)).await.unwrap();

    // consolidate (uses NoOpProvider by default — returns 0 nodes)
    let report = store.consolidate().await.unwrap();
    assert_eq!(report.nodes_created, 0);

    // learn
    let report = store
        .learn(vec![
            NewSemanticNode {
                content: "User likes Rust".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![ep1],
                embedding: None,
            },
        ])
        .await
        .unwrap();
    assert_eq!(report.nodes_created, 1);

    // Set extraction provider so auto_consolidate doesn't error
    store
        .set_extraction_provider(Box::new(MockExtractionProvider::empty()))
        .await
        .unwrap();

    // auto_consolidate (empty mock returns 0 nodes)
    let report = store.auto_consolidate().await.unwrap();
    assert_eq!(report.nodes_created, 0);

    // transform
    let report = store.transform().await.unwrap();
    let _ = report;

    // forget
    let report = store.forget().await.unwrap();
    let _ = report;

    // dream
    let report = store.dream(None).await.unwrap();
    let _ = report;

    // dream with interaction
    let report = store
        .dream(Some(Interaction {
            text: "Tell me about Rust".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 4000,
            context: EpisodeContext::default(),
        }))
        .await
        .unwrap();
    let _ = report;

    // perfume
    let report = store
        .perfume(Interaction {
            text: "I prefer dark mode".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 5000,
            context: EpisodeContext::default(),
        })
        .await
        .unwrap();
    let _ = report;

    // purge (by session)
    store.store_episode(new_episode("to be purged", "purge-me", 6000)).await.unwrap();
    let report = store.purge(PurgeFilter::Session("purge-me".to_string())).await.unwrap();
    assert_eq!(report.episodes_deleted, 1);

    store.close().await.unwrap();
}

// ---------------------------------------------------------------------------
// Test: exercises provider configuration methods
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_async_set_providers() {
    let store = AsyncAlaya::open_in_memory().unwrap();

    // set_consolidation_provider
    store
        .set_consolidation_provider(Box::new(NoOpProvider))
        .await
        .unwrap();

    // set_embedding_provider
    store
        .set_embedding_provider(Box::new(MockEmbeddingProvider::new(3)))
        .await
        .unwrap();

    // set_extraction_provider
    store
        .set_extraction_provider(Box::new(MockExtractionProvider::empty()))
        .await
        .unwrap();

    // Verify extraction provider works by auto_consolidating
    store.store_episode(new_episode("test", "s1", 1000)).await.unwrap();
    let report = store.auto_consolidate().await.unwrap();
    assert_eq!(report.nodes_created, 0); // empty mock returns nothing

    store.close().await.unwrap();
}

// ---------------------------------------------------------------------------
// Test: knowledge and categories after learn + transform
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_async_knowledge_and_categories() {
    let store = AsyncAlaya::open_in_memory().unwrap();

    // Store episodes and learn knowledge
    for i in 0..5 {
        store
            .store_episode(new_episode(
                &format!("Cooking topic {i}"),
                "s1",
                1000 + i * 100,
            ))
            .await
            .unwrap();
    }

    store
        .learn(vec![
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
        ])
        .await
        .unwrap();

    // knowledge with data
    let knowledge = store.knowledge(None).await.unwrap();
    assert_eq!(knowledge.len(), 3);

    // knowledge_breakdown with data
    let breakdown = store.knowledge_breakdown().await.unwrap();
    assert_eq!(breakdown.get(&SemanticType::Fact), Some(&2));

    // transform to discover categories
    store.transform().await.unwrap();

    let cats = store.categories(None).await.unwrap();
    if !cats.is_empty() {
        // subcategories
        let subcats = store.subcategories(cats[0].id).await.unwrap();
        let _ = subcats;

        // node_category for a known semantic node
        let node_id = knowledge[0].id;
        let cat = store.node_category(node_id).await.unwrap();
        let _ = cat;

        // node_content for category
        let content = store.node_content(NodeRef::Category(cats[0].id)).await.unwrap();
        assert!(content.is_some());
    }

    // neighbors (may or may not have links depending on episode relationships)
    let neighbors = store.neighbors(NodeRef::Episode(EpisodeId(1)), 2).await.unwrap();
    let _ = neighbors; // just exercising the path

    // strongest_link with links
    let link = store.strongest_link().await.unwrap();
    assert!(link.is_some());

    store.close().await.unwrap();
}
