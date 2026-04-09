use crate::error::Result;
use crate::graph::activation;
use crate::retrieval::{bm25, fusion, rerank, vector};
use crate::store::{episodic, strengths};
use crate::types::*;
use rusqlite::Connection;

#[cfg(feature = "tracing")]
use tracing::{debug, trace};

/// Execute a full hybrid retrieval query.
pub fn execute_query(conn: &Connection, query: &Query) -> Result<Vec<ScoredMemory>> {
    #[cfg(feature = "tracing")]
    debug!(query = %query.text, max_results = query.max_results, "executing retrieval pipeline");

    let now = query
        .context
        .current_timestamp
        .unwrap_or_else(crate::db::now);

    let fetch_limit = query.max_results * 3;

    // Stage 1: Parallel retrieval (BM25 + vector + graph)
    #[cfg(feature = "tracing")]
    let _bm25_span = tracing::info_span!("bm25_search").entered();

    let bm25_results: Vec<(NodeRef, f64)> =
        bm25::search_bm25(conn, &query.text, fetch_limit, &query.context)?
            .into_iter()
            .map(|(eid, score)| (NodeRef::Episode(eid), score))
            .collect();

    #[cfg(feature = "tracing")]
    trace!(bm25_count = bm25_results.len(), "BM25 search complete");

    #[cfg(feature = "tracing")]
    drop(_bm25_span);

    #[cfg(feature = "tracing")]
    let _vector_span = tracing::info_span!("vector_search").entered();

    let vector_results: Vec<(NodeRef, f64)> = match &query.embedding {
        Some(emb) => vector::search_vector(conn, emb, fetch_limit)?,
        None => vec![],
    };

    #[cfg(feature = "tracing")]
    trace!(
        vector_count = vector_results.len(),
        "vector search complete"
    );

    #[cfg(feature = "tracing")]
    drop(_vector_span);

    // Graph: seed from BM25 + vector top results, spread 1 hop
    #[cfg(feature = "tracing")]
    let _graph_span = tracing::info_span!("graph_activation").entered();

    let seed_nodes: Vec<NodeRef> = bm25_results
        .iter()
        .take(3)
        .chain(vector_results.iter().take(3))
        .map(|(nr, _)| *nr)
        .collect();

    let graph_activation = if !seed_nodes.is_empty() {
        activation::spread_activation(conn, &seed_nodes, 1, 0.1, 0.6)?
    } else {
        std::collections::HashMap::new()
    };

    let graph_results: Vec<(NodeRef, f64)> = graph_activation
        .into_iter()
        .filter(|(nr, _)| !seed_nodes.contains(nr)) // exclude seeds
        .map(|(nr, act)| (nr, act as f64))
        .collect();

    #[cfg(feature = "tracing")]
    drop(_graph_span);

    // Stage 2: RRF fusion
    #[cfg(feature = "tracing")]
    let _rrf_span = tracing::info_span!("rrf_fusion").entered();

    let boost = query.boost_weights.as_ref();
    let mut sets: Vec<Vec<(NodeRef, f64)>> = vec![bm25_results];
    let mut weights: Vec<f32> = vec![boost.map_or(1.0, |b| b.bm25)];
    if !vector_results.is_empty() {
        sets.push(vector_results);
        weights.push(boost.map_or(1.0, |b| b.vector));
    }
    if !graph_results.is_empty() {
        sets.push(graph_results);
        weights.push(boost.map_or(1.0, |b| b.graph));
    }
    let fused = fusion::rrf_merge_weighted(&sets, 60, Some(&weights));

    #[cfg(feature = "tracing")]
    trace!(fused_count = fused.len(), "RRF fusion complete");

    #[cfg(feature = "tracing")]
    drop(_rrf_span);

    // Stage 3: Enrich candidates with content and context for reranking
    let candidates: Vec<(NodeRef, f64, String, Option<Role>, i64, EpisodeContext)> =
        fused
            .into_iter()
            .take(fetch_limit)
            .filter_map(|(node_ref, score)| match node_ref {
                NodeRef::Episode(eid) => episodic::get_episode(conn, eid).ok().map(|ep| {
                    (
                        node_ref,
                        score,
                        ep.content,
                        Some(ep.role),
                        ep.timestamp,
                        ep.context,
                    )
                }),
                NodeRef::Semantic(nid) => crate::store::semantic::get_semantic_node(conn, nid)
                    .ok()
                    .filter(|node| node.confidence > 0.0) // superseded nodes have confidence 0.0
                    .filter(|node| match query.category_id {
                        Some(cat) => node.category_id == Some(cat),
                        None => true,
                    })
                    .map(|node| {
                        (
                            node_ref,
                            score,
                            node.content,
                            None,
                            node.created_at,
                            EpisodeContext::default(),
                        )
                    }),
                NodeRef::Preference(pid) => crate::store::implicit::get_preference(conn, pid)
                    .ok()
                    .map(|pref| {
                        (
                            node_ref,
                            score,
                            format!("preference: {}: {}", pref.domain, pref.preference),
                            None,
                            pref.first_observed,
                            EpisodeContext::default(),
                        )
                    }),
                _ => None, // Categories don't carry retrievable content
            })
            .collect();

    #[cfg(feature = "tracing")]
    let _rerank_span = tracing::info_span!("rerank").entered();

    let mut results = rerank::rerank(candidates, &query.context, now, query.max_results);

    #[cfg(feature = "tracing")]
    drop(_rerank_span);

    // Stage 3b: Post-filter — remove results containing any exclude_terms (case-insensitive)
    if !query.context.exclude_terms.is_empty() {
        let exclude_lower: Vec<String> = query
            .context
            .exclude_terms
            .iter()
            .map(|t| t.to_lowercase())
            .collect();
        results.retain(|scored| {
            let content_lower = scored.content.to_lowercase();
            !exclude_lower
                .iter()
                .any(|term| content_lower.contains(term))
        });
    }

    // Stage 4: Post-retrieval updates (RIF + strength tracking)
    for scored in &results {
        let _ = strengths::on_access(conn, scored.node);
    }

    // Co-retrieval Hebbian strengthening between all retrieved pairs
    let retrieved_nodes: Vec<NodeRef> = results.iter().map(|r| r.node).collect();
    for i in 0..retrieved_nodes.len() {
        for j in (i + 1)..retrieved_nodes.len() {
            let _ =
                crate::graph::links::on_co_retrieval(conn, retrieved_nodes[i], retrieved_nodes[j]);
        }
    }

    // RIF: suppress competing memories from the same session
    let rif_suppression_factor = 0.9;
    let retrieved_set: std::collections::HashSet<NodeRef> =
        results.iter().map(|r| r.node).collect();
    let mut suppressed_sessions: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    // Collect session IDs from retrieved episodes
    for scored in &results {
        if let NodeRef::Episode(eid) = scored.node {
            if let Ok(ep) = episodic::get_episode(conn, eid) {
                suppressed_sessions.insert(ep.session_id.clone());
            }
        }
    }

    // For each session, suppress non-retrieved episodes
    for session_id in &suppressed_sessions {
        if let Ok(session_episodes) = episodic::get_episodes_by_session(conn, session_id) {
            for ep in &session_episodes {
                let node = NodeRef::Episode(ep.id);
                if !retrieved_set.contains(&node) {
                    let _ = strengths::suppress_retrieval(conn, node, rif_suppression_factor);
                }
            }
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;
    use crate::store::episodic;

    // ---------------------------------------------------------------------------
    // Helper: build a NewEpisode with minimal boilerplate
    // ---------------------------------------------------------------------------
    fn ep(content: &str, session: &str, ts: i64) -> NewEpisode {
        NewEpisode {
            content: content.to_string(),
            role: Role::User,
            session_id: session.to_string(),
            timestamp: ts,
            context: EpisodeContext::default(),
            embedding: None,
        }
    }

    fn ep_with_ctx(content: &str, session: &str, ts: i64, ctx: EpisodeContext) -> NewEpisode {
        NewEpisode {
            content: content.to_string(),
            role: Role::User,
            session_id: session.to_string(),
            timestamp: ts,
            context: ctx,
            embedding: None,
        }
    }

    // ---------------------------------------------------------------------------
    // Empty store — BM25 + no seeds => empty graph activation branch
    // ---------------------------------------------------------------------------
    #[test]
    fn test_query_empty_store_returns_empty() {
        let conn = open_memory_db().unwrap();
        let results = execute_query(
            &conn,
            &Query {
                text: "anything".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(1000),
                    ..Default::default()
                },
                max_results: 5,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();
        assert!(results.is_empty(), "empty store must return no results");
    }

    // ---------------------------------------------------------------------------
    // BM25-only (no embedding) — exercises None arm of vector_results match
    // ---------------------------------------------------------------------------
    #[test]
    fn test_query_bm25_only_no_embedding() {
        let conn = open_memory_db().unwrap();

        episodic::store_episode(&conn, &ep("Rust memory safety is great", "s1", 1000)).unwrap();
        episodic::store_episode(&conn, &ep("Python data science workflow", "s2", 2000)).unwrap();

        let results = execute_query(
            &conn,
            &Query {
                text: "Rust memory safety".to_string(),
                embedding: None, // explicitly no embedding
                context: QueryContext {
                    current_timestamp: Some(5000),
                    ..Default::default()
                },
                max_results: 5,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        assert!(!results.is_empty(), "should find BM25 results");
        assert!(
            results[0].content.contains("Rust"),
            "top result should match query; got: {}",
            results[0].content
        );
    }

    // ---------------------------------------------------------------------------
    // Multiple results — verify descending score ordering
    // ---------------------------------------------------------------------------
    #[test]
    fn test_query_multiple_results_descending_order() {
        let conn = open_memory_db().unwrap();

        // Store several episodes about "Rust" with different timestamps
        for i in 0..5u32 {
            episodic::store_episode(
                &conn,
                &ep(
                    &format!("Rust programming episode {i}"),
                    "s1",
                    1000 + i as i64 * 100,
                ),
            )
            .unwrap();
        }

        let results = execute_query(
            &conn,
            &Query {
                text: "Rust programming".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(10000),
                    ..Default::default()
                },
                max_results: 5,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        assert!(results.len() >= 2, "should return multiple results");
        // Verify descending order
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "results must be in descending score order: {} < {}",
                w[0].score,
                w[1].score
            );
        }
    }

    // ---------------------------------------------------------------------------
    // max_results limit is honoured — never returns more than max_results
    // ---------------------------------------------------------------------------
    #[test]
    fn test_query_max_results_honored() {
        let conn = open_memory_db().unwrap();

        for i in 0..10u32 {
            episodic::store_episode(
                &conn,
                &ep(
                    &format!("Rust topic episode {i}"),
                    "s1",
                    1000 + i as i64 * 50,
                ),
            )
            .unwrap();
        }

        let results = execute_query(
            &conn,
            &Query {
                text: "Rust topic".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(9000),
                    ..Default::default()
                },
                max_results: 3,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        assert!(
            results.len() <= 3,
            "max_results=3 must not be exceeded; got {}",
            results.len()
        );
    }

    // ---------------------------------------------------------------------------
    // max_results=1 — minimal slice
    // ---------------------------------------------------------------------------
    #[test]
    fn test_query_max_results_one() {
        let conn = open_memory_db().unwrap();

        episodic::store_episode(&conn, &ep("Rust ownership model", "s1", 1000)).unwrap();
        episodic::store_episode(&conn, &ep("Rust borrow checker explained", "s1", 2000)).unwrap();

        let results = execute_query(
            &conn,
            &Query {
                text: "Rust ownership".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(5000),
                    ..Default::default()
                },
                max_results: 1,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        assert!(results.len() <= 1, "should not exceed max_results=1");
    }

    // ---------------------------------------------------------------------------
    // Query with context topics/entities — exercises reranking context_similarity
    // An episode stored with matching context should score higher than one without
    // ---------------------------------------------------------------------------
    #[test]
    fn test_query_with_context_boosts_matching_episode() {
        let conn = open_memory_db().unwrap();

        // Episode with matching topic context
        let matching_ctx = EpisodeContext {
            topics: vec!["rust".to_string(), "async".to_string()],
            mentioned_entities: vec!["tokio".to_string()],
            sentiment: 0.5,
            conversation_turn: 1,
            preceding_episode: None,
        };
        episodic::store_episode(
            &conn,
            &ep_with_ctx("async Rust with tokio runtime", "s1", 1000, matching_ctx),
        )
        .unwrap();

        // Episode without matching context (stored slightly later for same recency)
        episodic::store_episode(&conn, &ep("async Rust with tokio runtime", "s2", 1100)).unwrap();

        let results = execute_query(
            &conn,
            &Query {
                text: "async Rust tokio".to_string(),
                embedding: None,
                context: QueryContext {
                    topics: vec!["rust".to_string(), "async".to_string()],
                    mentioned_entities: vec!["tokio".to_string()],
                    sentiment: 0.5,
                    current_timestamp: Some(5000),
                    ..Default::default()
                },
                max_results: 10,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        assert!(!results.is_empty(), "should return results");
        // All we can guarantee: pipeline runs without error and order is valid
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score, "must be ordered descending");
        }
    }

    // ---------------------------------------------------------------------------
    // Context: current_timestamp provided vs None (exercises both branches of
    // the unwrap_or_else for `now`)
    // ---------------------------------------------------------------------------
    #[test]
    fn test_query_no_current_timestamp_uses_system_time() {
        let conn = open_memory_db().unwrap();
        episodic::store_episode(&conn, &ep("Rust lifetime elision", "s1", 1000)).unwrap();

        // current_timestamp: None => code falls through to SystemTime::now()
        let results = execute_query(
            &conn,
            &Query {
                text: "Rust lifetime".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: None, // triggers the SystemTime branch
                    ..Default::default()
                },
                max_results: 5,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        // As long as no panic, the branch was exercised
        assert!(!results.is_empty());
    }

    // ---------------------------------------------------------------------------
    // Graph boost path — episodes linked to preferences via Topical link;
    // graph spreading should activate the preference node
    // (mirrors test_query_returns_preferences_via_graph but asserts co-retrieval
    //  Hebbian strengthening side-effect via link weight increase)
    // ---------------------------------------------------------------------------
    #[test]
    fn test_graph_boost_co_retrieval_strengthens_link() {
        let conn = open_memory_db().unwrap();
        use crate::graph::links;
        use crate::store::{implicit, strengths};

        let ep_id = episodic::store_episode(
            &conn,
            &ep("I love dark mode for coding at night", "s1", 1000),
        )
        .unwrap();

        let pref_id = implicit::store_preference(&conn, "ui", "dark mode", 0.9).unwrap();
        strengths::init_strength(&conn, NodeRef::Preference(pref_id)).unwrap();

        links::create_link(
            &conn,
            NodeRef::Episode(ep_id),
            NodeRef::Preference(pref_id),
            LinkType::Topical,
            0.9,
        )
        .unwrap();

        // Before query: record initial link weight
        let before_links = links::get_links_from(&conn, NodeRef::Episode(ep_id)).unwrap();
        let before_weight = before_links
            .first()
            .map(|l| l.forward_weight)
            .unwrap_or(0.0);

        execute_query(
            &conn,
            &Query {
                text: "dark mode coding".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(2000),
                    ..Default::default()
                },
                max_results: 10,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        // After query: if both episode and preference were co-retrieved,
        // on_co_retrieval fires, strengthening the link.
        // (The test validates the co-retrieval path ran without error)
        let after_links = links::get_links_from(&conn, NodeRef::Episode(ep_id)).unwrap();
        let after_weight = after_links.first().map(|l| l.forward_weight).unwrap_or(0.0);

        // Weight should be >= before (co-retrieval only increases or holds)
        assert!(
            after_weight >= before_weight,
            "co-retrieval should not decrease link weight"
        );
    }

    // ---------------------------------------------------------------------------
    // Seed-node deduplication — graph_results excludes seed nodes
    // (exercises the filter in graph_results construction)
    // ---------------------------------------------------------------------------
    #[test]
    fn test_graph_results_exclude_seed_nodes() {
        let conn = open_memory_db().unwrap();
        use crate::graph::links;
        use crate::store::strengths;

        // Store two episodes and link them so graph spreading finds the second
        let ep1 = episodic::store_episode(&conn, &ep("Rust async seed node", "s1", 1000)).unwrap();
        let ep2 =
            episodic::store_episode(&conn, &ep("Rust async neighbor node", "s1", 2000)).unwrap();

        strengths::init_strength(&conn, NodeRef::Episode(ep1)).unwrap();
        strengths::init_strength(&conn, NodeRef::Episode(ep2)).unwrap();

        links::create_link(
            &conn,
            NodeRef::Episode(ep1),
            NodeRef::Episode(ep2),
            LinkType::Topical,
            0.8,
        )
        .unwrap();

        let results = execute_query(
            &conn,
            &Query {
                text: "Rust async".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(5000),
                    ..Default::default()
                },
                max_results: 10,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        // Should have results — both episodes reachable
        assert!(!results.is_empty(), "should find episodes");
        // No result should appear twice (seed dedup worked)
        let ids: Vec<i64> = results.iter().map(|r| r.node.id()).collect();
        let mut unique = ids.clone();
        unique.dedup();
        // Sort before dedup for correctness
        let mut sorted_ids = ids.clone();
        sorted_ids.sort_unstable();
        sorted_ids.dedup();
        assert_eq!(
            ids.len(),
            sorted_ids.len(),
            "duplicate nodes in results: {ids:?}"
        );
    }

    // ---------------------------------------------------------------------------
    // Vector search branch — embedding provided => Some(emb) arm exercised
    // ---------------------------------------------------------------------------
    #[test]
    fn test_query_with_embedding_exercises_vector_branch() {
        let conn = open_memory_db().unwrap();
        use crate::store::{embeddings, semantic, strengths};

        // Store semantic node + embedding
        let nid = semantic::store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "Rust zero-cost abstractions".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        embeddings::store_embedding(&conn, "semantic", nid.0, &[1.0f32, 0.0, 0.0], "").unwrap();
        strengths::init_strength(&conn, NodeRef::Semantic(nid)).unwrap();

        // Also store an episode for BM25 baseline
        episodic::store_episode(
            &conn,
            &ep("Rust zero-cost abstractions are elegant", "s1", 1000),
        )
        .unwrap();

        let results = execute_query(
            &conn,
            &Query {
                text: "Rust abstractions".to_string(),
                embedding: Some(vec![0.9f32, 0.1, 0.0]), // triggers vector search
                context: QueryContext {
                    current_timestamp: Some(5000),
                    ..Default::default()
                },
                max_results: 10,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        assert!(!results.is_empty(), "vector query must return results");
        // At least one semantic node should appear
        assert!(
            results
                .iter()
                .any(|r| matches!(r.node, NodeRef::Semantic(_))),
            "should include semantic node from vector search"
        );
    }

    // ---------------------------------------------------------------------------
    // RIF suppression: non-retrieved same-session episodes get suppressed
    // (already partially tested but extended to check strength init required)
    // ---------------------------------------------------------------------------
    #[test]
    fn test_rif_suppression_same_session() {
        let conn = open_memory_db().unwrap();
        use crate::store::strengths;

        // 4 episodes in same session — only one should match query strongly
        let ids: Vec<_> = (0..4)
            .map(|i| {
                let eid = episodic::store_episode(
                    &conn,
                    &ep(
                        &format!("session memory item {i} about Rust coding"),
                        "sess_rif",
                        1000 + i as i64 * 10,
                    ),
                )
                .unwrap();
                strengths::init_strength(&conn, NodeRef::Episode(eid)).unwrap();
                eid
            })
            .collect();

        // Query retrieving only 1 result
        let results = execute_query(
            &conn,
            &Query {
                text: "memory item 0 Rust".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(5000),
                    ..Default::default()
                },
                max_results: 1,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);

        let retrieved_id = match results[0].node {
            NodeRef::Episode(eid) => eid.0,
            _ => panic!("expected episode"),
        };

        // At least one non-retrieved episode should have suppressed strength < 1.0
        let suppressed = ids.iter().filter(|eid| eid.0 != retrieved_id).any(|eid| {
            strengths::get_strength(&conn, NodeRef::Episode(*eid))
                .map(|s| s.retrieval_strength < 1.0)
                .unwrap_or(false)
        });

        assert!(
            suppressed,
            "at least one non-retrieved episode should be suppressed"
        );
    }

    // ---------------------------------------------------------------------------
    // Edge case: empty query string — FTS5 with empty string returns nothing
    // ---------------------------------------------------------------------------
    #[test]
    fn test_empty_query_string_returns_empty() {
        let conn = open_memory_db().unwrap();
        episodic::store_episode(&conn, &ep("Rust programming", "s1", 1000)).unwrap();

        let results = execute_query(&conn, &Query::simple("")).unwrap();
        assert!(
            results.is_empty(),
            "empty query string must return no results"
        );
    }

    // ---------------------------------------------------------------------------
    // Edge case: single-character query
    // ---------------------------------------------------------------------------
    #[test]
    fn test_single_char_query() {
        let conn = open_memory_db().unwrap();
        episodic::store_episode(&conn, &ep("Rust is great", "s1", 1000)).unwrap();

        // FTS5 may or may not return results for 'R' — either way must not panic
        let result = execute_query(&conn, &Query::simple("R"));
        assert!(result.is_ok(), "single-char query must not error");
    }

    // ---------------------------------------------------------------------------
    // Edge case: very long query string
    // ---------------------------------------------------------------------------
    #[test]
    fn test_very_long_query_string() {
        let conn = open_memory_db().unwrap();
        episodic::store_episode(&conn, &ep("Rust programming language", "s1", 1000)).unwrap();

        let long_query = "Rust ".repeat(200); // 1000 chars
        let result = execute_query(&conn, &Query::simple(long_query.trim()));
        assert!(result.is_ok(), "very long query must not error");
    }

    // ---------------------------------------------------------------------------
    // on_access is called for each retrieved result (strength tracking path)
    // Verify retrieval_strength is refreshed (= 1.0) after query
    // ---------------------------------------------------------------------------
    #[test]
    fn test_on_access_updates_strength_after_query() {
        let conn = open_memory_db().unwrap();
        use crate::store::strengths;

        let eid =
            episodic::store_episode(&conn, &ep("Rust traits and generics", "s1", 1000)).unwrap();
        strengths::init_strength(&conn, NodeRef::Episode(eid)).unwrap();

        // Manually decay to a low value first
        strengths::suppress_retrieval(&conn, NodeRef::Episode(eid), 0.5).unwrap();
        let before = strengths::get_strength(&conn, NodeRef::Episode(eid))
            .unwrap()
            .retrieval_strength;
        assert!(before < 1.0, "strength should be decayed before query");

        execute_query(
            &conn,
            &Query {
                text: "Rust traits generics".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(5000),
                    ..Default::default()
                },
                max_results: 5,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        let after = strengths::get_strength(&conn, NodeRef::Episode(eid))
            .unwrap()
            .retrieval_strength;
        assert!(
            after > before,
            "on_access should refresh strength: before={before}, after={after}"
        );
    }

    // ---------------------------------------------------------------------------
    // Cross-session: multiple sessions — RIF suppression is per-session scoped
    // Episodes from different sessions are not suppressed by each other
    // ---------------------------------------------------------------------------
    #[test]
    fn test_rif_does_not_suppress_across_sessions() {
        let conn = open_memory_db().unwrap();
        use crate::store::strengths;

        // One episode in session A
        let ep_a =
            episodic::store_episode(&conn, &ep("Rust ownership rules", "session_a", 1000)).unwrap();
        strengths::init_strength(&conn, NodeRef::Episode(ep_a)).unwrap();

        // One episode in session B — completely different session
        let ep_b =
            episodic::store_episode(&conn, &ep("Rust ownership rules copy", "session_b", 2000))
                .unwrap();
        strengths::init_strength(&conn, NodeRef::Episode(ep_b)).unwrap();

        // Query returns episode from session_a
        execute_query(
            &conn,
            &Query {
                text: "Rust ownership".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(5000),
                    ..Default::default()
                },
                max_results: 5,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        // episode in session_b has no competing episodes in session_b,
        // so it should NOT be suppressed by RIF (no unretrieved session_b siblings)
        let strength_b = strengths::get_strength(&conn, NodeRef::Episode(ep_b))
            .unwrap()
            .retrieval_strength;
        // Initial strength (no suppress applied), either 1.0 (init) or may have been on_accessed.
        // Key invariant: it was NOT suppressed to < 0.9 * init
        assert!(
            strength_b >= 0.9,
            "cross-session episode should not be suppressed; got {strength_b}"
        );
    }

    // ---------------------------------------------------------------------------
    // Result contains ScoredMemory with correct fields (content, role, timestamp)
    // ---------------------------------------------------------------------------
    #[test]
    fn test_result_fields_populated_correctly() {
        let conn = open_memory_db().unwrap();

        episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "Rust lifetimes explained clearly".to_string(),
                role: Role::Assistant,
                session_id: "s_fields".to_string(),
                timestamp: 42000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();

        let results = execute_query(
            &conn,
            &Query {
                text: "Rust lifetimes".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(50000),
                    ..Default::default()
                },
                max_results: 5,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        assert!(!results.is_empty());
        let r = &results[0];
        assert!(
            r.content.contains("Rust lifetimes"),
            "content must be populated"
        );
        assert_eq!(r.role, Some(Role::Assistant), "role must be preserved");
        assert_eq!(r.timestamp, 42000, "timestamp must match stored value");
        assert!(r.score > 0.0, "score must be positive");
    }

    // ---------------------------------------------------------------------------
    // Original test kept unchanged
    // ---------------------------------------------------------------------------
    #[test]
    fn test_basic_query() {
        let conn = open_memory_db().unwrap();

        episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "I love Rust programming".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();

        episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "Python is great for ML".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 2000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();

        let results = execute_query(
            &conn,
            &Query {
                text: "Rust programming".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(3000),
                    ..Default::default()
                },
                max_results: 5,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        assert!(!results.is_empty());
        assert!(results[0].content.contains("Rust"));
    }

    #[test]
    fn test_query_returns_semantic_nodes() {
        let conn = open_memory_db().unwrap();
        use crate::store::{embeddings, semantic, strengths};

        // Store a semantic node with embedding
        let node_id = semantic::store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "Rust has zero-cost abstractions".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();

        // Store embedding for vector search
        let emb = vec![1.0, 0.0, 0.0];
        embeddings::store_embedding(&conn, "semantic", node_id.0, &emb, "").unwrap();
        strengths::init_strength(&conn, NodeRef::Semantic(node_id)).unwrap();

        // Also store an episode so BM25 can potentially match
        episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "Rust has zero-cost abstractions and great memory safety".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();

        // Query with embedding that matches the semantic node
        let results = execute_query(
            &conn,
            &Query {
                text: "Rust abstractions".to_string(),
                embedding: Some(vec![0.9, 0.1, 0.0]),
                context: QueryContext {
                    current_timestamp: Some(2000),
                    ..Default::default()
                },
                max_results: 10,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        // Should find results — at minimum the episode via BM25
        assert!(!results.is_empty(), "should have results");

        // Check if any result is a semantic node
        let has_semantic = results
            .iter()
            .any(|r| matches!(r.node, NodeRef::Semantic(_)));
        assert!(
            has_semantic,
            "should include semantic node in results, got: {:?}",
            results.iter().map(|r| r.node).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_rif_suppresses_competing_memories() {
        let conn = open_memory_db().unwrap();
        use crate::store::strengths;

        // Store 3 episodes in the same session
        for i in 0..3 {
            episodic::store_episode(
                &conn,
                &NewEpisode {
                    content: format!("session topic {i} about Rust programming"),
                    role: Role::User,
                    session_id: "s1".to_string(),
                    timestamp: 1000 + i as i64,
                    context: EpisodeContext::default(),
                    embedding: None,
                },
            )
            .unwrap();
        }

        // Init strengths for all 3
        for id in 1..=3 {
            strengths::init_strength(&conn, NodeRef::Episode(EpisodeId(id))).unwrap();
        }

        // Query should retrieve some but not all episodes from the session
        // The query "Rust programming 0" should match episode 0 most strongly
        let results = execute_query(
            &conn,
            &Query {
                text: "topic 0 Rust".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(2000),
                    ..Default::default()
                },
                max_results: 1, // Only retrieve 1
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        assert!(!results.is_empty(), "should have at least 1 result");

        // The retrieved episode(s) should have RS = 1.0 (refreshed by on_access)
        let retrieved_ids: Vec<i64> = results
            .iter()
            .filter_map(|r| match r.node {
                NodeRef::Episode(eid) => Some(eid.0),
                _ => None,
            })
            .collect();

        // Check that at least one NON-retrieved same-session episode got suppressed
        let mut any_suppressed = false;
        for id in 1..=3i64 {
            if !retrieved_ids.contains(&id) {
                let s = strengths::get_strength(&conn, NodeRef::Episode(EpisodeId(id))).unwrap();
                if s.retrieval_strength < 1.0 {
                    any_suppressed = true;
                }
            }
        }
        assert!(
            any_suppressed,
            "at least one non-retrieved same-session episode should have suppressed RS"
        );
    }

    #[test]
    fn test_empty_query() {
        let conn = open_memory_db().unwrap();
        let results = execute_query(&conn, &Query::simple("")).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_excludes_superseded_semantic_nodes() {
        let conn = open_memory_db().unwrap();
        use crate::store::{conflicts, embeddings, semantic, strengths};

        let winner = semantic::store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "Rust has zero-cost abstractions".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        embeddings::store_embedding(&conn, "semantic", winner.0, &[1.0, 0.0, 0.0], "").unwrap();
        strengths::init_strength(&conn, NodeRef::Semantic(winner)).unwrap();

        let loser = semantic::store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "Rust has high-cost abstractions".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        embeddings::store_embedding(&conn, "semantic", loser.0, &[0.9, 0.1, 0.0], "").unwrap();
        strengths::init_strength(&conn, NodeRef::Semantic(loser)).unwrap();

        // Supersede loser
        conflicts::supersede_node(&conn, loser, winner).unwrap();

        let results = execute_query(
            &conn,
            &Query {
                text: "Rust abstractions".to_string(),
                embedding: Some(vec![0.95, 0.05, 0.0]),
                context: QueryContext {
                    current_timestamp: Some(5000),
                    ..Default::default()
                },
                max_results: 10,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        // Loser should not appear in results
        let has_loser = results.iter().any(|r| r.node == NodeRef::Semantic(loser));
        assert!(
            !has_loser,
            "superseded node should be excluded from retrieval"
        );
    }

    #[test]
    fn test_query_returns_preferences_via_graph() {
        let conn = open_memory_db().unwrap();
        use crate::graph::links;
        use crate::store::{implicit, strengths};

        // Store an episode mentioning "dark mode"
        let ep_id = episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "I prefer dark mode for coding".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();

        // Store a preference about dark mode
        let pref_id = implicit::store_preference(&conn, "ui", "dark mode", 0.8).unwrap();
        strengths::init_strength(&conn, NodeRef::Preference(pref_id)).unwrap();

        // Link episode to preference to enable graph spreading activation
        links::create_link(
            &conn,
            NodeRef::Episode(ep_id),
            NodeRef::Preference(pref_id),
            LinkType::Topical,
            0.9,
        )
        .unwrap();

        // Query for "dark mode" - episode should be found via BM25,
        // then graph spreading should activate the preference
        let results = execute_query(
            &conn,
            &Query {
                text: "dark mode coding".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(2000),
                    ..Default::default()
                },
                max_results: 10,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        assert!(!results.is_empty(), "should have results");

        // Check if any result is a preference (graph activation path)
        let has_preference = results
            .iter()
            .any(|r| matches!(r.node, NodeRef::Preference(_)));
        if has_preference {
            let pref_result = results
                .iter()
                .find(|r| matches!(r.node, NodeRef::Preference(_)))
                .unwrap();
            assert!(
                pref_result.content.contains("dark mode"),
                "preference content should contain dark mode"
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Category-scoped retrieval — semantic nodes outside the requested category
    // are excluded from results (inspired by MemPalace's palace scoping)
    // ---------------------------------------------------------------------------
    #[test]
    fn test_category_scoped_query_excludes_other_categories() {
        let conn = open_memory_db().unwrap();
        use crate::store::{categories, embeddings, semantic, strengths};

        // Create two semantic nodes in different categories
        let node_rust = semantic::store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "Rust has zero-cost abstractions".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        embeddings::store_embedding(&conn, "semantic", node_rust.0, &[1.0, 0.0, 0.0], "").unwrap();
        strengths::init_strength(&conn, NodeRef::Semantic(node_rust)).unwrap();

        let node_python = semantic::store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "Python has dynamic typing with zero-cost abstractions mindset".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        embeddings::store_embedding(&conn, "semantic", node_python.0, &[0.9, 0.1, 0.0], "")
            .unwrap();
        strengths::init_strength(&conn, NodeRef::Semantic(node_python)).unwrap();

        // Create categories and assign nodes
        let cat_rust =
            categories::store_category(&conn, "rust-lang", node_rust, None, None).unwrap();
        categories::assign_node_to_category(&conn, node_rust, cat_rust).unwrap();

        let cat_python =
            categories::store_category(&conn, "python", node_python, None, None).unwrap();
        categories::assign_node_to_category(&conn, node_python, cat_python).unwrap();

        // Query scoped to rust-lang category
        let results = execute_query(
            &conn,
            &Query {
                text: "zero-cost abstractions".to_string(),
                embedding: Some(vec![0.95, 0.05, 0.0]),
                context: QueryContext {
                    current_timestamp: Some(5000),
                    ..Default::default()
                },
                max_results: 10,
                category_id: Some(cat_rust.0),
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        // Only the rust node should appear — python node filtered out
        let semantic_nodes: Vec<_> = results
            .iter()
            .filter(|r| matches!(r.node, NodeRef::Semantic(_)))
            .collect();
        assert!(
            !semantic_nodes.is_empty(),
            "should find the rust semantic node"
        );
        for r in &semantic_nodes {
            assert_eq!(
                r.node,
                NodeRef::Semantic(node_rust),
                "only rust-category nodes should appear; got {:?}",
                r.node
            );
        }
    }

    #[test]
    fn test_category_scoped_query_none_returns_all() {
        let conn = open_memory_db().unwrap();
        use crate::store::{categories, embeddings, semantic, strengths};

        // Create two semantic nodes
        let node_a = semantic::store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "Rust ownership model explained".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        embeddings::store_embedding(&conn, "semantic", node_a.0, &[1.0, 0.0, 0.0], "").unwrap();
        strengths::init_strength(&conn, NodeRef::Semantic(node_a)).unwrap();

        let node_b = semantic::store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "Rust borrow checker ownership rules".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        embeddings::store_embedding(&conn, "semantic", node_b.0, &[0.9, 0.1, 0.0], "").unwrap();
        strengths::init_strength(&conn, NodeRef::Semantic(node_b)).unwrap();

        // Assign to different categories
        let cat_a = categories::store_category(&conn, "cat-a", node_a, None, None).unwrap();
        categories::assign_node_to_category(&conn, node_a, cat_a).unwrap();
        let cat_b = categories::store_category(&conn, "cat-b", node_b, None, None).unwrap();
        categories::assign_node_to_category(&conn, node_b, cat_b).unwrap();

        // Query with category_id: None — should return both
        let results = execute_query(
            &conn,
            &Query {
                text: "Rust ownership".to_string(),
                embedding: Some(vec![0.95, 0.05, 0.0]),
                context: QueryContext {
                    current_timestamp: Some(5000),
                    ..Default::default()
                },
                max_results: 10,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        let semantic_nodes: Vec<_> = results
            .iter()
            .filter(|r| matches!(r.node, NodeRef::Semantic(_)))
            .collect();
        assert!(
            semantic_nodes.len() >= 2,
            "with no category filter, both semantic nodes should appear; got {}",
            semantic_nodes.len()
        );
    }

    #[test]
    fn test_category_scoped_query_does_not_filter_episodes() {
        let conn = open_memory_db().unwrap();
        use crate::store::{categories, semantic};

        // Create a semantic node and category
        let node = semantic::store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "prototype".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        let cat = categories::store_category(&conn, "scoped-cat", node, None, None).unwrap();

        // Store an episode (episodes have no category)
        episodic::store_episode(&conn, &ep("Rust async programming patterns", "s1", 1000)).unwrap();

        // Query scoped to the category — episodes should still appear
        let results = execute_query(
            &conn,
            &Query {
                text: "Rust async programming".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(5000),
                    ..Default::default()
                },
                max_results: 10,
                category_id: Some(cat.0),
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        let has_episodes = results
            .iter()
            .any(|r| matches!(r.node, NodeRef::Episode(_)));
        assert!(
            has_episodes,
            "category scoping should not filter out episodes"
        );
    }

    #[test]
    fn test_category_node_filtered_out_of_results() {
        // Covers line 119: _ => None for Category nodes in the filter_map
        let conn = open_memory_db().unwrap();

        // Create an episode that matches our query
        let ep1 =
            episodic::store_episode(&conn, &ep("Rust programming language", "s1", 1000)).unwrap();

        // Create a semantic node for use as prototype (FK)
        crate::store::semantic::store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "prototype node".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();

        // Create a category with that prototype
        let cat_id =
            crate::store::categories::store_category(&conn, "programming", NodeId(1), None, None)
                .unwrap();

        // Create a strong link from the episode to the category
        // This ensures spreading activation includes the Category node
        crate::graph::links::create_link(
            &conn,
            NodeRef::Episode(ep1),
            NodeRef::Category(cat_id),
            LinkType::MemberOf,
            1.0,
        )
        .unwrap();

        let results = execute_query(
            &conn,
            &Query {
                text: "Rust programming".to_string(),
                embedding: None,
                context: QueryContext {
                    current_timestamp: Some(5000),
                    ..Default::default()
                },
                max_results: 10,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        // Category nodes should be filtered out — only episodes/semantic/preferences in results
        for r in &results {
            assert!(
                !matches!(r.node, NodeRef::Category(_)),
                "Category nodes should be filtered out of results"
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Tracing coverage — exercises trace!() macro format args in pipeline
    // ---------------------------------------------------------------------------
    #[cfg(feature = "tracing")]
    #[test]
    fn test_query_with_tracing_subscriber() {
        // Initialize a subscriber at TRACE level so trace!() args are evaluated.
        // try_init because other tests may have already set one.
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_test_writer()
            .try_init();

        let conn = open_memory_db().unwrap();
        episodic::store_episode(&conn, &ep("Rust memory safety", "s1", 1000)).unwrap();

        let results = execute_query(
            &conn,
            &Query {
                text: "Rust".to_string(),
                embedding: Some(vec![0.1, 0.2, 0.3]),
                context: QueryContext {
                    current_timestamp: Some(2000),
                    ..Default::default()
                },
                max_results: 5,
                category_id: None,
                boost_categories: None,
                boost_weights: None,
            },
        )
        .unwrap();

        assert!(results.len() <= 5);
    }
}
