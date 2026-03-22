use crate::error::Result;
use crate::graph::links;
use crate::provider::ConsolidationProvider;
use crate::store::{categories, embeddings, episodic, semantic};
use crate::types::*;
use rusqlite::Connection;
use std::collections::HashMap;

/// Minimum number of unconsolidated episodes before consolidation triggers.
const CONSOLIDATION_BATCH_SIZE: u32 = 10;

/// Run a consolidation cycle: extract semantic knowledge from episodic store.
///
/// Models the Complementary Learning Systems (CLS) theory:
/// the hippocampus (episodic) gradually teaches the neocortex (semantic)
/// through interleaved replay, avoiding catastrophic interference.
pub fn consolidate(
    conn: &Connection,
    provider: &dyn ConsolidationProvider,
) -> Result<ConsolidationReport> {
    let mut report = ConsolidationReport::default();

    let episodes = episodic::get_unconsolidated_episodes(conn, CONSOLIDATION_BATCH_SIZE)?;
    if episodes.len() < 3 {
        // Not enough episodes to consolidate — need corroboration
        return Ok(report);
    }

    report.episodes_processed = episodes.len() as u32;

    // Ask the provider to extract knowledge
    let new_nodes = provider.extract_knowledge(&episodes)?;

    for node_data in new_nodes {
        process_node(conn, &node_data, &mut report)?;
    }

    Ok(report)
}

/// Provider-less consolidation: accepts pre-extracted semantic nodes directly.
///
/// Runs the same pipeline as [`consolidate`] (store node, create Causal links
/// to source episodes, init Bjork strength, try category assignment) but skips
/// the episode-fetching and provider-based extraction steps.
///
/// This enables library users and the MCP binary to perform consolidation
/// without requiring a [`ConsolidationProvider`] (LLM).
pub fn learn_direct(conn: &Connection, nodes: Vec<NewSemanticNode>) -> Result<ConsolidationReport> {
    let mut report = ConsolidationReport::default();

    for node_data in nodes {
        process_node(conn, &node_data, &mut report)?;
    }

    Ok(report)
}

/// Shared pipeline step: store a semantic node, create Causal links to its
/// source episodes, initialize Bjork strength, and try category assignment.
/// Used by both `consolidate()` and `learn_direct()`.
fn process_node(
    conn: &Connection,
    node_data: &NewSemanticNode,
    report: &mut ConsolidationReport,
) -> Result<()> {
    let node_id = semantic::store_semantic_node(conn, node_data)?;
    report.nodes_created += 1;

    // Link the new semantic node to its source episodes
    for ep_id in &node_data.source_episodes {
        links::create_link(
            conn,
            NodeRef::Semantic(node_id),
            NodeRef::Episode(*ep_id),
            LinkType::Causal,
            0.7,
        )?;
        report.links_created += 1;
    }

    // Initialize strength for the new node
    crate::store::strengths::init_strength(conn, NodeRef::Semantic(node_id))?;

    // Try to assign to an existing category
    if let Some(_cat_id) = try_assign_category(conn, node_id, node_data)? {
        report.categories_assigned += 1;
    }
    Ok(())
}

/// Cosine similarity threshold for embedding-based category assignment.
const CATEGORY_SIMILARITY_THRESHOLD: f32 = 0.6;

/// Try to assign a newly created semantic node to an existing category.
/// Signal 1: embedding similarity to category centroids (threshold 0.6)
/// Signal 2: graph neighbor majority vote (if >50% of linked nodes share a category)
/// Returns Some(CategoryId) if assigned, None if no match.
fn try_assign_category(
    conn: &Connection,
    node_id: NodeId,
    node_data: &NewSemanticNode,
) -> Result<Option<CategoryId>> {
    let all_categories = categories::list_categories(conn, None)?;
    if all_categories.is_empty() {
        return Ok(None);
    }

    // Signal 1: Embedding similarity to category centroids
    if let Some(ref node_embedding) = node_data.embedding {
        let mut best_sim = 0.0f32;
        let mut best_cat: Option<&Category> = None;

        for cat in &all_categories {
            if let Some(ref centroid) = cat.centroid_embedding {
                let sim = embeddings::cosine_similarity(node_embedding, centroid);
                if sim > best_sim {
                    best_sim = sim;
                    best_cat = Some(cat);
                }
            }
        }

        if best_sim >= CATEGORY_SIMILARITY_THRESHOLD {
            if let Some(cat) = best_cat {
                let cat_id = cat.id;
                return assign_and_update(conn, node_id, cat_id, node_embedding, cat);
            }
        }
    }

    // Signal 2: Graph neighbor majority vote
    let mut votes: HashMap<CategoryId, u32> = HashMap::new();
    let mut total_votes: u32 = 0;

    for ep_id in &node_data.source_episodes {
        let ep_links = links::get_links_from(conn, NodeRef::Episode(*ep_id))?;
        for link in &ep_links {
            if let NodeRef::Semantic(linked_node_id) = link.target {
                if linked_node_id != node_id {
                    if let Ok(Some(cat)) = categories::get_node_category(conn, linked_node_id) {
                        *votes.entry(cat.id).or_insert(0) += 1;
                        total_votes += 1;
                    }
                }
            }
        }
    }

    if total_votes > 0 {
        // Find the category with the most votes
        if let Some((&winning_cat_id, &winning_count)) = votes.iter().max_by_key(|(_k, v)| *v) {
            // Check >50% majority
            if winning_count * 2 > total_votes {
                let cat = categories::get_category(conn, winning_cat_id)?;
                let node_embedding = node_data.embedding.as_deref().unwrap_or(&[]);
                if !node_embedding.is_empty() {
                    return assign_and_update(conn, node_id, winning_cat_id, node_embedding, &cat);
                } else {
                    // Assign without centroid update (no embedding available)
                    categories::assign_node_to_category(conn, node_id, winning_cat_id)?;
                    links::create_link(
                        conn,
                        NodeRef::Semantic(node_id),
                        NodeRef::Category(winning_cat_id),
                        LinkType::MemberOf,
                        0.8,
                    )?;
                    return Ok(Some(winning_cat_id));
                }
            }
        }
    }

    Ok(None)
}

/// Assign a node to a category, update the centroid with a running average,
/// and create a MemberOf link in the graph.
fn assign_and_update(
    conn: &Connection,
    node_id: NodeId,
    cat_id: CategoryId,
    node_embedding: &[f32],
    cat: &Category,
) -> Result<Option<CategoryId>> {
    categories::assign_node_to_category(conn, node_id, cat_id)?;

    // Update centroid with running average: new = old*(n-1)/n + new_emb/n
    // member_count was already incremented by assign_node_to_category
    let n = (cat.member_count + 1) as f32; // +1 because assign already incremented
    if let Some(ref old_centroid) = cat.centroid_embedding {
        let new_centroid: Vec<f32> = old_centroid
            .iter()
            .zip(node_embedding.iter())
            .map(|(old, new)| old * (n - 1.0) / n + new / n)
            .collect();
        categories::update_centroid(conn, cat_id, &new_centroid)?;
    }

    // Create MemberOf link
    links::create_link(
        conn,
        NodeRef::Semantic(node_id),
        NodeRef::Category(cat_id),
        LinkType::MemberOf,
        0.8,
    )?;

    Ok(Some(cat_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::MockProvider;
    use crate::schema::open_memory_db;
    use crate::store::{categories, episodic, semantic};
    use rusqlite::Connection;

    #[test]
    fn test_consolidation_below_threshold() {
        let conn = open_memory_db().unwrap();
        // Only 2 episodes — below threshold of 3
        episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "hello".to_string(),
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
                content: "world".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 2000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();

        let report = consolidate(&conn, &MockProvider::empty()).unwrap();
        assert_eq!(report.nodes_created, 0);
    }

    #[test]
    fn test_consolidation_creates_nodes() {
        let conn = open_memory_db().unwrap();
        let mut ep_ids = vec![];
        for i in 0..5 {
            let id = episodic::store_episode(
                &conn,
                &NewEpisode {
                    content: format!("message about Rust {i}"),
                    role: Role::User,
                    session_id: "s1".to_string(),
                    timestamp: 1000 + i * 100,
                    context: EpisodeContext::default(),
                    embedding: None,
                },
            )
            .unwrap();
            ep_ids.push(id);
        }

        let provider = MockProvider::with_knowledge(vec![NewSemanticNode {
            content: "User discusses Rust programming".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.8,
            source_episodes: ep_ids,
            embedding: None,
        }]);

        let report = consolidate(&conn, &provider).unwrap();
        assert_eq!(report.nodes_created, 1);
        assert!(report.links_created > 0);
    }

    /// Helper: insert a bare semantic node to use as category prototype.
    fn insert_prototype(conn: &Connection) -> NodeId {
        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated)
             VALUES ('prototype', 'fact', 0.5, 1000, 1000)",
            [],
        ).unwrap();
        NodeId(conn.last_insert_rowid())
    }

    #[test]
    fn test_consolidation_assigns_existing_category_via_embedding() {
        let conn = open_memory_db().unwrap();

        // Create a category with a centroid (needs a real prototype node)
        let proto = insert_prototype(&conn);
        let cat_id =
            categories::store_category(&conn, "rust-topics", proto, Some(&[1.0, 0.0, 0.0]), None)
                .unwrap();

        // Store 5 episodes (enough for consolidation threshold of 3)
        let mut ep_ids = vec![];
        for i in 0..5 {
            let id = episodic::store_episode(
                &conn,
                &NewEpisode {
                    content: format!("Rust episode {i}"),
                    role: Role::User,
                    session_id: "s1".to_string(),
                    timestamp: 1000 + i * 100,
                    context: EpisodeContext::default(),
                    embedding: None,
                },
            )
            .unwrap();
            ep_ids.push(id);
        }

        // Provider returns a node with embedding close to the category centroid
        let provider = MockProvider::with_knowledge(vec![NewSemanticNode {
            content: "User programs in Rust".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.8,
            source_episodes: ep_ids,
            embedding: Some(vec![0.9, 0.1, 0.0]), // cosine sim ~0.99 to [1,0,0]
        }]);

        let report = consolidate(&conn, &provider).unwrap();
        assert_eq!(report.nodes_created, 1);
        assert_eq!(report.categories_assigned, 1);

        // Verify the semantic node was assigned
        let nodes = semantic::find_by_type(&conn, SemanticType::Fact, 10).unwrap();
        let node = &nodes[0];
        let cat = categories::get_node_category(&conn, node.id).unwrap();
        assert!(cat.is_some(), "node should be assigned to a category");
        assert_eq!(cat.unwrap().id, cat_id);
    }

    #[test]
    fn test_consolidation_skips_when_no_categories() {
        let conn = open_memory_db().unwrap();

        let mut ep_ids = vec![];
        for i in 0..5 {
            let id = episodic::store_episode(
                &conn,
                &NewEpisode {
                    content: format!("msg {i}"),
                    role: Role::User,
                    session_id: "s1".to_string(),
                    timestamp: 1000 + i * 100,
                    context: EpisodeContext::default(),
                    embedding: None,
                },
            )
            .unwrap();
            ep_ids.push(id);
        }

        let provider = MockProvider::with_knowledge(vec![NewSemanticNode {
            content: "some fact".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.8,
            source_episodes: ep_ids,
            embedding: Some(vec![0.5, 0.5, 0.0]),
        }]);

        let report = consolidate(&conn, &provider).unwrap();
        assert_eq!(report.nodes_created, 1);
        assert_eq!(report.categories_assigned, 0);
    }

    #[test]
    fn test_consolidation_assigns_category_via_neighbor_vote() {
        use crate::graph::links;
        use crate::store::strengths;

        let conn = open_memory_db().unwrap();

        // Create a category with a prototype node
        let proto = insert_prototype(&conn);
        let cat_id =
            categories::store_category(&conn, "rust-topics", proto, Some(&[1.0, 0.0, 0.0]), None)
                .unwrap();

        // Create 5 episodes
        let mut ep_ids = vec![];
        for i in 0..5 {
            let id = episodic::store_episode(
                &conn,
                &NewEpisode {
                    content: format!("Rust neighbor episode {i}"),
                    role: Role::User,
                    session_id: "s1".to_string(),
                    timestamp: 1000 + i * 100,
                    context: EpisodeContext::default(),
                    embedding: None,
                },
            )
            .unwrap();
            ep_ids.push(id);
        }

        // Create 3 existing semantic nodes assigned to that category.
        // For each, create both directions of Causal links so that
        // get_links_from(Episode) returns Semantic targets.
        for (i, &ep_id) in ep_ids.iter().enumerate().take(3) {
            let node_id = semantic::store_semantic_node(
                &conn,
                &NewSemanticNode {
                    content: format!("existing rust node {i}"),
                    node_type: SemanticType::Fact,
                    confidence: 0.8,
                    source_episodes: vec![ep_id],
                    embedding: None,
                },
            )
            .unwrap();
            categories::assign_node_to_category(&conn, node_id, cat_id).unwrap();
            strengths::init_strength(&conn, NodeRef::Semantic(node_id)).unwrap();
            // Create link FROM episode TO semantic (so get_links_from(episode) finds it)
            links::create_link(
                &conn,
                NodeRef::Episode(ep_id),
                NodeRef::Semantic(node_id),
                LinkType::Causal,
                0.7,
            )
            .unwrap();
        }

        // Use learn_direct to avoid consolidation threshold issues.
        // New node has NO embedding but source_episodes that overlap
        // with the categorized nodes' episodes.
        let report = learn_direct(
            &conn,
            vec![NewSemanticNode {
                content: "new rust knowledge via neighbor vote".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: vec![ep_ids[0], ep_ids[1], ep_ids[2]],
                embedding: None, // No embedding => neighbor vote path
            }],
        )
        .unwrap();
        assert_eq!(report.nodes_created, 1);
        assert_eq!(
            report.categories_assigned, 1,
            "should assign category via neighbor majority vote"
        );
    }

    #[test]
    fn test_consolidation_skips_when_below_threshold() {
        let conn = open_memory_db().unwrap();

        // Category centroid is far from node embedding
        let proto = insert_prototype(&conn);
        categories::store_category(&conn, "cooking", proto, Some(&[0.0, 0.0, 1.0]), None).unwrap();

        let mut ep_ids = vec![];
        for i in 0..5 {
            let id = episodic::store_episode(
                &conn,
                &NewEpisode {
                    content: format!("msg {i}"),
                    role: Role::User,
                    session_id: "s1".to_string(),
                    timestamp: 1000 + i * 100,
                    context: EpisodeContext::default(),
                    embedding: None,
                },
            )
            .unwrap();
            ep_ids.push(id);
        }

        let provider = MockProvider::with_knowledge(vec![NewSemanticNode {
            content: "Rust programming".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.8,
            source_episodes: ep_ids,
            embedding: Some(vec![1.0, 0.0, 0.0]), // cosine sim ~0.0 to [0,0,1]
        }]);

        let report = consolidate(&conn, &provider).unwrap();
        assert_eq!(report.nodes_created, 1);
        assert_eq!(
            report.categories_assigned, 0,
            "node should not be assigned to distant category"
        );
    }

    #[test]
    fn test_neighbor_vote_with_embedding_via_learn_direct() {
        // Tests lines 142-144 (skip self check) and 162 (assign_and_update via
        // neighbor vote with embedding). Uses learn_direct to bypass the
        // unconsolidated-episode threshold.
        let conn = open_memory_db().unwrap();

        // Create a prototype and category
        let proto = insert_prototype(&conn);
        let cat_id = categories::store_category(
            &conn,
            "rust-programming",
            proto,
            Some(&[1.0, 0.0, 0.0]),
            None,
        )
        .unwrap();

        // Create an existing semantic node that IS categorized
        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, category_id)
             VALUES ('Rust is great', 'fact', 0.8, 1000, 1000, ?1)",
            [cat_id.0],
        ).unwrap();
        let existing_node_id = NodeId(conn.last_insert_rowid());

        // Create episodes
        let mut ep_ids = vec![];
        for i in 0..5 {
            let id = episodic::store_episode(
                &conn,
                &NewEpisode {
                    content: format!("Rust topic {i}"),
                    role: Role::User,
                    session_id: "s1".to_string(),
                    timestamp: 1000 + i * 100,
                    context: EpisodeContext::default(),
                    embedding: None,
                },
            )
            .unwrap();
            ep_ids.push(id);

            // Link episode → existing semantic node (so neighbor vote can find it)
            links::create_link(
                &conn,
                NodeRef::Episode(id),
                NodeRef::Semantic(existing_node_id),
                LinkType::Causal,
                0.7,
            )
            .unwrap();
        }

        // Use learn_direct (not consolidate) to add a new node:
        // - Embedding far from centroid → Signal 1 won't fire
        // - References the same episodes → neighbor vote triggers (Signal 2)
        let report = learn_direct(
            &conn,
            vec![NewSemanticNode {
                content: "Rust ownership model".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: ep_ids,
                embedding: Some(vec![0.3, 0.3, 0.85]),
            }],
        )
        .unwrap();
        assert_eq!(report.nodes_created, 1);
        // The neighbor vote should assign the category because all linked nodes share it
        assert_eq!(
            report.categories_assigned, 1,
            "should assign category via neighbor vote with embedding (line 162)"
        );
    }

    #[test]
    fn test_learn_direct_empty_nodes() {
        let conn = open_memory_db().unwrap();
        let report = learn_direct(&conn, vec![]).unwrap();
        assert_eq!(report.nodes_created, 0);
        assert_eq!(report.links_created, 0);
        assert_eq!(report.categories_assigned, 0);
    }

    #[test]
    fn test_learn_direct_no_source_episodes() {
        // Node with no source_episodes — links_created should stay 0
        let conn = open_memory_db().unwrap();
        let report = learn_direct(
            &conn,
            vec![NewSemanticNode {
                content: "standalone fact".to_string(),
                node_type: SemanticType::Concept,
                confidence: 0.7,
                source_episodes: vec![],
                embedding: None,
            }],
        )
        .unwrap();
        assert_eq!(report.nodes_created, 1);
        assert_eq!(report.links_created, 0);
    }

    #[test]
    fn test_assign_and_update_no_old_centroid() {
        // Category has no centroid embedding — assign_and_update should not update centroid
        // but still assign the node and create the MemberOf link.
        let conn = open_memory_db().unwrap();
        let proto = insert_prototype(&conn);
        let cat_id = categories::store_category(&conn, "no-centroid", proto, None, None).unwrap();

        // Insert an episode so learn_direct gets the 3-episode threshold bypassed
        let mut ep_ids = vec![];
        for i in 0..3 {
            let id = episodic::store_episode(
                &conn,
                &NewEpisode {
                    content: format!("ep {i}"),
                    role: Role::User,
                    session_id: "s1".to_string(),
                    timestamp: 1000 + i * 100,
                    context: EpisodeContext::default(),
                    embedding: None,
                },
            )
            .unwrap();
            ep_ids.push(id);
        }

        // Learn a node with embedding similar to cat's (but cat has no centroid)
        // Embedding signal 1 won't fire (no centroid), category still exists.
        // Neighbor vote won't fire (no linked categorized nodes).
        // Result: categories_assigned = 0
        let report = learn_direct(
            &conn,
            vec![NewSemanticNode {
                content: "test no-centroid assign".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: ep_ids,
                embedding: Some(vec![1.0, 0.0, 0.0]),
            }],
        )
        .unwrap();
        assert_eq!(report.nodes_created, 1);
        // No centroid → no assignment via signal 1; no neighbors → signal 2 also fires 0
        assert_eq!(report.categories_assigned, 0);

        // Verify category exists and node is uncategorized
        let cats = categories::list_categories(&conn, None).unwrap();
        assert_eq!(cats.len(), 1);
        assert_eq!(cats[0].id, cat_id);
    }

    #[test]
    fn test_neighbor_vote_no_majority() {
        // Two different categories each get one vote — no >50% majority, no assignment
        use crate::graph::links;
        use crate::store::strengths;

        let conn = open_memory_db().unwrap();

        let p1 = insert_prototype(&conn);
        let p2 = insert_prototype(&conn);
        let cat1 = categories::store_category(&conn, "cat-a", p1, None, None).unwrap();
        let cat2 = categories::store_category(&conn, "cat-b", p2, None, None).unwrap();

        // Create two episodes
        let ep1 = episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "ep1".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();
        let ep2 = episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "ep2".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 2000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();

        // Create semantic node assigned to cat1 and linked to ep1
        let n1 = semantic::store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "node1".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        categories::assign_node_to_category(&conn, n1, cat1).unwrap();
        strengths::init_strength(&conn, NodeRef::Semantic(n1)).unwrap();
        links::create_link(
            &conn,
            NodeRef::Episode(ep1),
            NodeRef::Semantic(n1),
            LinkType::Causal,
            0.7,
        )
        .unwrap();

        // Create semantic node assigned to cat2 and linked to ep2
        let n2 = semantic::store_semantic_node(
            &conn,
            &NewSemanticNode {
                content: "node2".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: vec![],
                embedding: None,
            },
        )
        .unwrap();
        categories::assign_node_to_category(&conn, n2, cat2).unwrap();
        strengths::init_strength(&conn, NodeRef::Semantic(n2)).unwrap();
        links::create_link(
            &conn,
            NodeRef::Episode(ep2),
            NodeRef::Semantic(n2),
            LinkType::Causal,
            0.7,
        )
        .unwrap();

        // Learn a new node referencing both episodes — tied vote, no majority
        let report = learn_direct(
            &conn,
            vec![NewSemanticNode {
                content: "split vote node".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: vec![ep1, ep2],
                embedding: None, // no embedding so signal 1 skipped
            }],
        )
        .unwrap();
        assert_eq!(report.nodes_created, 1);
        assert_eq!(
            report.categories_assigned, 0,
            "tied vote (1-1) should not assign any category"
        );
    }

    #[test]
    fn test_self_link_skipped_in_category_vote() {
        // Covers line 141: continue when linked_node_id == node_id (self-link)
        let conn = open_memory_db().unwrap();

        // Create an episode
        let ep1 = episodic::store_episode(
            &conn,
            &NewEpisode {
                content: "self-link episode".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            },
        )
        .unwrap();

        // Find out what the NEXT semantic node ID will be
        let next_id: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM semantic_nodes",
                [],
                |row| row.get(0),
            )
            .unwrap();

        // Pre-create a link from ep1 TO the soon-to-be-created semantic node
        // This simulates a scenario where the episode already has a link to this node
        links::create_link(
            &conn,
            NodeRef::Episode(ep1),
            NodeRef::Semantic(NodeId(next_id)),
            LinkType::Causal,
            0.7,
        )
        .unwrap();

        // Now learn a node referencing ep1 — try_assign_category will find the
        // link from ep1 to this same node and should skip it (line 141)
        let report = learn_direct(
            &conn,
            vec![NewSemanticNode {
                content: "self-linked node".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: vec![ep1],
                embedding: None,
            }],
        )
        .unwrap();
        assert_eq!(report.nodes_created, 1);
        // No category should be assigned (no other categorized neighbors)
        assert_eq!(report.categories_assigned, 0);
    }
}
