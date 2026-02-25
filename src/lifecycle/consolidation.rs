use rusqlite::Connection;
use crate::error::Result;
use crate::provider::ConsolidationProvider;
use crate::store::{episodic, semantic};
use crate::graph::links;
use crate::types::*;

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
        let node_id = semantic::store_semantic_node(conn, &node_data)?;
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
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;
    use crate::provider::MockProvider;
    use crate::store::episodic;

    #[test]
    fn test_consolidation_below_threshold() {
        let conn = open_memory_db().unwrap();
        // Only 2 episodes — below threshold of 3
        episodic::store_episode(&conn, &NewEpisode {
            content: "hello".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 1000,
            context: EpisodeContext::default(),
            embedding: None,
        }).unwrap();
        episodic::store_episode(&conn, &NewEpisode {
            content: "world".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 2000,
            context: EpisodeContext::default(),
            embedding: None,
        }).unwrap();

        let report = consolidate(&conn, &MockProvider::empty()).unwrap();
        assert_eq!(report.nodes_created, 0);
    }

    #[test]
    fn test_consolidation_creates_nodes() {
        let conn = open_memory_db().unwrap();
        let mut ep_ids = vec![];
        for i in 0..5 {
            let id = episodic::store_episode(&conn, &NewEpisode {
                content: format!("message about Rust {}", i),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000 + i * 100,
                context: EpisodeContext::default(),
                embedding: None,
            }).unwrap();
            ep_ids.push(id);
        }

        let provider = MockProvider::with_knowledge(vec![
            NewSemanticNode {
                content: "User discusses Rust programming".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.8,
                source_episodes: ep_ids,
                embedding: None,
            },
        ]);

        let report = consolidate(&conn, &provider).unwrap();
        assert_eq!(report.nodes_created, 1);
        assert!(report.links_created > 0);
    }
}
