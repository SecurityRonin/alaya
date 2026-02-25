use rusqlite::Connection;
use crate::error::Result;
use crate::types::*;
use crate::retrieval::{bm25, vector, fusion, rerank};
use crate::graph::activation;
use crate::store::{episodic, strengths};

/// Execute a full hybrid retrieval query.
pub fn execute_query(conn: &Connection, query: &Query) -> Result<Vec<ScoredMemory>> {
    let now = query.context.current_timestamp.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    });

    let fetch_limit = query.max_results * 3;

    // Stage 1: Parallel retrieval (BM25 + vector + graph)
    let bm25_results: Vec<(NodeRef, f64)> = bm25::search_bm25(conn, &query.text, fetch_limit)?
        .into_iter()
        .map(|(eid, score)| (NodeRef::Episode(eid), score))
        .collect();

    let vector_results: Vec<(NodeRef, f64)> = match &query.embedding {
        Some(emb) => vector::search_vector(conn, emb, fetch_limit)?,
        None => vec![],
    };

    // Graph: seed from BM25 + vector top results, spread 1 hop
    let seed_nodes: Vec<NodeRef> = bm25_results.iter().take(3)
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

    // Stage 2: RRF fusion
    let mut sets: Vec<Vec<(NodeRef, f64)>> = vec![bm25_results];
    if !vector_results.is_empty() {
        sets.push(vector_results);
    }
    if !graph_results.is_empty() {
        sets.push(graph_results);
    }
    let fused = fusion::rrf_merge(&sets, 60);

    // Stage 3: Enrich candidates with content and context for reranking
    let candidates: Vec<(NodeRef, f64, String, Option<Role>, i64, EpisodeContext)> = fused
        .into_iter()
        .take(fetch_limit)
        .filter_map(|(node_ref, score)| {
            match node_ref {
                NodeRef::Episode(eid) => {
                    episodic::get_episode(conn, eid).ok().map(|ep| {
                        (node_ref, score, ep.content, Some(ep.role), ep.timestamp, ep.context)
                    })
                }
                _ => {
                    // For semantic/preference nodes, use minimal context
                    None // TODO: enrich semantic and preference nodes
                }
            }
        })
        .collect();

    let results = rerank::rerank(candidates, &query.context, now, query.max_results);

    // Stage 4: Post-retrieval updates (RIF + strength tracking)
    for scored in &results {
        let _ = strengths::on_access(conn, scored.node);
    }

    // Co-retrieval Hebbian strengthening between all retrieved pairs
    let retrieved_nodes: Vec<NodeRef> = results.iter().map(|r| r.node).collect();
    for i in 0..retrieved_nodes.len() {
        for j in (i + 1)..retrieved_nodes.len() {
            let _ = crate::graph::links::on_co_retrieval(conn, retrieved_nodes[i], retrieved_nodes[j]);
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;
    use crate::store::episodic;

    #[test]
    fn test_basic_query() {
        let conn = open_memory_db().unwrap();

        episodic::store_episode(&conn, &NewEpisode {
            content: "I love Rust programming".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 1000,
            context: EpisodeContext::default(),
            embedding: None,
        }).unwrap();

        episodic::store_episode(&conn, &NewEpisode {
            content: "Python is great for ML".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 2000,
            context: EpisodeContext::default(),
            embedding: None,
        }).unwrap();

        let results = execute_query(&conn, &Query {
            text: "Rust programming".to_string(),
            embedding: None,
            context: QueryContext {
                current_timestamp: Some(3000),
                ..Default::default()
            },
            max_results: 5,
        }).unwrap();

        assert!(!results.is_empty());
        assert!(results[0].content.contains("Rust"));
    }

    #[test]
    fn test_empty_query() {
        let conn = open_memory_db().unwrap();
        let results = execute_query(&conn, &Query::simple("")).unwrap();
        assert!(results.is_empty());
    }
}
