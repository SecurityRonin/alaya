use crate::types::*;

/// Rerank candidates using context similarity and recency.
pub fn rerank(
    candidates: Vec<(NodeRef, f64, String, Option<Role>, i64, EpisodeContext)>,
    query_context: &QueryContext,
    now: i64,
    max_results: usize,
) -> Vec<ScoredMemory> {
    let mut scored: Vec<ScoredMemory> = candidates
        .into_iter()
        .map(|(node, base_score, content, role, timestamp, ctx)| {
            let recency = recency_decay(timestamp, now);
            let context_sim = context_similarity(&ctx, query_context);
            let final_score = base_score * (1.0 + 0.3 * context_sim) * (1.0 + 0.2 * recency);

            ScoredMemory {
                node,
                content,
                score: final_score,
                role,
                timestamp,
            }
        })
        .collect();

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(max_results);
    scored
}

/// Exponential decay: exp(-age_days / 30.0)
/// Recent = ~1.0, 30 days = ~0.37, 90 days = ~0.05
fn recency_decay(timestamp: i64, now: i64) -> f64 {
    let age_secs = (now - timestamp).max(0) as f64;
    let age_days = age_secs / 86400.0;
    (-age_days / 30.0).exp()
}

/// Compute context similarity between a candidate's encoding context and the query context.
fn context_similarity(candidate: &EpisodeContext, query: &QueryContext) -> f64 {
    let topic_sim = jaccard(&candidate.topics, &query.topics);
    let entity_sim = jaccard(&candidate.mentioned_entities, &query.mentioned_entities);
    let sentiment_sim = 1.0 - ((candidate.sentiment - query.sentiment).abs() as f64 / 2.0);

    topic_sim * 0.5 + entity_sim * 0.25 + sentiment_sim * 0.25
}

fn jaccard(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let set_a: std::collections::HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let set_b: std::collections::HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let intersection = set_a.intersection(&set_b).count() as f64;
    let union = set_a.union(&set_b).count() as f64;
    if union == 0.0 { 0.0 } else { intersection / union }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recency_recent() {
        let now = 1000000;
        let recent = recency_decay(now - 3600, now); // 1 hour ago
        assert!(recent > 0.99);
    }

    #[test]
    fn test_recency_old() {
        let now = 1000000;
        let old = recency_decay(now - 86400 * 90, now); // 90 days ago
        assert!(old < 0.1);
    }

    #[test]
    fn test_jaccard() {
        let a = vec!["rust".to_string(), "async".to_string()];
        let b = vec!["rust".to_string(), "tokio".to_string()];
        let sim = jaccard(&a, &b);
        assert!((sim - 1.0 / 3.0).abs() < 0.01);
    }
}
