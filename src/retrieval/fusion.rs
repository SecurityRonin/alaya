use crate::types::NodeRef;
use std::collections::HashMap;

/// Reciprocal Rank Fusion (RRF) merges multiple ranked result sets.
///
/// For each document d: score(d) = sum(1.0 / (k + rank_i + 1))
/// where rank_i is the 0-based rank of d in result set i.
///
/// Reference: Cormack, Clarke & Buettcher (2009)
pub fn rrf_merge(result_sets: &[Vec<(NodeRef, f64)>], k: u32) -> Vec<(NodeRef, f64)> {
    let mut scores: HashMap<NodeRef, f64> = HashMap::new();

    for result_set in result_sets {
        for (rank, (node_ref, _original_score)) in result_set.iter().enumerate() {
            *scores.entry(*node_ref).or_default() += 1.0 / (k as f64 + rank as f64 + 1.0);
        }
    }

    let mut merged: Vec<(NodeRef, f64)> = scores.into_iter().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn test_rrf_single_set() {
        let set = vec![
            (NodeRef::Episode(EpisodeId(1)), 0.9),
            (NodeRef::Episode(EpisodeId(2)), 0.5),
        ];
        let merged = rrf_merge(&[set], 60);
        assert_eq!(merged.len(), 2);
        // First item should have higher score
        assert!(merged[0].1 > merged[1].1);
    }

    #[test]
    fn test_rrf_two_sets_overlap() {
        let set_a = vec![
            (NodeRef::Episode(EpisodeId(1)), 0.9),
            (NodeRef::Episode(EpisodeId(2)), 0.5),
        ];
        let set_b = vec![
            (NodeRef::Episode(EpisodeId(2)), 0.8),
            (NodeRef::Episode(EpisodeId(3)), 0.3),
        ];
        let merged = rrf_merge(&[set_a, set_b], 60);
        // Episode 2 appears in both sets, should have highest combined score
        assert_eq!(merged[0].0, NodeRef::Episode(EpisodeId(2)));
    }

    #[test]
    fn test_rrf_disjoint() {
        let set_a = vec![(NodeRef::Episode(EpisodeId(1)), 0.9)];
        let set_b = vec![(NodeRef::Episode(EpisodeId(2)), 0.8)];
        let merged = rrf_merge(&[set_a, set_b], 60);
        assert_eq!(merged.len(), 2);
        // Both at rank 0, so equal RRF scores
        assert!((merged[0].1 - merged[1].1).abs() < 1e-10);
    }

    #[test]
    fn test_rrf_empty_sets() {
        let sets: Vec<Vec<(NodeRef, f64)>> = vec![];
        let result = rrf_merge(&sets, 60);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rrf_single_item() {
        let sets = vec![vec![(NodeRef::Episode(EpisodeId(1)), 1.0)]];
        let result = rrf_merge(&sets, 60);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, NodeRef::Episode(EpisodeId(1)));
    }
}
