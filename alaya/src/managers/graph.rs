use super::Graph;
use crate::error::Result;
use crate::graph;
use crate::types::*;

impl Graph<'_> {
    /// Get graph neighbors of a node up to `depth` hops via spreading activation.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn neighbors(&self, node: NodeRef, depth: u32) -> Result<Vec<(NodeRef, f32)>> {
        let result =
            graph::activation::spread_activation(self.conn, &[node], depth, 0.05, 0.6)?;
        let mut pairs: Vec<(NodeRef, f32)> =
            result.into_iter().filter(|(nr, _)| *nr != node).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(pairs)
    }

    /// Returns the link with the highest forward weight, if any exist.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn strongest_link(&self) -> Result<Option<(NodeRef, NodeRef, f32)>> {
        graph::links::strongest_link(self.conn)
    }
}

#[cfg(test)]
mod tests {
    use crate::testutil::fixtures::*;
    use crate::types::*;
    use crate::Alaya;

    #[test]
    fn neighbors_empty_for_nonexistent_node() {
        let alaya = Alaya::open_in_memory().unwrap();
        let neighbors = alaya
            .graph()
            .neighbors(NodeRef::Episode(EpisodeId(999)), 2)
            .unwrap();
        assert!(neighbors.is_empty());
    }

    #[test]
    fn neighbors_finds_linked_episodes() {
        let alaya = Alaya::open_in_memory().unwrap();
        let ep1 = episode("first message");
        let id1 = alaya.episodes().store(&ep1).unwrap();

        let mut ep2 = episode("second message");
        ep2.context.preceding_episode = Some(id1);
        let id2 = alaya.episodes().store(&ep2).unwrap();

        let neighbors = alaya
            .graph()
            .neighbors(NodeRef::Episode(id1), 2)
            .unwrap();
        assert!(
            neighbors.iter().any(|(nr, _)| *nr == NodeRef::Episode(id2)),
            "expected id2 in neighbors: {neighbors:?}"
        );
    }

    #[test]
    fn strongest_link_none_when_empty() {
        let alaya = Alaya::open_in_memory().unwrap();
        assert!(alaya.graph().strongest_link().unwrap().is_none());
    }

    #[test]
    fn strongest_link_finds_temporal() {
        let alaya = Alaya::open_in_memory().unwrap();
        let id1 = alaya.episodes().store(&episode("msg 1")).unwrap();
        let mut ep2 = episode("msg 2");
        ep2.context.preceding_episode = Some(id1);
        alaya.episodes().store(&ep2).unwrap();

        let link = alaya.graph().strongest_link().unwrap();
        assert!(link.is_some());
    }
}
