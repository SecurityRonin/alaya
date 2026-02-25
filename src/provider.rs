use crate::error::Result;
use crate::types::*;

/// The agent provides this trait to support intelligent consolidation.
/// Alaya never calls an LLM directly — the agent owns the LLM connection.
pub trait ConsolidationProvider {
    /// Extract semantic knowledge from a batch of episodes.
    fn extract_knowledge(&self, episodes: &[Episode]) -> Result<Vec<NewSemanticNode>>;

    /// Extract behavioral impressions from an interaction.
    fn extract_impressions(&self, interaction: &Interaction) -> Result<Vec<NewImpression>>;

    /// Detect whether two semantic nodes contradict each other.
    fn detect_contradiction(&self, a: &SemanticNode, b: &SemanticNode) -> Result<bool>;
}

/// A no-op provider for when no LLM is available.
/// Consolidation and perfuming simply skip the LLM-dependent steps.
pub struct NoOpProvider;

impl ConsolidationProvider for NoOpProvider {
    fn extract_knowledge(&self, _episodes: &[Episode]) -> Result<Vec<NewSemanticNode>> {
        Ok(vec![])
    }

    fn extract_impressions(&self, _interaction: &Interaction) -> Result<Vec<NewImpression>> {
        Ok(vec![])
    }

    fn detect_contradiction(&self, _a: &SemanticNode, _b: &SemanticNode) -> Result<bool> {
        Ok(false)
    }
}

#[cfg(test)]
pub struct MockProvider {
    pub knowledge: Vec<NewSemanticNode>,
    pub impressions: Vec<NewImpression>,
}

#[cfg(test)]
impl MockProvider {
    pub fn empty() -> Self {
        Self {
            knowledge: vec![],
            impressions: vec![],
        }
    }

    pub fn with_knowledge(knowledge: Vec<NewSemanticNode>) -> Self {
        Self {
            knowledge,
            impressions: vec![],
        }
    }

    pub fn with_impressions(impressions: Vec<NewImpression>) -> Self {
        Self {
            knowledge: vec![],
            impressions,
        }
    }
}

#[cfg(test)]
impl ConsolidationProvider for MockProvider {
    fn extract_knowledge(&self, _episodes: &[Episode]) -> Result<Vec<NewSemanticNode>> {
        Ok(self.knowledge.clone())
    }

    fn extract_impressions(&self, _interaction: &Interaction) -> Result<Vec<NewImpression>> {
        Ok(self.impressions.clone())
    }

    fn detect_contradiction(&self, _a: &SemanticNode, _b: &SemanticNode) -> Result<bool> {
        Ok(false)
    }
}
