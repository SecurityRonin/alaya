use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ID newtypes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EpisodeId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PreferenceId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImpressionId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LinkId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CategoryId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConflictId(pub i64);

// ---------------------------------------------------------------------------
// Node reference — polymorphic pointer into any store
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NodeRef {
    Episode(EpisodeId),
    Semantic(NodeId),
    Preference(PreferenceId),
    Category(CategoryId),
}

impl NodeRef {
    pub fn type_str(&self) -> &'static str {
        match self {
            NodeRef::Episode(_) => "episode",
            NodeRef::Semantic(_) => "semantic",
            NodeRef::Preference(_) => "preference",
            NodeRef::Category(_) => "category",
        }
    }

    pub fn id(&self) -> i64 {
        match self {
            NodeRef::Episode(EpisodeId(id))
            | NodeRef::Semantic(NodeId(id))
            | NodeRef::Preference(PreferenceId(id))
            | NodeRef::Category(CategoryId(id)) => *id,
        }
    }

    pub fn from_parts(node_type: &str, id: i64) -> Option<Self> {
        match node_type {
            "episode" => Some(NodeRef::Episode(EpisodeId(id))),
            "semantic" => Some(NodeRef::Semantic(NodeId(id))),
            "preference" => Some(NodeRef::Preference(PreferenceId(id))),
            "category" => Some(NodeRef::Category(CategoryId(id))),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Role {
    User,
    Assistant,
    System,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "system" => Some(Role::System),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum SemanticType {
    Fact,
    Relationship,
    Event,
    Concept,
}

impl SemanticType {
    pub fn as_str(&self) -> &'static str {
        match self {
            SemanticType::Fact => "fact",
            SemanticType::Relationship => "relationship",
            SemanticType::Event => "event",
            SemanticType::Concept => "concept",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "fact" => Some(SemanticType::Fact),
            "relationship" => Some(SemanticType::Relationship),
            "event" => Some(SemanticType::Event),
            "concept" => Some(SemanticType::Concept),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum LinkType {
    Temporal,
    Topical,
    Entity,
    Causal,
    CoRetrieval,
    MemberOf,
    Supersedes,
}

impl LinkType {
    pub fn as_str(&self) -> &'static str {
        match self {
            LinkType::Temporal => "temporal",
            LinkType::Topical => "topical",
            LinkType::Entity => "entity",
            LinkType::Causal => "causal",
            LinkType::CoRetrieval => "co_retrieval",
            LinkType::MemberOf => "member_of",
            LinkType::Supersedes => "supersedes",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "temporal" => Some(LinkType::Temporal),
            "topical" => Some(LinkType::Topical),
            "entity" => Some(LinkType::Entity),
            "causal" => Some(LinkType::Causal),
            "co_retrieval" => Some(LinkType::CoRetrieval),
            "member_of" => Some(LinkType::MemberOf),
            "supersedes" => Some(LinkType::Supersedes),
            _ => None,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConflictStatus {
    Detected,
    Verified,
    Resolved,
    Dismissed,
}

impl ConflictStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ConflictStatus::Detected => "detected",
            ConflictStatus::Verified => "verified",
            ConflictStatus::Resolved => "resolved",
            ConflictStatus::Dismissed => "dismissed",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "detected" => Some(ConflictStatus::Detected),
            "verified" => Some(ConflictStatus::Verified),
            "resolved" => Some(ConflictStatus::Resolved),
            "dismissed" => Some(ConflictStatus::Dismissed),
            _ => None,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConflictStrategy {
    #[default]
    Recency,
    Confidence,
    Manual,
}

// ---------------------------------------------------------------------------
// Episode types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpisodeContext {
    #[serde(default)]
    pub topics: Vec<String>,
    #[serde(default)]
    pub sentiment: f32,
    #[serde(default)]
    pub conversation_turn: u32,
    #[serde(default)]
    pub mentioned_entities: Vec<String>,
    #[serde(default)]
    pub preceding_episode: Option<EpisodeId>,
}

#[derive(Debug, Clone)]
pub struct NewEpisode {
    pub content: String,
    pub role: Role,
    pub session_id: String,
    pub timestamp: i64,
    pub context: EpisodeContext,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: EpisodeId,
    pub content: String,
    pub role: Role,
    pub session_id: String,
    pub timestamp: i64,
    pub context: EpisodeContext,
}

// ---------------------------------------------------------------------------
// Semantic types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NewSemanticNode {
    pub content: String,
    pub node_type: SemanticType,
    pub confidence: f32,
    pub source_episodes: Vec<EpisodeId>,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticNode {
    pub id: NodeId,
    pub content: String,
    pub node_type: SemanticType,
    pub confidence: f32,
    pub source_episodes: Vec<EpisodeId>,
    pub created_at: i64,
    pub last_corroborated: i64,
    pub corroboration_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    pub id: ConflictId,
    pub node_a: NodeId,
    pub node_b: NodeId,
    pub similarity: f32,
    pub status: ConflictStatus,
    pub detected_at: i64,
}

// ---------------------------------------------------------------------------
// Implicit types (vasana)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NewImpression {
    pub domain: String,
    pub observation: String,
    pub valence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Impression {
    pub id: ImpressionId,
    pub domain: String,
    pub observation: String,
    pub valence: f32,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preference {
    pub id: PreferenceId,
    pub domain: String,
    pub preference: String,
    pub confidence: f32,
    pub evidence_count: u32,
    pub first_observed: i64,
    pub last_reinforced: i64,
}

// ---------------------------------------------------------------------------
// Graph types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub id: LinkId,
    pub source: NodeRef,
    pub target: NodeRef,
    pub forward_weight: f32,
    pub backward_weight: f32,
    pub link_type: LinkType,
    pub created_at: i64,
    pub last_activated: i64,
    pub activation_count: u32,
}

// ---------------------------------------------------------------------------
// Category types (emergent ontology)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Category {
    pub id: CategoryId,
    pub label: String,
    pub prototype_node: NodeId,
    pub member_count: u32,
    pub centroid_embedding: Option<Vec<f32>>,
    pub created_at: i64,
    pub last_updated: i64,
    pub stability: f32,
    pub parent_id: Option<CategoryId>,
}

// ---------------------------------------------------------------------------
// Node strength (Bjork dual-strength model)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStrength {
    pub node: NodeRef,
    pub storage_strength: f32,
    pub retrieval_strength: f32,
    pub access_count: u32,
    pub last_accessed: i64,
}

// ---------------------------------------------------------------------------
// Retrieval types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Query {
    pub text: String,
    pub embedding: Option<Vec<f32>>,
    pub context: QueryContext,
    pub max_results: usize,
    pub boost_categories: Option<Vec<String>>,
}

impl Query {
    /// Create a simple text query with default settings.
    ///
    /// # Examples
    ///
    /// ```
    /// let q = alaya::Query::simple("What is Rust?");
    /// assert_eq!(q.max_results, 5);
    /// ```
    pub fn simple(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            embedding: None,
            context: QueryContext::default(),
            max_results: 5,
            boost_categories: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct QueryContext {
    pub topics: Vec<String>,
    pub sentiment: f32,
    pub mentioned_entities: Vec<String>,
    pub current_timestamp: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredMemory {
    pub node: NodeRef,
    pub content: String,
    pub score: f64,
    pub role: Option<Role>,
    pub timestamp: i64,
}

// ---------------------------------------------------------------------------
// Filter types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct KnowledgeFilter {
    pub node_type: Option<SemanticType>,
    pub min_confidence: Option<f32>,
    pub limit: Option<usize>,
    pub category: Option<String>,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PurgeFilter {
    /// Delete everything for this session
    Session(String),
    /// Delete all episodes older than this timestamp
    OlderThan(i64),
    /// Delete everything (nuclear option)
    All,
}

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsolidationReport {
    pub episodes_processed: u32,
    pub nodes_created: u32,
    pub links_created: u32,
    pub categories_assigned: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerfumingReport {
    pub impressions_stored: u32,
    pub preferences_crystallized: u32,
    pub preferences_reinforced: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransformationReport {
    pub duplicates_merged: u32,
    pub links_decayed: u32,
    pub links_pruned: u32,
    pub preferences_decayed: u32,
    pub impressions_pruned: u32,
    pub categories_discovered: u32,
    pub categories_merged: u32,
    pub categories_dissolved: u32,
    pub categories_split: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ForgettingReport {
    pub nodes_decayed: u32,
    pub nodes_archived: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReconcileReport {
    pub conflicts_detected: u32,
    pub conflicts_resolved: u32,
    pub conflicts_pending: u32,
    pub nodes_superseded: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DreamReport {
    pub consolidation: ConsolidationReport,
    pub perfuming: Option<PerfumingReport>,
    pub transformation: TransformationReport,
    pub forgetting: ForgettingReport,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PurgeReport {
    pub episodes_deleted: u32,
    pub nodes_deleted: u32,
    pub links_deleted: u32,
    pub embeddings_deleted: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatus {
    pub episode_count: u64,
    pub semantic_node_count: u64,
    pub preference_count: u64,
    pub impression_count: u64,
    pub link_count: u64,
    pub embedding_count: u64,
    pub category_count: u64,
}

// ---------------------------------------------------------------------------
// Display impls for report types
// ---------------------------------------------------------------------------

impl std::fmt::Display for MemoryStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "episodes: {}, semantic_nodes: {}, preferences: {}, links: {}",
            self.episode_count, self.semantic_node_count, self.preference_count, self.link_count
        )
    }
}

impl std::fmt::Display for ConsolidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "consolidated {} episodes, created {} nodes",
            self.episodes_processed, self.nodes_created
        )
    }
}

impl std::fmt::Display for TransformationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "merged: {}, decayed: {}, pruned: {}",
            self.duplicates_merged, self.links_decayed, self.links_pruned
        )
    }
}

impl std::fmt::Display for ForgettingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "archived: {}, decayed: {}",
            self.nodes_archived, self.nodes_decayed
        )
    }
}

impl std::fmt::Display for DreamReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "dream: {}, ", self.consolidation)?;
        match &self.perfuming {
            Some(p) => write!(f, "perfuming: {} impressions, ", p.impressions_stored)?,
            None => write!(f, "perfuming: skipped, ")?,
        }
        write!(f, "{}, {}", self.transformation, self.forgetting)
    }
}

// ---------------------------------------------------------------------------
// Provider types
// ---------------------------------------------------------------------------

/// Input to the perfuming process. The agent constructs this from whatever
/// interaction format it uses (Signal message, Discord message, HTTP request, etc.)
#[derive(Debug, Clone)]
pub struct Interaction {
    pub text: String,
    pub role: Role,
    pub session_id: String,
    pub timestamp: i64,
    pub context: EpisodeContext,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_ref_episode_roundtrip() {
        let nr = NodeRef::Episode(EpisodeId(42));
        assert_eq!(nr.type_str(), "episode");
        assert_eq!(nr.id(), 42);
        assert_eq!(NodeRef::from_parts("episode", 42), Some(nr));
    }

    #[test]
    fn test_node_ref_semantic_roundtrip() {
        let nr = NodeRef::Semantic(NodeId(7));
        assert_eq!(nr.type_str(), "semantic");
        assert_eq!(nr.id(), 7);
        assert_eq!(NodeRef::from_parts("semantic", 7), Some(nr));
    }

    #[test]
    fn test_node_ref_preference_roundtrip() {
        let nr = NodeRef::Preference(PreferenceId(99));
        assert_eq!(nr.type_str(), "preference");
        assert_eq!(nr.id(), 99);
        assert_eq!(NodeRef::from_parts("preference", 99), Some(nr));
    }

    #[test]
    fn test_node_ref_from_parts_invalid() {
        assert_eq!(NodeRef::from_parts("unknown", 1), None);
        assert_eq!(NodeRef::from_parts("", 0), None);
    }

    #[test]
    fn test_role_roundtrip() {
        for (role, s) in [
            (Role::User, "user"),
            (Role::Assistant, "assistant"),
            (Role::System, "system"),
        ] {
            assert_eq!(role.as_str(), s);
            assert_eq!(Role::from_str(s), Some(role));
        }
        assert_eq!(Role::from_str("invalid"), None);
    }

    #[test]
    fn test_semantic_type_roundtrip() {
        for (st, s) in [
            (SemanticType::Fact, "fact"),
            (SemanticType::Relationship, "relationship"),
            (SemanticType::Event, "event"),
            (SemanticType::Concept, "concept"),
        ] {
            assert_eq!(st.as_str(), s);
            assert_eq!(SemanticType::from_str(s), Some(st));
        }
        assert_eq!(SemanticType::from_str("bogus"), None);
    }

    #[test]
    fn test_link_type_roundtrip() {
        for (lt, s) in [
            (LinkType::Temporal, "temporal"),
            (LinkType::Topical, "topical"),
            (LinkType::Entity, "entity"),
            (LinkType::Causal, "causal"),
            (LinkType::CoRetrieval, "co_retrieval"),
        ] {
            assert_eq!(lt.as_str(), s);
            assert_eq!(LinkType::from_str(s), Some(lt));
        }
        assert_eq!(LinkType::from_str("unknown"), None);
    }

    #[test]
    fn test_query_simple_defaults() {
        let q = Query::simple("hello world");
        assert_eq!(q.text, "hello world");
        assert_eq!(q.max_results, 5);
        assert!(q.embedding.is_none());
    }

    #[test]
    fn test_category_id_newtype() {
        let id = CategoryId(42);
        assert_eq!(id.0, 42);
        let id2 = CategoryId(42);
        assert_eq!(id, id2);
    }

    #[test]
    fn test_node_ref_category_roundtrip() {
        let nr = NodeRef::Category(CategoryId(5));
        assert_eq!(nr.type_str(), "category");
        assert_eq!(nr.id(), 5);
        assert_eq!(NodeRef::from_parts("category", 5), Some(nr));
    }

    #[test]
    fn test_link_type_member_of_roundtrip() {
        assert_eq!(LinkType::MemberOf.as_str(), "member_of");
        assert_eq!(LinkType::from_str("member_of"), Some(LinkType::MemberOf));
    }

    #[test]
    fn test_episode_context_default() {
        let ctx = EpisodeContext::default();
        assert!(ctx.topics.is_empty());
        assert_eq!(ctx.sentiment, 0.0);
        assert_eq!(ctx.conversation_turn, 0);
        assert!(ctx.mentioned_entities.is_empty());
        assert!(ctx.preceding_episode.is_none());
    }

    #[test]
    fn test_knowledge_filter_default() {
        let f = KnowledgeFilter::default();
        assert!(f.node_type.is_none());
        assert!(f.min_confidence.is_none());
        assert!(f.limit.is_none());
        assert!(f.category.is_none());
    }

    #[test]
    fn test_purge_filter_variants() {
        // Session variant
        let s = PurgeFilter::Session("sess-1".to_string());
        if let PurgeFilter::Session(id) = s {
            assert_eq!(id, "sess-1");
        } else {
            panic!("expected Session variant");
        }

        // OlderThan variant
        let o = PurgeFilter::OlderThan(12345);
        if let PurgeFilter::OlderThan(ts) = o {
            assert_eq!(ts, 12345);
        } else {
            panic!("expected OlderThan variant");
        }

        // All variant
        let a = PurgeFilter::All;
        assert!(matches!(a, PurgeFilter::All));
    }

    #[test]
    fn test_report_defaults() {
        let cr = ConsolidationReport::default();
        assert_eq!(cr.episodes_processed, 0);
        assert_eq!(cr.nodes_created, 0);
        assert_eq!(cr.links_created, 0);
        assert_eq!(cr.categories_assigned, 0);

        let pr = PerfumingReport::default();
        assert_eq!(pr.impressions_stored, 0);
        assert_eq!(pr.preferences_crystallized, 0);
        assert_eq!(pr.preferences_reinforced, 0);

        let tr = TransformationReport::default();
        assert_eq!(tr.duplicates_merged, 0);
        assert_eq!(tr.links_decayed, 0);
        assert_eq!(tr.categories_discovered, 0);

        let fr = ForgettingReport::default();
        assert_eq!(fr.nodes_decayed, 0);
        assert_eq!(fr.nodes_archived, 0);

        let purge = PurgeReport::default();
        assert_eq!(purge.episodes_deleted, 0);
        assert_eq!(purge.nodes_deleted, 0);
        assert_eq!(purge.links_deleted, 0);
        assert_eq!(purge.embeddings_deleted, 0);
    }

    #[test]
    fn test_memory_status_fields() {
        let ms = MemoryStatus {
            episode_count: 1,
            semantic_node_count: 2,
            preference_count: 3,
            impression_count: 4,
            link_count: 5,
            embedding_count: 6,
            category_count: 7,
        };
        assert_eq!(ms.episode_count, 1);
        assert_eq!(ms.category_count, 7);
    }

    #[test]
    fn test_display_memory_status() {
        let s = MemoryStatus {
            episode_count: 10,
            semantic_node_count: 5,
            preference_count: 3,
            link_count: 7,
            impression_count: 0,
            embedding_count: 0,
            category_count: 0,
        };
        assert!(s.to_string().contains("episodes: 10"));
        assert!(s.to_string().contains("semantic_nodes: 5"));
        assert!(s.to_string().contains("preferences: 3"));
        assert!(s.to_string().contains("links: 7"));
    }

    #[test]
    fn test_display_consolidation_report() {
        let r = ConsolidationReport {
            episodes_processed: 4,
            nodes_created: 2,
            ..Default::default()
        };
        assert!(r.to_string().contains("consolidated 4 episodes"));
        assert!(r.to_string().contains("created 2 nodes"));
    }

    #[test]
    fn test_display_transformation_report() {
        let r = TransformationReport {
            duplicates_merged: 3,
            links_decayed: 1,
            links_pruned: 2,
            ..Default::default()
        };
        assert!(r.to_string().contains("merged: 3"));
        assert!(r.to_string().contains("decayed: 1"));
        assert!(r.to_string().contains("pruned: 2"));
    }

    #[test]
    fn test_display_forgetting_report() {
        let r = ForgettingReport {
            nodes_archived: 5,
            nodes_decayed: 2,
        };
        assert!(r.to_string().contains("archived: 5"));
        assert!(r.to_string().contains("decayed: 2"));
    }

    #[test]
    fn test_display_dream_report_without_perfuming() {
        let r = DreamReport {
            consolidation: ConsolidationReport {
                episodes_processed: 5,
                nodes_created: 2,
                links_created: 1,
                categories_assigned: 0,
            },
            perfuming: None,
            transformation: TransformationReport {
                duplicates_merged: 1,
                links_decayed: 0,
                links_pruned: 0,
                preferences_decayed: 0,
                impressions_pruned: 0,
                categories_discovered: 0,
                categories_merged: 0,
                categories_dissolved: 0,
                categories_split: 0,
            },
            forgetting: ForgettingReport {
                nodes_archived: 3,
                nodes_decayed: 1,
            },
        };
        let s = r.to_string();
        assert!(s.contains("consolidated 5 episodes"));
        assert!(s.contains("perfuming: skipped"));
        assert!(s.contains("archived: 3"));
    }

    #[test]
    fn test_display_dream_report_with_perfuming() {
        let r = DreamReport {
            consolidation: ConsolidationReport::default(),
            perfuming: Some(PerfumingReport {
                impressions_stored: 2,
                preferences_crystallized: 1,
                preferences_reinforced: 0,
            }),
            transformation: TransformationReport::default(),
            forgetting: ForgettingReport::default(),
        };
        let s = r.to_string();
        assert!(s.contains("2 impressions"));
        assert!(!s.contains("skipped"));
    }

    #[test]
    fn test_query_context_default() {
        let qc = QueryContext::default();
        assert!(qc.topics.is_empty());
        assert_eq!(qc.sentiment, 0.0);
        assert!(qc.mentioned_entities.is_empty());
        assert!(qc.current_timestamp.is_none());
    }

    #[test]
    fn test_node_strength_fields() {
        let ns = NodeStrength {
            node: NodeRef::Episode(EpisodeId(1)),
            storage_strength: 0.5,
            retrieval_strength: 1.0,
            access_count: 3,
            last_accessed: 9999,
        };
        assert_eq!(ns.access_count, 3);
        assert!((ns.storage_strength - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_interaction_fields() {
        let i = Interaction {
            text: "hello".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 42,
            context: EpisodeContext::default(),
        };
        assert_eq!(i.text, "hello");
        assert_eq!(i.timestamp, 42);
    }

    #[test]
    fn test_node_ref_id_all_variants() {
        assert_eq!(NodeRef::Episode(EpisodeId(10)).id(), 10);
        assert_eq!(NodeRef::Semantic(NodeId(20)).id(), 20);
        assert_eq!(NodeRef::Preference(PreferenceId(30)).id(), 30);
        assert_eq!(NodeRef::Category(CategoryId(40)).id(), 40);
    }

    #[test]
    fn test_link_type_from_str_all_variants() {
        assert_eq!(LinkType::from_str("temporal"), Some(LinkType::Temporal));
        assert_eq!(LinkType::from_str("topical"), Some(LinkType::Topical));
        assert_eq!(LinkType::from_str("entity"), Some(LinkType::Entity));
        assert_eq!(LinkType::from_str("causal"), Some(LinkType::Causal));
        assert_eq!(
            LinkType::from_str("co_retrieval"),
            Some(LinkType::CoRetrieval)
        );
        assert_eq!(LinkType::from_str("member_of"), Some(LinkType::MemberOf));
        assert_eq!(LinkType::from_str("bogus"), None);
    }

    #[test]
    fn test_scored_memory_fields() {
        let sm = ScoredMemory {
            node: NodeRef::Episode(EpisodeId(1)),
            content: "some content".to_string(),
            score: 0.75,
            role: Some(Role::Assistant),
            timestamp: 1000,
        };
        assert_eq!(sm.content, "some content");
        assert!((sm.score - 0.75).abs() < 1e-6);
        assert_eq!(sm.role, Some(Role::Assistant));
    }

    #[test]
    fn test_conflict_id_newtype() {
        let id = ConflictId(42);
        assert_eq!(id.0, 42);
        let id2 = ConflictId(42);
        assert_eq!(id, id2);
    }

    #[test]
    fn test_conflict_status_roundtrip() {
        for (status, s) in [
            (ConflictStatus::Detected, "detected"),
            (ConflictStatus::Verified, "verified"),
            (ConflictStatus::Resolved, "resolved"),
            (ConflictStatus::Dismissed, "dismissed"),
        ] {
            assert_eq!(status.as_str(), s);
            assert_eq!(ConflictStatus::from_str(s), Some(status));
        }
        assert_eq!(ConflictStatus::from_str("bogus"), None);
    }

    #[test]
    fn test_conflict_strategy_default() {
        let strategy = ConflictStrategy::default();
        assert_eq!(strategy, ConflictStrategy::Recency);
    }

    #[test]
    fn test_conflict_fields() {
        let c = Conflict {
            id: ConflictId(1),
            node_a: NodeId(10),
            node_b: NodeId(20),
            similarity: 0.92,
            status: ConflictStatus::Detected,
            detected_at: 1000,
        };
        assert_eq!(c.id.0, 1);
        assert!((c.similarity - 0.92).abs() < 1e-6);
    }

    #[test]
    fn test_reconcile_report_default() {
        let r = ReconcileReport::default();
        assert_eq!(r.conflicts_detected, 0);
        assert_eq!(r.conflicts_resolved, 0);
        assert_eq!(r.conflicts_pending, 0);
        assert_eq!(r.nodes_superseded, 0);
    }

    #[test]
    fn test_link_type_supersedes_roundtrip() {
        assert_eq!(LinkType::Supersedes.as_str(), "supersedes");
        assert_eq!(LinkType::from_str("supersedes"), Some(LinkType::Supersedes));
    }
}
