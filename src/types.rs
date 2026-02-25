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

// ---------------------------------------------------------------------------
// Node reference — polymorphic pointer into any store
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeRef {
    Episode(EpisodeId),
    Semantic(NodeId),
    Preference(PreferenceId),
}

impl NodeRef {
    pub fn type_str(&self) -> &'static str {
        match self {
            NodeRef::Episode(_) => "episode",
            NodeRef::Semantic(_) => "semantic",
            NodeRef::Preference(_) => "preference",
        }
    }

    pub fn id(&self) -> i64 {
        match self {
            NodeRef::Episode(EpisodeId(id))
            | NodeRef::Semantic(NodeId(id))
            | NodeRef::Preference(PreferenceId(id)) => *id,
        }
    }

    pub fn from_parts(node_type: &str, id: i64) -> Option<Self> {
        match node_type {
            "episode" => Some(NodeRef::Episode(EpisodeId(id))),
            "semantic" => Some(NodeRef::Semantic(NodeId(id))),
            "preference" => Some(NodeRef::Preference(PreferenceId(id))),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
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

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "system" => Some(Role::System),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
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
pub enum LinkType {
    Temporal,
    Topical,
    Entity,
    Causal,
    CoRetrieval,
}

impl LinkType {
    pub fn as_str(&self) -> &'static str {
        match self {
            LinkType::Temporal => "temporal",
            LinkType::Topical => "topical",
            LinkType::Entity => "entity",
            LinkType::Causal => "causal",
            LinkType::CoRetrieval => "co_retrieval",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "temporal" => Some(LinkType::Temporal),
            "topical" => Some(LinkType::Topical),
            "entity" => Some(LinkType::Entity),
            "causal" => Some(LinkType::Causal),
            "co_retrieval" => Some(LinkType::CoRetrieval),
            _ => None,
        }
    }
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
}

impl Query {
    pub fn simple(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            embedding: None,
            context: QueryContext::default(),
            max_results: 5,
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
}

#[derive(Debug, Clone)]
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
    pub links_pruned: u32,
    pub preferences_decayed: u32,
    pub impressions_pruned: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ForgettingReport {
    pub nodes_decayed: u32,
    pub nodes_archived: u32,
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
