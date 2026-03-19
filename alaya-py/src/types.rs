use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Helper: convert a PyO3 error for invalid string values
// ---------------------------------------------------------------------------

fn invalid_value(val: &str, expected: &str) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "invalid {expected}: {val:?}"
    ))
}

// ---------------------------------------------------------------------------
// Output-only helper: NodeRef → (type_str, id) Python tuple
// ---------------------------------------------------------------------------

pub(crate) fn node_ref_to_py(py: Python<'_>, nr: ::alaya::NodeRef) -> PyObject {
    (nr.type_str().to_string(), nr.id()).into_pyobject(py).unwrap().unbind().into()
}

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyConsolidationReport {
    #[pyo3(get)]
    pub episodes_processed: u32,
    #[pyo3(get)]
    pub nodes_created: u32,
    #[pyo3(get)]
    pub links_created: u32,
    #[pyo3(get)]
    pub categories_assigned: u32,
}

#[pymethods]
impl PyConsolidationReport {
    fn __repr__(&self) -> String {
        format!(
            "ConsolidationReport(episodes_processed={}, nodes_created={}, links_created={}, categories_assigned={})",
            self.episodes_processed, self.nodes_created, self.links_created, self.categories_assigned
        )
    }
}

impl From<::alaya::ConsolidationReport> for PyConsolidationReport {
    fn from(r: ::alaya::ConsolidationReport) -> Self {
        Self {
            episodes_processed: r.episodes_processed,
            nodes_created: r.nodes_created,
            links_created: r.links_created,
            categories_assigned: r.categories_assigned,
        }
    }
}

// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyPerfumingReport {
    #[pyo3(get)]
    pub impressions_stored: u32,
    #[pyo3(get)]
    pub preferences_crystallized: u32,
    #[pyo3(get)]
    pub preferences_reinforced: u32,
}

#[pymethods]
impl PyPerfumingReport {
    fn __repr__(&self) -> String {
        format!(
            "PerfumingReport(impressions_stored={}, preferences_crystallized={}, preferences_reinforced={})",
            self.impressions_stored, self.preferences_crystallized, self.preferences_reinforced
        )
    }
}

impl From<::alaya::PerfumingReport> for PyPerfumingReport {
    fn from(r: ::alaya::PerfumingReport) -> Self {
        Self {
            impressions_stored: r.impressions_stored,
            preferences_crystallized: r.preferences_crystallized,
            preferences_reinforced: r.preferences_reinforced,
        }
    }
}

// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyTransformationReport {
    #[pyo3(get)]
    pub duplicates_merged: u32,
    #[pyo3(get)]
    pub links_decayed: u32,
    #[pyo3(get)]
    pub links_pruned: u32,
    #[pyo3(get)]
    pub preferences_decayed: u32,
    #[pyo3(get)]
    pub impressions_pruned: u32,
    #[pyo3(get)]
    pub categories_discovered: u32,
    #[pyo3(get)]
    pub categories_merged: u32,
    #[pyo3(get)]
    pub categories_dissolved: u32,
    #[pyo3(get)]
    pub categories_split: u32,
}

#[pymethods]
impl PyTransformationReport {
    fn __repr__(&self) -> String {
        format!(
            "TransformationReport(duplicates_merged={}, links_decayed={}, links_pruned={})",
            self.duplicates_merged, self.links_decayed, self.links_pruned
        )
    }
}

impl From<::alaya::TransformationReport> for PyTransformationReport {
    fn from(r: ::alaya::TransformationReport) -> Self {
        Self {
            duplicates_merged: r.duplicates_merged,
            links_decayed: r.links_decayed,
            links_pruned: r.links_pruned,
            preferences_decayed: r.preferences_decayed,
            impressions_pruned: r.impressions_pruned,
            categories_discovered: r.categories_discovered,
            categories_merged: r.categories_merged,
            categories_dissolved: r.categories_dissolved,
            categories_split: r.categories_split,
        }
    }
}

// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyForgettingReport {
    #[pyo3(get)]
    pub nodes_decayed: u32,
    #[pyo3(get)]
    pub nodes_archived: u32,
}

#[pymethods]
impl PyForgettingReport {
    fn __repr__(&self) -> String {
        format!(
            "ForgettingReport(nodes_decayed={}, nodes_archived={})",
            self.nodes_decayed, self.nodes_archived
        )
    }
}

impl From<::alaya::ForgettingReport> for PyForgettingReport {
    fn from(r: ::alaya::ForgettingReport) -> Self {
        Self {
            nodes_decayed: r.nodes_decayed,
            nodes_archived: r.nodes_archived,
        }
    }
}

// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyDreamReport {
    #[pyo3(get)]
    pub consolidation: PyConsolidationReport,
    #[pyo3(get)]
    pub perfuming: Option<PyPerfumingReport>,
    #[pyo3(get)]
    pub transformation: PyTransformationReport,
    #[pyo3(get)]
    pub forgetting: PyForgettingReport,
}

#[pymethods]
impl PyDreamReport {
    fn __repr__(&self) -> String {
        format!(
            "DreamReport(consolidation={:?}, perfuming={}, transformation={:?}, forgetting={:?})",
            self.consolidation.__repr__(),
            if self.perfuming.is_some() { "..." } else { "None" },
            self.transformation.__repr__(),
            self.forgetting.__repr__(),
        )
    }
}

impl From<::alaya::DreamReport> for PyDreamReport {
    fn from(r: ::alaya::DreamReport) -> Self {
        Self {
            consolidation: r.consolidation.into(),
            perfuming: r.perfuming.map(Into::into),
            transformation: r.transformation.into(),
            forgetting: r.forgetting.into(),
        }
    }
}

// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyPurgeReport {
    #[pyo3(get)]
    pub episodes_deleted: u32,
    #[pyo3(get)]
    pub nodes_deleted: u32,
    #[pyo3(get)]
    pub links_deleted: u32,
    #[pyo3(get)]
    pub embeddings_deleted: u32,
}

#[pymethods]
impl PyPurgeReport {
    fn __repr__(&self) -> String {
        format!(
            "PurgeReport(episodes_deleted={}, nodes_deleted={}, links_deleted={}, embeddings_deleted={})",
            self.episodes_deleted, self.nodes_deleted, self.links_deleted, self.embeddings_deleted
        )
    }
}

impl From<::alaya::PurgeReport> for PyPurgeReport {
    fn from(r: ::alaya::PurgeReport) -> Self {
        Self {
            episodes_deleted: r.episodes_deleted,
            nodes_deleted: r.nodes_deleted,
            links_deleted: r.links_deleted,
            embeddings_deleted: r.embeddings_deleted,
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryStatus
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyMemoryStatus {
    #[pyo3(get)]
    pub episode_count: u64,
    #[pyo3(get)]
    pub semantic_node_count: u64,
    #[pyo3(get)]
    pub preference_count: u64,
    #[pyo3(get)]
    pub impression_count: u64,
    #[pyo3(get)]
    pub link_count: u64,
    #[pyo3(get)]
    pub embedding_count: u64,
    #[pyo3(get)]
    pub category_count: u64,
}

#[pymethods]
impl PyMemoryStatus {
    fn __repr__(&self) -> String {
        format!(
            "MemoryStatus(episodes={}, semantic_nodes={}, preferences={}, impressions={}, links={}, embeddings={}, categories={})",
            self.episode_count, self.semantic_node_count, self.preference_count,
            self.impression_count, self.link_count, self.embedding_count, self.category_count
        )
    }
}

impl From<::alaya::MemoryStatus> for PyMemoryStatus {
    fn from(s: ::alaya::MemoryStatus) -> Self {
        Self {
            episode_count: s.episode_count,
            semantic_node_count: s.semantic_node_count,
            preference_count: s.preference_count,
            impression_count: s.impression_count,
            link_count: s.link_count,
            embedding_count: s.embedding_count,
            category_count: s.category_count,
        }
    }
}

// ---------------------------------------------------------------------------
// Data types — Episode
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyEpisode {
    #[pyo3(get)]
    pub id: i64,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub role: String,
    #[pyo3(get)]
    pub session_id: String,
    #[pyo3(get)]
    pub timestamp: i64,
}

#[pymethods]
impl PyEpisode {
    fn __repr__(&self) -> String {
        format!(
            "Episode(id={}, role={:?}, session_id={:?})",
            self.id, self.role, self.session_id
        )
    }
}

impl From<::alaya::Episode> for PyEpisode {
    fn from(e: ::alaya::Episode) -> Self {
        Self {
            id: e.id.0,
            content: e.content,
            role: e.role.as_str().to_string(),
            session_id: e.session_id,
            timestamp: e.timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// ScoredMemory
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyScoredMemory {
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub score: f64,
    /// (type_str, id) tuple identifying the source node
    #[pyo3(get)]
    pub node_type: String,
    #[pyo3(get)]
    pub node_id: i64,
    #[pyo3(get)]
    pub role: Option<String>,
    #[pyo3(get)]
    pub timestamp: i64,
}

#[pymethods]
impl PyScoredMemory {
    fn __repr__(&self) -> String {
        format!(
            "ScoredMemory(score={:.4}, node_type={:?}, node_id={})",
            self.score, self.node_type, self.node_id
        )
    }
}

impl From<::alaya::ScoredMemory> for PyScoredMemory {
    fn from(m: ::alaya::ScoredMemory) -> Self {
        Self {
            content: m.content,
            score: m.score,
            node_type: m.node.type_str().to_string(),
            node_id: m.node.id(),
            role: m.role.map(|r| r.as_str().to_string()),
            timestamp: m.timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// SemanticNode
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PySemanticNode {
    #[pyo3(get)]
    pub id: i64,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub node_type: String,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub source_episodes: Vec<i64>,
    #[pyo3(get)]
    pub created_at: i64,
    #[pyo3(get)]
    pub last_corroborated: i64,
    #[pyo3(get)]
    pub corroboration_count: u32,
}

#[pymethods]
impl PySemanticNode {
    fn __repr__(&self) -> String {
        format!(
            "SemanticNode(id={}, node_type={:?}, confidence={:.3})",
            self.id, self.node_type, self.confidence
        )
    }
}

impl From<::alaya::SemanticNode> for PySemanticNode {
    fn from(n: ::alaya::SemanticNode) -> Self {
        Self {
            id: n.id.0,
            content: n.content,
            node_type: n.node_type.as_str().to_string(),
            confidence: n.confidence,
            source_episodes: n.source_episodes.into_iter().map(|e| e.0).collect(),
            created_at: n.created_at,
            last_corroborated: n.last_corroborated,
            corroboration_count: n.corroboration_count,
        }
    }
}

// ---------------------------------------------------------------------------
// Preference
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyPreference {
    #[pyo3(get)]
    pub id: i64,
    #[pyo3(get)]
    pub domain: String,
    #[pyo3(get)]
    pub preference: String,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub evidence_count: u32,
    #[pyo3(get)]
    pub first_observed: i64,
    #[pyo3(get)]
    pub last_reinforced: i64,
}

#[pymethods]
impl PyPreference {
    fn __repr__(&self) -> String {
        format!(
            "Preference(id={}, domain={:?}, confidence={:.3})",
            self.id, self.domain, self.confidence
        )
    }
}

impl From<::alaya::Preference> for PyPreference {
    fn from(p: ::alaya::Preference) -> Self {
        Self {
            id: p.id.0,
            domain: p.domain,
            preference: p.preference,
            confidence: p.confidence,
            evidence_count: p.evidence_count,
            first_observed: p.first_observed,
            last_reinforced: p.last_reinforced,
        }
    }
}

// ---------------------------------------------------------------------------
// Impression
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyImpression {
    #[pyo3(get)]
    pub id: i64,
    #[pyo3(get)]
    pub domain: String,
    #[pyo3(get)]
    pub observation: String,
    #[pyo3(get)]
    pub valence: f32,
    #[pyo3(get)]
    pub timestamp: i64,
}

#[pymethods]
impl PyImpression {
    fn __repr__(&self) -> String {
        format!(
            "Impression(id={}, domain={:?}, valence={:.3})",
            self.id, self.domain, self.valence
        )
    }
}

impl From<::alaya::Impression> for PyImpression {
    fn from(i: ::alaya::Impression) -> Self {
        Self {
            id: i.id.0,
            domain: i.domain,
            observation: i.observation,
            valence: i.valence,
            timestamp: i.timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// Category
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyCategory {
    #[pyo3(get)]
    pub id: i64,
    #[pyo3(get)]
    pub label: String,
    #[pyo3(get)]
    pub prototype_node: i64,
    #[pyo3(get)]
    pub member_count: u32,
    #[pyo3(get)]
    pub created_at: i64,
    #[pyo3(get)]
    pub last_updated: i64,
    #[pyo3(get)]
    pub stability: f32,
    #[pyo3(get)]
    pub parent_id: Option<i64>,
}

#[pymethods]
impl PyCategory {
    fn __repr__(&self) -> String {
        format!(
            "Category(id={}, label={:?}, member_count={})",
            self.id, self.label, self.member_count
        )
    }
}

impl From<::alaya::Category> for PyCategory {
    fn from(c: ::alaya::Category) -> Self {
        Self {
            id: c.id.0,
            label: c.label,
            prototype_node: c.prototype_node.0,
            member_count: c.member_count,
            created_at: c.created_at,
            last_updated: c.last_updated,
            stability: c.stability,
            parent_id: c.parent_id.map(|p| p.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Link
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyLink {
    #[pyo3(get)]
    pub id: i64,
    /// (type_str, id) for source node
    #[pyo3(get)]
    pub source_type: String,
    #[pyo3(get)]
    pub source_id: i64,
    /// (type_str, id) for target node
    #[pyo3(get)]
    pub target_type: String,
    #[pyo3(get)]
    pub target_id: i64,
    #[pyo3(get)]
    pub link_type: String,
    #[pyo3(get)]
    pub forward_weight: f32,
    #[pyo3(get)]
    pub backward_weight: f32,
    #[pyo3(get)]
    pub created_at: i64,
    #[pyo3(get)]
    pub last_activated: i64,
    #[pyo3(get)]
    pub activation_count: u32,
}

#[pymethods]
impl PyLink {
    fn __repr__(&self) -> String {
        format!(
            "Link(id={}, {}:{} -[{}]-> {}:{})",
            self.id, self.source_type, self.source_id,
            self.link_type, self.target_type, self.target_id
        )
    }
}

impl From<::alaya::Link> for PyLink {
    fn from(l: ::alaya::Link) -> Self {
        Self {
            id: l.id.0,
            source_type: l.source.type_str().to_string(),
            source_id: l.source.id(),
            target_type: l.target.type_str().to_string(),
            target_id: l.target.id(),
            link_type: l.link_type.as_str().to_string(),
            forward_weight: l.forward_weight,
            backward_weight: l.backward_weight,
            created_at: l.created_at,
            last_activated: l.last_activated,
            activation_count: l.activation_count,
        }
    }
}

// ---------------------------------------------------------------------------
// Input types — NewEpisode
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyNewEpisode {
    #[pyo3(get, set)]
    pub content: String,
    #[pyo3(get, set)]
    pub role: String,
    #[pyo3(get, set)]
    pub session_id: String,
    #[pyo3(get, set)]
    pub timestamp: i64,
}

#[pymethods]
impl PyNewEpisode {
    #[new]
    #[pyo3(signature = (content, role, session_id, timestamp))]
    fn new(content: String, role: String, session_id: String, timestamp: i64) -> Self {
        Self { content, role, session_id, timestamp }
    }

    fn __repr__(&self) -> String {
        format!(
            "NewEpisode(role={:?}, session_id={:?})",
            self.role, self.session_id
        )
    }
}

impl TryFrom<PyNewEpisode> for ::alaya::NewEpisode {
    type Error = PyErr;

    fn try_from(e: PyNewEpisode) -> Result<Self, Self::Error> {
        let role = ::alaya::Role::from_str(&e.role)
            .ok_or_else(|| invalid_value(&e.role, "role"))?;
        Ok(::alaya::NewEpisode {
            content: e.content,
            role,
            session_id: e.session_id,
            timestamp: e.timestamp,
            context: ::alaya::EpisodeContext::default(),
            embedding: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Input types — Query
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyQuery {
    #[pyo3(get, set)]
    pub text: String,
    #[pyo3(get, set)]
    pub max_results: usize,
}

#[pymethods]
impl PyQuery {
    #[new]
    #[pyo3(signature = (text, max_results = 5))]
    fn new(text: String, max_results: usize) -> Self {
        Self { text, max_results }
    }

    fn __repr__(&self) -> String {
        format!("Query(text={:?}, max_results={})", self.text, self.max_results)
    }
}

impl From<PyQuery> for ::alaya::Query {
    fn from(q: PyQuery) -> Self {
        let mut query = ::alaya::Query::simple(q.text);
        query.max_results = q.max_results;
        query
    }
}

// ---------------------------------------------------------------------------
// Input types — Interaction
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyInteraction {
    #[pyo3(get, set)]
    pub text: String,
    #[pyo3(get, set)]
    pub role: String,
    #[pyo3(get, set)]
    pub session_id: String,
    #[pyo3(get, set)]
    pub timestamp: i64,
}

#[pymethods]
impl PyInteraction {
    #[new]
    #[pyo3(signature = (text, role, session_id, timestamp))]
    fn new(text: String, role: String, session_id: String, timestamp: i64) -> Self {
        Self { text, role, session_id, timestamp }
    }

    fn __repr__(&self) -> String {
        format!(
            "Interaction(role={:?}, session_id={:?})",
            self.role, self.session_id
        )
    }
}

impl TryFrom<PyInteraction> for ::alaya::Interaction {
    type Error = PyErr;

    fn try_from(i: PyInteraction) -> Result<Self, Self::Error> {
        let role = ::alaya::Role::from_str(&i.role)
            .ok_or_else(|| invalid_value(&i.role, "role"))?;
        Ok(::alaya::Interaction {
            text: i.text,
            role,
            session_id: i.session_id,
            timestamp: i.timestamp,
            context: ::alaya::EpisodeContext::default(),
        })
    }
}

impl From<::alaya::Interaction> for PyInteraction {
    fn from(i: ::alaya::Interaction) -> Self {
        Self {
            text: i.text,
            role: i.role.as_str().to_string(),
            session_id: i.session_id,
            timestamp: i.timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// Input types — KnowledgeFilter
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct PyKnowledgeFilter {
    #[pyo3(get, set)]
    pub node_type: Option<String>,
    #[pyo3(get, set)]
    pub min_confidence: Option<f32>,
    #[pyo3(get, set)]
    pub limit: Option<usize>,
    #[pyo3(get, set)]
    pub category: Option<String>,
}

#[pymethods]
impl PyKnowledgeFilter {
    #[new]
    #[pyo3(signature = (node_type = None, min_confidence = None, limit = None, category = None))]
    fn new(
        node_type: Option<String>,
        min_confidence: Option<f32>,
        limit: Option<usize>,
        category: Option<String>,
    ) -> Self {
        Self { node_type, min_confidence, limit, category }
    }

    fn __repr__(&self) -> String {
        format!(
            "KnowledgeFilter(node_type={:?}, min_confidence={:?}, limit={:?})",
            self.node_type, self.min_confidence, self.limit
        )
    }
}

impl TryFrom<PyKnowledgeFilter> for ::alaya::KnowledgeFilter {
    type Error = PyErr;

    fn try_from(f: PyKnowledgeFilter) -> Result<Self, Self::Error> {
        let node_type = f.node_type
            .map(|s| ::alaya::SemanticType::from_str(&s)
                .ok_or_else(|| invalid_value(&s, "node_type")))
            .transpose()?;
        Ok(::alaya::KnowledgeFilter {
            node_type,
            min_confidence: f.min_confidence,
            limit: f.limit,
            category: f.category,
        })
    }
}
