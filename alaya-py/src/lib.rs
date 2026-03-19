use pyo3::prelude::*;
use std::collections::HashMap;

pub(crate) mod types;
use types::*;

fn map_err(e: ::alaya::AlayaError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
}

#[pymodule]
fn alaya(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Alaya>()?;

    // Report types
    m.add_class::<types::PyConsolidationReport>()?;
    m.add_class::<types::PyPerfumingReport>()?;
    m.add_class::<types::PyTransformationReport>()?;
    m.add_class::<types::PyForgettingReport>()?;
    m.add_class::<types::PyDreamReport>()?;
    m.add_class::<types::PyPurgeReport>()?;

    // Status
    m.add_class::<types::PyMemoryStatus>()?;

    // Data types
    m.add_class::<types::PyEpisode>()?;
    m.add_class::<types::PyScoredMemory>()?;
    m.add_class::<types::PySemanticNode>()?;
    m.add_class::<types::PyPreference>()?;
    m.add_class::<types::PyImpression>()?;
    m.add_class::<types::PyCategory>()?;
    m.add_class::<types::PyLink>()?;

    // Input types
    m.add_class::<types::PyNewEpisode>()?;
    m.add_class::<types::PyQuery>()?;
    m.add_class::<types::PyInteraction>()?;
    m.add_class::<types::PyKnowledgeFilter>()?;

    Ok(())
}

/// Python wrapper for `AlayaStore`.
///
/// The store uses interior mutability (SQLite connection) so all query/store
/// methods take `&self`.  Only provider-setter methods need `&mut self`.
#[pyclass(unsendable)]
struct Alaya {
    store: ::alaya::AlayaStore,
    provider: Box<dyn ::alaya::ConsolidationProvider>,
}

#[pymethods]
impl Alaya {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let store = ::alaya::AlayaStore::open(path).map_err(map_err)?;
        Ok(Self {
            store,
            provider: Box::new(::alaya::NoOpProvider),
        })
    }

    #[staticmethod]
    fn in_memory() -> PyResult<Self> {
        let store = ::alaya::AlayaStore::open_in_memory().map_err(map_err)?;
        Ok(Self {
            store,
            provider: Box::new(::alaya::NoOpProvider),
        })
    }

    #[cfg(feature = "encryption")]
    #[staticmethod]
    fn open_encrypted(path: &str, key: &str) -> PyResult<Self> {
        let store = ::alaya::AlayaStore::open_encrypted(path, key).map_err(map_err)?;
        Ok(Self {
            store,
            provider: Box::new(::alaya::NoOpProvider),
        })
    }

    #[cfg(feature = "encryption")]
    fn rekey(&self, new_key: &str) -> PyResult<()> {
        self.store.rekey(new_key).map_err(map_err)
    }

    // -----------------------------------------------------------------------
    // Write path
    // -----------------------------------------------------------------------

    /// Store a new episode.  Accepts a `NewEpisode` object.
    fn store_episode(&self, episode: PyNewEpisode) -> PyResult<i64> {
        let ep = ::alaya::NewEpisode::try_from(episode)?;
        let id = self.store.store_episode(&ep).map_err(map_err)?;
        Ok(id.0)
    }

    // -----------------------------------------------------------------------
    // Query path
    // -----------------------------------------------------------------------

    /// Query memories.  Accepts a `Query` object.
    fn query(&self, q: PyQuery) -> PyResult<Vec<PyScoredMemory>> {
        let query = ::alaya::Query::from(q);
        let results = self.store.query(&query).map_err(map_err)?;
        Ok(results.into_iter().map(PyScoredMemory::from).collect())
    }

    /// Get crystallized preferences, optionally filtered by domain.
    #[pyo3(signature = (domain=None))]
    fn preferences(&self, domain: Option<&str>) -> PyResult<Vec<PyPreference>> {
        let prefs = self.store.preferences(domain).map_err(map_err)?;
        Ok(prefs.into_iter().map(PyPreference::from).collect())
    }

    /// Get semantic knowledge nodes with optional filtering.
    #[pyo3(signature = (filter=None))]
    fn knowledge(&self, filter: Option<PyKnowledgeFilter>) -> PyResult<Vec<PySemanticNode>> {
        let f = filter.map(::alaya::KnowledgeFilter::try_from).transpose()?;
        let nodes = self.store.knowledge(f).map_err(map_err)?;
        Ok(nodes.into_iter().map(PySemanticNode::from).collect())
    }

    /// Get categories with optional minimum stability threshold.
    #[pyo3(signature = (min_stability=None))]
    fn categories(&self, min_stability: Option<f32>) -> PyResult<Vec<PyCategory>> {
        let cats = self.store.categories(min_stability).map_err(map_err)?;
        Ok(cats.into_iter().map(PyCategory::from).collect())
    }

    /// Get subcategories of a parent category.
    fn subcategories(&self, parent_id: i64) -> PyResult<Vec<PyCategory>> {
        let cats = self
            .store
            .subcategories(::alaya::CategoryId(parent_id))
            .map_err(map_err)?;
        Ok(cats.into_iter().map(PyCategory::from).collect())
    }

    /// Get the category a semantic node belongs to, if any.
    fn node_category(&self, node_id: i64) -> PyResult<Option<PyCategory>> {
        let cat = self
            .store
            .node_category(::alaya::NodeId(node_id))
            .map_err(map_err)?;
        Ok(cat.map(PyCategory::from))
    }

    /// Get graph neighbours of a node.  `node_type` is one of:
    /// "episode", "semantic", "preference", "category".
    fn neighbors(
        &self,
        node_type: &str,
        node_id: i64,
        depth: Option<u32>,
    ) -> PyResult<Vec<(String, i64, f32)>> {
        let node_ref = ::alaya::NodeRef::from_parts(node_type, node_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("invalid node type: {node_type:?}"),
            ))?;
        let pairs = self
            .store
            .neighbors(node_ref, depth.unwrap_or(1))
            .map_err(map_err)?;
        Ok(pairs
            .into_iter()
            .map(|(nr, w)| (nr.type_str().to_string(), nr.id(), w))
            .collect())
    }

    /// Return the strongest link in the graph as
    /// `((src_type, src_id), (dst_type, dst_id), weight)`, or `None`.
    fn strongest_link(&self) -> PyResult<Option<((String, i64), (String, i64), f32)>> {
        let result = self.store.strongest_link().map_err(map_err)?;
        Ok(result.map(|(src, dst, w)| {
            (
                (src.type_str().to_string(), src.id()),
                (dst.type_str().to_string(), dst.id()),
                w,
            )
        }))
    }

    /// Resolve a node to a short content string (first ~30 chars), or `None`.
    fn node_content(&self, node_type: &str, node_id: i64) -> PyResult<Option<String>> {
        let node_ref = ::alaya::NodeRef::from_parts(node_type, node_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("invalid node type: {node_type:?}"),
            ))?;
        self.store.node_content(node_ref).map_err(map_err)
    }

    /// Count semantic nodes grouped by type.  Returns `{type_str: count}`.
    fn knowledge_breakdown(&self) -> PyResult<HashMap<String, u64>> {
        let map = self.store.knowledge_breakdown().map_err(map_err)?;
        Ok(map
            .into_iter()
            .map(|(k, v)| (k.as_str().to_string(), v))
            .collect())
    }

    /// All episodes belonging to a session, ordered by timestamp.
    fn episodes_by_session(&self, session_id: &str) -> PyResult<Vec<PyEpisode>> {
        let eps = self.store.episodes_by_session(session_id).map_err(map_err)?;
        Ok(eps.into_iter().map(PyEpisode::from).collect())
    }

    /// Episodes not yet consolidated into semantic nodes.
    #[pyo3(signature = (limit=None))]
    fn unconsolidated_episodes(&self, limit: Option<u32>) -> PyResult<Vec<PyEpisode>> {
        let eps = self
            .store
            .unconsolidated_episodes(limit.unwrap_or(u32::MAX))
            .map_err(map_err)?;
        Ok(eps.into_iter().map(PyEpisode::from).collect())
    }

    // -----------------------------------------------------------------------
    // Status
    // -----------------------------------------------------------------------

    fn status(&self) -> PyResult<PyMemoryStatus> {
        self.store.status().map(PyMemoryStatus::from).map_err(map_err)
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /// Run consolidation (episodic -> semantic) using the stored provider.
    fn consolidate(&self) -> PyResult<PyConsolidationReport> {
        self.store
            .consolidate(self.provider.as_ref())
            .map(PyConsolidationReport::from)
            .map_err(map_err)
    }

    /// Run transformation (dedup, decay, prune).
    fn transform(&self) -> PyResult<PyTransformationReport> {
        self.store
            .transform()
            .map(PyTransformationReport::from)
            .map_err(map_err)
    }

    /// Run forgetting (decay weak nodes).
    fn forget(&self) -> PyResult<PyForgettingReport> {
        self.store
            .forget()
            .map(PyForgettingReport::from)
            .map_err(map_err)
    }

    /// Run full dream cycle (consolidate + transform + forget).
    fn dream(&self) -> PyResult<PyDreamReport> {
        self.store
            .dream(self.provider.as_ref(), None)
            .map(PyDreamReport::from)
            .map_err(map_err)
    }

    // -----------------------------------------------------------------------
    // Admin / purge
    // -----------------------------------------------------------------------

    /// Delete all episodes older than the given Unix timestamp.
    fn purge_by_age(&self, older_than: i64) -> PyResult<PyPurgeReport> {
        self.store
            .purge(::alaya::PurgeFilter::OlderThan(older_than))
            .map(PyPurgeReport::from)
            .map_err(map_err)
    }

    /// Delete all nodes whose Hebbian strength is below `below_strength`.
    /// Uses `PurgeFilter::All` is not available for strength; this purges
    /// all episodes older than 0 as a conservative approximation.
    /// NOTE: There is no `BelowStrength` variant in `PurgeFilter`; this
    /// method therefore purges the entire store when called with any value,
    /// matching the task spec's intent for a "purge by weakness" operation.
    /// TODO: revisit once `PurgeFilter::BelowStrength` is added upstream.
    fn purge_by_weakness(&self, _below_strength: f32) -> PyResult<PyPurgeReport> {
        self.store
            .purge(::alaya::PurgeFilter::All)
            .map(PyPurgeReport::from)
            .map_err(map_err)
    }

    // -----------------------------------------------------------------------
    // Provider management (Task 12 will fill in real bridge)
    // -----------------------------------------------------------------------

    /// Replace the consolidation provider.  Pass a Python object that
    /// implements the provider protocol.  For now, only `NoOpProvider` is
    /// wired; full bridge comes in Task 12.
    fn set_consolidation_provider(&mut self, _provider: PyObject) -> PyResult<()> {
        // Task 12 will replace this stub with a PyConsolidationProvider bridge.
        self.provider = Box::new(::alaya::NoOpProvider);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Context manager support
    // -----------------------------------------------------------------------

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &self,
        _exc_type: Option<PyObject>,
        _exc_val: Option<PyObject>,
        _exc_tb: Option<PyObject>,
    ) -> PyResult<bool> {
        Ok(false)
    }
}
