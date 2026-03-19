use alaya::{
    ConsolidationProvider, Episode, EpisodeId, Interaction, NewImpression, NewSemanticNode,
    SemanticNode, SemanticType,
};
use pyo3::prelude::*;

pub struct PyConsolidationProvider {
    py_obj: PyObject,
}

impl PyConsolidationProvider {
    pub fn new(py_obj: PyObject) -> Self {
        Self { py_obj }
    }
}

// SAFETY: PyObject is Send, so PyConsolidationProvider is Send.
// The GIL is acquired inside each method before touching Python state.
unsafe impl Send for PyConsolidationProvider {}

impl ConsolidationProvider for PyConsolidationProvider {
    fn extract_knowledge(&self, episodes: &[Episode]) -> alaya::Result<Vec<NewSemanticNode>> {
        Python::with_gil(|py| {
            // Convert episodes to Python objects
            let py_episodes: Vec<PyObject> = episodes
                .iter()
                .map(|ep| {
                    let py_ep = crate::types::PyEpisode::from(ep.clone());
                    Py::new(py, py_ep)
                        .unwrap()
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind()
                })
                .collect();

            // Call Python method
            let result = self
                .py_obj
                .call_method1(py, "extract_knowledge", (py_episodes,))
                .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;

            // Expect a list of objects with attributes: content, node_type, confidence,
            // source_episodes (list of i64), embedding (optional list of f32)
            let py_list = result
                .downcast_bound::<pyo3::types::PyList>(py)
                .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;

            let mut nodes = Vec::new();
            for item in py_list.iter() {
                let content: String = item
                    .getattr("content")
                    .and_then(|v| v.extract())
                    .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;

                let node_type_str: String = item
                    .getattr("node_type")
                    .and_then(|v| v.extract())
                    .unwrap_or_else(|_| "fact".to_string());
                let node_type =
                    SemanticType::from_str(&node_type_str).unwrap_or(SemanticType::Fact);

                let confidence: f32 = item
                    .getattr("confidence")
                    .and_then(|v| v.extract())
                    .unwrap_or(1.0);

                let source_episodes: Vec<i64> = item
                    .getattr("source_episodes")
                    .and_then(|v| v.extract())
                    .unwrap_or_default();

                let embedding: Option<Vec<f32>> = item.getattr("embedding").ok().and_then(|v| {
                    if v.is_none() {
                        None
                    } else {
                        v.extract().ok()
                    }
                });

                nodes.push(NewSemanticNode {
                    content,
                    node_type,
                    confidence,
                    source_episodes: source_episodes.into_iter().map(EpisodeId).collect(),
                    embedding,
                });
            }
            Ok(nodes)
        })
    }

    fn extract_impressions(&self, interaction: &Interaction) -> alaya::Result<Vec<NewImpression>> {
        Python::with_gil(|py| {
            // Convert interaction to Python
            let py_interaction: PyObject =
                Py::new(py, crate::types::PyInteraction::from(interaction.clone()))
                    .unwrap()
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind();

            let result = self
                .py_obj
                .call_method1(py, "extract_impressions", (py_interaction,))
                .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;

            let py_list = result
                .downcast_bound::<pyo3::types::PyList>(py)
                .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;

            let mut impressions = Vec::new();
            for item in py_list.iter() {
                let domain: String = item
                    .getattr("domain")
                    .and_then(|v| v.extract())
                    .unwrap_or_else(|_| "general".to_string());

                let observation: String = item
                    .getattr("observation")
                    .and_then(|v| v.extract())
                    .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;

                let valence: f32 = item
                    .getattr("valence")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.0);

                impressions.push(NewImpression {
                    domain,
                    observation,
                    valence,
                });
            }
            Ok(impressions)
        })
    }

    fn detect_contradiction(&self, a: &SemanticNode, b: &SemanticNode) -> alaya::Result<bool> {
        Python::with_gil(|py| {
            let py_a: PyObject = Py::new(py, crate::types::PySemanticNode::from(a.clone()))
                .unwrap()
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind();
            let py_b: PyObject = Py::new(py, crate::types::PySemanticNode::from(b.clone()))
                .unwrap()
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind();

            let result = self
                .py_obj
                .call_method1(py, "detect_contradiction", (py_a, py_b))
                .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;

            result
                .extract::<bool>(py)
                .map_err(|e| alaya::AlayaError::Provider(e.to_string()))
        })
    }
}
