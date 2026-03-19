use pyo3::prelude::*;

pub(crate) mod types;

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

#[pyclass(unsendable)]
struct Alaya {
    store: ::alaya::AlayaStore,
}

#[pymethods]
impl Alaya {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let store = ::alaya::AlayaStore::open(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { store })
    }

    #[staticmethod]
    fn in_memory() -> PyResult<Self> {
        let store = ::alaya::AlayaStore::open_in_memory()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { store })
    }
}
