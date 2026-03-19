use pyo3::prelude::*;

mod types;

#[pymodule]
fn alaya(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Alaya>()?;
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
