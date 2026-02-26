use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AlayaError {
    #[error("database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("provider error: {0}")]
    Provider(String),
}

pub type Result<T> = std::result::Result<T, AlayaError>;
