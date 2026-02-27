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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_not_found() {
        let e = AlayaError::NotFound("episode 42".into());
        assert_eq!(e.to_string(), "not found: episode 42");
    }

    #[test]
    fn test_display_invalid_input() {
        let e = AlayaError::InvalidInput("empty content".into());
        assert_eq!(e.to_string(), "invalid input: empty content");
    }

    #[test]
    fn test_display_provider() {
        let e = AlayaError::Provider("LLM timeout".into());
        assert_eq!(e.to_string(), "provider error: LLM timeout");
    }

    #[test]
    fn test_from_rusqlite_error() {
        let sqlite_err = rusqlite::Error::QueryReturnedNoRows;
        let e: AlayaError = sqlite_err.into();
        assert!(matches!(e, AlayaError::Db(_)));
        assert!(e.to_string().contains("database error"));
    }

    #[test]
    fn test_from_serde_error() {
        let bad_json = serde_json::from_str::<String>("not valid json");
        let serde_err = bad_json.unwrap_err();
        let e: AlayaError = serde_err.into();
        assert!(matches!(e, AlayaError::Serialization(_)));
        assert!(e.to_string().contains("serialization error"));
    }
}
