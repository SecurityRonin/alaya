//! Shared MCP handler helpers — parameter extraction and error formatting.

#![allow(dead_code, unused_macros, unused_imports)]

use serde_json::Value;

pub(crate) fn require_str<'a>(params: &'a Value, key: &str) -> std::result::Result<&'a str, String> {
    params
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("missing required parameter '{key}'"))
}

pub(crate) fn require_u64(params: &Value, key: &str) -> std::result::Result<u64, String> {
    params
        .get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("missing required parameter '{key}'"))
}

pub(crate) fn optional_str<'a>(params: &'a Value, key: &str) -> Option<&'a str> {
    params.get(key).and_then(|v| v.as_str())
}

pub(crate) fn optional_u64(params: &Value, key: &str) -> Option<u64> {
    params.get(key).and_then(|v| v.as_u64())
}

pub(crate) fn optional_f64(params: &Value, key: &str) -> Option<f64> {
    params.get(key).and_then(|v| v.as_f64())
}

pub(crate) fn optional_bool(params: &Value, key: &str) -> Option<bool> {
    params.get(key).and_then(|v| v.as_bool())
}

/// Run a closure against the Alaya store and serialize the result.
/// Returns a JSON-formatted success string or an error response string.
pub(crate) fn run<F, T>(server: &super::AlayaMcp, f: F) -> String
where
    F: FnOnce(&crate::Alaya) -> crate::Result<T>,
    T: serde::Serialize,
{
    match server.with_store(f) {
        Ok(result) => serde_json::to_string_pretty(&result).unwrap_or_default(),
        Err(e) => super::error_response(&e.to_string()),
    }
}

/// Early-return macro for MCP handlers: converts a `Result<T, String>` from
/// `require_str`/`require_u64` into either the success value or an error
/// response string. Used in handler functions that return `String`.
///
/// Usage: `let val = try_or_err!(handler::require_str(&params, "key"));`
macro_rules! try_or_err {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(msg) => return $crate::mcp::error_response(&msg),
        }
    };
}
pub(crate) use try_or_err;
