//! Shared parameter validation helpers for MCP tool handlers.

use crate::Role;

/// Parse a role string into the Role enum.
pub fn parse_role(role: &str) -> Result<Role, String> {
    match role.to_lowercase().as_str() {
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "system" => Ok(Role::System),
        _ => Err(format!(
            "invalid role '{role}'. Use: user, assistant, system"
        )),
    }
}

/// Get the current Unix timestamp.
pub fn now_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use super::*;

    #[test]
    fn parse_role_user() {
        assert!(matches!(parse_role("user"), Ok(Role::User)));
    }
    #[test]
    fn parse_role_assistant() {
        assert!(matches!(parse_role("assistant"), Ok(Role::Assistant)));
    }
    #[test]
    fn parse_role_system() {
        assert!(matches!(parse_role("system"), Ok(Role::System)));
    }
    #[test]
    fn parse_role_case_insensitive() {
        assert!(matches!(parse_role("USER"), Ok(Role::User)));
    }
    #[test]
    fn parse_role_invalid() {
        let err = parse_role("invalid").unwrap_err();
        assert!(err.contains("invalid role"));
    }
    #[test]
    fn now_timestamp_reasonable() {
        let ts = now_timestamp();
        assert!(ts > 1_700_000_000); // After 2023
    }
}
