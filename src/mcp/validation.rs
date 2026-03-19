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

/// Expand `~/` prefix to the user's home directory.
pub fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/{rest}")
    } else {
        path.to_string()
    }
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
    #[test]
    fn expand_tilde_with_home_prefix() {
        let result = expand_tilde("~/foo/bar");
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        assert_eq!(result, format!("{home}/foo/bar"));
    }
    #[test]
    fn expand_tilde_absolute_path_unchanged() {
        assert_eq!(expand_tilde("/abs/path"), "/abs/path");
    }
    #[test]
    fn expand_tilde_relative_path_unchanged() {
        assert_eq!(expand_tilde("relative/path"), "relative/path");
    }
    #[test]
    fn expand_tilde_just_tilde_slash() {
        let result = expand_tilde("~/");
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        assert_eq!(result, format!("{home}/"));
    }
    #[test]
    fn expand_tilde_bare_tilde_no_slash_unchanged() {
        assert_eq!(expand_tilde("~"), "~");
    }
    #[test]
    fn expand_tilde_empty_string_unchanged() {
        assert_eq!(expand_tilde(""), "");
    }
    #[test]
    fn expand_tilde_deep_nested_path() {
        let result = expand_tilde("~/a/b/c/d/e.txt");
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        assert_eq!(result, format!("{home}/a/b/c/d/e.txt"));
    }
}
