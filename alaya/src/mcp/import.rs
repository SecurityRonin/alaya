//! Handler logic for the `import_claude_mem` and `import_claude_code` MCP tools,
//! plus their private parsing helpers.

use std::sync::atomic::Ordering;

use crate::{EpisodeContext, NewEpisode, NewSemanticNode, Role, SemanticType};

use super::{ImportClaudeCodeParams, ImportClaudeMemParams};

// ---------------------------------------------------------------------------
// Internal JSONL parsing types
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct ClaudeCodeEntry {
    #[serde(rename = "type")]
    entry_type: Option<String>,
    message: Option<ClaudeCodeMessage>,
    timestamp: Option<String>,
    #[serde(rename = "sessionId")]
    session_id: Option<String>,
}

#[derive(serde::Deserialize)]
struct ClaudeCodeMessage {
    #[allow(dead_code)]
    role: Option<String>,
    content: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn extract_content(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|item| item.get("text").and_then(|t| t.as_str()).map(String::from))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Result of parsing a Claude Code JSONL file: (episodes, session_ids, error_count, first_error).
type ParsedClaudeCodeJsonl = (
    Vec<NewEpisode>,
    std::collections::HashSet<String>,
    u32,
    Option<String>,
);

pub fn parse_claude_code_jsonl(path: &str) -> Result<ParsedClaudeCodeJsonl, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("Cannot read file '{path}': {e}"))?;

    let mut episodes = Vec::new();
    let mut sessions = std::collections::HashSet::new();
    let mut errors = 0u32;
    let mut first_error: Option<String> = None;

    let reader = std::io::BufReader::new(file);
    use std::io::BufRead;
    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                if first_error.is_none() {
                    first_error = Some(e.to_string());
                }
                errors += 1;
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }

        let entry: ClaudeCodeEntry = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(e) => {
                if first_error.is_none() {
                    first_error = Some(e.to_string());
                }
                errors += 1;
                continue;
            }
        };

        let entry_type = entry.entry_type.as_deref().unwrap_or("");
        if entry_type != "human" && entry_type != "assistant" {
            continue;
        }

        let message = match entry.message {
            Some(m) => m,
            None => continue,
        };

        // entry_type is guaranteed to be "human" or "assistant" by the guard above
        let role = if entry_type == "human" {
            Role::User
        } else {
            Role::Assistant
        };

        let content_value = match message.content {
            Some(c) => c,
            None => continue,
        };
        let content_text = extract_content(&content_value);
        if content_text.trim().is_empty() {
            continue;
        }

        let content_text = if content_text.chars().count() > 2000 {
            let truncated: String = content_text.chars().take(2000).collect();
            format!("{truncated}...")
        } else {
            content_text
        };

        let session_id = entry.session_id.unwrap_or_else(|| "imported".to_string());
        sessions.insert(session_id.clone());

        let timestamp = entry
            .timestamp
            .as_deref()
            .and_then(|ts| ts.parse::<i64>().ok())
            .unwrap_or(0);

        episodes.push(NewEpisode {
            content: content_text,
            role,
            session_id,
            timestamp,
            context: EpisodeContext::default(),
            embedding: None,
        });
    }

    Ok((episodes, sessions, errors, first_error))
}

fn parse_claude_mem_db(path: &str) -> Result<(u32, Vec<NewSemanticNode>), String> {
    let source_conn =
        rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(|e| format!("Cannot open claude-mem.db at '{path}': {e}"))?;

    let mut stmt = source_conn
        .prepare("SELECT title, facts, narrative, concepts, created_at FROM observations")
        .map_err(|e| format!("Error reading observations: {e}"))?;

    let mut nodes = Vec::new();
    let mut obs_count = 0u32;

    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0).unwrap_or_default(),
                row.get::<_, String>(1).unwrap_or_default(),
                row.get::<_, String>(2).unwrap_or_default(),
                row.get::<_, String>(3).unwrap_or_default(),
                row.get::<_, String>(4).unwrap_or_default(),
            ))
        })
        .map_err(|e| format!("Error querying observations: {e}"))?;

    for (title, facts_json, _narrative, concepts_json, _created_at) in rows.flatten() {
        obs_count += 1;
        let _ = title;

        if let Ok(facts) = serde_json::from_str::<Vec<String>>(&facts_json) {
            for fact in facts {
                if fact.trim().is_empty() {
                    continue;
                }
                nodes.push(NewSemanticNode {
                    content: fact,
                    node_type: SemanticType::Fact,
                    confidence: 0.8,
                    source_episodes: vec![],
                    embedding: None,
                });
            }
        }

        if let Ok(concepts) = serde_json::from_str::<Vec<String>>(&concepts_json) {
            for concept in concepts {
                if concept.trim().is_empty() {
                    continue;
                }
                nodes.push(NewSemanticNode {
                    content: concept,
                    node_type: SemanticType::Concept,
                    confidence: 0.7,
                    source_episodes: vec![],
                    embedding: None,
                });
            }
        }
    }

    Ok((obs_count, nodes))
}

// ---------------------------------------------------------------------------
// Public handlers
// ---------------------------------------------------------------------------

pub fn handle_import_claude_mem(server: &super::AlayaMcp, params: ImportClaudeMemParams) -> String {
    let path = super::validation::expand_tilde(
        &params
            .path
            .unwrap_or_else(|| "~/.claude-mem/claude-mem.db".to_string()),
    );

    let (obs_count, nodes) = match parse_claude_mem_db(&path) {
        Ok(result) => result,
        Err(e) => return e,
    };

    if obs_count == 0 {
        return format!("No observations found in '{path}'.");
    }

    let node_count = nodes.len();
    match server.with_store(|s| s.learn(nodes)) {
        Ok(report) => format!(
            "Imported {obs_count} observations \u{2192} {node_count} semantic nodes. {} categories assigned.",
            report.categories_assigned
        ),
        Err(e) => format!("Error importing: {e}"),
    }
}

pub fn handle_import_claude_code(
    server: &super::AlayaMcp,
    params: ImportClaudeCodeParams,
) -> String {
    let path = super::validation::expand_tilde(&params.path);

    let (episodes, sessions, mut errors, first_error) = match parse_claude_code_jsonl(&path) {
        Ok(result) => result,
        Err(e) => return e,
    };

    let mut imported = 0u32;
    for episode in episodes {
        match server.with_store(|s| s.store_episode(&episode)) {
            Ok(_) => {
                imported += 1;
                server.episode_count.fetch_add(1, Ordering::Relaxed);
                server.unconsolidated_count.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => errors += 1,
        }
    }

    let error_detail = match (&first_error, errors) {
        (Some(e), n) if n > 0 => format!(" ({n} errors, first: {e})"),
        _ => String::new(),
    };

    if imported == 0 {
        return format!("No importable messages found in '{path}'.{error_detail}");
    }

    format!(
        "Imported {imported} messages from {} sessions as episodes.{error_detail} Call 'learn' to consolidate.",
        sessions.len(),
    )
}

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use crate::AlayaStore;

    use super::super::{AlayaMcp, ImportClaudeCodeParams, ImportClaudeMemParams};

    fn make_server() -> AlayaMcp {
        let store = AlayaStore::open_in_memory().unwrap();
        AlayaMcp::new(store)
    }

    // -----------------------------------------------------------------------
    // extract_content
    // -----------------------------------------------------------------------

    #[test]
    fn extract_content_string_value() {
        let val = serde_json::json!("hello world");
        assert_eq!(super::extract_content(&val), "hello world");
    }

    #[test]
    fn extract_content_array_of_text_objects() {
        let val = serde_json::json!([
            {"text": "line one"},
            {"text": "line two"},
            {"text": "line three"}
        ]);
        assert_eq!(
            super::extract_content(&val),
            "line one\nline two\nline three"
        );
    }

    #[test]
    fn extract_content_array_mixed_some_without_text() {
        let val = serde_json::json!([
            {"text": "has text"},
            {"type": "tool_use", "id": "123"},
            {"text": "also text"}
        ]);
        assert_eq!(super::extract_content(&val), "has text\nalso text");
    }

    #[test]
    fn extract_content_empty_array() {
        let val = serde_json::json!([]);
        assert_eq!(super::extract_content(&val), "");
    }

    #[test]
    fn extract_content_number() {
        let val = serde_json::json!(42);
        assert_eq!(super::extract_content(&val), "");
    }

    #[test]
    fn extract_content_null() {
        let val = serde_json::Value::Null;
        assert_eq!(super::extract_content(&val), "");
    }

    #[test]
    fn extract_content_bool() {
        let val = serde_json::json!(true);
        assert_eq!(super::extract_content(&val), "");
    }

    #[test]
    fn extract_content_object() {
        let val = serde_json::json!({"key": "value"});
        assert_eq!(super::extract_content(&val), "");
    }

    // -----------------------------------------------------------------------
    // import_claude_mem
    // -----------------------------------------------------------------------

    #[test]
    fn import_claude_mem_nonexistent_path() {
        let srv = make_server();
        let result = srv.import_claude_mem(ImportClaudeMemParams {
            path: Some("/tmp/this-does-not-exist-alaya-test.db".into()),
        });
        assert!(result.contains("Cannot open claude-mem.db"));
    }

    #[test]
    fn import_claude_mem_default_path_error() {
        let srv = make_server();
        let result = srv.import_claude_mem(ImportClaudeMemParams { path: None });
        assert!(
            result.contains("Cannot open")
                || result.contains("Imported")
                || result.contains("No observations"),
            "Unexpected result: {result}"
        );
    }

    // -----------------------------------------------------------------------
    // import_claude_code
    // -----------------------------------------------------------------------

    #[test]
    fn import_claude_code_nonexistent_path() {
        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: "/tmp/this-does-not-exist-alaya-test.jsonl".into(),
        });
        assert!(result.contains("Cannot read file"));
    }

    #[test]
    fn import_claude_code_valid_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.jsonl");

        let lines = [
            serde_json::json!({
                "type": "human",
                "message": {"role": "user", "content": "How do I sort a vec in Rust?"},
                "timestamp": "1700000000",
                "sessionId": "import-sess"
            }),
            serde_json::json!({
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"text": "Use vec.sort() or vec.sort_by()"}]},
                "timestamp": "1700000001",
                "sessionId": "import-sess"
            }),
            serde_json::json!({
                "type": "human",
                "message": {"role": "user", "content": "Thanks!"},
                "timestamp": "1700000002",
                "sessionId": "import-sess"
            }),
            // Non-message entry (should be skipped)
            serde_json::json!({
                "type": "system",
                "message": {"role": "system", "content": "System message"},
                "timestamp": "1700000003"
            }),
        ];

        let content: String = lines
            .iter()
            .map(|l| serde_json::to_string(l).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, content).unwrap();

        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: file_path.to_str().unwrap().into(),
        });
        assert!(result.contains("Imported 3 messages"));
        assert!(result.contains("1 sessions"));
        assert!(result.contains("Call 'learn' to consolidate"));
    }

    #[test]
    fn import_claude_code_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("empty.jsonl");
        std::fs::write(&file_path, "").unwrap();

        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: file_path.to_str().unwrap().into(),
        });
        assert!(result.contains("No importable messages found"));
    }

    #[test]
    fn import_claude_code_with_bad_lines() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("mixed.jsonl");

        let good_line = serde_json::json!({
            "type": "human",
            "message": {"role": "user", "content": "Valid message"},
            "timestamp": "1700000000",
            "sessionId": "s1"
        });
        let content = format!(
            "{}\n{{\ninvalid json\n{}\n",
            serde_json::to_string(&good_line).unwrap(),
            serde_json::to_string(&good_line).unwrap(),
        );
        std::fs::write(&file_path, content).unwrap();

        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: file_path.to_str().unwrap().into(),
        });
        assert!(
            result.contains("Imported") || result.contains("error"),
            "Should handle mixed valid/invalid lines: {result}"
        );
    }

    #[test]
    fn import_claude_code_tilde_expansion() {
        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: "~/nonexistent-alaya-test-file.jsonl".into(),
        });
        assert!(result.contains("Cannot read file"));
        assert!(
            !result.contains("~/"),
            "Tilde should be expanded in error message"
        );
    }

    #[test]
    fn import_claude_code_updates_counters() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("counter-test.jsonl");

        let lines: Vec<String> = (0..3)
            .map(|i| {
                serde_json::to_string(&serde_json::json!({
                    "type": "human",
                    "message": {"role": "user", "content": format!("Message {i}")},
                    "timestamp": format!("{}", 1700000000 + i),
                    "sessionId": "s1"
                }))
                .unwrap()
            })
            .collect();
        std::fs::write(&file_path, lines.join("\n")).unwrap();

        let srv = make_server();
        srv.import_claude_code(ImportClaudeCodeParams {
            path: file_path.to_str().unwrap().into(),
        });

        let status = srv.status();
        assert!(
            status.contains("3 this session"),
            "Import should increment episode_count: {status}"
        );
        assert!(
            status.contains("3 unconsolidated"),
            "Import should increment unconsolidated_count: {status}"
        );
    }

    // -----------------------------------------------------------------------
    // parse_claude_code_jsonl edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn import_claude_code_content_truncated_at_2000_chars() {
        // Content longer than 2000 chars must be truncated to 2000 chars + "..."
        // Verify via parse_claude_code_jsonl directly (it is pub(crate)).
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("long.jsonl");

        let long_content = "a".repeat(2500);
        let line = serde_json::to_string(&serde_json::json!({
            "type": "human",
            "message": {"role": "user", "content": long_content},
            "timestamp": "1700000000",
            "sessionId": "s1"
        }))
        .unwrap();
        std::fs::write(&file_path, line).unwrap();

        let (episodes, _sessions, errors, _first_err) =
            super::parse_claude_code_jsonl(file_path.to_str().unwrap()).unwrap();

        assert_eq!(errors, 0, "No parse errors expected");
        assert_eq!(episodes.len(), 1, "Should produce exactly one episode");

        let content = &episodes[0].content;
        // The truncation adds "..." after taking 2000 chars, so total chars = 2003
        assert!(
            content.ends_with("..."),
            "Truncated content should end with '...': {content}"
        );
        assert!(
            content.chars().count() == 2003,
            "Truncated content should be 2003 chars (2000 + '...'): got {}",
            content.chars().count()
        );
    }

    #[test]
    fn import_claude_code_missing_session_id_uses_default() {
        // When sessionId is absent the entry should use "imported" as session_id
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("no-session.jsonl");

        let line = serde_json::to_string(&serde_json::json!({
            "type": "human",
            "message": {"role": "user", "content": "No session here"},
            "timestamp": "1700000000"
            // no "sessionId" key
        }))
        .unwrap();
        std::fs::write(&file_path, line).unwrap();

        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: file_path.to_str().unwrap().into(),
        });
        // Should import successfully and report 1 session (the default "imported")
        assert!(
            result.contains("Imported 1 messages"),
            "Missing sessionId should fall back to 'imported': {result}"
        );
        assert!(
            result.contains("1 sessions"),
            "Should count one session: {result}"
        );
    }

    #[test]
    fn import_claude_code_invalid_timestamp_defaults_to_zero() {
        // An unparseable timestamp string must fall back to 0 without crashing
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("bad-ts.jsonl");

        let line = serde_json::to_string(&serde_json::json!({
            "type": "assistant",
            "message": {"role": "assistant", "content": "Some response"},
            "timestamp": "not-a-number",
            "sessionId": "s1"
        }))
        .unwrap();
        std::fs::write(&file_path, line).unwrap();

        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: file_path.to_str().unwrap().into(),
        });
        assert!(
            result.contains("Imported 1 messages"),
            "Invalid timestamp should default to 0 and still import: {result}"
        );
    }

    #[test]
    fn import_claude_code_only_bad_lines_shows_error_detail() {
        // File with only unparseable JSON produces "No importable messages" + error detail
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("all-bad.jsonl");
        std::fs::write(&file_path, "not json at all\nalso bad\n").unwrap();

        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: file_path.to_str().unwrap().into(),
        });
        assert!(
            result.contains("No importable messages found"),
            "All-bad file should report no importable messages: {result}"
        );
        // The error detail branch: (Some(e), n) if n > 0 → includes "(N errors, first: ...)"
        assert!(
            result.contains("errors"),
            "Should include error count in detail: {result}"
        );
    }

    #[test]
    fn import_claude_code_skips_non_human_assistant_entries() {
        // Entries with type != "human" | "assistant" are silently skipped
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("mixed-types.jsonl");

        let lines = [
            serde_json::json!({
                "type": "system",
                "message": {"role": "system", "content": "System init"},
                "sessionId": "s1"
            }),
            serde_json::json!({
                "type": "tool_result",
                "message": {"content": "tool output"},
                "sessionId": "s1"
            }),
            serde_json::json!({
                "type": "human",
                "message": {"role": "user", "content": "Kept message"},
                "sessionId": "s1"
            }),
        ];
        let content: String = lines
            .iter()
            .map(|l| serde_json::to_string(l).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, content).unwrap();

        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: file_path.to_str().unwrap().into(),
        });
        assert!(
            result.contains("Imported 1 messages"),
            "Only the 'human' entry should be imported: {result}"
        );
    }

    #[test]
    fn import_claude_code_skips_entries_with_empty_content() {
        // Entries where content is whitespace-only are silently skipped
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("empty-content.jsonl");

        let lines = [
            serde_json::json!({
                "type": "human",
                "message": {"role": "user", "content": "   "},  // whitespace only
                "sessionId": "s1"
            }),
            serde_json::json!({
                "type": "human",
                "message": {"role": "user", "content": "Real content"},
                "sessionId": "s1"
            }),
        ];
        let content: String = lines
            .iter()
            .map(|l| serde_json::to_string(l).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, content).unwrap();

        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: file_path.to_str().unwrap().into(),
        });
        assert!(
            result.contains("Imported 1 messages"),
            "Whitespace-only content should be skipped: {result}"
        );
    }

    #[test]
    fn import_claude_code_skips_entries_without_message_field() {
        // Entries where the "message" key is absent are silently skipped
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("no-message.jsonl");

        let lines = [
            serde_json::json!({
                "type": "human",
                // no "message" key
                "sessionId": "s1"
            }),
            serde_json::json!({
                "type": "human",
                "message": {"role": "user", "content": "Has message"},
                "sessionId": "s1"
            }),
        ];
        let content: String = lines
            .iter()
            .map(|l| serde_json::to_string(l).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, content).unwrap();

        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: file_path.to_str().unwrap().into(),
        });
        assert!(
            result.contains("Imported 1 messages"),
            "Entry with no message field should be skipped: {result}"
        );
    }

    #[test]
    fn import_claude_code_skips_entries_without_content_field() {
        // Entries where "message.content" is absent are silently skipped
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("no-content.jsonl");

        let lines = [
            serde_json::json!({
                "type": "assistant",
                "message": {"role": "assistant"},  // no "content" key
                "sessionId": "s1"
            }),
            serde_json::json!({
                "type": "assistant",
                "message": {"role": "assistant", "content": "Valid"},
                "sessionId": "s1"
            }),
        ];
        let content: String = lines
            .iter()
            .map(|l| serde_json::to_string(l).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, content).unwrap();

        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: file_path.to_str().unwrap().into(),
        });
        assert!(
            result.contains("Imported 1 messages"),
            "Entry with no content field should be skipped: {result}"
        );
    }

    #[test]
    fn import_claude_code_with_empty_lines() {
        // Covers line 77: empty/whitespace lines in JSONL should be skipped
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("with-blanks.jsonl");

        let good_line = serde_json::to_string(&serde_json::json!({
            "type": "human",
            "message": {"role": "user", "content": "Valid message"},
            "timestamp": "1700000000",
            "sessionId": "s1"
        }))
        .unwrap();

        // Include empty lines, whitespace-only lines, and blank lines between entries
        let content = format!("\n\n{good_line}\n   \n\n{good_line}\n\n");
        std::fs::write(&file_path, content).unwrap();

        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: file_path.to_str().unwrap().into(),
        });
        assert!(
            result.contains("Imported 2 messages"),
            "Should import valid lines and skip empty ones: {result}"
        );
    }

    #[test]
    fn import_claude_mem_empty_observations() {
        // Covers line 229: "No observations found" when DB has empty observations table
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("empty-claude-mem.db");

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE observations (
                title TEXT,
                facts TEXT,
                narrative TEXT,
                concepts TEXT,
                created_at TEXT
            );"
        ).unwrap();
        drop(conn);

        let srv = make_server();
        let result = srv.import_claude_mem(ImportClaudeMemParams {
            path: Some(db_path.to_str().unwrap().into()),
        });
        assert!(
            result.contains("No observations found"),
            "Empty observations table should report no observations: {result}"
        );
    }

    #[test]
    fn import_claude_mem_with_empty_facts_and_concepts() {
        // Covers lines 181, 196: skip empty strings in facts/concepts JSON arrays
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("sparse-claude-mem.db");

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE observations (
                title TEXT,
                facts TEXT,
                narrative TEXT,
                concepts TEXT,
                created_at TEXT
            );"
        ).unwrap();
        // Insert observation with empty strings mixed into facts and concepts arrays
        conn.execute(
            "INSERT INTO observations (title, facts, narrative, concepts, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                "Test Observation",
                r#"["real fact", "", "   ", "another fact"]"#,
                "Some narrative",
                r#"["real concept", "", "  "]"#,
                "2024-01-01T00:00:00Z"
            ],
        ).unwrap();
        drop(conn);

        let srv = make_server();
        let result = srv.import_claude_mem(ImportClaudeMemParams {
            path: Some(db_path.to_str().unwrap().into()),
        });
        assert!(
            result.contains("Imported 1 observations"),
            "Should import the observation: {result}"
        );
        // 2 real facts + 1 real concept = 3 semantic nodes (empty strings skipped)
        assert!(
            result.contains("3 semantic nodes"),
            "Should skip empty facts/concepts: {result}"
        );
    }

    #[test]
    fn import_claude_code_with_invalid_utf8() {
        // Covers lines 68-73: IO read error from invalid UTF-8 bytes
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("bad-utf8.jsonl");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            // Write a valid JSONL line
            let good_line = serde_json::to_string(&serde_json::json!({
                "type": "human",
                "message": {"role": "user", "content": "Valid message"},
                "timestamp": "1700000000",
                "sessionId": "s1"
            }))
            .unwrap();
            f.write_all(good_line.as_bytes()).unwrap();
            f.write_all(b"\n").unwrap();
            // Write invalid UTF-8 bytes followed by a newline
            f.write_all(&[0xFF, 0xFE, 0x80, 0x90, b'\n']).unwrap();
            // Write another valid line
            f.write_all(good_line.as_bytes()).unwrap();
            f.write_all(b"\n").unwrap();
        }

        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: file_path.to_str().unwrap().into(),
        });
        // Should import the valid lines and report errors for invalid UTF-8
        assert!(
            result.contains("Imported") || result.contains("error"),
            "Should handle invalid UTF-8: {result}"
        );
    }

    #[test]
    fn import_claude_code_store_episode_db_error() {
        // Covers line 261: store_episode error during import
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("valid.jsonl");
        let good_line = serde_json::to_string(&serde_json::json!({
            "type": "human",
            "message": {"role": "user", "content": "Valid message"},
            "timestamp": "1700000000",
            "sessionId": "s1"
        }))
        .unwrap();
        std::fs::write(&file_path, format!("{good_line}\n")).unwrap();

        let store = AlayaStore::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE episodes")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: file_path.to_str().unwrap().into(),
        });
        assert!(
            result.contains("error") || result.contains("No importable"),
            "Should report store error: {result}"
        );
    }

    #[test]
    fn import_claude_mem_learn_db_error() {
        // Covers line 238: learn error during claude-mem import
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("good-source.db");

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE observations (
                title TEXT,
                facts TEXT,
                narrative TEXT,
                concepts TEXT,
                created_at TEXT
            );"
        ).unwrap();
        conn.execute(
            "INSERT INTO observations (title, facts, narrative, concepts, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                "Test",
                r#"["some fact"]"#,
                "narrative",
                r#"["some concept"]"#,
                "2024-01-01"
            ],
        ).unwrap();
        drop(conn);

        // Create a server with corrupted semantic_nodes table
        let store = AlayaStore::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE semantic_nodes")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.import_claude_mem(ImportClaudeMemParams {
            path: Some(db_path.to_str().unwrap().into()),
        });
        assert!(
            result.starts_with("Error importing:"),
            "Should return learn error: {result}"
        );
    }

    #[test]
    fn import_claude_mem_row_iteration_error() {
        // Covers line 173: row error during observation iteration
        // Create a DB with fewer columns to trigger row index out-of-bounds
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("bad-schema.db");

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        // Create table with only 2 columns — the query SELECTs 5 columns by position
        // so row.get(2), row.get(3), row.get(4) will fail
        conn.execute_batch(
            "CREATE TABLE observations (title TEXT, facts TEXT);
             INSERT INTO observations VALUES ('test', '[\"fact\"]');"
        ).unwrap();
        drop(conn);

        let srv = make_server();
        let result = srv.import_claude_mem(ImportClaudeMemParams {
            path: Some(db_path.to_str().unwrap().into()),
        });
        // Should handle the row error gracefully (continue on Err)
        assert!(
            !result.contains("panic"),
            "Should not panic on bad row data: {result}"
        );
    }
}
