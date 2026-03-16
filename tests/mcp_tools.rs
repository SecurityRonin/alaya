//! Integration tests for MCP tool functions.
//!
//! These test the AlayaMcp tool methods directly (without MCP transport)
//! using an in-memory store.

#![cfg(feature = "mcp")]

// We can't directly import AlayaMcp from the binary,
// so we test the underlying AlayaStore operations that the MCP tools wrap.
// This validates the data flow that the MCP tools rely on.

use std::collections::HashSet;

use alaya::{
    AlayaStore, EpisodeContext, EpisodeId, KnowledgeFilter, NewEpisode, NewSemanticNode, NodeRef,
    PurgeFilter, Query, Role, SemanticType,
};

fn make_episode(content: &str, role: Role, session: &str, ts: i64) -> NewEpisode {
    NewEpisode {
        content: content.to_string(),
        role,
        session_id: session.to_string(),
        timestamp: ts,
        context: EpisodeContext::default(),
        embedding: None,
    }
}

#[test]
fn test_mcp_remember_and_recall_flow() {
    let store = AlayaStore::open_in_memory().unwrap();

    // Simulate MCP "remember" tool
    let id = store
        .store_episode(&make_episode(
            "I love hiking in the mountains",
            Role::User,
            "session-1",
            1700000000,
        ))
        .unwrap();
    assert!(id.0 > 0);

    store
        .store_episode(&make_episode(
            "That sounds fun! Do you have a favorite trail?",
            Role::Assistant,
            "session-1",
            1700000001,
        ))
        .unwrap();

    store
        .store_episode(&make_episode(
            "Yes, I love the Appalachian Trail",
            Role::User,
            "session-1",
            1700000002,
        ))
        .unwrap();

    // Simulate MCP "recall" tool
    let results = store.query(&Query::simple("hiking")).unwrap();
    assert!(!results.is_empty(), "recall should find hiking memories");

    // Also search for "Appalachian"
    let results2 = store.query(&Query::simple("Appalachian")).unwrap();
    assert!(
        !results2.is_empty(),
        "recall should find Appalachian Trail memory"
    );
}

#[test]
fn test_mcp_status_flow() {
    let store = AlayaStore::open_in_memory().unwrap();

    // Empty status
    let status = store.status().unwrap();
    assert_eq!(status.episode_count, 0);
    assert_eq!(status.semantic_node_count, 0);
    assert_eq!(status.preference_count, 0);

    // Store episodes
    for i in 0..3 {
        store
            .store_episode(&make_episode(
                &format!("message {i}"),
                Role::User,
                "s1",
                1000 + i,
            ))
            .unwrap();
    }

    let status = store.status().unwrap();
    assert_eq!(status.episode_count, 3);
}

#[test]
fn test_mcp_preferences_flow() {
    let store = AlayaStore::open_in_memory().unwrap();

    // No preferences initially
    let prefs = store.preferences(None).unwrap();
    assert!(prefs.is_empty());

    // With domain filter
    let prefs = store.preferences(Some("style")).unwrap();
    assert!(prefs.is_empty());
}

#[test]
fn test_mcp_knowledge_flow() {
    let store = AlayaStore::open_in_memory().unwrap();

    // No knowledge initially
    let nodes = store.knowledge(None).unwrap();
    assert!(nodes.is_empty());

    // With type filter
    let nodes = store
        .knowledge(Some(KnowledgeFilter {
            node_type: Some(SemanticType::Fact),
            ..Default::default()
        }))
        .unwrap();
    assert!(nodes.is_empty());
}

#[test]
fn test_mcp_purge_session_flow() {
    let store = AlayaStore::open_in_memory().unwrap();

    // Store in two sessions
    store
        .store_episode(&make_episode("msg in s1", Role::User, "s1", 1000))
        .unwrap();
    store
        .store_episode(&make_episode("msg in s2", Role::User, "s2", 2000))
        .unwrap();

    assert_eq!(store.status().unwrap().episode_count, 2);

    // Purge session s1
    let report = store.purge(PurgeFilter::Session("s1".to_string())).unwrap();
    assert_eq!(report.episodes_deleted, 1);
    assert_eq!(store.status().unwrap().episode_count, 1);
}

#[test]
fn test_mcp_purge_all_flow() {
    let store = AlayaStore::open_in_memory().unwrap();

    store
        .store_episode(&make_episode("msg1", Role::User, "s1", 1000))
        .unwrap();
    store
        .store_episode(&make_episode("msg2", Role::User, "s1", 2000))
        .unwrap();

    store.purge(PurgeFilter::All).unwrap();
    assert_eq!(store.status().unwrap().episode_count, 0);
}

#[test]
fn test_mcp_maintain_flow() {
    let store = AlayaStore::open_in_memory().unwrap();

    // transform + forget on empty store should succeed
    let tr = store.transform().unwrap();
    assert_eq!(tr.duplicates_merged, 0);

    let fr = store.forget().unwrap();
    assert_eq!(fr.nodes_decayed, 0);
}

#[test]
fn test_mcp_recall_max_results() {
    let store = AlayaStore::open_in_memory().unwrap();

    for i in 0..10 {
        store
            .store_episode(&make_episode(
                &format!("Rust programming tip number {i}"),
                Role::User,
                "s1",
                1000 + i,
            ))
            .unwrap();
    }

    // Limit to 3 results
    let query = Query {
        text: "Rust programming".to_string(),
        embedding: None,
        context: alaya::QueryContext::default(),
        max_results: 3,
        boost_categories: None,
    };
    let results = store.query(&query).unwrap();
    assert!(results.len() <= 3, "should respect max_results limit");
}

#[test]
fn test_mcp_role_parsing() {
    let store = AlayaStore::open_in_memory().unwrap();

    // All three roles should work
    for (role, role_str) in [
        (Role::User, "user"),
        (Role::Assistant, "assistant"),
        (Role::System, "system"),
    ] {
        let id = store
            .store_episode(&make_episode("test", role, "s1", 1000))
            .unwrap();
        assert!(id.0 > 0, "role '{role_str}' should be accepted");
    }
}

#[test]
fn test_mcp_learn_creates_knowledge() {
    let store = AlayaStore::open_in_memory().unwrap();

    // Store 3 facts via learn()
    let nodes = vec![
        NewSemanticNode {
            content: "User prefers Rust".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.9,
            source_episodes: vec![],
            embedding: None,
        },
        NewSemanticNode {
            content: "User knows async programming".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.8,
            source_episodes: vec![],
            embedding: None,
        },
        NewSemanticNode {
            content: "Rust and async are related".to_string(),
            node_type: SemanticType::Relationship,
            confidence: 0.7,
            source_episodes: vec![],
            embedding: None,
        },
    ];
    let report = store.learn(nodes).unwrap();
    assert_eq!(report.nodes_created, 3);

    let knowledge = store.knowledge(None).unwrap();
    assert_eq!(knowledge.len(), 3);
}

#[test]
fn test_mcp_learn_with_session_links() {
    let store = AlayaStore::open_in_memory().unwrap();

    // Store episodes first
    let ep1 = store
        .store_episode(&make_episode("msg1", Role::User, "s1", 1000))
        .unwrap();
    let ep2 = store
        .store_episode(&make_episode("msg2", Role::User, "s1", 2000))
        .unwrap();

    // Learn with those episodes as sources
    let nodes = vec![NewSemanticNode {
        content: "User discussed topic X".to_string(),
        node_type: SemanticType::Fact,
        confidence: 0.9,
        source_episodes: vec![ep1, ep2],
        embedding: None,
    }];
    let report = store.learn(nodes).unwrap();
    assert_eq!(report.nodes_created, 1);
    assert_eq!(report.links_created, 2); // 2 Causal links

    // Verify episodes are now consolidated
    let unconsolidated = store.unconsolidated_episodes(100).unwrap();
    assert!(unconsolidated.is_empty());
}

#[test]
fn test_mcp_episodes_by_session() {
    let store = AlayaStore::open_in_memory().unwrap();

    // Store episodes in two sessions
    store
        .store_episode(&make_episode("msg1", Role::User, "s1", 1000))
        .unwrap();
    store
        .store_episode(&make_episode("msg2", Role::Assistant, "s1", 2000))
        .unwrap();
    store
        .store_episode(&make_episode("msg3", Role::User, "s2", 3000))
        .unwrap();

    // Query session s1
    let eps = store.episodes_by_session("s1").unwrap();
    assert_eq!(eps.len(), 2);
    assert_eq!(eps[0].content, "msg1");
    assert_eq!(eps[1].content, "msg2");

    // Query session s2
    let eps = store.episodes_by_session("s2").unwrap();
    assert_eq!(eps.len(), 1);

    // Query non-existent session
    let eps = store.episodes_by_session("s999").unwrap();
    assert!(eps.is_empty());
}

#[test]
fn test_unconsolidated_episodes_count() {
    let store = AlayaStore::open_in_memory().unwrap();
    // Store 10 episodes
    for i in 0..10 {
        store
            .store_episode(&make_episode(
                &format!("msg {i}"),
                Role::User,
                "s1",
                1000 + i,
            ))
            .unwrap();
    }
    // All 10 should be unconsolidated
    let uncons = store.unconsolidated_episodes(20).unwrap();
    assert_eq!(uncons.len(), 10);

    // Learn a fact linking to first 3 episodes
    let nodes = vec![NewSemanticNode {
        content: "test fact".to_string(),
        node_type: SemanticType::Fact,
        confidence: 0.8,
        source_episodes: vec![EpisodeId(1), EpisodeId(2), EpisodeId(3)],
        embedding: None,
    }];
    store.learn(nodes).unwrap();

    // Now only 7 should be unconsolidated
    let uncons = store.unconsolidated_episodes(20).unwrap();
    assert_eq!(uncons.len(), 7);
}

#[test]
fn test_transform_and_forget_on_empty() {
    // Verify transform+forget work without panicking (used by auto-maintenance)
    let store = AlayaStore::open_in_memory().unwrap();
    let tr = store.transform().unwrap();
    let fr = store.forget().unwrap();
    assert_eq!(tr.duplicates_merged, 0);
    assert_eq!(fr.nodes_decayed, 0);
}

#[test]
fn test_mcp_rich_status_fields() {
    let store = AlayaStore::open_in_memory().unwrap();

    // Empty breakdown
    let breakdown = store.knowledge_breakdown().unwrap();
    assert!(breakdown.is_empty());

    // Empty strongest link
    assert!(store.strongest_link().unwrap().is_none());

    // Add an episode
    store
        .store_episode(&make_episode("test content", Role::User, "s1", 1000))
        .unwrap();

    // Learn a fact and a relationship
    let nodes = vec![
        NewSemanticNode {
            content: "User likes Rust".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.8,
            source_episodes: vec![EpisodeId(1)],
            embedding: None,
        },
        NewSemanticNode {
            content: "Rust relates to async".to_string(),
            node_type: SemanticType::Relationship,
            confidence: 0.7,
            source_episodes: vec![EpisodeId(1)],
            embedding: None,
        },
    ];
    store.learn(nodes).unwrap();

    // Knowledge breakdown should reflect 1 fact, 1 relationship
    let breakdown = store.knowledge_breakdown().unwrap();
    assert_eq!(breakdown.get(&SemanticType::Fact), Some(&1));
    assert_eq!(breakdown.get(&SemanticType::Relationship), Some(&1));
    assert_eq!(breakdown.get(&SemanticType::Event), None);
    assert_eq!(breakdown.get(&SemanticType::Concept), None);

    // Strongest link should now exist (learn creates Causal links)
    let strongest = store.strongest_link().unwrap();
    assert!(strongest.is_some(), "learn should have created links");

    // node_content should resolve episode content
    let label = store.node_content(NodeRef::Episode(EpisodeId(1))).unwrap();
    assert_eq!(label, Some("test content".to_string()));

    // node_content for missing node
    let missing = store
        .node_content(NodeRef::Episode(EpisodeId(999)))
        .unwrap();
    assert!(missing.is_none());
}

#[test]
fn test_import_claude_mem_data_flow() {
    // Create a temp SQLite db with claude-mem schema
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("claude-mem.db");
    let source_conn = rusqlite::Connection::open(&db_path).unwrap();
    source_conn
        .execute_batch(
            "CREATE TABLE observations (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                facts TEXT NOT NULL DEFAULT '[]',
                narrative TEXT NOT NULL DEFAULT '',
                concepts TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT ''
            );",
        )
        .unwrap();

    // Insert test observations
    source_conn
        .execute(
            "INSERT INTO observations (title, facts, concepts, created_at) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![
                "Rust Programming",
                r#"["User prefers Rust", "User likes type safety"]"#,
                r#"["systems programming", "memory safety"]"#,
                "2024-01-01T00:00:00Z"
            ],
        )
        .unwrap();
    source_conn
        .execute(
            "INSERT INTO observations (title, facts, concepts, created_at) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![
                "Cooking",
                r#"["User enjoys Italian cuisine"]"#,
                r#"["cooking"]"#,
                "2024-01-02T00:00:00Z"
            ],
        )
        .unwrap();
    drop(source_conn);

    // Now simulate what the MCP tool does: read from source, create nodes, learn
    let source_conn =
        rusqlite::Connection::open_with_flags(&db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .unwrap();

    let mut stmt = source_conn
        .prepare("SELECT title, facts, narrative, concepts, created_at FROM observations")
        .unwrap();

    let mut nodes = Vec::new();
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0).unwrap_or_default(),
                row.get::<_, String>(1).unwrap_or_default(),
                row.get::<_, String>(2).unwrap_or_default(),
                row.get::<_, String>(3).unwrap_or_default(),
            ))
        })
        .unwrap();

    for row in rows {
        let (_title, facts_json, _narrative, concepts_json) = row.unwrap();
        if let Ok(facts) = serde_json::from_str::<Vec<String>>(&facts_json) {
            for fact in facts {
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
    drop(stmt);
    drop(source_conn);

    assert_eq!(nodes.len(), 6); // 3 facts + 3 concepts

    let store = AlayaStore::open_in_memory().unwrap();
    let report = store.learn(nodes).unwrap();
    assert_eq!(report.nodes_created, 6);

    let knowledge = store.knowledge(None).unwrap();
    assert_eq!(knowledge.len(), 6);

    // Verify types
    let facts: Vec<_> = knowledge
        .iter()
        .filter(|n| n.node_type == SemanticType::Fact)
        .collect();
    let concepts: Vec<_> = knowledge
        .iter()
        .filter(|n| n.node_type == SemanticType::Concept)
        .collect();
    assert_eq!(facts.len(), 3);
    assert_eq!(concepts.len(), 3);
}

#[test]
fn test_import_claude_code_data_flow() {
    // Create a temp JSONL file
    let dir = tempfile::tempdir().unwrap();
    let jsonl_path = dir.path().join("conversation.jsonl");
    let content = r#"{"type":"human","message":{"role":"user","content":"Hello, how are you?"},"timestamp":"1700000000","sessionId":"session-1"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"I'm doing well, thanks!"}]},"timestamp":"1700000001","sessionId":"session-1"}
{"type":"human","message":{"role":"user","content":"Tell me about Rust"},"timestamp":"1700000002","sessionId":"session-2"}
{"type":"summary","message":{"role":"assistant","content":"Summary of conversation"},"timestamp":"1700000003","sessionId":"session-1"}
"#;
    std::fs::write(&jsonl_path, content).unwrap();

    // Simulate the import data flow
    let file_content = std::fs::read_to_string(&jsonl_path).unwrap();
    let store = AlayaStore::open_in_memory().unwrap();

    let mut imported = 0u32;
    let mut sessions = HashSet::new();

    for line in file_content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line).unwrap();

        let entry_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if entry_type != "human" && entry_type != "assistant" {
            continue;
        }

        let message = value.get("message").unwrap();
        let role = match entry_type {
            "human" => Role::User,
            "assistant" => Role::Assistant,
            _ => continue,
        };

        let content_val = message.get("content").unwrap();
        let content_text = match content_val {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|item| item.get("text").and_then(|t| t.as_str()).map(String::from))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => continue,
        };

        let session_id = value
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("imported")
            .to_string();
        sessions.insert(session_id.clone());

        let timestamp = value
            .get("timestamp")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0);

        store
            .store_episode(&NewEpisode {
                content: content_text,
                role,
                session_id,
                timestamp,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();
        imported += 1;
    }

    assert_eq!(imported, 3); // human, assistant, human (summary is skipped)
    assert_eq!(sessions.len(), 2); // session-1, session-2
    assert_eq!(store.status().unwrap().episode_count, 3);

    // Verify content was parsed correctly
    let results = store.query(&Query::simple("Rust")).unwrap();
    assert!(!results.is_empty());

    let results = store.query(&Query::simple("Hello")).unwrap();
    assert!(!results.is_empty());
}
