//! MCP (Model Context Protocol) handler logic for Alaya.
//!
//! This module contains all the parameter types and the `AlayaMcp` server
//! struct with its tool handler methods. Handler logic is split into
//! domain-focused submodules:
//!
//! - `memory`      — `remember`, `recall`
//! - `lifecycle`   — `maintain`, `purge`, `reconcile_memories`, `list_conflicts`
//! - `preferences` — `learn`, `preferences`
//! - `query`       — `knowledge`, `categories`, `neighbors`, `node_category`
//! - `import`      — `import_claude_mem`, `import_claude_code`
//! - `status`      — `status`
//! - `validation`  — shared parameter validation helpers
//! - `serialization` — shared response formatting helpers
//!
//! The binary `src/bin/alaya-mcp.rs` is a thin wrapper that provides `main()`
//! and transport setup.

mod import;
mod lifecycle;
mod memory;
mod preferences;
mod query;
mod serialization;
mod status;
mod validation;

use std::sync::atomic::AtomicU32;
use std::sync::Mutex;

use crate::Alaya;
use rmcp::{model::ServerInfo, schemars, tool, ServerHandler};

// ---------------------------------------------------------------------------
// Parameter types (schemars::JsonSchema for MCP tool schemas)
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RememberParams {
    /// The message content to store
    #[schemars(description = "The message content to remember")]
    pub content: String,

    /// Role: "user", "assistant", or "system"
    #[schemars(description = "Who said it: user, assistant, or system")]
    pub role: String,

    /// Session identifier to group related messages
    #[schemars(description = "Session ID to group related messages")]
    pub session_id: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RecallParams {
    /// What to search for in memory
    #[schemars(description = "What to search for in memory")]
    pub query: String,

    /// Maximum number of results (default: 5)
    #[schemars(description = "Maximum results to return (default: 5)")]
    pub max_results: Option<usize>,

    /// Category ID to boost in results
    #[schemars(
        description = "Category ID to boost in ranking (memories in this category score higher)"
    )]
    pub boost_category: Option<i64>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct PreferencesParams {
    /// Optional domain filter (e.g. "style", "tone", "format")
    #[schemars(description = "Optional domain filter (e.g. style, tone, format)")]
    pub domain: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct KnowledgeParams {
    /// Filter by type: "fact", "relationship", "event", "concept"
    #[schemars(description = "Filter by type: fact, relationship, event, concept")]
    pub node_type: Option<String>,

    /// Minimum confidence threshold (0.0 to 1.0)
    #[schemars(description = "Minimum confidence threshold (0.0 to 1.0)")]
    pub min_confidence: Option<f32>,

    /// Maximum number of results
    #[schemars(description = "Maximum results to return (default: 20)")]
    pub limit: Option<usize>,

    /// Filter by category label
    #[schemars(description = "Filter by category label (exact match)")]
    pub category: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct PurgeParams {
    /// Purge scope: "session", "older_than", or "all"
    #[schemars(description = "Purge scope: session, older_than, or all")]
    pub scope: String,

    /// Session ID (required when scope is "session")
    #[schemars(description = "Session ID (required when scope is session)")]
    pub session_id: Option<String>,

    /// Unix timestamp (required when scope is "older_than")
    #[schemars(description = "Unix timestamp (required when scope is older_than)")]
    pub before_timestamp: Option<i64>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CategoriesParams {
    /// Minimum stability threshold (0.0 to 1.0)
    #[schemars(
        description = "Minimum stability threshold (0.0 to 1.0). Categories below this are filtered out."
    )]
    pub min_stability: Option<f32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct NeighborsParams {
    /// Node type: "episode", "semantic", "preference", "category"
    #[schemars(description = "Node type: episode, semantic, preference, or category")]
    pub node_type: String,

    /// Node ID
    #[schemars(description = "The numeric ID of the node")]
    pub node_id: i64,

    /// Traversal depth (default: 1)
    #[schemars(description = "How many hops to traverse (default: 1)")]
    pub depth: Option<u32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct NodeCategoryParams {
    /// Semantic node ID
    #[schemars(description = "The numeric ID of the semantic node")]
    pub node_id: i64,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct LearnFactEntry {
    /// The knowledge content
    #[schemars(description = "The knowledge content")]
    pub content: String,

    /// Type: fact, relationship, event, or concept
    #[schemars(description = "Type: fact, relationship, event, or concept")]
    pub node_type: String,

    /// Confidence level 0.0-1.0 (default: 0.8)
    #[schemars(description = "Confidence level 0.0-1.0 (default: 0.8)")]
    pub confidence: Option<f32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct LearnParams {
    /// Facts to learn
    #[schemars(description = "Facts to learn: [{content, node_type, confidence?}]")]
    pub facts: Vec<LearnFactEntry>,

    /// Session ID to link facts to (episodes in this session become source episodes)
    #[schemars(
        description = "Session ID to link facts to (episodes in this session become source episodes)"
    )]
    pub session_id: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ImportClaudeMemParams {
    /// Path to claude-mem.db (default: ~/.claude-mem/claude-mem.db)
    #[schemars(description = "Path to claude-mem.db (default: ~/.claude-mem/claude-mem.db)")]
    pub path: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ImportClaudeCodeParams {
    /// Path to Claude Code JSONL conversation file
    #[schemars(
        description = "Path to Claude Code JSONL conversation file (e.g., ~/.claude/projects/-Users-me-myproject/{uuid}.jsonl)"
    )]
    pub path: String,
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

pub struct AlayaMcp {
    store: Mutex<Alaya>,
    /// Total episodes stored this session.
    pub(crate) episode_count: AtomicU32,
    /// Episodes stored since last `learn` call.
    pub(crate) unconsolidated_count: AtomicU32,
}

#[cfg(not(tarpaulin_include))]
impl Clone for AlayaMcp {
    fn clone(&self) -> Self {
        // MCP servers are single-instance; clone should not be called in practice.
        // This satisfies the derive requirement from rmcp.
        panic!("AlayaMcp should not be cloned \u{2014} single-instance server")
    }
}

impl AlayaMcp {
    pub fn new(store: Alaya) -> Self {
        Self {
            store: Mutex::new(store),
            episode_count: AtomicU32::new(0),
            unconsolidated_count: AtomicU32::new(0),
        }
    }

    pub(crate) fn with_store<F, T>(&self, f: F) -> Result<T, String>
    where
        F: FnOnce(&Alaya) -> crate::Result<T>,
    {
        let store = self.store.lock().map_err(|e| format!("lock error: {e}"))?;
        f(&store).map_err(|e| format!("{e}"))
    }
}

// ---------------------------------------------------------------------------
// Tool handlers — thin wrappers delegating to domain modules
// ---------------------------------------------------------------------------

#[tool(tool_box)]
impl AlayaMcp {
    /// Store a conversation message in memory.
    #[tool(
        description = "Store a conversation message in Alaya's episodic memory. Call this for each message in the conversation that should be remembered."
    )]
    fn remember(&self, #[tool(aggr)] params: RememberParams) -> String {
        memory::handle_remember(self, params)
    }

    /// Search memory for relevant information.
    #[tool(
        description = "Search Alaya's memory using hybrid retrieval (BM25 + vector + graph + RRF fusion). Returns the most relevant memories matching the query."
    )]
    fn recall(&self, #[tool(aggr)] params: RecallParams) -> String {
        memory::handle_recall(self, params)
    }

    /// Get memory statistics.
    #[tool(
        description = "Get Alaya memory statistics: episode counts, knowledge breakdown by type, categories, preferences, graph links with strongest connection, and embedding coverage."
    )]
    fn status(&self) -> String {
        status::handle_status(self)
    }

    /// Get user preferences.
    #[tool(
        description = "Get crystallized user preferences learned from past interactions. Optionally filter by domain (e.g. 'style', 'tone', 'format')."
    )]
    fn preferences(&self, #[tool(aggr)] params: PreferencesParams) -> String {
        preferences::handle_preferences(self, params)
    }

    /// Get semantic knowledge.
    #[tool(
        description = "Get distilled semantic knowledge (facts, relationships, events, concepts) extracted from past conversations."
    )]
    fn knowledge(&self, #[tool(aggr)] params: KnowledgeParams) -> String {
        query::handle_knowledge(self, params)
    }

    /// Run memory maintenance (dedup, prune weak links, decay preferences).
    #[tool(
        description = "Run memory maintenance: deduplicates nodes, prunes weak links, decays stale preferences. Call periodically to keep memory healthy."
    )]
    fn maintain(&self) -> String {
        lifecycle::handle_maintain(self)
    }

    /// List emergent categories.
    #[tool(
        description = "List emergent categories discovered from semantic knowledge clusters. Categories form automatically and evolve through use."
    )]
    fn categories(&self, #[tool(aggr)] params: CategoriesParams) -> String {
        query::handle_categories(self, params)
    }

    /// Get graph neighbors of a node.
    #[tool(
        description = "Get graph neighbors of a memory node via spreading activation. Shows connected memories with link weights."
    )]
    fn neighbors(&self, #[tool(aggr)] params: NeighborsParams) -> String {
        query::handle_neighbors(self, params)
    }

    /// Get the category of a semantic node.
    #[tool(
        description = "Get which category a semantic knowledge node belongs to. Returns the category or 'uncategorized'."
    )]
    fn node_category(&self, #[tool(aggr)] params: NodeCategoryParams) -> String {
        query::handle_node_category(self, params)
    }

    /// Teach Alaya extracted knowledge directly.
    #[tool(
        description = "Teach Alaya extracted knowledge directly. The agent extracts facts from conversation and calls this tool. Each fact becomes a semantic node with full lifecycle wiring (strength, categories, graph links)."
    )]
    fn learn(&self, #[tool(aggr)] params: LearnParams) -> String {
        preferences::handle_learn(self, params)
    }

    /// Import memories from claude-mem SQLite database.
    #[tool(
        description = "Import memories from claude-mem (claude-mem.db SQLite database). Reads observations and converts facts/concepts into Alaya semantic nodes."
    )]
    fn import_claude_mem(&self, #[tool(aggr)] params: ImportClaudeMemParams) -> String {
        import::handle_import_claude_mem(self, params)
    }

    /// Import conversation history from Claude Code JSONL files.
    #[tool(
        description = "Import conversation history from Claude Code JSONL files. Reads messages and stores them as episodes."
    )]
    fn import_claude_code(&self, #[tool(aggr)] params: ImportClaudeCodeParams) -> String {
        import::handle_import_claude_code(self, params)
    }

    /// Purge memories by session, timestamp, or everything.
    #[tool(
        description = "Purge memories. Scope: 'session' (requires session_id), 'older_than' (requires before_timestamp), or 'all' (deletes everything)."
    )]
    fn purge(&self, #[tool(aggr)] params: PurgeParams) -> String {
        lifecycle::handle_purge(self, params)
    }

    /// Run conflict detection and resolution.
    #[tool(
        description = "Run conflict detection and resolution on semantic knowledge. Finds contradictory facts via embedding similarity, resolves using the configured strategy (recency by default), and archives superseded nodes."
    )]
    fn reconcile_memories(&self) -> String {
        lifecycle::handle_reconcile(self)
    }

    /// List unresolved conflicts.
    #[tool(
        description = "List unresolved conflicts between semantic knowledge nodes. Use after reconcile with manual strategy, or to review detected contradictions."
    )]
    fn list_conflicts(&self) -> String {
        lifecycle::handle_conflicts(self)
    }
}

#[tool(tool_box)]
impl ServerHandler for AlayaMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Alaya is a memory engine for AI agents. Use 'remember' to store messages, \
                 'recall' to search memory, 'learn' to teach extracted knowledge directly, \
                 'status' to check stats, 'preferences' for user preferences, 'knowledge' for \
                 semantic facts, 'categories' for emergent clusters, 'neighbors' for graph \
                 traversal, 'node_category' to check a node's category, 'maintain' for cleanup, \
                 'import_claude_mem' to import from claude-mem.db, \
                 'import_claude_code' to import from Claude Code JSONL files, \
                 'purge' to delete data, 'reconcile_memories' to detect and resolve \
                 contradictions, and 'list_conflicts' to review unresolved conflicts."
                    .into(),
            ),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Integration / cross-tool tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use super::*;

    fn make_server() -> AlayaMcp {
        let store = Alaya::open_in_memory().unwrap();
        AlayaMcp::new(store)
    }

    #[test]
    fn get_info_returns_instructions() {
        use rmcp::ServerHandler;
        let srv = make_server();
        let info = srv.get_info();
        let instructions = info.instructions.expect("should have instructions");
        assert!(instructions.contains("Alaya is a memory engine"));
        assert!(instructions.contains("remember"));
        assert!(instructions.contains("recall"));
        assert!(instructions.contains("learn"));
    }

    #[test]
    fn full_lifecycle_remember_learn_recall() {
        let srv = make_server();

        // 1. Store episodes
        srv.remember(RememberParams {
            content: "The capital of France is Paris".into(),
            role: "user".into(),
            session_id: "geo".into(),
        });
        srv.remember(RememberParams {
            content: "Paris has the Eiffel Tower".into(),
            role: "assistant".into(),
            session_id: "geo".into(),
        });

        // 2. Extract knowledge
        let learn_result = srv.learn(LearnParams {
            facts: vec![
                LearnFactEntry {
                    content: "France capital is Paris".into(),
                    node_type: "fact".into(),
                    confidence: Some(0.95),
                },
                LearnFactEntry {
                    content: "Paris has Eiffel Tower".into(),
                    node_type: "fact".into(),
                    confidence: Some(0.9),
                },
            ],
            session_id: Some("geo".into()),
        });
        assert!(learn_result.contains("Learned 2 facts:"));

        // 3. Recall finds the knowledge
        let recall_result = srv.recall(RecallParams {
            query: "Paris France".into(),
            max_results: Some(5),
            boost_category: None,
        });
        assert!(recall_result.contains("Found"));

        // 4. Status reflects stored data
        let status = srv.status();
        assert!(status.contains("Episodes: 2"));
    }
}
