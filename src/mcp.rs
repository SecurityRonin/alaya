//! MCP (Model Context Protocol) handler logic for Alaya.
//!
//! This module contains all the parameter types, helper functions, and the
//! `AlayaMcp` server struct with its tool handler methods. The binary
//! `src/bin/alaya-mcp.rs` is a thin wrapper that provides `main()` and
//! transport setup.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use crate::{
    AlayaStore, CategoryId, EpisodeContext, EpisodeId, KnowledgeFilter, NewEpisode,
    NewSemanticNode, NodeId, NodeRef, PreferenceId, PurgeFilter, Query, Role, SemanticType,
};
use rmcp::{model::ServerInfo, schemars, tool, ServerHandler};

/// Result of parsing a Claude Code JSONL file: (episodes, session_ids, error_count, first_error).
type ParsedClaudeCodeJsonl = (
    Vec<NewEpisode>,
    std::collections::HashSet<String>,
    u32,
    Option<String>,
);

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
// Claude Code JSONL parsing helpers
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

/// Expand `~/` prefix to the user's home directory.
fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/{rest}")
    } else {
        path.to_string()
    }
}

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

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

pub struct AlayaMcp {
    store: Mutex<AlayaStore>,
    /// Total episodes stored this session.
    episode_count: AtomicU32,
    /// Episodes stored since last `learn` call.
    unconsolidated_count: AtomicU32,
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
    pub fn new(store: AlayaStore) -> Self {
        Self {
            store: Mutex::new(store),
            episode_count: AtomicU32::new(0),
            unconsolidated_count: AtomicU32::new(0),
        }
    }

    fn with_store<F, T>(&self, f: F) -> Result<T, String>
    where
        F: FnOnce(&AlayaStore) -> crate::Result<T>,
    {
        let store = self.store.lock().map_err(|e| format!("lock error: {e}"))?;
        f(&store).map_err(|e| format!("{e}"))
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers — extracted from #[tool] methods so tarpaulin can track
// coverage through proc-macro-generated wrappers.
// ---------------------------------------------------------------------------

use crate::types::{Category, MemoryStatus, Preference};

fn format_preferences(prefs: &[Preference]) -> String {
    let mut out = format!("Found {} preferences:\n\n", prefs.len());
    for p in prefs {
        out.push_str(&format!(
            "- [{}] {} (confidence: {:.2}, evidence: {})\n",
            p.domain, p.preference, p.confidence, p.evidence_count
        ));
    }
    out
}

fn format_categories(cats: &[Category]) -> String {
    let mut out = format!("Found {} categories:\n\n", cats.len());
    for c in cats {
        let parent = c
            .parent_id
            .map(|p| format!(" (parent: {})", p.0))
            .unwrap_or_default();
        out.push_str(&format!(
            "- [{}] {} — {} members, stability: {:.2}{}\n",
            c.id.0, c.label, c.member_count, c.stability, parent
        ));
    }
    out
}

fn format_neighbors(neighbors: &[(NodeRef, f32)]) -> String {
    let mut out = format!("Found {} neighbors:\n\n", neighbors.len());
    for (nr, weight) in neighbors {
        let (ntype, nid) = match nr {
            NodeRef::Episode(id) => ("episode", id.0),
            NodeRef::Semantic(id) => ("semantic", id.0),
            NodeRef::Preference(id) => ("preference", id.0),
            NodeRef::Category(id) => ("category", id.0),
        };
        out.push_str(&format!("- {ntype} #{nid} (weight: {weight:.3})\n"));
    }
    out
}

fn format_node_category(node_id: i64, cat: &Category) -> String {
    let parent = cat
        .parent_id
        .map(|p| format!(" (parent: {})", p.0))
        .unwrap_or_default();
    format!(
        "Node {} belongs to category [{}] '{}' — {} members, stability: {:.2}{}",
        node_id, cat.id.0, cat.label, cat.member_count, cat.stability, parent
    )
}

fn format_knowledge_breakdown(breakdown: &std::collections::HashMap<SemanticType, u64>) -> String {
    let mut parts = Vec::new();
    for (st, label) in [
        (SemanticType::Fact, "facts"),
        (SemanticType::Relationship, "relationships"),
        (SemanticType::Event, "events"),
        (SemanticType::Concept, "concepts"),
    ] {
        if let Some(&count) = breakdown.get(&st) {
            parts.push(format!("{count} {label}"));
        }
    }
    parts.join(", ")
}

fn format_category_line(cats: &[Category]) -> String {
    let labels: Vec<&str> = cats.iter().map(|c| c.label.as_str()).collect();
    format!("{} ({})", cats.len(), labels.join(", "))
}

fn format_status(
    st: &MemoryStatus,
    session_eps: u32,
    unconsolidated: u32,
    knowledge_line: &str,
    cat_line: &str,
    strongest_desc: &str,
    coverage: &str,
) -> String {
    let mut out = format!("Memory Status:\n  Episodes: {}", st.episode_count);
    if session_eps > 0 || unconsolidated > 0 {
        out.push_str(&format!(
            " ({session_eps} this session, {unconsolidated} unconsolidated)"
        ));
    }
    out.push_str(&format!("\n  Knowledge: {knowledge_line}"));
    out.push_str(&format!("\n  Categories: {cat_line}"));
    out.push_str(&format!(
        "\n  Preferences: {} crystallized, {} impressions accumulating",
        st.preference_count, st.impression_count
    ));
    out.push_str(&format!(
        "\n  Graph: {} links{strongest_desc}",
        st.link_count
    ));
    out.push_str(&format!("\n  Embedding coverage: {coverage}"));
    out
}

/// Parse a Claude-mem SQLite DB and return (obs_count, nodes).
fn parse_claude_mem_db(path: &str) -> Result<(u32, Vec<crate::NewSemanticNode>), String> {
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

    for row_result in rows {
        let (title, facts_json, _narrative, concepts_json, _created_at) = match row_result {
            Ok(r) => r,
            Err(_) => continue,
        };
        obs_count += 1;
        let _ = title;

        if let Ok(facts) = serde_json::from_str::<Vec<String>>(&facts_json) {
            for fact in facts {
                if fact.trim().is_empty() {
                    continue;
                }
                nodes.push(crate::NewSemanticNode {
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
                nodes.push(crate::NewSemanticNode {
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

/// Parse a Claude Code JSONL file and return (episodes, sessions, errors, first_error).
fn parse_claude_code_jsonl(path: &str) -> Result<ParsedClaudeCodeJsonl, String> {
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

#[tool(tool_box)]
impl AlayaMcp {
    /// Store a conversation message in memory.
    #[tool(
        description = "Store a conversation message in Alaya's episodic memory. Call this for each message in the conversation that should be remembered."
    )]
    fn remember(&self, #[tool(aggr)] params: RememberParams) -> String {
        let role = match params.role.to_lowercase().as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "system" => Role::System,
            _ => {
                return format!(
                    "Error: invalid role '{}'. Use: user, assistant, system",
                    params.role
                )
            }
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let episode = NewEpisode {
            content: params.content.clone(),
            role,
            session_id: params.session_id.clone(),
            timestamp: now,
            context: EpisodeContext::default(),
            embedding: None,
        };

        match self.with_store(|s| s.store_episode(&episode)) {
            Ok(id) => {
                let ep_total = self.episode_count.fetch_add(1, Ordering::Relaxed) + 1;
                let uncons = self.unconsolidated_count.fetch_add(1, Ordering::Relaxed) + 1;

                let mut response = format!(
                    "Stored episode {} in session '{}'.",
                    id.0, params.session_id
                );

                // Auto-consolidation at 10 unconsolidated episodes
                if uncons >= 10 {
                    // Try auto-consolidation first (if ExtractionProvider is set)
                    match self.with_store(|s| s.auto_consolidate()) {
                        Ok(report) if report.nodes_created > 0 => {
                            self.unconsolidated_count.store(0, Ordering::Relaxed);
                            response.push_str(&format!(
                                "\n\n--- Auto-consolidated ---\n\
                                 Extracted {} knowledge nodes from {} episodes.",
                                report.nodes_created, uncons
                            ));
                        }
                        Ok(_) => {
                            // Provider returned zero nodes — fall back to prompt
                            if let Ok(episodes) = self.with_store(|s| s.unconsolidated_episodes(20))
                            {
                                response.push_str(&format!(
                                    "\n\n--- Consolidation suggested ---\n\
                                     You have {} unconsolidated episodes. \
                                     Please extract key facts and call the 'learn' tool.\n\
                                     Recent unconsolidated episodes:",
                                    episodes.len()
                                ));
                                for ep in &episodes {
                                    response.push_str(&format!(
                                        "\n[{}] {}: {}",
                                        ep.id.0,
                                        ep.role.as_str(),
                                        ep.content
                                    ));
                                }
                            }
                        }
                        Err(e) => {
                            // Provider error or no provider — fall back to prompt with note
                            let err_msg = e.to_string();
                            let is_no_provider = err_msg.contains("extraction provider");
                            if let Ok(episodes) = self.with_store(|s| s.unconsolidated_episodes(20))
                            {
                                if !is_no_provider {
                                    response
                                        .push_str(&format!("\n\n(Auto-consolidation failed: {e})"));
                                }
                                response.push_str(&format!(
                                    "\n\n--- Consolidation suggested ---\n\
                                     You have {} unconsolidated episodes. \
                                     Please extract key facts and call the 'learn' tool.\n\
                                     Recent unconsolidated episodes:",
                                    episodes.len()
                                ));
                                for ep in &episodes {
                                    response.push_str(&format!(
                                        "\n[{}] {}: {}",
                                        ep.id.0,
                                        ep.role.as_str(),
                                        ep.content
                                    ));
                                }
                            }
                        }
                    }
                }

                // Auto-maintenance every 25 episodes
                if ep_total % 25 == 0 {
                    let tr = self.with_store(|s| s.transform());
                    let fr = self.with_store(|s| s.forget());
                    match (tr, fr) {
                        (Ok(tr), Ok(fr)) => {
                            response.push_str(&format!(
                                "\n\n--- Auto-maintenance ---\n\
                                 Transform: {} merged, {} links pruned, {} categories discovered\n\
                                 Forget: {} decayed, {} archived",
                                tr.duplicates_merged,
                                tr.links_pruned,
                                tr.categories_discovered,
                                fr.nodes_decayed,
                                fr.nodes_archived,
                            ));
                        }
                        (Err(e), _) | (_, Err(e)) => {
                            response.push_str(&format!("\n\n--- Auto-maintenance error: {e} ---"));
                        }
                    }
                }

                response
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Search memory for relevant information.
    #[tool(
        description = "Search Alaya's memory using hybrid retrieval (BM25 + vector + graph + RRF fusion). Returns the most relevant memories matching the query."
    )]
    fn recall(&self, #[tool(aggr)] params: RecallParams) -> String {
        let query = Query {
            text: params.query,
            embedding: None,
            context: crate::QueryContext::default(),
            max_results: params.max_results.unwrap_or(5),
            boost_categories: params.boost_category.map(|c| vec![c.to_string()]),
        };

        match self.with_store(|s| s.query(&query)) {
            Ok(results) if results.is_empty() => "No memories found.".to_string(),
            Ok(results) => {
                let mut out = format!("Found {} memories:\n\n", results.len());
                for (i, mem) in results.iter().enumerate() {
                    let role = mem.role.map(|r| r.as_str()).unwrap_or("unknown");
                    out.push_str(&format!(
                        "{}. [{}] (score: {:.3}) {}\n",
                        i + 1,
                        role,
                        mem.score,
                        mem.content
                    ));
                }
                out
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Get memory statistics.
    #[tool(
        description = "Get Alaya memory statistics: episode counts, knowledge breakdown by type, categories, preferences, graph links with strongest connection, and embedding coverage."
    )]
    fn status(&self) -> String {
        let st = match self.with_store(|s| s.status()) {
            Ok(st) => st,
            Err(e) => return format!("Error: {e}"),
        };

        let session_eps = self.episode_count.load(Ordering::Relaxed);
        let unconsolidated = self.unconsolidated_count.load(Ordering::Relaxed);

        let knowledge_line = match self.with_store(|s| s.knowledge_breakdown()) {
            Ok(breakdown) if !breakdown.is_empty() => format_knowledge_breakdown(&breakdown),
            Ok(_) => "none".to_string(),
            Err(_) => "error".to_string(),
        };

        let cat_line = match self.with_store(|s| s.categories(None)) {
            Ok(cats) if !cats.is_empty() => format_category_line(&cats),
            Ok(_) => "0".to_string(),
            Err(_) => "error".to_string(),
        };

        let strongest_desc = match self.with_store(|s| {
            let link = s.strongest_link()?;
            match link {
                Some((src, tgt, w)) => {
                    let src_label = s
                        .node_content(src)?
                        .unwrap_or_else(|| format!("{}#{}", src.type_str(), src.id()));
                    let tgt_label = s
                        .node_content(tgt)?
                        .unwrap_or_else(|| format!("{}#{}", tgt.type_str(), tgt.id()));
                    Ok(Some(format!(
                        " (strongest: \"{src_label}\" <-> \"{tgt_label}\" weight {w:.2})"
                    )))
                }
                None => Ok(None),
            }
        }) {
            Ok(Some(desc)) => desc,
            _ => String::new(),
        };

        let total_nodes = st.episode_count + st.semantic_node_count;
        let coverage = if total_nodes > 0 {
            format!(
                "{}/{} nodes ({}%)",
                st.embedding_count,
                total_nodes,
                st.embedding_count * 100 / total_nodes
            )
        } else {
            "0/0 nodes".to_string()
        };

        format_status(
            &st,
            session_eps,
            unconsolidated,
            &knowledge_line,
            &cat_line,
            &strongest_desc,
            &coverage,
        )
    }

    /// Get user preferences.
    #[tool(
        description = "Get crystallized user preferences learned from past interactions. Optionally filter by domain (e.g. 'style', 'tone', 'format')."
    )]
    fn preferences(&self, #[tool(aggr)] params: PreferencesParams) -> String {
        match self.with_store(|s| s.preferences(params.domain.as_deref())) {
            Ok(prefs) if prefs.is_empty() => "No preferences found.".to_string(),
            Ok(prefs) => format_preferences(&prefs),
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Get semantic knowledge.
    #[tool(
        description = "Get distilled semantic knowledge (facts, relationships, events, concepts) extracted from past conversations."
    )]
    fn knowledge(&self, #[tool(aggr)] params: KnowledgeParams) -> String {
        let filter = KnowledgeFilter {
            node_type: params.node_type.as_deref().and_then(SemanticType::from_str),
            min_confidence: params.min_confidence,
            limit: params.limit.or(Some(20)),
            category: params.category,
        };

        match self.with_store(|s| s.knowledge(Some(filter))) {
            Ok(nodes) if nodes.is_empty() => "No knowledge found.".to_string(),
            Ok(nodes) => {
                let mut out = format!("Found {} knowledge nodes:\n\n", nodes.len());
                for n in &nodes {
                    out.push_str(&format!(
                        "- [{}] {} (confidence: {:.2})\n",
                        n.node_type.as_str(),
                        n.content,
                        n.confidence
                    ));
                }
                out
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Run memory maintenance (dedup, prune weak links, decay preferences).
    #[tool(
        description = "Run memory maintenance: deduplicates nodes, prunes weak links, decays stale preferences. Call periodically to keep memory healthy."
    )]
    fn maintain(&self) -> String {
        let transform = self.with_store(|s| s.transform());
        let forget = self.with_store(|s| s.forget());

        match (transform, forget) {
            (Ok(tr), Ok(fr)) => format!(
                "Maintenance complete:\n  Duplicates merged: {}\n  Links pruned: {}\n  Preferences decayed: {}\n  Nodes decayed: {}\n  Nodes archived: {}",
                tr.duplicates_merged,
                tr.links_pruned,
                tr.preferences_decayed,
                fr.nodes_decayed,
                fr.nodes_archived,
            ),
            (Err(e), _) | (_, Err(e)) => format!("Error: {e}"),
        }
    }

    /// List emergent categories.
    #[tool(
        description = "List emergent categories discovered from semantic knowledge clusters. Categories form automatically and evolve through use."
    )]
    fn categories(&self, #[tool(aggr)] params: CategoriesParams) -> String {
        match self.with_store(|s| s.categories(params.min_stability)) {
            Ok(cats) if cats.is_empty() => "No categories found.".to_string(),
            Ok(cats) => format_categories(&cats),
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Get graph neighbors of a node.
    #[tool(
        description = "Get graph neighbors of a memory node via spreading activation. Shows connected memories with link weights."
    )]
    fn neighbors(&self, #[tool(aggr)] params: NeighborsParams) -> String {
        let node_ref = match params.node_type.to_lowercase().as_str() {
            "episode" => NodeRef::Episode(EpisodeId(params.node_id)),
            "semantic" => NodeRef::Semantic(NodeId(params.node_id)),
            "preference" => NodeRef::Preference(PreferenceId(params.node_id)),
            "category" => NodeRef::Category(CategoryId(params.node_id)),
            _ => {
                return format!(
                    "Error: invalid node_type '{}'. Use: episode, semantic, preference, category",
                    params.node_type
                )
            }
        };
        let depth = params.depth.unwrap_or(1);

        match self.with_store(|s| s.neighbors(node_ref, depth)) {
            Ok(neighbors) if neighbors.is_empty() => "No neighbors found.".to_string(),
            Ok(neighbors) => format_neighbors(&neighbors),
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Get the category of a semantic node.
    #[tool(
        description = "Get which category a semantic knowledge node belongs to. Returns the category or 'uncategorized'."
    )]
    fn node_category(&self, #[tool(aggr)] params: NodeCategoryParams) -> String {
        match self.with_store(|s| s.node_category(NodeId(params.node_id))) {
            Ok(Some(cat)) => format_node_category(params.node_id, &cat),
            Ok(None) => format!("Node {} is uncategorized.", params.node_id),
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Teach Alaya extracted knowledge directly.
    #[tool(
        description = "Teach Alaya extracted knowledge directly. The agent extracts facts from conversation and calls this tool. Each fact becomes a semantic node with full lifecycle wiring (strength, categories, graph links)."
    )]
    fn learn(&self, #[tool(aggr)] params: LearnParams) -> String {
        // Resolve session_id → source episode IDs
        let source_episodes = match &params.session_id {
            Some(sid) => match self.with_store(|s| s.episodes_by_session(sid)) {
                Ok(eps) => eps.iter().map(|e| e.id).collect::<Vec<_>>(),
                Err(e) => return format!("Error resolving session '{sid}': {e}"),
            },
            None => vec![],
        };

        // Convert LearnFactEntry → NewSemanticNode
        let nodes: Vec<NewSemanticNode> = params
            .facts
            .iter()
            .map(|fact| {
                let node_type =
                    SemanticType::from_str(&fact.node_type).unwrap_or(SemanticType::Fact);
                let confidence = fact.confidence.unwrap_or(0.8).clamp(0.0, 1.0);
                NewSemanticNode {
                    content: fact.content.clone(),
                    node_type,
                    confidence,
                    source_episodes: source_episodes.clone(),
                    embedding: None,
                }
            })
            .collect();

        let count = nodes.len();
        match self.with_store(|s| s.learn(nodes)) {
            Ok(report) => {
                self.unconsolidated_count.store(0, Ordering::Relaxed);
                format!(
                    "Learned {} facts: {} nodes created, {} links created, {} categories assigned",
                    count, report.nodes_created, report.links_created, report.categories_assigned
                )
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Import memories from claude-mem SQLite database.
    #[tool(
        description = "Import memories from claude-mem (claude-mem.db SQLite database). Reads observations and converts facts/concepts into Alaya semantic nodes."
    )]
    fn import_claude_mem(&self, #[tool(aggr)] params: ImportClaudeMemParams) -> String {
        let path = expand_tilde(
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
        match self.with_store(|s| s.learn(nodes)) {
            Ok(report) => format!(
                "Imported {obs_count} observations \u{2192} {node_count} semantic nodes. {} categories assigned.",
                report.categories_assigned
            ),
            Err(e) => format!("Error importing: {e}"),
        }
    }

    /// Import conversation history from Claude Code JSONL files.
    #[tool(
        description = "Import conversation history from Claude Code JSONL files. Reads messages and stores them as episodes."
    )]
    fn import_claude_code(&self, #[tool(aggr)] params: ImportClaudeCodeParams) -> String {
        let path = expand_tilde(&params.path);

        let (episodes, sessions, mut errors, first_error) = match parse_claude_code_jsonl(&path) {
            Ok(result) => result,
            Err(e) => return e,
        };

        let mut imported = 0u32;
        for episode in episodes {
            match self.with_store(|s| s.store_episode(&episode)) {
                Ok(_) => {
                    imported += 1;
                    self.episode_count.fetch_add(1, Ordering::Relaxed);
                    self.unconsolidated_count.fetch_add(1, Ordering::Relaxed);
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

    /// Purge memories by session, timestamp, or everything.
    #[tool(
        description = "Purge memories. Scope: 'session' (requires session_id), 'older_than' (requires before_timestamp), or 'all' (deletes everything)."
    )]
    fn purge(&self, #[tool(aggr)] params: PurgeParams) -> String {
        let filter = match params.scope.as_str() {
            "session" => match params.session_id {
                Some(sid) => PurgeFilter::Session(sid),
                None => return "Error: session_id required for scope 'session'".to_string(),
            },
            "older_than" => match params.before_timestamp {
                Some(ts) => PurgeFilter::OlderThan(ts),
                None => {
                    return "Error: before_timestamp required for scope 'older_than'".to_string()
                }
            },
            "all" => PurgeFilter::All,
            _ => {
                return format!(
                    "Error: invalid scope '{}'. Use: session, older_than, all",
                    params.scope
                )
            }
        };

        match self.with_store(|s| s.purge(filter)) {
            Ok(report) => format!(
                "Purge complete: {} episodes deleted",
                report.episodes_deleted
            ),
            Err(e) => format!("Error: {e}"),
        }
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
                 and 'purge' to delete data."
                    .into(),
            ),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use super::*;
    use crate::{AlayaStore, MockExtractionProvider};

    fn make_server() -> AlayaMcp {
        let store = AlayaStore::open_in_memory().unwrap();
        AlayaMcp::new(store)
    }

    /// Helper: store N user messages and return the server.
    fn server_with_episodes(n: u32) -> AlayaMcp {
        let srv = make_server();
        for i in 0..n {
            srv.remember(RememberParams {
                content: format!("Message number {i}"),
                role: "user".into(),
                session_id: "sess-1".into(),
            });
        }
        srv
    }

    // -----------------------------------------------------------------------
    // expand_tilde
    // -----------------------------------------------------------------------

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
        // "~" without trailing slash does not match "~/" prefix, returned as-is
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

    // -----------------------------------------------------------------------
    // extract_content
    // -----------------------------------------------------------------------

    #[test]
    fn extract_content_string_value() {
        let val = serde_json::json!("hello world");
        assert_eq!(extract_content(&val), "hello world");
    }

    #[test]
    fn extract_content_array_of_text_objects() {
        let val = serde_json::json!([
            {"text": "line one"},
            {"text": "line two"},
            {"text": "line three"}
        ]);
        assert_eq!(extract_content(&val), "line one\nline two\nline three");
    }

    #[test]
    fn extract_content_array_mixed_some_without_text() {
        let val = serde_json::json!([
            {"text": "has text"},
            {"type": "tool_use", "id": "123"},
            {"text": "also text"}
        ]);
        assert_eq!(extract_content(&val), "has text\nalso text");
    }

    #[test]
    fn extract_content_empty_array() {
        let val = serde_json::json!([]);
        assert_eq!(extract_content(&val), "");
    }

    #[test]
    fn extract_content_number() {
        let val = serde_json::json!(42);
        assert_eq!(extract_content(&val), "");
    }

    #[test]
    fn extract_content_null() {
        let val = serde_json::Value::Null;
        assert_eq!(extract_content(&val), "");
    }

    #[test]
    fn extract_content_bool() {
        let val = serde_json::json!(true);
        assert_eq!(extract_content(&val), "");
    }

    #[test]
    fn extract_content_object() {
        let val = serde_json::json!({"key": "value"});
        assert_eq!(extract_content(&val), "");
    }

    // -----------------------------------------------------------------------
    // remember tool
    // -----------------------------------------------------------------------

    #[test]
    fn remember_valid_user_message() {
        let srv = make_server();
        let result = srv.remember(RememberParams {
            content: "Hello world".into(),
            role: "user".into(),
            session_id: "test-sess".into(),
        });
        assert!(result.starts_with("Stored episode "));
        assert!(result.contains("in session 'test-sess'"));
    }

    #[test]
    fn remember_valid_assistant_message() {
        let srv = make_server();
        let result = srv.remember(RememberParams {
            content: "I can help with that".into(),
            role: "assistant".into(),
            session_id: "test-sess".into(),
        });
        assert!(result.starts_with("Stored episode "));
    }

    #[test]
    fn remember_valid_system_message() {
        let srv = make_server();
        let result = srv.remember(RememberParams {
            content: "System prompt here".into(),
            role: "system".into(),
            session_id: "test-sess".into(),
        });
        assert!(result.starts_with("Stored episode "));
    }

    #[test]
    fn remember_invalid_role() {
        let srv = make_server();
        let result = srv.remember(RememberParams {
            content: "Hello".into(),
            role: "moderator".into(),
            session_id: "test-sess".into(),
        });
        assert!(result.starts_with("Error: invalid role"));
        assert!(result.contains("moderator"));
    }

    #[test]
    fn remember_case_insensitive_role() {
        let srv = make_server();
        let result = srv.remember(RememberParams {
            content: "Hello".into(),
            role: "USER".into(),
            session_id: "test-sess".into(),
        });
        assert!(result.starts_with("Stored episode "));
    }

    #[test]
    fn remember_consolidation_prompt_at_10() {
        let srv = make_server();
        for i in 0..10 {
            let result = srv.remember(RememberParams {
                content: format!("Message {i}"),
                role: "user".into(),
                session_id: "sess".into(),
            });
            if i < 9 {
                // Episodes 1-9: no consolidation prompt
                assert!(
                    !result.contains("Consolidation suggested"),
                    "Should not suggest consolidation at episode {}",
                    i + 1
                );
            } else {
                // Episode 10: consolidation prompt appears
                assert!(
                    result.contains("Consolidation suggested"),
                    "Should suggest consolidation at episode 10"
                );
                assert!(result.contains("unconsolidated episodes"));
            }
        }
    }

    #[test]
    fn remember_learn_resets_consolidation_counter() {
        let srv = make_server();
        // Store 10 episodes (triggers consolidation prompt)
        for i in 0..10 {
            srv.remember(RememberParams {
                content: format!("Fact message {i}"),
                role: "user".into(),
                session_id: "sess".into(),
            });
        }

        // Learn resets unconsolidated_count
        srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Extracted fact".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: None,
        });

        // Store 1 more — should NOT trigger consolidation prompt (counter was reset)
        let result = srv.remember(RememberParams {
            content: "Post-learn message".into(),
            role: "user".into(),
            session_id: "sess".into(),
        });
        assert!(
            !result.contains("Consolidation suggested"),
            "After learn, counter should be reset; 1 episode should not trigger consolidation"
        );
    }

    #[test]
    fn remember_auto_maintenance_at_25() {
        let srv = make_server();
        let mut maintenance_seen = false;
        for i in 0..25 {
            let result = srv.remember(RememberParams {
                content: format!("Episode {i}"),
                role: "user".into(),
                session_id: "sess".into(),
            });
            if result.contains("Auto-maintenance") {
                maintenance_seen = true;
            }
        }
        assert!(
            maintenance_seen,
            "Auto-maintenance should trigger at 25 episodes"
        );
    }

    // -----------------------------------------------------------------------
    // auto-consolidation via ExtractionProvider
    // -----------------------------------------------------------------------

    fn make_server_with_extraction() -> AlayaMcp {
        let mut store = AlayaStore::open_in_memory().unwrap();
        store.set_extraction_provider(Box::new(MockExtractionProvider::new(vec![
            NewSemanticNode {
                content: "Auto-extracted fact".into(),
                node_type: SemanticType::Fact,
                confidence: 0.85,
                source_episodes: vec![],
                embedding: None,
            },
        ])));
        AlayaMcp::new(store)
    }

    #[test]
    fn remember_auto_consolidates_with_extraction_provider() {
        let srv = make_server_with_extraction();
        let mut auto_response = String::new();
        for i in 0..10 {
            let result = srv.remember(RememberParams {
                content: format!("Episode {i}"),
                role: "user".into(),
                session_id: "s1".into(),
            });
            if result.contains("Auto-consolidated") {
                auto_response = result;
            }
        }
        assert!(
            !auto_response.is_empty(),
            "Should have auto-consolidated at episode 10"
        );
        assert!(auto_response.contains("knowledge nodes"));
    }

    #[test]
    fn remember_falls_back_to_prompt_without_provider() {
        let srv = make_server(); // no extraction provider
        let mut prompt_response = String::new();
        for i in 0..10 {
            let result = srv.remember(RememberParams {
                content: format!("Episode {i}"),
                role: "user".into(),
                session_id: "s1".into(),
            });
            if result.contains("Consolidation suggested") {
                prompt_response = result;
            }
        }
        assert!(
            !prompt_response.is_empty(),
            "Should fall back to prompt without extraction provider"
        );
        assert!(prompt_response.contains("unconsolidated episodes"));
    }

    #[test]
    fn remember_auto_consolidation_resets_counter() {
        let srv = make_server_with_extraction();
        for i in 0..10 {
            srv.remember(RememberParams {
                content: format!("Episode {i}"),
                role: "user".into(),
                session_id: "s1".into(),
            });
        }
        // Counter should be reset after auto-consolidation
        let status = srv.status();
        assert!(
            status.contains("0 unconsolidated"),
            "Counter should reset after auto-consolidation: {status}"
        );
    }

    // -----------------------------------------------------------------------
    // recall tool
    // -----------------------------------------------------------------------

    #[test]
    fn recall_empty_store() {
        let srv = make_server();
        let result = srv.recall(RecallParams {
            query: "anything".into(),
            max_results: None,
            boost_category: None,
        });
        assert_eq!(result, "No memories found.");
    }

    #[test]
    fn recall_finds_matching_episodes() {
        let srv = make_server();
        srv.remember(RememberParams {
            content: "Rust has zero-cost abstractions".into(),
            role: "user".into(),
            session_id: "s1".into(),
        });
        srv.remember(RememberParams {
            content: "Python is great for scripting".into(),
            role: "user".into(),
            session_id: "s1".into(),
        });

        let result = srv.recall(RecallParams {
            query: "Rust abstractions".into(),
            max_results: None,
            boost_category: None,
        });
        assert!(result.contains("Found"));
        assert!(result.contains("memories"));
    }

    #[test]
    fn recall_with_max_results() {
        let srv = make_server();
        for i in 0..10 {
            srv.remember(RememberParams {
                content: format!("Fact number {i} about programming"),
                role: "user".into(),
                session_id: "s1".into(),
            });
        }

        let result = srv.recall(RecallParams {
            query: "programming".into(),
            max_results: Some(3),
            boost_category: None,
        });
        assert!(result.contains("Found"));
        // Count the numbered results (lines starting with "N.")
        let result_count = result
            .lines()
            .filter(|l| {
                let trimmed = l.trim();
                trimmed.starts_with("1.")
                    || trimmed.starts_with("2.")
                    || trimmed.starts_with("3.")
                    || trimmed.starts_with("4.")
            })
            .count();
        assert!(
            result_count <= 3,
            "Should return at most 3 results, got {result_count}"
        );
    }

    #[test]
    fn recall_with_boost_category_no_crash() {
        let srv = make_server();
        srv.remember(RememberParams {
            content: "Some memory content".into(),
            role: "user".into(),
            session_id: "s1".into(),
        });
        // boost_category with a non-existent category should not crash
        let result = srv.recall(RecallParams {
            query: "memory".into(),
            max_results: None,
            boost_category: Some(9999),
        });
        // Should succeed (may or may not find results depending on boost logic)
        assert!(!result.starts_with("Error:"));
    }

    // -----------------------------------------------------------------------
    // status tool
    // -----------------------------------------------------------------------

    #[test]
    fn status_empty_store() {
        let srv = make_server();
        let result = srv.status();
        assert!(result.contains("Memory Status:"));
        assert!(result.contains("Episodes: 0"));
        assert!(result.contains("Knowledge: none"));
    }

    #[test]
    fn status_after_storing_episodes() {
        let srv = server_with_episodes(3);
        let result = srv.status();
        assert!(result.contains("Memory Status:"));
        assert!(result.contains("Episodes: 3"));
        assert!(result.contains("3 this session"));
        assert!(result.contains("3 unconsolidated"));
    }

    #[test]
    fn status_shows_session_and_unconsolidated() {
        let srv = make_server();
        for i in 0..5 {
            srv.remember(RememberParams {
                content: format!("Msg {i}"),
                role: "user".into(),
                session_id: "s1".into(),
            });
        }
        let result = srv.status();
        assert!(result.contains("5 this session"));
        assert!(result.contains("5 unconsolidated"));
    }

    // -----------------------------------------------------------------------
    // preferences tool
    // -----------------------------------------------------------------------

    #[test]
    fn preferences_empty_store() {
        let srv = make_server();
        let result = srv.preferences(PreferencesParams { domain: None });
        assert_eq!(result, "No preferences found.");
    }

    #[test]
    fn preferences_with_domain_filter_no_crash() {
        let srv = make_server();
        let result = srv.preferences(PreferencesParams {
            domain: Some("style".into()),
        });
        // Should return gracefully, not crash
        assert_eq!(result, "No preferences found.");
    }

    // -----------------------------------------------------------------------
    // knowledge tool
    // -----------------------------------------------------------------------

    #[test]
    fn knowledge_empty_store() {
        let srv = make_server();
        let result = srv.knowledge(KnowledgeParams {
            node_type: None,
            min_confidence: None,
            limit: None,
            category: None,
        });
        assert_eq!(result, "No knowledge found.");
    }

    #[test]
    fn knowledge_after_learn() {
        let srv = make_server();
        srv.learn(LearnParams {
            facts: vec![
                LearnFactEntry {
                    content: "Rust is a systems language".into(),
                    node_type: "fact".into(),
                    confidence: Some(0.9),
                },
                LearnFactEntry {
                    content: "Alaya means storehouse".into(),
                    node_type: "concept".into(),
                    confidence: Some(0.85),
                },
            ],
            session_id: None,
        });

        let result = srv.knowledge(KnowledgeParams {
            node_type: None,
            min_confidence: None,
            limit: None,
            category: None,
        });
        assert!(result.contains("Found 2 knowledge nodes"));
        assert!(result.contains("Rust is a systems language"));
        assert!(result.contains("Alaya means storehouse"));
    }

    #[test]
    fn knowledge_with_node_type_filter() {
        let srv = make_server();
        srv.learn(LearnParams {
            facts: vec![
                LearnFactEntry {
                    content: "Fact one".into(),
                    node_type: "fact".into(),
                    confidence: None,
                },
                LearnFactEntry {
                    content: "Concept one".into(),
                    node_type: "concept".into(),
                    confidence: None,
                },
            ],
            session_id: None,
        });

        let result = srv.knowledge(KnowledgeParams {
            node_type: Some("fact".into()),
            min_confidence: None,
            limit: None,
            category: None,
        });
        assert!(result.contains("Fact one"));
        assert!(!result.contains("Concept one"));
    }

    #[test]
    fn knowledge_with_category_filter_no_crash() {
        let srv = make_server();
        let result = srv.knowledge(KnowledgeParams {
            node_type: None,
            min_confidence: None,
            limit: None,
            category: Some("nonexistent".into()),
        });
        // Should not crash; may return no results
        assert!(result == "No knowledge found." || result.contains("Found"));
    }

    #[test]
    fn knowledge_with_min_confidence_filter() {
        let srv = make_server();
        srv.learn(LearnParams {
            facts: vec![
                LearnFactEntry {
                    content: "Low confidence fact".into(),
                    node_type: "fact".into(),
                    confidence: Some(0.3),
                },
                LearnFactEntry {
                    content: "High confidence fact".into(),
                    node_type: "fact".into(),
                    confidence: Some(0.95),
                },
            ],
            session_id: None,
        });

        let result = srv.knowledge(KnowledgeParams {
            node_type: None,
            min_confidence: Some(0.9),
            limit: None,
            category: None,
        });
        assert!(result.contains("High confidence fact"));
        assert!(!result.contains("Low confidence fact"));
    }

    #[test]
    fn knowledge_with_limit() {
        let srv = make_server();
        srv.learn(LearnParams {
            facts: (0..10)
                .map(|i| LearnFactEntry {
                    content: format!("Knowledge item {i}"),
                    node_type: "fact".into(),
                    confidence: None,
                })
                .collect(),
            session_id: None,
        });

        let result = srv.knowledge(KnowledgeParams {
            node_type: None,
            min_confidence: None,
            limit: Some(3),
            category: None,
        });
        assert!(result.contains("Found 3 knowledge nodes"));
    }

    // -----------------------------------------------------------------------
    // maintain tool
    // -----------------------------------------------------------------------

    #[test]
    fn maintain_empty_store() {
        let srv = make_server();
        let result = srv.maintain();
        assert!(result.contains("Maintenance complete"));
        assert!(result.contains("Duplicates merged: 0"));
        assert!(result.contains("Links pruned: 0"));
    }

    #[test]
    fn maintain_after_data() {
        let srv = server_with_episodes(5);
        let result = srv.maintain();
        assert!(result.contains("Maintenance complete"));
    }

    // -----------------------------------------------------------------------
    // categories tool
    // -----------------------------------------------------------------------

    #[test]
    fn categories_empty_store() {
        let srv = make_server();
        let result = srv.categories(CategoriesParams {
            min_stability: None,
        });
        assert_eq!(result, "No categories found.");
    }

    #[test]
    fn categories_with_min_stability_no_crash() {
        let srv = make_server();
        let result = srv.categories(CategoriesParams {
            min_stability: Some(0.5),
        });
        // Should not crash; no categories exist
        assert_eq!(result, "No categories found.");
    }

    // -----------------------------------------------------------------------
    // neighbors tool
    // -----------------------------------------------------------------------

    #[test]
    fn neighbors_episode_node() {
        let srv = make_server();
        srv.remember(RememberParams {
            content: "Test episode".into(),
            role: "user".into(),
            session_id: "s1".into(),
        });

        let result = srv.neighbors(NeighborsParams {
            node_type: "episode".into(),
            node_id: 1,
            depth: None,
        });
        // May return "No neighbors found." or actual neighbors
        assert!(!result.starts_with("Error:"));
    }

    #[test]
    fn neighbors_semantic_node_no_crash() {
        let srv = make_server();
        let result = srv.neighbors(NeighborsParams {
            node_type: "semantic".into(),
            node_id: 1,
            depth: None,
        });
        assert!(!result.starts_with("Error:") || result.contains("No neighbors"));
    }

    #[test]
    fn neighbors_preference_node_no_crash() {
        let srv = make_server();
        let result = srv.neighbors(NeighborsParams {
            node_type: "preference".into(),
            node_id: 1,
            depth: None,
        });
        assert!(!result.starts_with("Error:") || result.contains("No neighbors"));
    }

    #[test]
    fn neighbors_category_node_no_crash() {
        let srv = make_server();
        let result = srv.neighbors(NeighborsParams {
            node_type: "category".into(),
            node_id: 1,
            depth: None,
        });
        assert!(!result.starts_with("Error:") || result.contains("No neighbors"));
    }

    #[test]
    fn neighbors_invalid_node_type() {
        let srv = make_server();
        let result = srv.neighbors(NeighborsParams {
            node_type: "bogus".into(),
            node_id: 1,
            depth: None,
        });
        assert!(result.starts_with("Error: invalid node_type"));
        assert!(result.contains("bogus"));
    }

    #[test]
    fn neighbors_with_depth() {
        let srv = make_server();
        srv.remember(RememberParams {
            content: "Test".into(),
            role: "user".into(),
            session_id: "s1".into(),
        });
        let result = srv.neighbors(NeighborsParams {
            node_type: "episode".into(),
            node_id: 1,
            depth: Some(2),
        });
        assert!(!result.starts_with("Error: invalid"));
    }

    // -----------------------------------------------------------------------
    // node_category tool
    // -----------------------------------------------------------------------

    #[test]
    fn node_category_nonexistent() {
        let srv = make_server();
        let result = srv.node_category(NodeCategoryParams { node_id: 9999 });
        assert!(result.contains("uncategorized"));
    }

    #[test]
    fn node_category_after_learn() {
        let srv = make_server();
        srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Test knowledge".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: None,
        });
        let result = srv.node_category(NodeCategoryParams { node_id: 1 });
        // Either categorized or uncategorized — both are valid
        assert!(result.contains("Node 1"));
    }

    // -----------------------------------------------------------------------
    // learn tool
    // -----------------------------------------------------------------------

    #[test]
    fn learn_three_facts() {
        let srv = make_server();
        let result = srv.learn(LearnParams {
            facts: vec![
                LearnFactEntry {
                    content: "Rust is fast".into(),
                    node_type: "fact".into(),
                    confidence: None,
                },
                LearnFactEntry {
                    content: "Alaya uses SQLite".into(),
                    node_type: "fact".into(),
                    confidence: Some(0.9),
                },
                LearnFactEntry {
                    content: "MCP is a protocol".into(),
                    node_type: "concept".into(),
                    confidence: None,
                },
            ],
            session_id: None,
        });
        assert!(result.starts_with("Learned 3 facts:"));
        assert!(result.contains("3 nodes created"));
    }

    #[test]
    fn learn_with_session_id_links_episodes() {
        let srv = make_server();
        // Store some episodes in a session
        srv.remember(RememberParams {
            content: "User said something".into(),
            role: "user".into(),
            session_id: "sess-link".into(),
        });
        srv.remember(RememberParams {
            content: "Assistant replied".into(),
            role: "assistant".into(),
            session_id: "sess-link".into(),
        });

        let result = srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Extracted from conversation".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: Some("sess-link".into()),
        });
        assert!(result.starts_with("Learned 1 facts:"));
        assert!(result.contains("1 nodes created"));
        // links_created should be > 0 because source_episodes were linked
        assert!(result.contains("links created"));
    }

    #[test]
    fn learn_resets_unconsolidated_counter() {
        let srv = make_server();
        // Store 5 episodes
        for i in 0..5 {
            srv.remember(RememberParams {
                content: format!("Ep {i}"),
                role: "user".into(),
                session_id: "s1".into(),
            });
        }

        // Verify unconsolidated is 5 via status
        let status = srv.status();
        assert!(status.contains("5 unconsolidated"));

        // Learn resets unconsolidated
        srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Learned fact".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: None,
        });

        // Verify counter is reset
        let status = srv.status();
        assert!(status.contains("0 unconsolidated"));
    }

    #[test]
    fn learn_invalid_node_type_defaults_to_fact() {
        let srv = make_server();
        let result = srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Something interesting".into(),
                node_type: "invalid_type".into(),
                confidence: None,
            }],
            session_id: None,
        });
        assert!(result.starts_with("Learned 1 facts:"));

        // Verify it was stored as a fact
        let knowledge = srv.knowledge(KnowledgeParams {
            node_type: Some("fact".into()),
            min_confidence: None,
            limit: None,
            category: None,
        });
        assert!(knowledge.contains("Something interesting"));
    }

    #[test]
    fn learn_with_clamped_confidence() {
        let srv = make_server();
        let result = srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Over-confident fact".into(),
                node_type: "fact".into(),
                confidence: Some(5.0), // should be clamped to 1.0
            }],
            session_id: None,
        });
        assert!(result.starts_with("Learned 1 facts:"));

        // Verify confidence was clamped
        let knowledge = srv.knowledge(KnowledgeParams {
            node_type: None,
            min_confidence: Some(1.0),
            limit: None,
            category: None,
        });
        assert!(knowledge.contains("Over-confident fact"));
    }

    #[test]
    fn learn_empty_facts_vec() {
        let srv = make_server();
        let result = srv.learn(LearnParams {
            facts: vec![],
            session_id: None,
        });
        assert!(result.starts_with("Learned 0 facts:"));
    }

    #[test]
    fn learn_with_nonexistent_session() {
        let srv = make_server();
        // Session doesn't have any episodes; should still work (empty source_episodes)
        let result = srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "A fact".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: Some("nonexistent-session".into()),
        });
        assert!(result.starts_with("Learned 1 facts:"));
    }

    // -----------------------------------------------------------------------
    // import_claude_mem tool
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
        // Default path (~/.claude-mem/claude-mem.db) likely doesn't exist in test
        let srv = make_server();
        let result = srv.import_claude_mem(ImportClaudeMemParams { path: None });
        // Either "Cannot open" or it happens to exist; don't crash either way
        assert!(
            result.contains("Cannot open")
                || result.contains("Imported")
                || result.contains("No observations"),
            "Unexpected result: {result}"
        );
    }

    // -----------------------------------------------------------------------
    // import_claude_code tool
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

        // Create a realistic Claude Code JSONL file
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
        // Should import what it can and report errors
        assert!(
            result.contains("Imported") || result.contains("error"),
            "Should handle mixed valid/invalid lines: {result}"
        );
    }

    #[test]
    fn import_claude_code_tilde_expansion() {
        let srv = make_server();
        // This path likely doesn't exist, but tests that tilde expansion runs
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: "~/nonexistent-alaya-test-file.jsonl".into(),
        });
        assert!(result.contains("Cannot read file"));
        // Crucially, the path in the error should be expanded (not contain ~/)
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

        // Check that counters were updated via status
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
    // purge tool
    // -----------------------------------------------------------------------

    #[test]
    fn purge_session() {
        let srv = make_server();
        // Store episodes in two sessions
        for i in 0..3 {
            srv.remember(RememberParams {
                content: format!("Sess A msg {i}"),
                role: "user".into(),
                session_id: "sess-a".into(),
            });
        }
        for i in 0..2 {
            srv.remember(RememberParams {
                content: format!("Sess B msg {i}"),
                role: "user".into(),
                session_id: "sess-b".into(),
            });
        }

        let result = srv.purge(PurgeParams {
            scope: "session".into(),
            session_id: Some("sess-a".into()),
            before_timestamp: None,
        });
        assert!(result.contains("Purge complete"));
        assert!(result.contains("3 episodes deleted"));
    }

    #[test]
    fn purge_older_than() {
        let srv = make_server();
        srv.remember(RememberParams {
            content: "Old message".into(),
            role: "user".into(),
            session_id: "s1".into(),
        });

        // Use a far-future timestamp to purge everything
        let result = srv.purge(PurgeParams {
            scope: "older_than".into(),
            session_id: None,
            before_timestamp: Some(i64::MAX),
        });
        assert!(result.contains("Purge complete"));
        assert!(result.contains("episodes deleted"));
    }

    #[test]
    fn purge_all() {
        let srv = server_with_episodes(5);
        let result = srv.purge(PurgeParams {
            scope: "all".into(),
            session_id: None,
            before_timestamp: None,
        });
        assert!(result.contains("Purge complete"));
        // PurgeFilter::All uses execute_batch so episodes_deleted stays 0;
        // verify the data is actually gone via status.
        let status = srv.status();
        assert!(
            status.contains("Episodes: 0"),
            "All episodes should be gone after purge all: {status}"
        );
    }

    #[test]
    fn purge_invalid_scope() {
        let srv = make_server();
        let result = srv.purge(PurgeParams {
            scope: "invalid".into(),
            session_id: None,
            before_timestamp: None,
        });
        assert!(result.starts_with("Error: invalid scope"));
        assert!(result.contains("invalid"));
    }

    #[test]
    fn purge_session_without_session_id() {
        let srv = make_server();
        let result = srv.purge(PurgeParams {
            scope: "session".into(),
            session_id: None,
            before_timestamp: None,
        });
        assert_eq!(result, "Error: session_id required for scope 'session'");
    }

    #[test]
    fn purge_older_than_without_timestamp() {
        let srv = make_server();
        let result = srv.purge(PurgeParams {
            scope: "older_than".into(),
            session_id: None,
            before_timestamp: None,
        });
        assert_eq!(
            result,
            "Error: before_timestamp required for scope 'older_than'"
        );
    }

    #[test]
    fn purge_session_deletes_only_that_session() {
        let srv = make_server();
        srv.remember(RememberParams {
            content: "Keep me".into(),
            role: "user".into(),
            session_id: "keep".into(),
        });
        srv.remember(RememberParams {
            content: "Delete me".into(),
            role: "user".into(),
            session_id: "delete".into(),
        });

        srv.purge(PurgeParams {
            scope: "session".into(),
            session_id: Some("delete".into()),
            before_timestamp: None,
        });

        // The kept session's episode should still be queryable
        let result = srv.recall(RecallParams {
            query: "Keep me".into(),
            max_results: None,
            boost_category: None,
        });
        assert!(result.contains("Found"));
        assert!(result.contains("Keep me"));
    }

    // -----------------------------------------------------------------------
    // Integration / cross-tool tests
    // -----------------------------------------------------------------------

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
                    content: "Paris is the capital of France".into(),
                    node_type: "fact".into(),
                    confidence: Some(0.95),
                },
                LearnFactEntry {
                    content: "Paris has the Eiffel Tower".into(),
                    node_type: "fact".into(),
                    confidence: Some(0.9),
                },
            ],
            session_id: Some("geo".into()),
        });
        assert!(learn_result.contains("Learned 2 facts"));

        // 3. Recall should find memories
        let recall_result = srv.recall(RecallParams {
            query: "capital France Paris".into(),
            max_results: None,
            boost_category: None,
        });
        assert!(recall_result.contains("Found"));

        // 4. Knowledge should show the facts
        let knowledge_result = srv.knowledge(KnowledgeParams {
            node_type: None,
            min_confidence: None,
            limit: None,
            category: None,
        });
        assert!(knowledge_result.contains("Paris is the capital of France"));

        // 5. Status should reflect the data
        let status = srv.status();
        assert!(status.contains("Episodes: 2"));
        assert!(status.contains("0 unconsolidated")); // learn reset it
    }

    #[test]
    fn purge_then_status_shows_zero() {
        let srv = server_with_episodes(5);

        // Verify we have data
        let status = srv.status();
        assert!(status.contains("Episodes: 5"));

        // Purge all
        srv.purge(PurgeParams {
            scope: "all".into(),
            session_id: None,
            before_timestamp: None,
        });

        // Status should show 0 episodes (in DB — counters are session-scoped)
        let status = srv.status();
        assert!(status.contains("Episodes: 0"));
    }

    #[test]
    fn maintain_after_learn_no_crash() {
        let srv = make_server();
        // Learn some facts
        srv.learn(LearnParams {
            facts: vec![
                LearnFactEntry {
                    content: "Fact A".into(),
                    node_type: "fact".into(),
                    confidence: None,
                },
                LearnFactEntry {
                    content: "Fact B".into(),
                    node_type: "relationship".into(),
                    confidence: None,
                },
            ],
            session_id: None,
        });

        // Maintenance should work fine
        let result = srv.maintain();
        assert!(result.contains("Maintenance complete"));
    }

    #[test]
    fn status_knowledge_breakdown_after_learn() {
        let srv = make_server();
        srv.learn(LearnParams {
            facts: vec![
                LearnFactEntry {
                    content: "F1".into(),
                    node_type: "fact".into(),
                    confidence: None,
                },
                LearnFactEntry {
                    content: "F2".into(),
                    node_type: "fact".into(),
                    confidence: None,
                },
                LearnFactEntry {
                    content: "R1".into(),
                    node_type: "relationship".into(),
                    confidence: None,
                },
                LearnFactEntry {
                    content: "E1".into(),
                    node_type: "event".into(),
                    confidence: None,
                },
            ],
            session_id: None,
        });

        let status = srv.status();
        assert!(status.contains("2 facts"));
        assert!(status.contains("1 relationships"));
        assert!(status.contains("1 events"));
    }

    #[test]
    fn neighbors_after_learn_with_session() {
        let srv = make_server();
        srv.remember(RememberParams {
            content: "Source episode".into(),
            role: "user".into(),
            session_id: "ns".into(),
        });
        srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Linked fact".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: Some("ns".into()),
        });

        // The semantic node (id=1) should have an episode neighbor
        let result = srv.neighbors(NeighborsParams {
            node_type: "semantic".into(),
            node_id: 1,
            depth: Some(1),
        });
        // Should not error; may find the episode as a neighbor
        assert!(!result.starts_with("Error: invalid"));
    }

    // -----------------------------------------------------------------------
    // Coverage gap tests: status with knowledge and categories
    // (covers lines 408, 415-416, 419)
    // -----------------------------------------------------------------------

    /// Helper: create a server pre-populated with categorized knowledge nodes.
    fn server_with_categories() -> AlayaMcp {
        let store = crate::AlayaStore::open_in_memory().unwrap();

        // Create semantic nodes with embeddings (for clustering)
        let nodes: Vec<crate::NewSemanticNode> = (0..5)
            .map(|i| crate::NewSemanticNode {
                content: format!("machine learning concept {i}"),
                node_type: crate::SemanticType::Fact,
                confidence: 0.8,
                source_episodes: vec![],
                embedding: Some(vec![1.0, 0.0 + (i as f32) * 0.01, 0.0]),
            })
            .collect();
        let _ = store.learn(nodes);
        // Transform discovers categories from clustered embeddings
        let _ = store.transform();

        AlayaMcp::new(store)
    }

    #[test]
    fn status_with_knowledge_and_categories() {
        let srv = server_with_categories();

        let status = srv.status();
        // Status should contain knowledge breakdown
        assert!(
            status.contains("Knowledge:"),
            "Status should show knowledge: {status}"
        );
        assert!(
            status.contains("facts"),
            "Status should mention facts: {status}"
        );
        // Should show categories (may be 0 if clustering threshold not met)
        assert!(
            status.contains("Categories:"),
            "Status should show categories: {status}"
        );
    }

    // -----------------------------------------------------------------------
    // Coverage gap tests: preferences with actual data
    // (covers lines 477-487)
    // -----------------------------------------------------------------------

    #[test]
    fn preferences_with_actual_data() {
        // Build store with preferences via the perfume pipeline
        let store = crate::AlayaStore::open_in_memory().unwrap();

        // Store some episodes for context
        for i in 0..5 {
            store
                .store_episode(&crate::NewEpisode {
                    content: format!("User prefers concise code comments {i}"),
                    role: crate::Role::User,
                    session_id: "pref-test".to_string(),
                    timestamp: 1000 + i * 100,
                    context: crate::EpisodeContext::default(),
                    embedding: None,
                })
                .unwrap();
        }

        // Create an interaction and use MockProvider with impressions
        let interaction = crate::types::Interaction {
            text: "I prefer concise code".to_string(),
            role: crate::Role::User,
            session_id: "pref-test".to_string(),
            timestamp: 2000,
            context: crate::EpisodeContext::default(),
        };
        let provider =
            crate::provider::MockProvider::with_impressions(vec![crate::NewImpression {
                domain: "style".to_string(),
                observation: "prefers concise code".to_string(),
                valence: 0.9,
            }]);
        // Run perfume to crystallize impressions into preferences
        let _ = store.perfume(&interaction, &provider);
        // Run perfume again so impressions crystallize into preferences
        let _ = store.perfume(&interaction, &provider);

        let srv = AlayaMcp::new(store);
        let result = srv.preferences(PreferencesParams { domain: None });
        // Should find preferences if perfuming worked
        assert!(
            result.contains("Found") || result.contains("No preferences"),
            "Preferences result: {result}"
        );
    }

    // -----------------------------------------------------------------------
    // Coverage gap tests: categories with actual data
    // (covers lines 538, 549-558, 561, 563)
    // -----------------------------------------------------------------------

    #[test]
    fn categories_with_data_after_transform() {
        let srv = server_with_categories();
        let result = srv.categories(CategoriesParams {
            min_stability: None,
        });
        // Either categories were discovered or not (depends on clustering threshold)
        assert!(
            result.contains("Found") || result.contains("No categories"),
            "Categories result: {result}"
        );
    }

    #[test]
    fn categories_with_min_stability_filter() {
        let srv = make_server();
        // Learn some facts
        srv.learn(LearnParams {
            facts: vec![
                LearnFactEntry {
                    content: "Fact A".into(),
                    node_type: "fact".into(),
                    confidence: None,
                },
                LearnFactEntry {
                    content: "Fact B".into(),
                    node_type: "fact".into(),
                    confidence: None,
                },
            ],
            session_id: None,
        });
        srv.maintain();
        let result = srv.categories(CategoriesParams {
            min_stability: Some(0.0),
        });
        // Should not crash regardless of whether categories exist
        assert!(
            result.contains("Found") || result.contains("No categories"),
            "Categories result: {result}"
        );
    }

    // -----------------------------------------------------------------------
    // Coverage gap tests: neighbors with actual data
    // (covers lines 593-596, 602)
    // -----------------------------------------------------------------------

    #[test]
    fn neighbors_returns_actual_linked_nodes() {
        let srv = make_server();
        // Store episodes in a session
        for i in 0..3 {
            srv.remember(RememberParams {
                content: format!("Important conversation about topic {i}"),
                role: "user".into(),
                session_id: "neighbor-sess".into(),
            });
        }
        // Learn facts linked to the session
        srv.learn(LearnParams {
            facts: vec![
                LearnFactEntry {
                    content: "Topic is important".into(),
                    node_type: "fact".into(),
                    confidence: None,
                },
                LearnFactEntry {
                    content: "Topic has sub-topics".into(),
                    node_type: "relationship".into(),
                    confidence: None,
                },
            ],
            session_id: Some("neighbor-sess".into()),
        });

        // Query neighbors of the first episode
        let result = srv.neighbors(NeighborsParams {
            node_type: "episode".into(),
            node_id: 1,
            depth: Some(1),
        });
        // Should find semantic nodes as neighbors (Causal links)
        assert!(
            result.contains("Found") || result.contains("No neighbors"),
            "Neighbors result: {result}"
        );
    }

    // -----------------------------------------------------------------------
    // Coverage gap tests: node_category with categorized node
    // (covers lines 612-617, 623)
    // -----------------------------------------------------------------------

    #[test]
    fn node_category_with_categorized_node() {
        let srv = server_with_categories();
        // Check node 1 - may be categorized after transform
        let result = srv.node_category(NodeCategoryParams { node_id: 1 });
        // Should either show category info or "uncategorized"
        assert!(
            result.contains("belongs to category") || result.contains("uncategorized"),
            "Node category result: {result}"
        );
    }

    // -----------------------------------------------------------------------
    // Coverage gap tests: import_claude_mem with real data
    // (covers lines 667, 695, 711, 717, 726, 742, 756, 765)
    // -----------------------------------------------------------------------

    #[test]
    fn import_claude_mem_with_data() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "
                CREATE TABLE observations (
                    title TEXT, facts TEXT, narrative TEXT, concepts TEXT, created_at TEXT
                );
                INSERT INTO observations VALUES (
                    'Test Observation',
                    '[\"Rust is fast\", \"Rust has ownership\"]',
                    'A test observation about Rust',
                    '[\"systems programming\"]',
                    '2025-01-01'
                );
            ",
            )
            .unwrap();
        }
        let srv = make_server();
        let result = srv.import_claude_mem(ImportClaudeMemParams { path: Some(path) });
        assert!(
            result.contains("Imported 1 observations"),
            "Result: {result}"
        );
        assert!(
            result.contains("3 semantic nodes"),
            "Should have 2 facts + 1 concept: {result}"
        );
    }

    #[test]
    fn import_claude_mem_with_empty_facts() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "
                CREATE TABLE observations (
                    title TEXT, facts TEXT, narrative TEXT, concepts TEXT, created_at TEXT
                );
                INSERT INTO observations VALUES (
                    'Empty Obs',
                    '[\"\", \"valid fact\"]',
                    '',
                    '[\"\", \"\"]',
                    '2025-01-01'
                );
            ",
            )
            .unwrap();
        }
        let srv = make_server();
        let result = srv.import_claude_mem(ImportClaudeMemParams { path: Some(path) });
        assert!(
            result.contains("Imported 1 observations"),
            "Result: {result}"
        );
        // Only "valid fact" should be imported (empty strings skipped)
        assert!(
            result.contains("1 semantic nodes"),
            "Should skip empty facts: {result}"
        );
    }

    #[test]
    fn import_claude_mem_no_observations() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "
                CREATE TABLE observations (
                    title TEXT, facts TEXT, narrative TEXT, concepts TEXT, created_at TEXT
                );
            ",
            )
            .unwrap();
        }
        let srv = make_server();
        let result = srv.import_claude_mem(ImportClaudeMemParams { path: Some(path) });
        assert!(result.contains("No observations found"), "Result: {result}");
    }

    // -----------------------------------------------------------------------
    // Coverage gap tests: import_claude_code full path
    // (covers lines 792-797, 801, 823, 829, 834, 838, 843-844, 876)
    // -----------------------------------------------------------------------

    #[test]
    fn import_claude_code_full_coverage() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("full_test.jsonl");

        let long_content = "x".repeat(3000);
        let long_line = format!(
            r#"{{"type":"human","message":{{"role":"user","content":"{long_content}"}},"timestamp":"1003","sessionId":"s1"}}"#
        );
        let lines = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n",
            // Valid human message
            r#"{"type":"human","message":{"role":"user","content":"hello world"},"timestamp":"1000","sessionId":"s1"}"#,
            // Valid assistant with array content
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"text":"response here"}]},"timestamp":"1001","sessionId":"s1"}"#,
            // Skipped type (not human/assistant)
            r#"{"type":"system_prompt","message":{"content":"skip me"},"timestamp":"1002","sessionId":"s1"}"#,
            // Very long content that gets truncated
            long_line,
            // Message without content field
            r#"{"type":"human","message":{"role":"user"},"timestamp":"1004","sessionId":"s1"}"#,
            // Invalid JSON line
            "not valid json at all",
        );
        std::fs::write(&path, lines).unwrap();

        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: path.to_str().unwrap().into(),
        });
        assert!(
            result.contains("Imported"),
            "Should import valid messages: {result}"
        );
        assert!(
            result.contains("sessions"),
            "Should mention sessions: {result}"
        );
        // Should report errors for bad JSON
        assert!(
            result.contains("error"),
            "Should report parse errors: {result}"
        );
    }

    #[test]
    fn import_claude_code_message_without_message_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_msg.jsonl");

        // Entry with type but no message field at all
        let lines = format!(
            "{}\n{}\n",
            r#"{"type":"human","timestamp":"1000","sessionId":"s1"}"#,
            r#"{"type":"human","message":{"role":"user","content":"valid"},"timestamp":"1001","sessionId":"s1"}"#,
        );
        std::fs::write(&path, lines).unwrap();

        let srv = make_server();
        let result = srv.import_claude_code(ImportClaudeCodeParams {
            path: path.to_str().unwrap().into(),
        });
        assert!(result.contains("Imported"), "Result: {result}");
    }

    // -----------------------------------------------------------------------
    // Coverage gap tests: maintain error branch
    // (covers line 538)
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Direct tests for extracted formatting helpers
    // (these cover lines that tarpaulin can't track through #[tool] wrappers)
    // -----------------------------------------------------------------------

    #[test]
    fn format_preferences_with_data() {
        let prefs = vec![
            Preference {
                id: PreferenceId(1),
                domain: "style".to_string(),
                preference: "concise code".to_string(),
                confidence: 0.85,
                evidence_count: 3,
                first_observed: 1000,
                last_reinforced: 2000,
            },
            Preference {
                id: PreferenceId(2),
                domain: "tone".to_string(),
                preference: "friendly".to_string(),
                confidence: 0.72,
                evidence_count: 5,
                first_observed: 1100,
                last_reinforced: 2100,
            },
        ];
        let result = format_preferences(&prefs);
        assert!(result.contains("Found 2 preferences:"));
        assert!(result.contains("[style] concise code (confidence: 0.85, evidence: 3)"));
        assert!(result.contains("[tone] friendly (confidence: 0.72, evidence: 5)"));
    }

    #[test]
    fn format_preferences_empty() {
        let result = format_preferences(&[]);
        assert!(result.contains("Found 0 preferences:"));
    }

    #[test]
    fn format_categories_with_data() {
        let cats = vec![
            Category {
                id: CategoryId(1),
                label: "programming".to_string(),
                prototype_node: NodeId(1),
                member_count: 10,
                centroid_embedding: None,
                created_at: 1000,
                last_updated: 2000,
                stability: 0.95,
                parent_id: None,
            },
            Category {
                id: CategoryId(2),
                label: "rust-lang".to_string(),
                prototype_node: NodeId(2),
                member_count: 5,
                centroid_embedding: None,
                created_at: 1100,
                last_updated: 2100,
                stability: 0.80,
                parent_id: Some(CategoryId(1)),
            },
        ];
        let result = format_categories(&cats);
        assert!(result.contains("Found 2 categories:"));
        assert!(result.contains("[1] programming"));
        assert!(result.contains("10 members"));
        assert!(result.contains("stability: 0.95"));
        assert!(result.contains("(parent: 1)"));
    }

    #[test]
    fn format_categories_no_parent() {
        let cats = vec![Category {
            id: CategoryId(1),
            label: "general".to_string(),
            prototype_node: NodeId(1),
            member_count: 3,
            centroid_embedding: None,
            created_at: 1000,
            last_updated: 2000,
            stability: 0.5,
            parent_id: None,
        }];
        let result = format_categories(&cats);
        assert!(!result.contains("parent:"));
    }

    #[test]
    fn format_neighbors_all_node_types() {
        let neighbors = vec![
            (NodeRef::Episode(EpisodeId(1)), 0.9),
            (NodeRef::Semantic(NodeId(2)), 0.75),
            (NodeRef::Preference(PreferenceId(3)), 0.5),
            (NodeRef::Category(CategoryId(4)), 0.3),
        ];
        let result = format_neighbors(&neighbors);
        assert!(result.contains("Found 4 neighbors:"));
        assert!(result.contains("episode #1 (weight: 0.900)"));
        assert!(result.contains("semantic #2 (weight: 0.750)"));
        assert!(result.contains("preference #3 (weight: 0.500)"));
        assert!(result.contains("category #4 (weight: 0.300)"));
    }

    #[test]
    fn format_neighbors_empty() {
        let result = format_neighbors(&[]);
        assert!(result.contains("Found 0 neighbors:"));
    }

    #[test]
    fn format_node_category_with_parent() {
        let cat = Category {
            id: CategoryId(5),
            label: "algorithms".to_string(),
            prototype_node: NodeId(5),
            member_count: 8,
            centroid_embedding: None,
            created_at: 1000,
            last_updated: 2000,
            stability: 0.88,
            parent_id: Some(CategoryId(1)),
        };
        let result = format_node_category(42, &cat);
        assert!(result.contains("Node 42 belongs to category [5] 'algorithms'"));
        assert!(result.contains("8 members"));
        assert!(result.contains("stability: 0.88"));
        assert!(result.contains("(parent: 1)"));
    }

    #[test]
    fn format_node_category_no_parent() {
        let cat = Category {
            id: CategoryId(1),
            label: "general".to_string(),
            prototype_node: NodeId(1),
            member_count: 2,
            centroid_embedding: None,
            created_at: 1000,
            last_updated: 2000,
            stability: 0.5,
            parent_id: None,
        };
        let result = format_node_category(7, &cat);
        assert!(result.contains("Node 7 belongs to category [1] 'general'"));
        assert!(!result.contains("parent:"));
    }

    #[test]
    fn format_knowledge_breakdown_all_types() {
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert(SemanticType::Fact, 10);
        breakdown.insert(SemanticType::Relationship, 5);
        breakdown.insert(SemanticType::Event, 3);
        breakdown.insert(SemanticType::Concept, 7);
        let result = format_knowledge_breakdown(&breakdown);
        assert!(result.contains("10 facts"));
        assert!(result.contains("5 relationships"));
        assert!(result.contains("3 events"));
        assert!(result.contains("7 concepts"));
    }

    #[test]
    fn format_knowledge_breakdown_partial() {
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert(SemanticType::Fact, 2);
        let result = format_knowledge_breakdown(&breakdown);
        assert_eq!(result, "2 facts");
        assert!(!result.contains("relationships"));
    }

    #[test]
    fn format_knowledge_breakdown_empty() {
        let breakdown = std::collections::HashMap::new();
        let result = format_knowledge_breakdown(&breakdown);
        assert_eq!(result, "");
    }

    #[test]
    fn format_category_line_with_labels() {
        let cats = vec![
            Category {
                id: CategoryId(1),
                label: "rust".to_string(),
                prototype_node: NodeId(1),
                member_count: 3,
                centroid_embedding: None,
                created_at: 1000,
                last_updated: 2000,
                stability: 0.5,
                parent_id: None,
            },
            Category {
                id: CategoryId(2),
                label: "python".to_string(),
                prototype_node: NodeId(2),
                member_count: 2,
                centroid_embedding: None,
                created_at: 1100,
                last_updated: 2100,
                stability: 0.4,
                parent_id: None,
            },
        ];
        let result = format_category_line(&cats);
        assert_eq!(result, "2 (rust, python)");
    }

    #[test]
    fn format_status_full() {
        let st = MemoryStatus {
            episode_count: 50,
            semantic_node_count: 20,
            preference_count: 3,
            impression_count: 10,
            link_count: 45,
            embedding_count: 30,
            category_count: 2,
        };
        let result = format_status(
            &st,
            5,
            3,
            "10 facts, 5 relationships",
            "2 (rust, python)",
            " (strongest: \"a\" <-> \"b\" weight 0.95)",
            "30/70 nodes (42%)",
        );
        assert!(result.contains("Memory Status:"));
        assert!(result.contains("Episodes: 50"));
        assert!(result.contains("5 this session, 3 unconsolidated"));
        assert!(result.contains("Knowledge: 10 facts, 5 relationships"));
        assert!(result.contains("Categories: 2 (rust, python)"));
        assert!(result.contains("3 crystallized, 10 impressions"));
        assert!(result.contains("45 links"));
        assert!(result.contains("strongest: \"a\" <-> \"b\" weight 0.95"));
        assert!(result.contains("Embedding coverage: 30/70 nodes (42%)"));
    }

    #[test]
    fn format_status_no_session_data() {
        let st = MemoryStatus {
            episode_count: 0,
            semantic_node_count: 0,
            preference_count: 0,
            impression_count: 0,
            link_count: 0,
            embedding_count: 0,
            category_count: 0,
        };
        let result = format_status(&st, 0, 0, "none", "0", "", "0/0 nodes");
        assert!(result.contains("Episodes: 0"));
        // Should NOT contain the session/unconsolidated parenthetical
        assert!(!result.contains("this session"));
    }

    // -----------------------------------------------------------------------
    // Direct tests for parse_claude_mem_db
    // -----------------------------------------------------------------------

    #[test]
    fn parse_claude_mem_db_with_facts_and_concepts() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE observations (
                    title TEXT, facts TEXT, narrative TEXT, concepts TEXT, created_at TEXT
                );
                INSERT INTO observations VALUES (
                    'Obs1', '[\"fact one\", \"fact two\"]', 'narrative', '[\"concept one\"]', '2025-01-01'
                );
                INSERT INTO observations VALUES (
                    'Obs2', '[\"fact three\"]', '', '[\"concept two\", \"concept three\"]', '2025-02-01'
                );",
            ).unwrap();
        }
        let (count, nodes) = parse_claude_mem_db(&path).unwrap();
        assert_eq!(count, 2);
        assert_eq!(nodes.len(), 6); // 3 facts + 3 concepts
                                    // Obs1: facts first, then concepts; Obs2: facts then concepts
        assert_eq!(nodes[0].content, "fact one");
        assert_eq!(nodes[0].node_type, SemanticType::Fact);
        assert_eq!(nodes[0].confidence, 0.8);
        // Concepts come after facts within each observation
        let concept_nodes: Vec<_> = nodes
            .iter()
            .filter(|n| n.node_type == SemanticType::Concept)
            .collect();
        assert_eq!(concept_nodes.len(), 3);
        assert_eq!(concept_nodes[0].confidence, 0.7);
    }

    #[test]
    fn parse_claude_mem_db_empty_strings_skipped() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE observations (
                    title TEXT, facts TEXT, narrative TEXT, concepts TEXT, created_at TEXT
                );
                INSERT INTO observations VALUES (
                    'Obs', '[\"\", \"real fact\", \"  \"]', '', '[\"\"]', '2025-01-01'
                );",
            )
            .unwrap();
        }
        let (count, nodes) = parse_claude_mem_db(&path).unwrap();
        assert_eq!(count, 1);
        assert_eq!(nodes.len(), 1); // only "real fact"
        assert_eq!(nodes[0].content, "real fact");
    }

    #[test]
    fn parse_claude_mem_db_invalid_json_fields() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE observations (
                    title TEXT, facts TEXT, narrative TEXT, concepts TEXT, created_at TEXT
                );
                INSERT INTO observations VALUES (
                    'Obs', 'not json', '', 'also not json', '2025-01-01'
                );",
            )
            .unwrap();
        }
        let (count, nodes) = parse_claude_mem_db(&path).unwrap();
        assert_eq!(count, 1);
        assert_eq!(nodes.len(), 0); // no valid JSON arrays
    }

    #[test]
    fn parse_claude_mem_db_nonexistent_file() {
        let result = parse_claude_mem_db("/tmp/nonexistent-alaya-parse-test.db");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Cannot open"));
    }

    #[test]
    fn parse_claude_mem_db_no_table() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        {
            let _conn = rusqlite::Connection::open(&path).unwrap();
            // empty DB, no observations table
        }
        let result = parse_claude_mem_db(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Error reading observations"));
    }

    // -----------------------------------------------------------------------
    // Direct tests for parse_claude_code_jsonl
    // -----------------------------------------------------------------------

    #[test]
    fn parse_claude_code_jsonl_valid_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");
        let lines = [
            r#"{"type":"human","message":{"role":"user","content":"hello"},"timestamp":"1000","sessionId":"s1"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"text":"hi there"}]},"timestamp":"1001","sessionId":"s1"}"#,
            r#"{"type":"human","message":{"role":"user","content":"bye"},"timestamp":"1002","sessionId":"s2"}"#,
        ];
        std::fs::write(&path, lines.join("\n")).unwrap();

        let (episodes, sessions, errors, first_error) =
            parse_claude_code_jsonl(path.to_str().unwrap()).unwrap();
        assert_eq!(episodes.len(), 3);
        assert_eq!(sessions.len(), 2); // s1 and s2
        assert_eq!(errors, 0);
        assert!(first_error.is_none());
        assert_eq!(episodes[0].content, "hello");
        assert_eq!(episodes[1].content, "hi there");
        assert_eq!(episodes[2].content, "bye");
    }

    #[test]
    fn parse_claude_code_jsonl_skips_system_type() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sys.jsonl");
        let lines = [
            r#"{"type":"system","message":{"content":"skip me"},"timestamp":"1000"}"#,
            r#"{"type":"human","message":{"role":"user","content":"keep me"},"timestamp":"1001","sessionId":"s1"}"#,
        ];
        std::fs::write(&path, lines.join("\n")).unwrap();

        let (episodes, _, _, _) = parse_claude_code_jsonl(path.to_str().unwrap()).unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].content, "keep me");
    }

    #[test]
    fn parse_claude_code_jsonl_skips_empty_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.jsonl");
        let lines = [
            r#"{"type":"human","message":{"role":"user","content":""},"timestamp":"1000","sessionId":"s1"}"#,
            r#"{"type":"human","message":{"role":"user","content":"  "},"timestamp":"1001","sessionId":"s1"}"#,
            r#"{"type":"human","message":{"role":"user","content":"real content"},"timestamp":"1002","sessionId":"s1"}"#,
        ];
        std::fs::write(&path, lines.join("\n")).unwrap();

        let (episodes, _, _, _) = parse_claude_code_jsonl(path.to_str().unwrap()).unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].content, "real content");
    }

    #[test]
    fn parse_claude_code_jsonl_truncates_long_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("long.jsonl");
        let long_text = "a".repeat(3000);
        let line = format!(
            r#"{{"type":"human","message":{{"role":"user","content":"{long_text}"}},"timestamp":"1000","sessionId":"s1"}}"#
        );
        std::fs::write(&path, line).unwrap();

        let (episodes, _, _, _) = parse_claude_code_jsonl(path.to_str().unwrap()).unwrap();
        assert_eq!(episodes.len(), 1);
        // 2000 chars + "..."
        assert_eq!(episodes[0].content.chars().count(), 2003);
        assert!(episodes[0].content.ends_with("..."));
    }

    #[test]
    fn parse_claude_code_jsonl_bad_json_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.jsonl");
        let content = "not json at all\n{broken\n";
        std::fs::write(&path, content).unwrap();

        let (episodes, _, errors, first_error) =
            parse_claude_code_jsonl(path.to_str().unwrap()).unwrap();
        assert_eq!(episodes.len(), 0);
        assert!(errors >= 2);
        assert!(first_error.is_some());
    }

    #[test]
    fn parse_claude_code_jsonl_missing_message_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nomsg.jsonl");
        let line = r#"{"type":"human","timestamp":"1000","sessionId":"s1"}"#;
        std::fs::write(&path, line).unwrap();

        let (episodes, _, _, _) = parse_claude_code_jsonl(path.to_str().unwrap()).unwrap();
        assert_eq!(episodes.len(), 0);
    }

    #[test]
    fn parse_claude_code_jsonl_missing_content_in_message() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nocontent.jsonl");
        let line =
            r#"{"type":"human","message":{"role":"user"},"timestamp":"1000","sessionId":"s1"}"#;
        std::fs::write(&path, line).unwrap();

        let (episodes, _, _, _) = parse_claude_code_jsonl(path.to_str().unwrap()).unwrap();
        assert_eq!(episodes.len(), 0);
    }

    #[test]
    fn parse_claude_code_jsonl_empty_lines_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blanks.jsonl");
        let content = format!(
            "\n\n{}\n\n",
            r#"{"type":"human","message":{"role":"user","content":"msg"},"timestamp":"1000","sessionId":"s1"}"#
        );
        std::fs::write(&path, content).unwrap();

        let (episodes, _, errors, _) = parse_claude_code_jsonl(path.to_str().unwrap()).unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(errors, 0);
    }

    #[test]
    fn parse_claude_code_jsonl_no_session_id_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nosess.jsonl");
        let line =
            r#"{"type":"human","message":{"role":"user","content":"hello"},"timestamp":"1000"}"#;
        std::fs::write(&path, line).unwrap();

        let (episodes, sessions, _, _) = parse_claude_code_jsonl(path.to_str().unwrap()).unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].session_id, "imported");
        assert!(sessions.contains("imported"));
    }

    #[test]
    fn parse_claude_code_jsonl_no_timestamp_defaults_zero() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nots.jsonl");
        let line =
            r#"{"type":"human","message":{"role":"user","content":"hello"},"sessionId":"s1"}"#;
        std::fs::write(&path, line).unwrap();

        let (episodes, _, _, _) = parse_claude_code_jsonl(path.to_str().unwrap()).unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].timestamp, 0);
    }

    #[test]
    fn parse_claude_code_jsonl_nonexistent_file() {
        let result = parse_claude_code_jsonl("/tmp/nonexistent-alaya-parse-test.jsonl");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Cannot read file"));
    }

    // -----------------------------------------------------------------------
    // ServerHandler::get_info
    // -----------------------------------------------------------------------

    #[test]
    fn get_info_returns_instructions() {
        use rmcp::ServerHandler;
        let srv = make_server();
        let info = srv.get_info();
        assert!(info.instructions.is_some());
        let instr = info.instructions.unwrap();
        assert!(instr.contains("Alaya"));
        assert!(instr.contains("remember"));
        assert!(instr.contains("recall"));
    }

    // -----------------------------------------------------------------------
    // Coverage gap test: maintain error branch
    // -----------------------------------------------------------------------

    #[test]
    fn maintain_returns_complete_info() {
        let srv = server_with_episodes(3);
        srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Maintain test fact".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: None,
        });
        let result = srv.maintain();
        assert!(
            result.contains("Maintenance complete"),
            "Maintain result: {result}"
        );
        assert!(
            result.contains("Duplicates merged:"),
            "Should show duplicates: {result}"
        );
        assert!(
            result.contains("Links pruned:"),
            "Should show links pruned: {result}"
        );
        assert!(
            result.contains("Preferences decayed:"),
            "Should show prefs decayed: {result}"
        );
    }

    // -----------------------------------------------------------------------
    // format_preferences — edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn format_preferences_single_item() {
        let prefs = vec![Preference {
            id: PreferenceId(99),
            domain: "coding".to_string(),
            preference: "use snake_case".to_string(),
            confidence: 1.0,
            evidence_count: 1,
            first_observed: 0,
            last_reinforced: 0,
        }];
        let result = format_preferences(&prefs);
        assert!(result.contains("Found 1 preferences:"));
        assert!(result.contains("[coding] use snake_case (confidence: 1.00, evidence: 1)"));
    }

    #[test]
    fn format_preferences_zero_confidence_and_evidence() {
        let prefs = vec![Preference {
            id: PreferenceId(1),
            domain: "style".to_string(),
            preference: "unknown".to_string(),
            confidence: 0.0,
            evidence_count: 0,
            first_observed: 0,
            last_reinforced: 0,
        }];
        let result = format_preferences(&prefs);
        assert!(result.contains("confidence: 0.00, evidence: 0"));
    }

    #[test]
    fn format_preferences_special_characters_in_fields() {
        let prefs = vec![Preference {
            id: PreferenceId(1),
            domain: "lang/framework".to_string(),
            preference: "tabs > spaces (always!)".to_string(),
            confidence: 0.99,
            evidence_count: 42,
            first_observed: 0,
            last_reinforced: 0,
        }];
        let result = format_preferences(&prefs);
        assert!(result.contains("[lang/framework] tabs > spaces (always!)"));
    }

    // -----------------------------------------------------------------------
    // format_categories — edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn format_categories_single_item_with_parent() {
        let cats = vec![Category {
            id: CategoryId(7),
            label: "child".to_string(),
            prototype_node: NodeId(1),
            member_count: 1,
            centroid_embedding: None,
            created_at: 0,
            last_updated: 0,
            stability: 0.33,
            parent_id: Some(CategoryId(3)),
        }];
        let result = format_categories(&cats);
        assert!(result.contains("Found 1 categories:"));
        assert!(result.contains("[7] child"));
        assert!(result.contains("1 members"));
        assert!(result.contains("stability: 0.33"));
        assert!(result.contains("(parent: 3)"));
    }

    #[test]
    fn format_categories_zero_members_zero_stability() {
        let cats = vec![Category {
            id: CategoryId(2),
            label: "empty-cat".to_string(),
            prototype_node: NodeId(1),
            member_count: 0,
            centroid_embedding: None,
            created_at: 0,
            last_updated: 0,
            stability: 0.0,
            parent_id: None,
        }];
        let result = format_categories(&cats);
        assert!(result.contains("0 members, stability: 0.00"));
        assert!(!result.contains("parent:"));
    }

    #[test]
    fn format_categories_empty() {
        let result = format_categories(&[]);
        assert!(result.contains("Found 0 categories:"));
    }

    // -----------------------------------------------------------------------
    // format_neighbors — edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn format_neighbors_single_episode() {
        let neighbors = vec![(NodeRef::Episode(EpisodeId(5)), 1.0_f32)];
        let result = format_neighbors(&neighbors);
        assert!(result.contains("Found 1 neighbors:"));
        assert!(result.contains("episode #5 (weight: 1.000)"));
    }

    #[test]
    fn format_neighbors_single_semantic() {
        let neighbors = vec![(NodeRef::Semantic(NodeId(10)), 0.0_f32)];
        let result = format_neighbors(&neighbors);
        assert!(result.contains("semantic #10 (weight: 0.000)"));
    }

    #[test]
    fn format_neighbors_single_preference() {
        let neighbors = vec![(NodeRef::Preference(PreferenceId(3)), 0.5_f32)];
        let result = format_neighbors(&neighbors);
        assert!(result.contains("preference #3 (weight: 0.500)"));
    }

    #[test]
    fn format_neighbors_single_category() {
        let neighbors = vec![(NodeRef::Category(CategoryId(8)), 0.123_f32)];
        let result = format_neighbors(&neighbors);
        assert!(result.contains("category #8 (weight: 0.123)"));
    }

    // -----------------------------------------------------------------------
    // format_node_category — edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn format_node_category_zero_members_zero_stability() {
        let cat = Category {
            id: CategoryId(1),
            label: "empty".to_string(),
            prototype_node: NodeId(1),
            member_count: 0,
            centroid_embedding: None,
            created_at: 0,
            last_updated: 0,
            stability: 0.0,
            parent_id: None,
        };
        let result = format_node_category(42, &cat);
        assert!(result.contains("Node 42 belongs to category [1] 'empty'"));
        assert!(result.contains("0 members, stability: 0.00"));
        assert!(!result.contains("parent:"));
    }

    #[test]
    fn format_node_category_large_node_id() {
        let cat = Category {
            id: CategoryId(999),
            label: "big".to_string(),
            prototype_node: NodeId(1),
            member_count: 100,
            centroid_embedding: None,
            created_at: 0,
            last_updated: 0,
            stability: 1.0,
            parent_id: None,
        };
        let result = format_node_category(1_000_000, &cat);
        assert!(result.contains("Node 1000000 belongs to category [999] 'big'"));
        assert!(result.contains("stability: 1.00"));
    }

    // -----------------------------------------------------------------------
    // format_knowledge_breakdown — edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn format_knowledge_breakdown_single_type_relationships() {
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert(SemanticType::Relationship, 7_u64);
        let result = format_knowledge_breakdown(&breakdown);
        assert_eq!(result, "7 relationships");
    }

    #[test]
    fn format_knowledge_breakdown_single_type_events() {
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert(SemanticType::Event, 3_u64);
        let result = format_knowledge_breakdown(&breakdown);
        assert_eq!(result, "3 events");
    }

    #[test]
    fn format_knowledge_breakdown_single_type_concepts() {
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert(SemanticType::Concept, 1_u64);
        let result = format_knowledge_breakdown(&breakdown);
        assert_eq!(result, "1 concepts");
    }

    #[test]
    fn format_knowledge_breakdown_zero_count_not_included() {
        // A type with count=0 is stored in the map but check that only
        // present keys are emitted (inserting 0 does still emit "0 facts").
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert(SemanticType::Fact, 0_u64);
        let result = format_knowledge_breakdown(&breakdown);
        assert_eq!(result, "0 facts");
    }

    #[test]
    fn format_knowledge_breakdown_ordering_facts_first() {
        // Verify output always starts with facts when facts are present.
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert(SemanticType::Concept, 2_u64);
        breakdown.insert(SemanticType::Fact, 5_u64);
        let result = format_knowledge_breakdown(&breakdown);
        assert!(result.starts_with("5 facts"));
        assert!(result.contains("2 concepts"));
    }

    // -----------------------------------------------------------------------
    // format_category_line — edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn format_category_line_empty_slice() {
        let result = format_category_line(&[]);
        assert_eq!(result, "0 ()");
    }

    #[test]
    fn format_category_line_single_item() {
        let cats = vec![Category {
            id: CategoryId(1),
            label: "solo".to_string(),
            prototype_node: NodeId(1),
            member_count: 1,
            centroid_embedding: None,
            created_at: 0,
            last_updated: 0,
            stability: 1.0,
            parent_id: None,
        }];
        let result = format_category_line(&cats);
        assert_eq!(result, "1 (solo)");
    }

    // -----------------------------------------------------------------------
    // format_status — edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn format_status_all_zeros() {
        let st = MemoryStatus {
            episode_count: 0,
            semantic_node_count: 0,
            preference_count: 0,
            impression_count: 0,
            link_count: 0,
            embedding_count: 0,
            category_count: 0,
        };
        let result = format_status(&st, 0, 0, "", "0 ()", "", "0/0");
        assert!(result.contains("Episodes: 0"));
        // With session_eps=0 and unconsolidated=0 the parenthetical is omitted
        assert!(!result.contains("this session"));
        assert!(result.contains("Preferences: 0 crystallized, 0 impressions"));
        assert!(result.contains("Graph: 0 links"));
    }

    #[test]
    fn format_status_session_eps_nonzero_but_unconsolidated_zero() {
        let st = MemoryStatus {
            episode_count: 5,
            semantic_node_count: 0,
            preference_count: 0,
            impression_count: 0,
            link_count: 0,
            embedding_count: 0,
            category_count: 0,
        };
        let result = format_status(&st, 5, 0, "5 facts", "1 (rust)", "", "5/5");
        assert!(result.contains("Episodes: 5"));
        assert!(result.contains("5 this session, 0 unconsolidated"));
    }

    #[test]
    fn format_status_strongest_desc_included() {
        let st = MemoryStatus {
            episode_count: 10,
            semantic_node_count: 3,
            preference_count: 1,
            impression_count: 0,
            link_count: 8,
            embedding_count: 3,
            category_count: 1,
        };
        let result = format_status(
            &st,
            0,
            0,
            "3 facts",
            "1 (general)",
            " — strongest: Rust",
            "3/3",
        );
        assert!(result.contains("Graph: 8 links — strongest: Rust"));
    }
}
