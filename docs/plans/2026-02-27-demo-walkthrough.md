# Demo Walkthrough Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `examples/demo.rs` -- a 6-chapter scripted walkthrough that showcases Alaya's neuroscience-inspired memory features (episodic store, Hebbian graph, consolidation, perfuming, transformation, forgetting) with a rule-based KeywordProvider.

**Architecture:** Single-file example binary. A `KeywordProvider` struct implements `ConsolidationProvider` using keyword matching (no LLM needed). Six chapter functions run sequentially, each printing annotated output. The `main()` function creates an in-memory AlayaStore and passes it through all chapters.

**Tech Stack:** Rust, alaya (this crate), no additional dependencies.

---

### Task 1: Create examples/demo.rs with KeywordProvider + Chapter 1

**Files:**
- Create: `examples/demo.rs`

**Step 1: Create examples directory**

Run: `mkdir -p examples`

**Step 2: Write the full demo.rs**

Create `examples/demo.rs` with the KeywordProvider, helper functions, main(), and Chapter 1:

```rust
//! # Alaya Demo: A Scripted Walkthrough
//!
//! This demo walks through Alaya's six core capabilities:
//! 1. Episodic Memory (store + query)
//! 2. Hebbian Graph (temporal links + co-retrieval + spreading activation)
//! 3. Consolidation (episodic -> semantic knowledge)
//! 4. Perfuming (vasana -> preference crystallization)
//! 5. Transformation (dedup, prune, decay)
//! 6. Forgetting (Bjork dual-strength model)
//!
//! Run: `cargo run --example demo`

use alaya::{
    AlayaStore, AlayaError, ConsolidationProvider, EpisodeContext, EpisodeId, Interaction,
    KnowledgeFilter, NewEpisode, NewImpression, NewSemanticNode, NodeRef, NoOpProvider,
    Query, Role, SemanticNode, SemanticType, Episode,
};

// ============================================================================
// KeywordProvider — rule-based ConsolidationProvider (no LLM needed)
// ============================================================================

/// A simple keyword-matching provider that extracts knowledge and impressions
/// from text using pattern matching. Replace with an LLM-backed provider
/// for production use.
struct KeywordProvider;

impl ConsolidationProvider for KeywordProvider {
    fn extract_knowledge(&self, episodes: &[Episode]) -> alaya::Result<Vec<NewSemanticNode>> {
        let mut nodes = Vec::new();
        let ep_ids: Vec<EpisodeId> = episodes.iter().map(|e| e.id).collect();
        let all_text: String = episodes.iter().map(|e| e.content.as_str()).collect::<Vec<_>>().join(" ");

        // Detect technology relationships
        let techs = ["Rust", "tokio", "SQLite", "rusqlite", "async"];
        let found: Vec<&str> = techs.iter().filter(|t| all_text.contains(*t)).copied().collect();
        if found.len() >= 2 {
            nodes.push(NewSemanticNode {
                content: format!("User works with {}", found.join(", ")),
                node_type: SemanticType::Relationship,
                confidence: 0.75,
                source_episodes: ep_ids.clone(),
                embedding: None,
            });
        }

        // Detect preference-like facts
        for ep in episodes {
            let lower = ep.content.to_lowercase();
            if lower.contains("prefer") || lower.contains("enjoy") || lower.contains("love") {
                nodes.push(NewSemanticNode {
                    content: ep.content.clone(),
                    node_type: SemanticType::Fact,
                    confidence: 0.65,
                    source_episodes: vec![ep.id],
                    embedding: None,
                });
            }
        }

        // Detect project-level concepts
        if all_text.contains("memory") && all_text.contains("agent") {
            nodes.push(NewSemanticNode {
                content: "User is building AI agent memory systems".to_string(),
                node_type: SemanticType::Concept,
                confidence: 0.70,
                source_episodes: ep_ids,
                embedding: None,
            });
        }

        Ok(nodes)
    }

    fn extract_impressions(&self, interaction: &Interaction) -> alaya::Result<Vec<NewImpression>> {
        let mut impressions = Vec::new();
        let text = interaction.text.to_lowercase();

        if text.contains("concise") || text.contains("brief") || text.contains("direct") {
            impressions.push(NewImpression {
                domain: "communication_style".to_string(),
                observation: "prefers concise, direct answers".to_string(),
                valence: 0.8,
            });
        }
        if text.contains("example") || text.contains("code") || text.contains("show me") {
            impressions.push(NewImpression {
                domain: "learning_style".to_string(),
                observation: "prefers code examples over explanations".to_string(),
                valence: 0.9,
            });
        }
        if text.contains("practical") || text.contains("real-world") {
            impressions.push(NewImpression {
                domain: "learning_style".to_string(),
                observation: "prefers practical over theoretical".to_string(),
                valence: 0.7,
            });
        }
        if text.contains("small") || text.contains("focused") || text.contains("modular") {
            impressions.push(NewImpression {
                domain: "code_style".to_string(),
                observation: "prefers small, focused modules".to_string(),
                valence: 0.8,
            });
        }

        Ok(impressions)
    }

    fn detect_contradiction(&self, _a: &SemanticNode, _b: &SemanticNode) -> alaya::Result<bool> {
        Ok(false)
    }
}

// ============================================================================
// Output helpers
// ============================================================================

fn print_chapter(n: u32, title: &str, subtitle: &str) {
    println!();
    println!("  ═══════════════════════════════════════════════════");
    println!("   Chapter {}: {} — {}", n, title, subtitle);
    println!("  ═══════════════════════════════════════════════════");
    println!();
}

fn print_status(store: &AlayaStore) {
    let s = store.status().unwrap();
    println!("  MemoryStatus:");
    println!("    episodes:       {}", s.episode_count);
    println!("    semantic_nodes: {}", s.semantic_node_count);
    println!("    preferences:    {}", s.preference_count);
    println!("    impressions:    {}", s.impression_count);
    println!("    links:          {}", s.link_count);
    println!("    embeddings:     {}", s.embedding_count);
    println!();
}

fn print_insight(text: &str) {
    println!("  \u{2605} Insight: {}", text);
    println!();
}

// ============================================================================
// Demo data
// ============================================================================

fn demo_episodes() -> Vec<(&'static str, &'static str, i64)> {
    vec![
        // (content, session_id, timestamp)
        // Session 1: Learning Rust
        ("I'm learning Rust and really enjoying the borrow checker. It catches so many bugs at compile time.", "day-1", 1000),
        ("Async programming in Rust with tokio is powerful but has a steep learning curve.", "day-1", 1100),
        ("I prefer using SQLite for embedded databases. It's simple and reliable.", "day-1", 1200),
        ("The Rust type system is amazing. Pattern matching with enums is my favorite feature.", "day-1", 1300),
        // Session 2: Building a project
        ("I'm building a memory engine for AI agents using rusqlite.", "day-2", 2000),
        ("Performance matters a lot for my use case. I need sub-millisecond queries.", "day-2", 2100),
        ("Can you show me code examples? I learn better from reading code than explanations.", "day-2", 2200),
        ("I always structure my projects with small, focused modules. Each module does one thing.", "day-2", 2300),
    ]
}

fn perfuming_interactions() -> Vec<&'static str> {
    vec![
        "I prefer concise answers, not long explanations.",
        "Show me a code example instead of describing the algorithm.",
        "Give me the direct answer please, keep it brief.",
        "I like seeing practical, real-world code patterns.",
        "Can you be more concise? Just the key points.",
        "Another code example would help me understand this better.",
        "I want practical advice, not theoretical background.",
    ]
}

// ============================================================================
// Chapters
// ============================================================================

fn chapter_1_episodic(store: &AlayaStore) -> Vec<EpisodeId> {
    print_chapter(1, "Episodic Memory", "Store + Query");

    println!("  Storing 8 conversation episodes across 2 sessions...");
    println!();

    let episodes = demo_episodes();
    let mut ids = Vec::new();
    let mut prev_id: Option<EpisodeId> = None;
    let mut last_session = "";

    for (content, session, ts) in &episodes {
        // Reset temporal chain when session changes
        if *session != last_session {
            prev_id = None;
            last_session = session;
        }

        let mut ctx = EpisodeContext::default();
        ctx.preceding_episode = prev_id;

        let id = store.store_episode(&NewEpisode {
            content: content.to_string(),
            role: Role::User,
            session_id: session.to_string(),
            timestamp: *ts,
            context: ctx,
            embedding: None,
        }).unwrap();

        println!("    [{}] ep#{}: \"{}\"",
            session, id.0,
            if content.len() > 60 { &content[..60] } else { content });
        prev_id = Some(id);
        ids.push(id);
    }

    println!();
    print_status(store);

    // Query
    println!("  Querying: \"Rust async programming\"");
    let results = store.query(&Query::simple("Rust async programming")).unwrap();
    println!("  Found {} results:", results.len());
    for (i, mem) in results.iter().enumerate() {
        println!("    {}. [score {:.4}] \"{}\"",
            i + 1, mem.score,
            if mem.content.len() > 55 { format!("{}...", &mem.content[..55]) } else { mem.content.clone() });
    }
    println!();

    print_insight(
        "Episodic memory stores raw experiences with full context.\n\
         \x20 Like the hippocampus, it captures everything — retrieval\n\
         \x20 is handled by the hybrid BM25 + graph pipeline."
    );

    ids
}

fn chapter_2_hebbian(store: &AlayaStore, episode_ids: &[EpisodeId]) {
    print_chapter(2, "Hebbian Graph", "Co-Retrieval + Spreading Activation");

    let status = store.status().unwrap();
    println!("  Links created during episode storage: {}", status.link_count);
    println!("  (Temporal links chain episodes within each session)");
    println!();

    // Run overlapping queries to trigger co-retrieval links
    println!("  Running overlapping queries to trigger Hebbian learning...");
    let _ = store.query(&Query::simple("Rust borrow checker")).unwrap();
    let _ = store.query(&Query::simple("Rust type system")).unwrap();
    let _ = store.query(&Query::simple("SQLite embedded database")).unwrap();

    let status2 = store.status().unwrap();
    let new_links = status2.link_count - status.link_count;
    println!("  Co-retrieval links created: {}", new_links);
    println!("  (Memories retrieved together strengthen their connection)");
    println!();

    // Show spreading activation from first episode
    if let Some(&seed) = episode_ids.first() {
        println!("  Spreading activation from episode #{}:", seed.0);
        let neighbors = store.neighbors(NodeRef::Episode(seed), 2).unwrap();
        if neighbors.is_empty() {
            println!("    (No neighbors yet — graph needs more co-retrieval events)");
        } else {
            for (node, activation) in neighbors.iter().take(5) {
                println!("    {} #{}: activation {:.3}",
                    node.type_str(), node.id(), activation);
            }
        }
    }
    println!();

    print_insight(
        "Hebbian learning: 'neurons that fire together wire together.'\n\
         \x20 When memories are retrieved together, their link weight\n\
         \x20 grows: w += 0.1 * (1 - w). This creates an associative\n\
         \x20 network that mirrors how human memory clusters related ideas."
    );
}

fn chapter_3_consolidation(store: &AlayaStore) {
    print_chapter(3, "Consolidation", "Episodic \u{2192} Semantic (CLS Replay)");

    let provider = KeywordProvider;

    println!("  Running CLS replay on unconsolidated episodes...");
    let report = store.consolidate(&provider).unwrap();
    println!();
    println!("  ConsolidationReport:");
    println!("    episodes_processed: {}", report.episodes_processed);
    println!("    nodes_created:      {}", report.nodes_created);
    println!("    links_created:      {}", report.links_created);
    println!();

    // Show extracted knowledge
    let knowledge = store.knowledge(None).unwrap();
    if !knowledge.is_empty() {
        println!("  Extracted Knowledge:");
        for node in &knowledge {
            println!("    [{:?}] \"{}\" (confidence: {:.2})",
                node.node_type, node.content, node.confidence);
        }
    } else {
        println!("  (No knowledge extracted — provider returned empty results)");
    }
    println!();

    print_status(store);

    print_insight(
        "Complementary Learning Systems (CLS) theory: the hippocampus\n\
         \x20 (episodic store) gradually teaches the neocortex (semantic\n\
         \x20 store) through interleaved replay. This avoids catastrophic\n\
         \x20 forgetting — new knowledge doesn't overwrite old memories."
    );
}

fn chapter_4_perfuming(store: &AlayaStore) {
    print_chapter(4, "Perfuming", "Vasana \u{2192} Preference Crystallization");

    let provider = KeywordProvider;
    let interactions = perfuming_interactions();

    println!("  Feeding {} interactions to extract behavioral impressions...", interactions.len());
    println!();

    for (i, text) in interactions.iter().enumerate() {
        let interaction = Interaction {
            text: text.to_string(),
            role: Role::User,
            session_id: "day-3".to_string(),
            timestamp: 3000 + (i as i64) * 100,
            context: EpisodeContext::default(),
        };

        let report = store.perfume(&interaction, &provider).unwrap();
        let marker = if report.preferences_crystallized > 0 {
            " \u{2728} CRYSTALLIZED!"
        } else if report.preferences_reinforced > 0 {
            " \u{2191} reinforced"
        } else {
            ""
        };
        println!("    [{}] impressions: {}, crystallized: {}, reinforced: {}{}",
            i + 1, report.impressions_stored,
            report.preferences_crystallized,
            report.preferences_reinforced,
            marker);
    }
    println!();

    // Show crystallized preferences
    let prefs = store.preferences(None).unwrap();
    if !prefs.is_empty() {
        println!("  Crystallized Preferences:");
        for pref in &prefs {
            println!("    [{}] \"{}\" (confidence: {:.2}, evidence: {})",
                pref.domain, pref.preference, pref.confidence, pref.evidence_count);
        }
    } else {
        println!("  (No preferences crystallized yet)");
    }
    println!();

    print_status(store);

    print_insight(
        "Vasana (Sanskrit: 'perfume/fragrance'): each interaction leaves\n\
         \x20 a subtle trace (impression). When 5+ traces accumulate in one\n\
         \x20 domain, a preference crystallizes — like incense gradually\n\
         \x20 permeating cloth. Preferences are emergent, not declared."
    );
}

fn chapter_5_transformation(store: &AlayaStore) {
    print_chapter(5, "Transformation", "Dedup + Prune + Decay (Asraya-Paravrtti)");

    println!("  Status before transformation:");
    print_status(store);

    let report = store.transform().unwrap();

    println!("  TransformationReport:");
    println!("    duplicates_merged:  {}", report.duplicates_merged);
    println!("    links_pruned:       {}", report.links_pruned);
    println!("    preferences_decayed: {}", report.preferences_decayed);
    println!("    impressions_pruned: {}", report.impressions_pruned);
    println!();

    println!("  Status after transformation:");
    print_status(store);

    print_insight(
        "Asraya-paravrtti ('transformation of the storehouse'): periodic\n\
         \x20 refinement removes duplicates, prunes weak links (< 0.02),\n\
         \x20 and decays old preferences (30-day half-life). The memory\n\
         \x20 system trends toward clarity, not accumulation."
    );
}

fn chapter_6_forgetting(store: &AlayaStore) {
    print_chapter(6, "Forgetting", "Bjork Dual-Strength Model");

    println!("  Running 5 forgetting cycles (retrieval strength decays 0.95x each)...");
    println!();

    for cycle in 1..=5 {
        let report = store.forget().unwrap();
        println!("    Cycle {}: nodes_decayed={}, nodes_archived={}",
            cycle, report.nodes_decayed, report.nodes_archived);
    }
    println!();

    // Demonstrate memory revival through retrieval
    println!("  Now querying 'Rust borrow checker' to revive fading memories...");
    let results = store.query(&Query::simple("Rust borrow checker")).unwrap();
    println!("  Found {} results (retrieval boosts strength on access)", results.len());
    println!();

    println!("  Final status:");
    print_status(store);

    print_insight(
        "Bjork & Bjork (1992) 'New Theory of Disuse':\n\
         \x20 - Storage strength: how well-learned (increases with practice)\n\
         \x20 - Retrieval strength: how accessible now (decays without use)\n\
         \x20 A memory can have high storage but low retrieval — it exists\n\
         \x20 but is hard to find. Retrieving it revives the retrieval\n\
         \x20 strength, modeling the 'tip of the tongue' phenomenon."
    );
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!();
    println!("  \u{256D}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{256E}");
    println!("  \u{2502}  ALAYA \u{2014} Memory Engine Demo                     \u{2502}");
    println!("  \u{2502}  Neuroscience-inspired memory for AI agents      \u{2502}");
    println!("  \u{2570}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{256F}");
    println!();

    let store = AlayaStore::open_in_memory().unwrap();

    let episode_ids = chapter_1_episodic(&store);
    chapter_2_hebbian(&store, &episode_ids);
    chapter_3_consolidation(&store);
    chapter_4_perfuming(&store);
    chapter_5_transformation(&store);
    chapter_6_forgetting(&store);

    println!("  ═══════════════════════════════════════════════════");
    println!("   Demo Complete");
    println!("  ═══════════════════════════════════════════════════");
    println!();
    println!("  To learn more:");
    println!("    - API docs: cargo doc --open");
    println!("    - Source: https://github.com/h4x0r/alaya");
    println!();
}
```

**Step 3: Verify it compiles**

Run: `cargo check --example demo`
Expected: Compiles with no errors (warnings about unused imports are OK — fix any actual errors)

Note: There may be import issues since `Episode` is re-exported via `pub use types::*` but some types might need adjusting. If `Episode` or `SemanticNode` aren't accessible, check what `pub use types::*` actually exports and adjust the import list.

**Step 4: Run the demo (if linker works)**

Run: `cargo run --example demo`
Expected: Prints all 6 chapters with formatted output. If the Xcode linker issue is present, `cargo check --example demo` passing is sufficient.

**Step 5: Commit**

```bash
git add examples/demo.rs
git commit -m "feat: add scripted walkthrough demo (examples/demo.rs)

Six-chapter demo showcasing episodic memory, Hebbian graph,
consolidation, perfuming, transformation, and forgetting.
Includes a rule-based KeywordProvider that extracts knowledge
and impressions via pattern matching (no LLM required).

Run: cargo run --example demo"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Full demo with KeywordProvider + 6 chapters | `examples/demo.rs` |

**Total commits:** 1

**Verification:**
```bash
cargo check --example demo   # compiles
cargo run --example demo     # runs the walkthrough (if linker works)
```
