//! # Alaya Demo: A Scripted Walkthrough
//!
//! This demo walks through Alaya's ten core capabilities:
//! 1. Episodic Memory (store + query)
//! 2. Hebbian Graph (temporal links + co-retrieval + spreading activation)
//! 3. Consolidation (episodic -> semantic knowledge)
//! 4. Perfuming (vasana -> preference crystallization)
//! 5. Transformation + LTD (dedup, prune, link decay, emergent categories)
//! 6. Emergent Ontology (categories emerge from clustering)
//! 7. Enriched Retrieval (semantic nodes in query results)
//! 8. Retrieval-Induced Forgetting (competitor suppression)
//! 9. Forgetting (Bjork dual-strength model)
//! 10. Purge (cascade deletion with tombstone tracking)
//!
//! Run: `cargo run --example demo`

use alaya::{
    AlayaStore, ConsolidationProvider, Episode, EpisodeContext, EpisodeId, Interaction, NewEpisode,
    NewImpression, NewSemanticNode, NodeRef, PurgeFilter, Query, QueryContext, Role, SemanticNode,
    SemanticType,
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
        let all_text: String = episodes
            .iter()
            .map(|e| e.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // Detect technology relationships
        let techs = ["Rust", "tokio", "SQLite", "rusqlite", "async"];
        let found: Vec<&str> = techs
            .iter()
            .filter(|t| all_text.contains(*t))
            .copied()
            .collect();
        if found.len() >= 2 {
            nodes.push(NewSemanticNode {
                content: format!("User works with {}", found.join(", ")),
                node_type: SemanticType::Relationship,
                confidence: 0.75,
                source_episodes: ep_ids.clone(),
                // Embedding: tech-tools cluster
                embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            });
        }

        // Detect "X is Y" fact patterns and preference-like facts
        for ep in episodes {
            let lower = ep.content.to_lowercase();
            if lower.contains(" is ")
                && (lower.contains("amazing")
                    || lower.contains("powerful")
                    || lower.contains("simple"))
            {
                // Assign embeddings that cluster together (cos >= 0.7)
                // but don't dedup (cos < 0.95)
                let emb = if lower.contains("powerful") {
                    vec![0.8, 0.5, 0.0, 0.0]
                } else if lower.contains("simple") {
                    vec![0.8, 0.0, 0.5, 0.0]
                } else {
                    vec![0.7, 0.3, 0.3, 0.0]
                };
                nodes.push(NewSemanticNode {
                    content: ep.content.clone(),
                    node_type: SemanticType::Fact,
                    confidence: 0.60,
                    source_episodes: vec![ep.id],
                    embedding: Some(emb),
                });
            }
            if lower.contains("prefer")
                || lower.contains("enjoy")
                || lower.contains("love")
                || lower.contains("like")
            {
                nodes.push(NewSemanticNode {
                    content: ep.content.clone(),
                    node_type: SemanticType::Fact,
                    confidence: 0.65,
                    source_episodes: vec![ep.id],
                    embedding: Some(vec![0.5, 0.2, 0.5, 0.0]),
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
                // Different cluster — won't group with tech nodes
                embedding: Some(vec![0.0, 0.0, 0.0, 1.0]),
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
        if text.contains("like") || text.contains("practical") || text.contains("real-world") {
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
    println!("   Chapter {n}: {title} — {subtitle}");
    println!("  ═══════════════════════════════════════════════════");
    println!();
}

fn print_status(store: &AlayaStore) {
    let s = store.status().expect("failed to get memory status");
    println!("  MemoryStatus:");
    println!("    episodes:       {}", s.episode_count);
    println!("    semantic_nodes: {}", s.semantic_node_count);
    println!("    preferences:    {}", s.preference_count);
    println!("    impressions:    {}", s.impression_count);
    println!("    links:          {}", s.link_count);
    println!("    embeddings:     {}", s.embedding_count);
    let cats = store.categories(None).unwrap_or_default();
    println!("    categories:     {}", cats.len());
    println!();
}

fn print_insight(text: &str) {
    println!("  \u{2605} Insight: {text}");
    println!();
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max])
    } else {
        s.to_string()
    }
}

// ============================================================================
// Demo data
// ============================================================================

fn demo_episodes() -> Vec<(&'static str, &'static str, i64)> {
    vec![
        // (content, session_id, timestamp)
        // Session 1: Learning Rust
        (
            "I'm learning Rust and really enjoying the borrow checker. It catches so many bugs at compile time.",
            "day-1",
            1000,
        ),
        (
            "Async programming in Rust with tokio is powerful but has a steep learning curve.",
            "day-1",
            1100,
        ),
        (
            "I prefer using SQLite for embedded databases. It's simple and reliable.",
            "day-1",
            1200,
        ),
        (
            "The Rust type system is amazing. Pattern matching with enums is my favorite feature.",
            "day-1",
            1300,
        ),
        // Session 2: Building a project
        (
            "I'm building a memory engine for AI agents using rusqlite.",
            "day-2",
            2000,
        ),
        (
            "Performance matters a lot for my use case. I need sub-millisecond queries.",
            "day-2",
            2100,
        ),
        (
            "Can you show me code examples? I learn better from reading code than explanations.",
            "day-2",
            2200,
        ),
        (
            "I always structure my projects with small, focused modules. Each module does one thing.",
            "day-2",
            2300,
        ),
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

        let ctx = EpisodeContext {
            preceding_episode: prev_id,
            ..Default::default()
        };

        let id = store
            .store_episode(&NewEpisode {
                content: content.to_string(),
                role: Role::User,
                session_id: session.to_string(),
                timestamp: *ts,
                context: ctx,
                embedding: None,
            })
            .expect("failed to store episode");

        println!(
            "    [{}] ep#{}: \"{}\"",
            session,
            id.0,
            if content.len() > 60 {
                &content[..60]
            } else {
                content
            }
        );
        prev_id = Some(id);
        ids.push(id);
    }

    println!();
    print_status(store);

    // Query
    println!("  Querying: \"Rust async programming\"");
    let results = store
        .query(&Query::simple("Rust async programming"))
        .expect("failed to query episodes");
    println!("  Found {} results:", results.len());
    for (i, mem) in results.iter().enumerate() {
        println!(
            "    {}. [score {:.4}] \"{}\"",
            i + 1,
            mem.score,
            truncate(&mem.content, 55)
        );
    }
    println!();

    print_insight(
        "Episodic memory stores raw experiences with full context.\n\
         \x20 Like the hippocampus, it captures everything -- retrieval\n\
         \x20 is handled by the hybrid BM25 + graph pipeline.",
    );

    ids
}

fn chapter_2_hebbian(store: &AlayaStore, episode_ids: &[EpisodeId]) {
    print_chapter(2, "Hebbian Graph", "Co-Retrieval + Spreading Activation");

    let status = store.status().expect("failed to get memory status");
    println!(
        "  Links created during episode storage: {}",
        status.link_count
    );
    println!("  (Temporal links chain episodes within each session)");
    println!();

    // Run overlapping queries to trigger co-retrieval links
    println!("  Running overlapping queries to trigger Hebbian learning...");
    let _ = store
        .query(&Query::simple("Rust borrow checker"))
        .expect("query failed");
    let _ = store
        .query(&Query::simple("Rust type system"))
        .expect("query failed");
    let _ = store
        .query(&Query::simple("SQLite embedded database"))
        .expect("query failed");

    let status2 = store.status().expect("failed to get memory status");
    let new_links = status2.link_count - status.link_count;
    println!("  Co-retrieval links created: {new_links}");
    println!("  (Memories retrieved together strengthen their connection)");
    println!();

    // Show spreading activation from first episode
    if let Some(&seed) = episode_ids.first() {
        println!("  Spreading activation from episode #{}:", seed.0);
        let neighbors = store
            .neighbors(NodeRef::Episode(seed), 2)
            .expect("failed to get neighbors");
        if neighbors.is_empty() {
            println!("    (No neighbors yet -- graph needs more co-retrieval events)");
        } else {
            for (node, activation) in neighbors.iter().take(5) {
                println!(
                    "    {} #{}: activation {:.3}",
                    node.type_str(),
                    node.id(),
                    activation
                );
            }
        }
    }
    println!();

    print_insight(
        "Hebbian learning: 'neurons that fire together wire together.'\n\
         \x20 When memories are retrieved together, their link weight\n\
         \x20 grows: w += 0.1 * (1 - w). This creates an associative\n\
         \x20 network that mirrors how human memory clusters related ideas.",
    );
}

fn chapter_3_consolidation(store: &AlayaStore) {
    print_chapter(3, "Consolidation", "Episodic -> Semantic (CLS Replay)");

    let provider = KeywordProvider;

    println!("  Running CLS replay on unconsolidated episodes...");
    let report = store.consolidate(&provider).expect("consolidation failed");
    println!();
    println!("  ConsolidationReport:");
    println!("    episodes_processed: {}", report.episodes_processed);
    println!("    nodes_created:      {}", report.nodes_created);
    println!("    links_created:      {}", report.links_created);
    println!();

    // Show extracted knowledge
    let knowledge = store.knowledge(None).expect("failed to get knowledge");
    if !knowledge.is_empty() {
        println!("  Extracted Knowledge:");
        for node in &knowledge {
            println!(
                "    [{:?}] \"{}\" (confidence: {:.2})",
                node.node_type, node.content, node.confidence
            );
        }
    } else {
        println!("  (No knowledge extracted -- provider returned empty results)");
    }
    println!();

    print_status(store);

    print_insight(
        "Complementary Learning Systems (CLS) theory: the hippocampus\n\
         \x20 (episodic store) gradually teaches the neocortex (semantic\n\
         \x20 store) through interleaved replay. This avoids catastrophic\n\
         \x20 forgetting -- new knowledge doesn't overwrite old memories.",
    );
}

fn chapter_4_perfuming(store: &AlayaStore) {
    print_chapter(4, "Perfuming", "Vasana -> Preference Crystallization");

    let provider = KeywordProvider;
    let interactions = perfuming_interactions();

    println!(
        "  Feeding {} interactions to extract behavioral impressions...",
        interactions.len()
    );
    println!();

    for (i, text) in interactions.iter().enumerate() {
        let interaction = Interaction {
            text: text.to_string(),
            role: Role::User,
            session_id: "day-3".to_string(),
            timestamp: 3000 + (i as i64) * 100,
            context: EpisodeContext::default(),
        };

        let report = store
            .perfume(&interaction, &provider)
            .expect("perfuming failed");
        let marker = if report.preferences_crystallized > 0 {
            " << CRYSTALLIZED!"
        } else if report.preferences_reinforced > 0 {
            " ^ reinforced"
        } else {
            ""
        };
        println!(
            "    [{}] impressions: {}, crystallized: {}, reinforced: {}{}",
            i + 1,
            report.impressions_stored,
            report.preferences_crystallized,
            report.preferences_reinforced,
            marker
        );
    }
    println!();

    // Show crystallized preferences
    let prefs = store.preferences(None).expect("failed to get preferences");
    if !prefs.is_empty() {
        println!("  Crystallized Preferences:");
        for pref in &prefs {
            println!(
                "    [{}] \"{}\" (confidence: {:.2}, evidence: {})",
                pref.domain, pref.preference, pref.confidence, pref.evidence_count
            );
        }
    } else {
        println!("  (No preferences crystallized yet)");
    }
    println!();

    print_status(store);

    print_insight(
        "Vasana (Sanskrit: 'perfume/fragrance'): each interaction leaves\n\
         \x20 a subtle trace (impression). When 5+ traces accumulate in one\n\
         \x20 domain, a preference crystallizes -- like incense gradually\n\
         \x20 permeating cloth. Preferences are emergent, not declared.",
    );
}

fn chapter_5_transformation(store: &AlayaStore) {
    print_chapter(
        5,
        "Transformation + LTD",
        "Dedup + Prune + Link Decay + Category Discovery",
    );

    println!("  Status before transformation:");
    print_status(store);

    let report = store.transform().expect("transformation failed");

    println!("  TransformationReport:");
    println!("    duplicates_merged:      {}", report.duplicates_merged);
    println!("    links_decayed:          {}", report.links_decayed);
    println!("    links_pruned:           {}", report.links_pruned);
    println!("    preferences_decayed:    {}", report.preferences_decayed);
    println!("    impressions_pruned:     {}", report.impressions_pruned);
    println!(
        "    categories_discovered:  {}",
        report.categories_discovered
    );
    println!("    categories_merged:      {}", report.categories_merged);
    println!("    categories_dissolved:   {}", report.categories_dissolved);
    println!();

    if report.links_decayed > 0 {
        println!("  LTD (Long-Term Depression): {} links decayed by factor 0.95", report.links_decayed);
        println!("  Unused associations weaken -- only active pathways persist.");
        println!();
    }

    if report.categories_discovered > 0 {
        println!(
            "  Emergent categories discovered: {} (from semantic node clustering)",
            report.categories_discovered
        );
        println!();
    }

    println!("  Status after transformation:");
    print_status(store);

    print_insight(
        "Asraya-paravrtti ('transformation of the storehouse'): periodic\n\
         \x20 refinement removes duplicates, prunes weak links (< 0.02),\n\
         \x20 decays old preferences (30-day half-life), and applies LTD\n\
         \x20 (Long-Term Depression) to unused graph links. Categories\n\
         \x20 emerge organically from semantic node embedding similarity.",
    );
}

fn chapter_6_emergent_ontology(store: &AlayaStore) {
    print_chapter(6, "Emergent Ontology", "Categories from Clustering");

    let categories = store.categories(None).expect("failed to get categories");

    if categories.is_empty() {
        println!("  (No categories formed yet -- need 3+ semantic nodes with");
        println!("   similar embeddings. Run more consolidation cycles.)");
    } else {
        println!("  {} categories emerged from semantic node clustering:", categories.len());
        println!();
        for cat in &categories {
            println!(
                "    [cat#{}] \"{}\" (members: {}, stability: {:.2})",
                cat.id.0, cat.label, cat.member_count, cat.stability
            );

            // Show which nodes belong to this category
            let knowledge = store.knowledge(None).expect("failed to get knowledge");
            for node in &knowledge {
                if let Ok(Some(node_cat)) = store.node_category(node.id) {
                    if node_cat.id == cat.id {
                        println!("      -> \"{}\"", truncate(&node.content, 50));
                    }
                }
            }
        }
    }
    println!();

    // Show node_category API
    let knowledge = store.knowledge(None).expect("failed to get knowledge");
    if let Some(node) = knowledge.first() {
        match store.node_category(node.id) {
            Ok(Some(cat)) => {
                println!(
                    "  node_category(node#{}): belongs to \"{}\"",
                    node.id.0, cat.label
                );
            }
            Ok(None) => {
                println!("  node_category(node#{}): uncategorized", node.id.0);
            }
            Err(e) => {
                println!("  node_category error: {e}");
            }
        }
    }
    println!();

    print_insight(
        "Vikalpa ('conceptual construction'): categories are not declared\n\
         \x20 -- they emerge when 3+ semantic nodes cluster by embedding\n\
         \x20 similarity (threshold 0.7). The label comes from the prototype\n\
         \x20 node's content. Categories gain stability with each member\n\
         \x20 added, and dissolve if stability drops below 0.1.",
    );
}

fn chapter_7_enriched_retrieval(store: &AlayaStore) {
    print_chapter(
        7,
        "Enriched Retrieval",
        "Semantic Nodes in Query Results",
    );

    println!("  Standard BM25 query returns only episodes:");
    let bm25_results = store
        .query(&Query::simple("Rust programming tools"))
        .expect("query failed");
    for (i, mem) in bm25_results.iter().enumerate() {
        println!(
            "    {}. [{}] \"{}\"",
            i + 1,
            mem.node.type_str(),
            truncate(&mem.content, 50)
        );
    }
    println!();

    println!("  Vector-enriched query also surfaces semantic knowledge:");
    let enriched_query = Query {
        text: "Rust programming tools".to_string(),
        // Embedding close to the tech-tools cluster
        embedding: Some(vec![0.9, 0.1, 0.0, 0.0]),
        context: QueryContext {
            current_timestamp: Some(5000),
            ..Default::default()
        },
        max_results: 10,
        boost_categories: None,
    };
    let enriched_results = store.query(&enriched_query).expect("enriched query failed");
    for (i, mem) in enriched_results.iter().enumerate() {
        let tag = match mem.node {
            NodeRef::Episode(_) => "episode   ",
            NodeRef::Semantic(_) => "SEMANTIC  ",
            NodeRef::Preference(_) => "PREFERENCE",
            NodeRef::Category(_) => "category  ",
            _ => "unknown   ",
        };
        println!(
            "    {}. [{}] [score {:.4}] \"{}\"",
            i + 1,
            tag,
            mem.score,
            truncate(&mem.content, 45)
        );
    }
    println!();

    let has_semantic = enriched_results
        .iter()
        .any(|r| matches!(r.node, NodeRef::Semantic(_)));
    if has_semantic {
        println!("  Semantic nodes surfaced alongside episodes in results.");
    } else {
        println!("  (Semantic enrichment requires graph links from episodes");
        println!("   to semantic nodes, created during consolidation.)");
    }
    println!();

    print_insight(
        "The retrieval pipeline enriches results beyond raw episodes.\n\
         \x20 When vector search finds semantic nodes (via embeddings),\n\
         \x20 their content appears alongside episodic results. This\n\
         \x20 surfaces consolidated knowledge -- not just raw memories.",
    );
}

fn chapter_8_rif(store: &AlayaStore) {
    print_chapter(
        8,
        "Retrieval-Induced Forgetting",
        "Competitor Suppression (Anderson et al. 1994)",
    );

    println!("  When you retrieve specific memories, competing memories");
    println!("  from the same session are suppressed (RS *= 0.9).");
    println!();

    // Query specifically for borrow checker content
    println!("  Query 1: \"Rust borrow checker\" (retrieves specific day-1 memories)");
    let results = store
        .query(&Query::simple("Rust borrow checker"))
        .expect("query failed");
    let retrieved: Vec<String> = results
        .iter()
        .map(|r| truncate(&r.content, 50))
        .collect();
    for (i, content) in retrieved.iter().enumerate() {
        println!("    Retrieved: {}. \"{}\"", i + 1, content);
    }
    println!();
    println!("  Behind the scenes: non-retrieved day-1 episodes now have");
    println!("  reduced retrieval strength (RS suppressed by factor 0.9).");
    println!();

    // Query for something that was suppressed
    println!("  Query 2: \"SQLite database\" (may show suppression effect)");
    let results2 = store
        .query(&Query::simple("SQLite embedded database"))
        .expect("query failed");
    for (i, mem) in results2.iter().enumerate() {
        println!(
            "    {}. [score {:.4}] \"{}\"",
            i + 1,
            mem.score,
            truncate(&mem.content, 50)
        );
    }
    println!();

    print_insight(
        "Retrieval-Induced Forgetting (Anderson et al. 1994): retrieving\n\
         \x20 one memory actively suppresses competitors sharing the same\n\
         \x20 cues (here: same session). This models the 'part-set cuing\n\
         \x20 inhibition' effect -- studying a subset of items impairs\n\
         \x20 recall of the unstudied items from the same category.",
    );
}

fn chapter_9_forgetting(store: &AlayaStore) {
    print_chapter(9, "Forgetting", "Bjork Dual-Strength Model");

    println!("  Running 5 forgetting cycles (retrieval strength decays 0.95x each)...");
    println!();

    for cycle in 1..=5 {
        let report = store.forget().expect("forgetting failed");
        println!(
            "    Cycle {}: nodes_decayed={}, nodes_archived={}",
            cycle, report.nodes_decayed, report.nodes_archived
        );
    }
    println!();

    // Demonstrate memory revival through retrieval
    println!("  Now querying 'Rust borrow checker' to revive fading memories...");
    let results = store
        .query(&Query::simple("Rust borrow checker"))
        .expect("failed to query after forgetting");
    println!(
        "  Found {} results (retrieval boosts strength on access)",
        results.len()
    );
    println!();

    println!("  Status after forgetting:");
    print_status(store);

    print_insight(
        "Bjork & Bjork (1992) 'New Theory of Disuse':\n\
         \x20 - Storage strength: how well-learned (increases with practice)\n\
         \x20 - Retrieval strength: how accessible now (decays without use)\n\
         \x20 A memory can have high storage but low retrieval -- it exists\n\
         \x20 but is hard to find. Retrieving it revives the retrieval\n\
         \x20 strength, modeling the 'tip of the tongue' phenomenon.",
    );
}

fn chapter_10_purge(store: &AlayaStore) {
    print_chapter(10, "Purge", "Cascade Deletion with Tombstone Tracking");

    println!("  Status before purge:");
    print_status(store);

    // Purge a specific session
    println!("  Purging session 'day-1' (cascade deletes episodes + tombstones)...");
    let report = store
        .purge(PurgeFilter::Session("day-1".into()))
        .expect("purge failed");

    println!();
    println!("  PurgeReport:");
    println!("    episodes_deleted:   {}", report.episodes_deleted);
    println!("    nodes_deleted:      {}", report.nodes_deleted);
    println!("    links_deleted:      {}", report.links_deleted);
    println!("    embeddings_deleted: {}", report.embeddings_deleted);
    println!();

    println!("  Status after purge:");
    print_status(store);

    println!("  Remaining episodes (day-2 preserved):");
    let remaining = store
        .query(&Query::simple("building memory engine"))
        .expect("query failed");
    for (i, mem) in remaining.iter().enumerate() {
        println!(
            "    {}. [score {:.4}] \"{}\"",
            i + 1,
            mem.score,
            truncate(&mem.content, 50)
        );
    }
    println!();

    print_insight(
        "Purge supports three modes: PurgeFilter::Session (delete one\n\
         \x20 session), PurgeFilter::OlderThan (TTL-based), and\n\
         \x20 PurgeFilter::All (factory reset). Each deleted episode\n\
         \x20 records a tombstone internally for audit trail. Tombstones\n\
         \x20 track node_type, node_id, deletion timestamp, and reason.",
    );
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!();
    println!("  +---------------------------------------------------+");
    println!("  |  ALAYA -- Memory Engine Demo                      |");
    println!("  |  Neuroscience-inspired memory for AI agents        |");
    println!("  +---------------------------------------------------+");
    println!();

    let store = AlayaStore::open_in_memory().expect("failed to open in-memory database");

    let episode_ids = chapter_1_episodic(&store);
    chapter_2_hebbian(&store, &episode_ids);
    chapter_3_consolidation(&store);
    chapter_4_perfuming(&store);
    chapter_5_transformation(&store);
    chapter_6_emergent_ontology(&store);
    chapter_7_enriched_retrieval(&store);
    chapter_8_rif(&store);
    chapter_9_forgetting(&store);
    chapter_10_purge(&store);

    println!("  ═══════════════════════════════════════════════════");
    println!("   Demo Complete — 10 Chapters");
    println!("  ═══════════════════════════════════════════════════");
    println!();
    println!("  To learn more:");
    println!("    - API docs: cargo doc --open");
    println!("    - Source: https://github.com/SecurityRonin/alaya");
    println!();
}
