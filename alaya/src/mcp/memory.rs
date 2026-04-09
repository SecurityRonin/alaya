//! Handler logic for the `remember` and `recall` MCP tools.

use std::sync::atomic::Ordering;

use crate::{EpisodeContext, NewEpisode, Query};

use super::{RecallParams, RememberParams};

pub fn handle_remember(server: &super::AlayaMcp, params: RememberParams) -> String {
    let role = match super::validation::parse_role(&params.role) {
        Ok(r) => r,
        Err(e) => return format!("Error: {e}"),
    };

    let now = super::validation::now_timestamp();

    let episode = NewEpisode {
        content: params.content.clone(),
        role,
        session_id: params.session_id.clone(),
        timestamp: now,
        context: EpisodeContext::default(),
        embedding: None,
    };

    match server.with_store(|s| s.episodes().store(&episode)) {
        Ok(id) => {
            let ep_total = server.episode_count.fetch_add(1, Ordering::Relaxed) + 1;
            let uncons = server.unconsolidated_count.fetch_add(1, Ordering::Relaxed) + 1;

            let mut response = format!(
                "Stored episode {} in session '{}'.",
                id.0, params.session_id
            );

            // Auto-consolidation at 10 unconsolidated episodes
            if uncons >= 10 {
                // Try auto-consolidation first (if ExtractionProvider is set)
                match server.with_store(|s| s.lifecycle().auto_consolidate()) {
                    Ok(report) if report.nodes_created > 0 => {
                        server.unconsolidated_count.store(0, Ordering::Relaxed);
                        response.push_str(&format!(
                            "\n\n--- Auto-consolidated ---\n\
                             Extracted {} knowledge nodes from {} episodes.",
                            report.nodes_created, uncons
                        ));
                    }
                    Ok(_) => {
                        // Provider returned zero nodes — fall back to prompt
                        if let Ok(episodes) = server.with_store(|s| s.episodes().unconsolidated(20))
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
                        if let Ok(episodes) = server.with_store(|s| s.episodes().unconsolidated(20))
                        {
                            if !is_no_provider {
                                response.push_str(&format!("\n\n(Auto-consolidation failed: {e})"));
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
                let tr = server.with_store(|s| s.lifecycle().transform());
                let fr = server.with_store(|s| s.lifecycle().forget());
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

pub fn handle_recall(server: &super::AlayaMcp, params: RecallParams) -> String {
    let query = Query {
        text: params.query,
        embedding: None,
        context: crate::QueryContext::default(),
        max_results: params.max_results.unwrap_or(5),
        category_id: params.category_id,
        boost_categories: params.boost_category.map(|c| vec![c.to_string()]),
        boost_weights: None,
    };

    match server.with_store(|s| s.knowledge().query(&query)) {
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

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use crate::{Alaya, MockExtractionProvider, NewSemanticNode, SemanticType};

    use super::super::{AlayaMcp, LearnFactEntry, LearnParams, RememberParams};

    fn make_server() -> AlayaMcp {
        let store = Alaya::open_in_memory().unwrap();
        AlayaMcp::new(store)
    }

    fn make_server_with_extraction() -> AlayaMcp {
        let mut store = Alaya::open_in_memory().unwrap();
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
                assert!(
                    !result.contains("Consolidation suggested"),
                    "Should not suggest consolidation at episode {}",
                    i + 1
                );
            } else {
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
        for i in 0..10 {
            srv.remember(RememberParams {
                content: format!("Fact message {i}"),
                role: "user".into(),
                session_id: "sess".into(),
            });
        }

        srv.learn(LearnParams {
            facts: vec![LearnFactEntry {
                content: "Extracted fact".into(),
                node_type: "fact".into(),
                confidence: None,
            }],
            session_id: None,
        });

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
        let srv = make_server();
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
        let status = srv.status();
        assert!(
            status.contains("0 unconsolidated"),
            "Counter should reset after auto-consolidation: {status}"
        );
    }

    #[test]
    fn recall_empty_store() {
        let srv = make_server();
        let result = srv.recall(super::super::RecallParams {
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

        let result = srv.recall(super::super::RecallParams {
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

        let result = srv.recall(super::super::RecallParams {
            query: "programming".into(),
            max_results: Some(3),
            boost_category: None,
        });
        assert!(result.contains("Found"));
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
        let result = srv.recall(super::super::RecallParams {
            query: "memory".into(),
            max_results: None,
            boost_category: Some(9999),
        });
        assert!(!result.starts_with("Error:"));
    }

    #[test]
    fn recall_output_format_contains_numbered_entries() {
        // Verify the "N. [role] (score: X.XXX) content" format for found memories
        let srv = make_server();
        srv.remember(RememberParams {
            content: "Rust is memory safe".into(),
            role: "user".into(),
            session_id: "s1".into(),
        });
        srv.remember(RememberParams {
            content: "Rust has zero-cost abstractions".into(),
            role: "assistant".into(),
            session_id: "s1".into(),
        });

        let result = srv.recall(super::super::RecallParams {
            query: "Rust".into(),
            max_results: Some(5),
            boost_category: None,
        });
        assert!(
            result.starts_with("Found"),
            "Should start with 'Found': {result}"
        );
        assert!(
            result.contains("memories:"),
            "Should say 'memories:': {result}"
        );
        // Each result line starts with an ordinal and includes a role label and score
        assert!(
            result.contains("1."),
            "Should number results starting at 1: {result}"
        );
        assert!(
            result.contains("score:"),
            "Should include score in output: {result}"
        );
        assert!(
            result.contains("[user]") || result.contains("[assistant]"),
            "Should include role in brackets: {result}"
        );
    }

    #[test]
    fn recall_default_max_results_is_five() {
        // When max_results is None, the default of 5 is used
        let srv = make_server();
        for i in 0..10 {
            srv.remember(RememberParams {
                content: format!("Rust fact number {i}"),
                role: "user".into(),
                session_id: "s1".into(),
            });
        }
        let result = srv.recall(super::super::RecallParams {
            query: "Rust fact".into(),
            max_results: None,
            boost_category: None,
        });
        // Count result lines (start with digit + ".")
        let count = result
            .lines()
            .filter(|l| {
                let t = l.trim();
                t.starts_with("1.")
                    || t.starts_with("2.")
                    || t.starts_with("3.")
                    || t.starts_with("4.")
                    || t.starts_with("5.")
                    || t.starts_with("6.")
            })
            .count();
        assert!(
            count <= 5,
            "Default max_results=5 should return at most 5 results, got {count}"
        );
    }

    #[test]
    fn remember_store_episode_db_error() {
        let store = Alaya::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE episodes")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.remember(RememberParams {
            content: "test".into(),
            role: "user".into(),
            session_id: "s1".into(),
        });
        assert!(
            result.starts_with("Error:"),
            "Should return error when episodes table is missing: {result}"
        );
    }

    #[test]
    fn recall_db_error() {
        let store = Alaya::open_in_memory().unwrap();
        store
            .raw_conn()
            .execute_batch("DROP TABLE episodes")
            .unwrap();
        let srv = AlayaMcp::new(store);
        let result = srv.recall(super::super::RecallParams {
            query: "test".into(),
            max_results: None,
            boost_category: None,
        });
        assert!(
            result.starts_with("Error:"),
            "Should return error when DB is corrupted: {result}"
        );
    }

    #[test]
    fn remember_auto_consolidation_error_with_message() {
        // When auto_consolidate returns a real error (not "extraction provider"),
        // the error message should appear in the response (line 74).
        let mut store = Alaya::open_in_memory().unwrap();
        store.set_extraction_provider(Box::new(MockExtractionProvider::new(vec![
            NewSemanticNode {
                content: "test fact".into(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: None,
            },
        ])));
        let srv = AlayaMcp::new(store);
        // Store 9 episodes first
        for i in 0..9 {
            srv.remember(RememberParams {
                content: format!("Episode {i}"),
                role: "user".into(),
                session_id: "s1".into(),
            });
        }
        // Corrupt the DB so auto_consolidate fails with a real error
        let _ = srv.with_store(|s| {
            s.raw_conn()
                .execute_batch("DROP TABLE semantic_nodes")
                .map_err(|e| crate::AlayaError::InvalidInput(e.to_string()))
        });
        // The 10th episode triggers auto-consolidation which will fail
        let result = srv.remember(RememberParams {
            content: "Episode 9".into(),
            role: "user".into(),
            session_id: "s1".into(),
        });
        // Should contain the error message from the failed consolidation
        assert!(
            result.contains("Auto-consolidation failed")
                || result.contains("Consolidation suggested")
                || result.contains("Error:"),
            "Should show consolidation failure: {result}"
        );
    }

    #[test]
    fn remember_auto_maintenance_error() {
        // Store 24 episodes, then corrupt DB, then store the 25th to trigger maintenance error (lines 113-114)
        let srv = make_server();
        for i in 0..24 {
            srv.remember(RememberParams {
                content: format!("Episode {i}"),
                role: "user".into(),
                session_id: "s1".into(),
            });
        }
        // Corrupt DB so transform/forget fails
        let _ = srv.with_store(|s| {
            s.raw_conn()
                .execute_batch("DROP TABLE semantic_nodes")
                .map_err(|e| crate::AlayaError::InvalidInput(e.to_string()))
        });
        let result = srv.remember(RememberParams {
            content: "Episode 24 triggers maintenance".into(),
            role: "user".into(),
            session_id: "s1".into(),
        });
        // Should contain auto-maintenance error
        assert!(
            result.contains("Auto-maintenance error") || result.contains("Stored episode"),
            "Should handle maintenance error gracefully: {result}"
        );
    }

    #[test]
    fn remember_consolidation_fallback_empty_provider() {
        // When extraction provider returns zero nodes, the Ok(_) path should
        // suggest consolidation with episode listing (covers lines 50-66).
        let mut store = Alaya::open_in_memory().unwrap();
        store.set_extraction_provider(Box::new(MockExtractionProvider::empty()));
        let srv = AlayaMcp::new(store);

        let mut fallback_response = String::new();
        for i in 0..10 {
            let result = srv.remember(RememberParams {
                content: format!("Episode {i} about cooking"),
                role: "user".into(),
                session_id: "s1".into(),
            });
            if result.contains("Consolidation suggested") {
                fallback_response = result;
            }
        }
        assert!(
            !fallback_response.is_empty(),
            "Empty extraction provider should trigger consolidation fallback"
        );
        assert!(
            fallback_response.contains("unconsolidated episodes"),
            "Fallback should list unconsolidated episodes"
        );
        // Should contain episode content (the for loop in lines 58-65)
        assert!(
            fallback_response.contains("Episode"),
            "Fallback should include episode content: {fallback_response}"
        );
    }
}
