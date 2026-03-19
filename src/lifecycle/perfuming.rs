use crate::error::Result;
use crate::provider::ConsolidationProvider;
use crate::store::implicit;
use crate::types::*;
use rusqlite::Connection;

/// Minimum impressions in a domain before we attempt crystallization.
const CRYSTALLIZATION_THRESHOLD: u64 = 5;

/// Run a perfuming cycle: extract impressions and crystallize preferences.
///
/// Models vasana (perfuming) from Yogacara Buddhism: each interaction
/// leaves a subtle trace. When enough traces accumulate in one domain,
/// a preference crystallizes — like incense gradually permeating cloth.
pub fn perfume(
    conn: &Connection,
    interaction: &Interaction,
    provider: &dyn ConsolidationProvider,
) -> Result<PerfumingReport> {
    let mut report = PerfumingReport::default();

    // Extract impressions from this interaction
    let impressions = provider.extract_impressions(interaction)?;

    for imp in &impressions {
        implicit::store_impression(conn, imp)?;
        report.impressions_stored += 1;
    }

    // Check each affected domain for crystallization
    let domains: std::collections::HashSet<&str> =
        impressions.iter().map(|i| i.domain.as_str()).collect();

    for domain in domains {
        let count = implicit::count_impressions_by_domain(conn, domain)?;
        if count >= CRYSTALLIZATION_THRESHOLD {
            // Check if we already have a preference for this domain
            let existing = implicit::get_preferences(conn, Some(domain))?;
            if existing.is_empty() {
                // Crystallize a new preference from accumulated impressions
                let recent = implicit::get_impressions_by_domain(conn, domain, 20)?;
                if let Some(pref_text) = summarize_impressions(&recent) {
                    let avg_valence: f32 =
                        recent.iter().map(|i| i.valence).sum::<f32>() / recent.len() as f32;
                    let confidence = (count as f32 / 20.0).min(0.9);
                    implicit::store_preference(conn, domain, &pref_text, confidence)?;
                    report.preferences_crystallized += 1;

                    // Initialize strength for the new preference
                    let prefs = implicit::get_preferences(conn, Some(domain))?;
                    if let Some(p) = prefs.first() {
                        crate::store::strengths::init_strength(conn, NodeRef::Preference(p.id))?;
                    }
                    let _ = avg_valence; // Will be used in future for valence-aware preferences
                }
            } else {
                // Reinforce existing preference
                for pref in &existing {
                    implicit::reinforce_preference(conn, pref.id, 1)?;
                    report.preferences_reinforced += 1;
                }
            }
        }
    }

    Ok(report)
}

/// Simple heuristic: pick the most common observation as the preference summary.
/// A real implementation would use the ConsolidationProvider LLM.
fn summarize_impressions(impressions: &[Impression]) -> Option<String> {
    if impressions.is_empty() {
        return None;
    }
    // Find most common observation (simplified: just pick the most recent)
    Some(impressions[0].observation.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::MockProvider;
    use crate::schema::open_memory_db;

    #[test]
    fn test_perfuming_stores_impressions() {
        let conn = open_memory_db().unwrap();
        let interaction = Interaction {
            text: "Can you be more concise?".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 1000,
            context: EpisodeContext::default(),
        };

        let provider = MockProvider::with_impressions(vec![NewImpression {
            domain: "communication".to_string(),
            observation: "prefers concise answers".to_string(),
            valence: 1.0,
        }]);

        let report = perfume(&conn, &interaction, &provider).unwrap();
        assert_eq!(report.impressions_stored, 1);
    }

    #[test]
    fn test_crystallization_after_threshold() {
        let conn = open_memory_db().unwrap();
        let provider = MockProvider::with_impressions(vec![NewImpression {
            domain: "style".to_string(),
            observation: "prefers code examples".to_string(),
            valence: 1.0,
        }]);

        // Perfume multiple times to reach threshold
        for i in 0..6 {
            let interaction = Interaction {
                text: format!("interaction {i}"),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000 + i * 100,
                context: EpisodeContext::default(),
            };
            perfume(&conn, &interaction, &provider).unwrap();
        }

        let prefs = implicit::get_preferences(&conn, Some("style")).unwrap();
        assert!(!prefs.is_empty(), "should have crystallized a preference");
    }

    #[test]
    fn test_summarize_impressions_empty() {
        // Directly test the summarize_impressions function
        let result = summarize_impressions(&[]);
        assert!(result.is_none(), "empty impressions should return None");
    }

    #[test]
    fn test_perfume_with_noop_provider() {
        let conn = open_memory_db().unwrap();
        let provider = MockProvider::empty();
        let interaction = Interaction {
            text: "no impressions expected".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 1000,
            context: EpisodeContext::default(),
        };
        let report = perfume(&conn, &interaction, &provider).unwrap();
        assert_eq!(report.impressions_stored, 0);
        assert_eq!(report.preferences_crystallized, 0);
        assert_eq!(report.preferences_reinforced, 0);
    }

    #[test]
    fn test_reinforce_existing_preference() {
        // First crystallize a preference, then perfume again — should reinforce, not re-crystallize
        let conn = open_memory_db().unwrap();
        let provider = MockProvider::with_impressions(vec![NewImpression {
            domain: "format".to_string(),
            observation: "prefers tables".to_string(),
            valence: 0.8,
        }]);

        // Crystallize by reaching threshold (5 impressions)
        for i in 0..6 {
            let interaction = Interaction {
                text: format!("interaction {i}"),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000 + i * 100,
                context: EpisodeContext::default(),
            };
            perfume(&conn, &interaction, &provider).unwrap();
        }

        // Now there should be a preference for "format"
        let prefs_before = implicit::get_preferences(&conn, Some("format")).unwrap();
        assert!(
            !prefs_before.is_empty(),
            "should have crystallized a preference"
        );
        let evidence_before = prefs_before[0].evidence_count;

        // Perfume again — should reinforce the existing preference
        let interaction = Interaction {
            text: "one more".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 9000,
            context: EpisodeContext::default(),
        };
        let report = perfume(&conn, &interaction, &provider).unwrap();
        assert!(
            report.preferences_reinforced >= 1,
            "should reinforce existing preference"
        );
        assert_eq!(
            report.preferences_crystallized, 0,
            "should not crystallize again"
        );

        // Evidence count should have increased
        let prefs_after = implicit::get_preferences(&conn, Some("format")).unwrap();
        assert!(
            prefs_after[0].evidence_count > evidence_before,
            "evidence_count should increase after reinforce"
        );
    }

    #[test]
    fn test_summarize_impressions_non_empty() {
        // summarize_impressions returns the first impression's observation
        let imps = vec![
            Impression {
                id: ImpressionId(1),
                domain: "style".to_string(),
                observation: "first observation".to_string(),
                valence: 0.9,
                timestamp: 1000,
            },
            Impression {
                id: ImpressionId(2),
                domain: "style".to_string(),
                observation: "second observation".to_string(),
                valence: 0.5,
                timestamp: 900,
            },
        ];
        let result = summarize_impressions(&imps);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "first observation");
    }

    #[test]
    fn test_perfume_below_threshold_no_crystallization() {
        // Only 3 impressions (below threshold of 5) — should not crystallize
        let conn = open_memory_db().unwrap();
        let provider = MockProvider::with_impressions(vec![NewImpression {
            domain: "tone".to_string(),
            observation: "formal tone".to_string(),
            valence: 0.7,
        }]);

        for i in 0..3 {
            let interaction = Interaction {
                text: format!("interaction {i}"),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000 + i * 100,
                context: EpisodeContext::default(),
            };
            perfume(&conn, &interaction, &provider).unwrap();
        }

        let prefs = implicit::get_preferences(&conn, Some("tone")).unwrap();
        assert!(
            prefs.is_empty(),
            "should not crystallize with only 3 impressions (threshold=5)"
        );
    }
}
