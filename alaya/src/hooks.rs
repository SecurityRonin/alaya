//! Webhook/callback hooks for memory lifecycle events.
//!
//! The [`MemoryHooks`] trait provides event callbacks that fire at key
//! lifecycle points, enabling agent developers to react to memory events
//! (logging, metrics, syncing, etc.) without modifying core logic.

use crate::types::*;

/// Trait for receiving memory lifecycle event callbacks.
///
/// All methods have default no-op implementations, so implementors
/// only need to override the events they care about.
///
/// # Examples
///
/// ```
/// use alaya::{MemoryHooks, EpisodeId, ConsolidationReport, ForgettingReport};
///
/// struct LoggingHooks;
/// impl MemoryHooks for LoggingHooks {
///     fn on_episode_stored(&self, id: EpisodeId) {
///         println!("Episode stored: {:?}", id);
///     }
/// }
/// ```
pub trait MemoryHooks: Send + Sync {
    /// Called after an episode is successfully stored.
    fn on_episode_stored(&self, _id: EpisodeId) {}

    /// Called after consolidation completes successfully.
    fn on_consolidated(&self, _report: &ConsolidationReport) {}

    /// Called after a preference crystallizes from accumulated impressions.
    fn on_preference_crystallized(&self, _pref: &Preference) {}

    /// Called after a new category emerges from transformation.
    fn on_category_formed(&self, _cat: &Category) {}

    /// Called after forgetting completes successfully.
    fn on_forgotten(&self, _report: &ForgettingReport) {}
}

/// Default no-op implementation of [`MemoryHooks`].
///
/// Used internally when no hooks are configured.
pub struct NoOpHooks;
impl MemoryHooks for NoOpHooks {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct RecordingHooks {
        episodes: Arc<Mutex<Vec<EpisodeId>>>,
        consolidations: Arc<Mutex<Vec<u32>>>,
        forgettings: Arc<Mutex<Vec<u32>>>,
    }

    impl MemoryHooks for RecordingHooks {
        fn on_episode_stored(&self, id: EpisodeId) {
            self.episodes.lock().unwrap().push(id);
        }
        fn on_consolidated(&self, report: &ConsolidationReport) {
            self.consolidations
                .lock()
                .unwrap()
                .push(report.nodes_created);
        }
        fn on_forgotten(&self, report: &ForgettingReport) {
            self.forgettings
                .lock()
                .unwrap()
                .push(report.nodes_archived);
        }
    }

    #[test]
    fn test_no_op_hooks_compile() {
        let hooks = NoOpHooks;
        hooks.on_episode_stored(EpisodeId(1));
        hooks.on_consolidated(&ConsolidationReport::default());
        hooks.on_forgotten(&ForgettingReport::default());
        hooks.on_preference_crystallized(&Preference {
            id: PreferenceId(1),
            domain: "test".into(),
            preference: "test".into(),
            confidence: 0.5,
            evidence_count: 1,
            first_observed: 0,
            last_reinforced: 0,
        });
        hooks.on_category_formed(&Category {
            id: CategoryId(1),
            label: "test".into(),
            prototype_node: NodeId(1),
            member_count: 0,
            centroid_embedding: None,
            created_at: 0,
            last_updated: 0,
            stability: 0.0,
            parent_id: None,
        });
    }

    #[test]
    fn test_recording_hooks_captures_events() {
        let eps = Arc::new(Mutex::new(Vec::new()));
        let hooks = RecordingHooks {
            episodes: eps.clone(),
            consolidations: Arc::new(Mutex::new(Vec::new())),
            forgettings: Arc::new(Mutex::new(Vec::new())),
        };
        hooks.on_episode_stored(EpisodeId(42));
        assert_eq!(eps.lock().unwrap().len(), 1);
        assert_eq!(eps.lock().unwrap()[0], EpisodeId(42));
    }

    #[test]
    fn test_recording_hooks_captures_consolidation() {
        let cons = Arc::new(Mutex::new(Vec::new()));
        let hooks = RecordingHooks {
            episodes: Arc::new(Mutex::new(Vec::new())),
            consolidations: cons.clone(),
            forgettings: Arc::new(Mutex::new(Vec::new())),
        };
        let report = ConsolidationReport {
            episodes_processed: 5,
            nodes_created: 3,
            links_created: 2,
            categories_assigned: 1,
        };
        hooks.on_consolidated(&report);
        assert_eq!(cons.lock().unwrap().len(), 1);
        assert_eq!(cons.lock().unwrap()[0], 3);
    }

    #[test]
    fn test_recording_hooks_captures_forgetting() {
        let forgets = Arc::new(Mutex::new(Vec::new()));
        let hooks = RecordingHooks {
            episodes: Arc::new(Mutex::new(Vec::new())),
            consolidations: Arc::new(Mutex::new(Vec::new())),
            forgettings: forgets.clone(),
        };
        let report = ForgettingReport {
            nodes_decayed: 2,
            nodes_archived: 5,
        };
        hooks.on_forgotten(&report);
        assert_eq!(forgets.lock().unwrap().len(), 1);
        assert_eq!(forgets.lock().unwrap()[0], 5);
    }

    #[test]
    fn test_multiple_events_accumulate() {
        let eps = Arc::new(Mutex::new(Vec::new()));
        let hooks = RecordingHooks {
            episodes: eps.clone(),
            consolidations: Arc::new(Mutex::new(Vec::new())),
            forgettings: Arc::new(Mutex::new(Vec::new())),
        };
        hooks.on_episode_stored(EpisodeId(1));
        hooks.on_episode_stored(EpisodeId(2));
        hooks.on_episode_stored(EpisodeId(3));
        assert_eq!(eps.lock().unwrap().len(), 3);
        assert_eq!(*eps.lock().unwrap(), vec![EpisodeId(1), EpisodeId(2), EpisodeId(3)]);
    }
}
