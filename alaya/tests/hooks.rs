mod common;

use alaya::{
    ConsolidationReport, EpisodeId, ForgettingReport, MemoryHooks, MockExtractionProvider,
    NoOpHooks, NoOpProvider,
};
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
        self.forgettings.lock().unwrap().push(report.nodes_archived);
    }
}

#[test]
fn hooks_fire_on_episode_store() {
    let mut alaya = common::empty_store();
    let eps = Arc::new(Mutex::new(Vec::new()));
    let hooks = RecordingHooks {
        episodes: eps.clone(),
        consolidations: Arc::new(Mutex::new(Vec::new())),
        forgettings: Arc::new(Mutex::new(Vec::new())),
    };
    alaya.set_hooks(Box::new(hooks));

    let id = alaya
        .episodes()
        .store(&common::make_episode("hello", "user", "s1", 1000))
        .unwrap();

    let recorded = eps.lock().unwrap();
    assert_eq!(recorded.len(), 1, "hook should fire exactly once");
    assert_eq!(
        recorded[0], id,
        "hook should receive the correct episode ID"
    );
}

#[test]
fn hooks_fire_on_consolidate() {
    let mut alaya = common::empty_store();
    let cons = Arc::new(Mutex::new(Vec::new()));
    let hooks = RecordingHooks {
        episodes: Arc::new(Mutex::new(Vec::new())),
        consolidations: cons.clone(),
        forgettings: Arc::new(Mutex::new(Vec::new())),
    };
    alaya.set_hooks(Box::new(hooks));

    let _report = alaya.lifecycle().consolidate(&NoOpProvider).unwrap();

    let recorded = cons.lock().unwrap();
    assert_eq!(recorded.len(), 1, "consolidation hook should fire once");
    assert_eq!(recorded[0], 0, "no nodes created on empty DB");
}

#[test]
fn hooks_fire_on_forget() {
    let mut alaya = common::empty_store();
    let forgets = Arc::new(Mutex::new(Vec::new()));
    let hooks = RecordingHooks {
        episodes: Arc::new(Mutex::new(Vec::new())),
        consolidations: Arc::new(Mutex::new(Vec::new())),
        forgettings: forgets.clone(),
    };
    alaya.set_hooks(Box::new(hooks));

    let _report = alaya.lifecycle().forget().unwrap();

    let recorded = forgets.lock().unwrap();
    assert_eq!(recorded.len(), 1, "forgetting hook should fire once");
    assert_eq!(recorded[0], 0, "no nodes archived on empty DB");
}

#[test]
fn hooks_fire_on_dream() {
    let mut alaya = common::empty_store();
    let cons = Arc::new(Mutex::new(Vec::new()));
    let forgets = Arc::new(Mutex::new(Vec::new()));
    let hooks = RecordingHooks {
        episodes: Arc::new(Mutex::new(Vec::new())),
        consolidations: cons.clone(),
        forgettings: forgets.clone(),
    };
    alaya.set_hooks(Box::new(hooks));

    // dream calls consolidate + forget internally
    alaya.lifecycle().dream(&NoOpProvider, None).unwrap();

    assert_eq!(
        cons.lock().unwrap().len(),
        1,
        "consolidation hook should fire once during dream"
    );
    assert_eq!(
        forgets.lock().unwrap().len(),
        1,
        "forgetting hook should fire once during dream"
    );
}

#[test]
fn no_hooks_configured_does_not_panic() {
    // Verify that operations work fine without any hooks set
    let alaya = common::empty_store();
    alaya
        .episodes()
        .store(&common::make_episode("hello", "user", "s1", 1000))
        .unwrap();
    alaya.lifecycle().consolidate(&NoOpProvider).unwrap();
    alaya.lifecycle().forget().unwrap();
}

#[test]
fn noop_hooks_do_not_affect_results() {
    let mut alaya = common::empty_store();
    alaya.set_hooks(Box::new(NoOpHooks));

    let id = alaya
        .episodes()
        .store(&common::make_episode("hello", "user", "s1", 1000))
        .unwrap();
    assert!(id.0 > 0);

    let report = alaya.lifecycle().consolidate(&NoOpProvider).unwrap();
    assert_eq!(report.nodes_created, 0);
}

#[test]
fn hooks_fire_on_consolidate_batch() {
    let mut alaya = common::empty_store();
    let cons = Arc::new(Mutex::new(Vec::new()));
    let hooks = RecordingHooks {
        episodes: Arc::new(Mutex::new(Vec::new())),
        consolidations: cons.clone(),
        forgettings: Arc::new(Mutex::new(Vec::new())),
    };
    alaya.set_hooks(Box::new(hooks));

    // Store some episodes so consolidation has work
    alaya
        .episodes()
        .store(&common::make_episode("batch test", "user", "s1", 1000))
        .unwrap();

    let _report = alaya
        .lifecycle()
        .consolidate_batch(&NoOpProvider, 5)
        .unwrap();

    let recorded = cons.lock().unwrap();
    assert_eq!(
        recorded.len(),
        1,
        "consolidation hook should fire on consolidate_batch"
    );
}

#[test]
fn hooks_fire_on_auto_consolidate_batch() {
    let mut alaya = common::empty_store();
    let cons = Arc::new(Mutex::new(Vec::new()));
    let hooks = RecordingHooks {
        episodes: Arc::new(Mutex::new(Vec::new())),
        consolidations: cons.clone(),
        forgettings: Arc::new(Mutex::new(Vec::new())),
    };
    alaya.set_hooks(Box::new(hooks));

    // Set extraction provider (required for auto_consolidate)
    alaya.set_extraction_provider(Box::new(MockExtractionProvider::empty()));

    // Store an episode so there's something to process
    alaya
        .episodes()
        .store(&common::make_episode("auto batch", "user", "s1", 2000))
        .unwrap();

    let _report = alaya.lifecycle().auto_consolidate_batch(5).unwrap();

    let recorded = cons.lock().unwrap();
    assert_eq!(
        recorded.len(),
        1,
        "consolidation hook should fire on auto_consolidate_batch"
    );
}

#[test]
fn multiple_episode_stores_fire_multiple_hooks() {
    let mut alaya = common::empty_store();
    let eps = Arc::new(Mutex::new(Vec::new()));
    let hooks = RecordingHooks {
        episodes: eps.clone(),
        consolidations: Arc::new(Mutex::new(Vec::new())),
        forgettings: Arc::new(Mutex::new(Vec::new())),
    };
    alaya.set_hooks(Box::new(hooks));

    for i in 0..5 {
        alaya
            .episodes()
            .store(&common::make_episode(
                &format!("msg {i}"),
                "user",
                "s1",
                1000 + i * 100,
            ))
            .unwrap();
    }

    let recorded = eps.lock().unwrap();
    assert_eq!(
        recorded.len(),
        5,
        "hook should fire once per episode stored"
    );
}
