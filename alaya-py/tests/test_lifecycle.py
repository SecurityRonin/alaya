import alaya


def test_dream_empty_store():
    store = alaya.Alaya.in_memory()
    report = store.dream()
    assert report.consolidation.episodes_processed == 0
    assert report.consolidation.nodes_created == 0
    assert report.consolidation.links_created == 0
    assert report.transformation.duplicates_merged == 0
    assert report.forgetting.nodes_decayed == 0


def test_dream_perfuming_is_none():
    # dream() passes interaction=None so perfuming is always None
    store = alaya.Alaya.in_memory()
    report = store.dream()
    assert report.perfuming is None


def test_consolidate_empty_store():
    store = alaya.Alaya.in_memory()
    report = store.consolidate()
    assert report.episodes_processed == 0
    assert report.nodes_created == 0
    assert report.links_created == 0
    assert report.categories_assigned == 0


def test_transform_empty_store():
    store = alaya.Alaya.in_memory()
    report = store.transform()
    assert report.duplicates_merged == 0
    assert report.links_decayed == 0
    assert report.links_pruned == 0


def test_forget_empty_store():
    store = alaya.Alaya.in_memory()
    report = store.forget()
    assert report.nodes_decayed == 0
    assert report.nodes_archived == 0


def test_dream_after_storing_episodes():
    store = alaya.Alaya.in_memory()
    for i in range(3):
        ep = alaya.PyNewEpisode(
            content=f"Rust ownership model episode {i}",
            role="user",
            session_id="s1",
            timestamp=1000 + i,
        )
        store.store_episode(ep)
    # dream with NoOpProvider processes no episodes semantically but runs the cycle
    report = store.dream()
    assert isinstance(report.consolidation, alaya.PyConsolidationReport)
    assert isinstance(report.transformation, alaya.PyTransformationReport)
    assert isinstance(report.forgetting, alaya.PyForgettingReport)


def test_consolidate_then_transform():
    store = alaya.Alaya.in_memory()
    c_report = store.consolidate()
    assert isinstance(c_report, alaya.PyConsolidationReport)
    t_report = store.transform()
    assert isinstance(t_report, alaya.PyTransformationReport)


def test_lifecycle_methods_return_correct_types():
    store = alaya.Alaya.in_memory()
    assert isinstance(store.consolidate(), alaya.PyConsolidationReport)
    assert isinstance(store.transform(), alaya.PyTransformationReport)
    assert isinstance(store.forget(), alaya.PyForgettingReport)
    assert isinstance(store.dream(), alaya.PyDreamReport)


def test_purge_by_age_empty_store():
    store = alaya.Alaya.in_memory()
    # purge everything older than a future timestamp
    report = store.purge_by_age(older_than=9_999_999_999)
    assert report.episodes_deleted == 0
    assert report.nodes_deleted == 0
    assert report.links_deleted == 0
    assert report.embeddings_deleted == 0


def test_purge_by_age_removes_old_episodes():
    store = alaya.Alaya.in_memory()
    # Store episodes with old timestamps
    for i in range(3):
        ep = alaya.PyNewEpisode(
            content=f"Old episode {i}",
            role="user",
            session_id="s1",
            timestamp=100 + i,
        )
        store.store_episode(ep)
    status_before = store.status()
    assert status_before.episode_count == 3
    # Purge everything older than timestamp 9999
    report = store.purge_by_age(older_than=9999)
    assert report.episodes_deleted == 3
    status_after = store.status()
    assert status_after.episode_count == 0


def test_purge_by_weakness_empty_store():
    store = alaya.Alaya.in_memory()
    report = store.purge_by_weakness(below_strength=0.5)
    assert isinstance(report, alaya.PyPurgeReport)
    assert report.episodes_deleted == 0
