import alaya


def test_open_in_memory():
    store = alaya.Alaya.in_memory()
    status = store.status()
    assert status.episode_count == 0
    assert status.semantic_node_count == 0


def test_store_episode_returns_positive_id():
    store = alaya.Alaya.in_memory()
    ep = alaya.PyNewEpisode(
        content="Rust has zero-cost abstractions",
        role="user",
        session_id="s1",
        timestamp=1000,
    )
    eid = store.store_episode(ep)
    assert eid > 0


def test_store_increments_episode_count():
    store = alaya.Alaya.in_memory()
    ep = alaya.PyNewEpisode(
        content="Python is great for scripting",
        role="assistant",
        session_id="s1",
        timestamp=2000,
    )
    store.store_episode(ep)
    status = store.status()
    assert status.episode_count == 1


def test_query_returns_results():
    store = alaya.Alaya.in_memory()
    ep = alaya.PyNewEpisode(
        content="Rust has zero-cost abstractions",
        role="user",
        session_id="s1",
        timestamp=1000,
    )
    store.store_episode(ep)
    q = alaya.PyQuery(text="Rust", max_results=5)
    results = store.query(q)
    assert len(results) > 0
    for r in results:
        assert isinstance(r.content, str)
        assert isinstance(r.score, float)
        assert isinstance(r.node_type, str)
        assert isinstance(r.node_id, int)


def test_query_respects_max_results():
    store = alaya.Alaya.in_memory()
    for i in range(10):
        ep = alaya.PyNewEpisode(
            content=f"Episode number {i} about memory systems",
            role="user",
            session_id="s1",
            timestamp=1000 + i,
        )
        store.store_episode(ep)
    q = alaya.PyQuery(text="memory", max_results=3)
    results = store.query(q)
    assert len(results) <= 3


def test_context_manager():
    with alaya.Alaya.in_memory() as store:
        status = store.status()
        assert status.episode_count == 0


def test_context_manager_store_and_query():
    with alaya.Alaya.in_memory() as store:
        ep = alaya.PyNewEpisode(
            content="Context managers are useful",
            role="user",
            session_id="ctx1",
            timestamp=5000,
        )
        eid = store.store_episode(ep)
        assert eid > 0
        q = alaya.PyQuery(text="context", max_results=5)
        results = store.query(q)
        assert len(results) > 0


def test_multiple_ids_are_unique():
    store = alaya.Alaya.in_memory()
    ids = []
    for i in range(5):
        ep = alaya.PyNewEpisode(
            content=f"Unique episode {i}",
            role="user",
            session_id="s1",
            timestamp=1000 + i,
        )
        ids.append(store.store_episode(ep))
    assert len(set(ids)) == 5


def test_knowledge_breakdown_empty():
    store = alaya.Alaya.in_memory()
    breakdown = store.knowledge_breakdown()
    assert isinstance(breakdown, dict)


def test_preferences_empty():
    store = alaya.Alaya.in_memory()
    prefs = store.preferences()
    assert prefs == []


def test_preferences_with_domain_filter():
    store = alaya.Alaya.in_memory()
    prefs = store.preferences(domain="coding")
    assert prefs == []


def test_knowledge_empty():
    store = alaya.Alaya.in_memory()
    nodes = store.knowledge()
    assert nodes == []


def test_knowledge_with_filter():
    store = alaya.Alaya.in_memory()
    f = alaya.PyKnowledgeFilter(node_type="fact", min_confidence=0.5)
    nodes = store.knowledge(filter=f)
    assert nodes == []


def test_categories_empty():
    store = alaya.Alaya.in_memory()
    cats = store.categories()
    assert cats == []


def test_categories_with_min_stability():
    store = alaya.Alaya.in_memory()
    cats = store.categories(min_stability=0.8)
    assert cats == []


def test_strongest_link_none_when_empty():
    store = alaya.Alaya.in_memory()
    result = store.strongest_link()
    assert result is None


def test_node_content_none_for_missing_node():
    store = alaya.Alaya.in_memory()
    result = store.node_content("episode", 999)
    assert result is None


def test_episodes_by_session_empty():
    store = alaya.Alaya.in_memory()
    eps = store.episodes_by_session("nonexistent-session")
    assert eps == []


def test_unconsolidated_episodes_empty():
    store = alaya.Alaya.in_memory()
    eps = store.unconsolidated_episodes()
    assert eps == []


def test_unconsolidated_episodes_with_limit():
    store = alaya.Alaya.in_memory()
    for i in range(5):
        ep = alaya.PyNewEpisode(
            content=f"Episode {i}",
            role="user",
            session_id="s1",
            timestamp=1000 + i,
        )
        store.store_episode(ep)
    eps = store.unconsolidated_episodes(limit=3)
    assert len(eps) <= 3


def test_episodes_by_session_filters_correctly():
    store = alaya.Alaya.in_memory()
    for i in range(3):
        ep = alaya.PyNewEpisode(
            content=f"Session A episode {i}",
            role="user",
            session_id="session-a",
            timestamp=1000 + i,
        )
        store.store_episode(ep)
    for i in range(2):
        ep = alaya.PyNewEpisode(
            content=f"Session B episode {i}",
            role="assistant",
            session_id="session-b",
            timestamp=2000 + i,
        )
        store.store_episode(ep)
    eps_a = store.episodes_by_session("session-a")
    assert len(eps_a) == 3
    for ep in eps_a:
        assert ep.session_id == "session-a"
    eps_b = store.episodes_by_session("session-b")
    assert len(eps_b) == 2
