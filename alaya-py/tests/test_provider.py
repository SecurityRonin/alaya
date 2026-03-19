import alaya


class SimpleProvider:
    """Minimal duck-typed provider that satisfies the Alaya provider protocol."""

    def extract_knowledge(self, episodes):
        return []

    def extract_impressions(self, interaction):
        return []

    def detect_contradiction(self, a, b):
        return False


class CountingProvider:
    """Provider that counts calls for verification."""

    def __init__(self):
        self.consolidate_calls = 0

    def extract_knowledge(self, episodes):
        self.consolidate_calls += 1
        return []

    def extract_impressions(self, interaction):
        return []

    def detect_contradiction(self, a, b):
        return False


def test_set_provider_no_error():
    store = alaya.Alaya.in_memory()
    store.set_consolidation_provider(SimpleProvider())


def test_consolidate_with_provider_empty_store():
    store = alaya.Alaya.in_memory()
    store.set_consolidation_provider(SimpleProvider())
    report = store.consolidate()
    assert report.episodes_processed == 0
    assert report.nodes_created == 0


def test_dream_with_provider_empty_store():
    store = alaya.Alaya.in_memory()
    store.set_consolidation_provider(SimpleProvider())
    report = store.dream()
    assert report.consolidation.episodes_processed == 0


def test_provider_can_be_replaced():
    store = alaya.Alaya.in_memory()
    store.set_consolidation_provider(SimpleProvider())
    store.set_consolidation_provider(SimpleProvider())
    report = store.consolidate()
    assert isinstance(report, alaya.PyConsolidationReport)


def test_consolidate_with_provider_and_episodes():
    store = alaya.Alaya.in_memory()
    store.set_consolidation_provider(SimpleProvider())
    for i in range(5):
        ep = alaya.PyNewEpisode(
            content=f"Knowledge item {i}: Python supports multiple paradigms",
            role="user",
            session_id="s1",
            timestamp=1000 + i,
        )
        store.store_episode(ep)
    # With SimpleProvider returning empty knowledge, no nodes are created
    report = store.consolidate()
    assert isinstance(report, alaya.PyConsolidationReport)
    assert report.nodes_created == 0


def test_provider_with_any_object():
    """Provider is duck-typed — any object with the right methods works."""

    class MinimalProvider:
        def extract_knowledge(self, eps):
            return []

        def extract_impressions(self, interaction):
            return []

        def detect_contradiction(self, a, b):
            return False

    store = alaya.Alaya.in_memory()
    store.set_consolidation_provider(MinimalProvider())
    report = store.consolidate()
    assert report.episodes_processed == 0


def test_default_provider_works_without_set():
    """Store uses NoOpProvider by default — consolidate should not raise."""
    store = alaya.Alaya.in_memory()
    report = store.consolidate()
    assert isinstance(report, alaya.PyConsolidationReport)


def test_full_lifecycle_with_provider():
    store = alaya.Alaya.in_memory()
    store.set_consolidation_provider(SimpleProvider())

    # Store some episodes
    for i in range(3):
        ep = alaya.PyNewEpisode(
            content=f"Episode {i} about distributed systems",
            role="user",
            session_id="lifecycle-test",
            timestamp=1000 + i,
        )
        store.store_episode(ep)

    # Run full dream cycle
    report = store.dream()
    assert isinstance(report, alaya.PyDreamReport)
    assert isinstance(report.consolidation, alaya.PyConsolidationReport)
    assert isinstance(report.transformation, alaya.PyTransformationReport)
    assert isinstance(report.forgetting, alaya.PyForgettingReport)
    assert report.perfuming is None
