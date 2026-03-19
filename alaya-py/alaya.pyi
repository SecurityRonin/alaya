from typing import Optional

# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

class PyNewEpisode:
    content: str
    role: str
    session_id: str
    timestamp: int

    def __init__(self, content: str, role: str, session_id: str, timestamp: int) -> None: ...
    def __repr__(self) -> str: ...

class PyQuery:
    text: str
    max_results: int

    def __init__(self, text: str, max_results: int = 5) -> None: ...
    def __repr__(self) -> str: ...

class PyInteraction:
    text: str
    role: str
    session_id: str

    def __repr__(self) -> str: ...

class PyKnowledgeFilter:
    node_type: Optional[str]
    min_confidence: Optional[float]
    limit: Optional[int]
    category: Optional[str]

    def __init__(
        self,
        node_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: Optional[int] = None,
        category: Optional[str] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------

class PyConsolidationReport:
    episodes_processed: int
    nodes_created: int
    links_created: int
    categories_assigned: int

    def __repr__(self) -> str: ...

class PyPerfumingReport:
    impressions_stored: int
    preferences_crystallized: int
    preferences_reinforced: int

    def __repr__(self) -> str: ...

class PyTransformationReport:
    duplicates_merged: int
    links_decayed: int
    links_pruned: int
    preferences_decayed: int
    impressions_pruned: int
    categories_discovered: int
    categories_merged: int
    categories_dissolved: int
    categories_split: int

    def __repr__(self) -> str: ...

class PyForgettingReport:
    nodes_decayed: int
    nodes_archived: int

    def __repr__(self) -> str: ...

class PyDreamReport:
    consolidation: PyConsolidationReport
    perfuming: Optional[PyPerfumingReport]
    transformation: PyTransformationReport
    forgetting: PyForgettingReport

    def __repr__(self) -> str: ...

class PyPurgeReport:
    episodes_deleted: int
    nodes_deleted: int
    links_deleted: int
    embeddings_deleted: int

    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class PyMemoryStatus:
    episode_count: int
    semantic_node_count: int
    preference_count: int
    impression_count: int
    link_count: int
    embedding_count: int
    category_count: int

    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class PyEpisode:
    id: int
    content: str
    role: str
    session_id: str
    timestamp: int

    def __repr__(self) -> str: ...

class PyScoredMemory:
    content: str
    score: float
    node_type: str
    node_id: int
    role: Optional[str]
    timestamp: int

    def __repr__(self) -> str: ...

class PySemanticNode:
    id: int
    content: str
    node_type: str
    confidence: float
    source_episodes: list[int]
    created_at: int
    last_corroborated: int
    corroboration_count: int

    def __repr__(self) -> str: ...

class PyPreference:
    id: int
    domain: str
    preference: str
    confidence: float
    evidence_count: int
    first_observed: int
    last_reinforced: int

    def __repr__(self) -> str: ...

class PyImpression:
    id: int
    domain: str
    observation: str
    valence: float
    timestamp: int

    def __repr__(self) -> str: ...

class PyCategory:
    id: int
    label: str
    prototype_node: int
    member_count: int
    created_at: int
    last_updated: int
    stability: float
    parent_id: Optional[int]

    def __repr__(self) -> str: ...

class PyLink:
    id: int
    source_type: str
    source_id: int
    target_type: str
    target_id: int
    link_type: str
    forward_weight: float
    backward_weight: float
    created_at: int
    last_activated: int
    activation_count: int

    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Main store class
# ---------------------------------------------------------------------------

class Alaya:
    def __init__(self, path: str) -> None: ...

    @staticmethod
    def in_memory() -> "Alaya": ...

    @staticmethod
    def open_encrypted(path: str, key: str) -> "Alaya": ...

    def rekey(self, new_key: str) -> None: ...

    # Write path
    def store_episode(self, episode: PyNewEpisode) -> int: ...

    # Query path
    def query(self, q: PyQuery) -> list[PyScoredMemory]: ...
    def preferences(self, domain: Optional[str] = None) -> list[PyPreference]: ...
    def knowledge(self, filter: Optional[PyKnowledgeFilter] = None) -> list[PySemanticNode]: ...
    def categories(self, min_stability: Optional[float] = None) -> list[PyCategory]: ...
    def subcategories(self, parent_id: int) -> list[PyCategory]: ...
    def node_category(self, node_id: int) -> Optional[PyCategory]: ...
    def neighbors(
        self,
        node_type: str,
        node_id: int,
        depth: Optional[int] = None,
    ) -> list[tuple[str, int, float]]: ...
    def strongest_link(self) -> Optional[tuple[tuple[str, int], tuple[str, int], float]]: ...
    def node_content(self, node_type: str, node_id: int) -> Optional[str]: ...
    def knowledge_breakdown(self) -> dict[str, int]: ...
    def episodes_by_session(self, session_id: str) -> list[PyEpisode]: ...
    def unconsolidated_episodes(self, limit: Optional[int] = None) -> list[PyEpisode]: ...

    # Status
    def status(self) -> PyMemoryStatus: ...

    # Lifecycle
    def consolidate(self) -> PyConsolidationReport: ...
    def transform(self) -> PyTransformationReport: ...
    def forget(self) -> PyForgettingReport: ...
    def dream(self) -> PyDreamReport: ...

    # Admin / purge
    def purge_by_age(self, older_than: int) -> PyPurgeReport: ...
    def purge_by_weakness(self, below_strength: float) -> PyPurgeReport: ...

    # Provider management
    def set_consolidation_provider(self, provider: object) -> None: ...

    # Context manager
    def __enter__(self) -> "Alaya": ...
    def __exit__(
        self,
        _exc_type: Optional[object] = None,
        _exc_val: Optional[object] = None,
        _exc_tb: Optional[object] = None,
    ) -> bool: ...
