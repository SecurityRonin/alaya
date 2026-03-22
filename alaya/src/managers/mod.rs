//! Sub-manager types for the Alaya coordinator.
//!
//! Each sub-manager borrows from the parent `Alaya` instance
//! and provides a focused API for a specific domain.

pub(crate) mod admin;
pub(crate) mod episodes;
pub(crate) mod graph;
pub(crate) mod knowledge;
pub(crate) mod lifecycle;

use rusqlite::Connection;

use crate::provider::{EmbeddingProvider, ExtractionProvider};
use crate::types::ConflictStrategy;

/// Episodes sub-manager: store and query conversation episodes.
#[non_exhaustive]
pub struct Episodes<'a> {
    pub(crate) conn: &'a Connection,
    pub(crate) embedding_provider: Option<&'a dyn EmbeddingProvider>,
}

/// Knowledge sub-manager: query and manage semantic knowledge.
#[non_exhaustive]
pub struct Knowledge<'a> {
    pub(crate) conn: &'a Connection,
    pub(crate) embedding_provider: Option<&'a dyn EmbeddingProvider>,
}

/// Lifecycle sub-manager: consolidation, transformation, forgetting, reconciliation.
///
/// **Provider note:** Only `extraction_provider` is stored (needed by `auto_consolidate`).
/// `consolidate`, `dream`, and `perfume` take `&dyn ConsolidationProvider` as a
/// parameter — no `embedding_provider` needed. `auto_consolidate` calls
/// `store::episodic::unconsolidated_episodes` and `lifecycle::consolidation::learn_direct`
/// directly via its borrowed `conn`.
#[non_exhaustive]
pub struct Lifecycle<'a> {
    pub(crate) conn: &'a Connection,
    pub(crate) extraction_provider: Option<&'a dyn ExtractionProvider>,
    pub(crate) conflict_strategy: ConflictStrategy,
}

/// Graph sub-manager: neighbors, link queries.
#[non_exhaustive]
pub struct Graph<'a> {
    pub(crate) conn: &'a Connection,
}

/// Admin sub-manager: status, purge, categories, preferences.
#[non_exhaustive]
pub struct Admin<'a> {
    pub(crate) conn: &'a Connection,
}
