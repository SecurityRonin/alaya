use super::Lifecycle;
use crate::db;
use crate::error::{AlayaError, Result};
use crate::provider::ConsolidationProvider;
use crate::types::*;
use crate::{graph, lifecycle, store};

impl Lifecycle<'_> {
    /// Run consolidation: episodic -> semantic (CLS replay).
    ///
    /// Processes all unconsolidated episodes (up to the internal batch size).
    ///
    /// ```
    /// use alaya::{Alaya, NoOpProvider};
    ///
    /// let alaya = Alaya::open_in_memory().unwrap();
    /// let report = alaya.lifecycle().consolidate(&NoOpProvider).unwrap();
    /// assert_eq!(report.nodes_created, 0); // no episodes to consolidate
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, provider)))]
    pub fn consolidate(&self, provider: &dyn ConsolidationProvider) -> Result<ConsolidationReport> {
        db::transact(self.conn, |tx| {
            lifecycle::consolidation::consolidate(tx, provider)
        })
    }

    /// Run consolidation on at most `batch_size` unconsolidated episodes.
    ///
    /// This enables incremental/streaming consolidation by processing episodes
    /// in fixed-size windows instead of all at once. A `batch_size` of 0 returns
    /// an empty report immediately.
    ///
    /// ```
    /// use alaya::{Alaya, NoOpProvider};
    ///
    /// let alaya = Alaya::open_in_memory().unwrap();
    /// let report = alaya.lifecycle().consolidate_batch(&NoOpProvider, 5).unwrap();
    /// assert_eq!(report.nodes_created, 0);
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, provider)))]
    pub fn consolidate_batch(
        &self,
        provider: &dyn ConsolidationProvider,
        batch_size: u32,
    ) -> Result<ConsolidationReport> {
        db::transact(self.conn, |tx| {
            lifecycle::consolidation::consolidate_batch(tx, provider, batch_size)
        })
    }

    /// Automatically extract knowledge from unconsolidated episodes using
    /// the configured ExtractionProvider, then learn the extracted nodes.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn auto_consolidate(&self) -> Result<ConsolidationReport> {
        self.auto_consolidate_batch(20)
    }

    /// Automatically extract knowledge from at most `batch_size` unconsolidated
    /// episodes using the configured ExtractionProvider, then learn the
    /// extracted nodes.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn auto_consolidate_batch(&self, batch_size: u32) -> Result<ConsolidationReport> {
        if batch_size == 0 {
            return Ok(ConsolidationReport::default());
        }

        let provider = self.extraction_provider.ok_or_else(|| {
            AlayaError::InvalidInput(
                "no extraction provider configured; call set_extraction_provider() first".into(),
            )
        })?;
        let episodes =
            store::episodic::get_unconsolidated_episodes(self.conn, batch_size)?;
        if episodes.is_empty() {
            return Ok(ConsolidationReport::default());
        }
        let nodes = provider.extract(&episodes)?;
        db::transact(self.conn, |tx| {
            lifecycle::consolidation::learn_direct(tx, nodes)
        })
    }

    /// Run transformation: dedup, prune, decay (asraya-paravrtti).
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn transform(&self) -> Result<TransformationReport> {
        db::transact(self.conn, |tx| lifecycle::transformation::transform(tx))
    }

    /// Run forgetting: decay retrieval strengths, archive weak nodes (Bjork).
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn forget(&self) -> Result<ForgettingReport> {
        db::transact(self.conn, |tx| lifecycle::forgetting::forget(tx))
    }

    /// Run perfuming: extract impressions, crystallize preferences (vasana).
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, provider)))]
    pub fn perfume(
        &self,
        interaction: &Interaction,
        provider: &dyn ConsolidationProvider,
    ) -> Result<PerfumingReport> {
        db::transact(self.conn, |tx| {
            lifecycle::perfuming::perfume(tx, interaction, provider)
        })
    }

    /// Run the full cognitive lifecycle in one call: consolidate, optionally
    /// perfume, transform, and forget.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, provider)))]
    pub fn dream(
        &self,
        provider: &dyn ConsolidationProvider,
        interaction: Option<&Interaction>,
    ) -> Result<DreamReport> {
        let consolidation = self.consolidate(provider)?;
        let perfuming = match interaction {
            Some(inter) => Some(self.perfume(inter, provider)?),
            None => None,
        };
        let transformation = self.transform()?;
        let forgetting = self.forget()?;

        Ok(DreamReport {
            consolidation,
            perfuming,
            transformation,
            forgetting,
        })
    }

    /// Run conflict detection and resolution.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn reconcile(&self) -> Result<ReconcileReport> {
        db::transact(self.conn, |tx| {
            lifecycle::reconciliation::reconcile(tx, self.conflict_strategy)
        })
    }

    /// Query unresolved conflicts (for Manual strategy).
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn conflicts(&self) -> Result<Vec<Conflict>> {
        store::conflicts::get_unresolved_conflicts(self.conn)
    }

    /// Manually resolve a specific conflict by choosing a winner.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn resolve_conflict(&self, conflict_id: ConflictId, winner_id: NodeId) -> Result<()> {
        db::transact(self.conn, |tx| {
            let now = crate::db::now();

            // Find the conflict to determine the loser
            let conflicts = store::conflicts::get_unresolved_conflicts(tx)?;
            let conflict = conflicts
                .iter()
                .find(|c| c.id == conflict_id)
                .ok_or_else(|| AlayaError::NotFound(format!("conflict {}", conflict_id.0)))?;

            let loser = if winner_id == conflict.node_a {
                conflict.node_b
            } else {
                conflict.node_a
            };

            store::conflicts::resolve_conflict(tx, conflict_id, winner_id, "manual", now)?;
            store::conflicts::supersede_node(tx, loser, winner_id)?;
            graph::links::create_link(
                tx,
                NodeRef::Semantic(winner_id),
                NodeRef::Semantic(loser),
                LinkType::Supersedes,
                1.0,
            )?;

            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::Alaya;

    #[test]
    fn consolidate_with_no_op() {
        let alaya = Alaya::open_in_memory().unwrap();
        let report = alaya.lifecycle().consolidate(&crate::NoOpProvider).unwrap();
        assert_eq!(report.nodes_created, 0);
    }

    #[test]
    fn auto_consolidate_requires_provider() {
        let alaya = Alaya::open_in_memory().unwrap();
        let result = alaya.lifecycle().auto_consolidate();
        assert!(result.is_err());
    }

    #[test]
    fn auto_consolidate_with_mock() {
        let mut alaya = Alaya::open_in_memory().unwrap();
        alaya.set_extraction_provider(Box::new(crate::MockExtractionProvider::empty()));
        let report = alaya.lifecycle().auto_consolidate().unwrap();
        assert_eq!(report.nodes_created, 0);
    }

    #[test]
    fn transform_on_empty_db() {
        let alaya = Alaya::open_in_memory().unwrap();
        let report = alaya.lifecycle().transform().unwrap();
        assert_eq!(report.duplicates_merged, 0);
    }

    #[test]
    fn forget_on_empty_db() {
        let alaya = Alaya::open_in_memory().unwrap();
        let report = alaya.lifecycle().forget().unwrap();
        assert_eq!(report.nodes_decayed, 0);
    }

    #[test]
    fn dream_runs_full_lifecycle() {
        let alaya = Alaya::open_in_memory().unwrap();
        let report = alaya.lifecycle().dream(&crate::NoOpProvider, None).unwrap();
        assert_eq!(report.consolidation.episodes_processed, 0);
        assert!(report.perfuming.is_none());
    }

    #[test]
    fn dream_with_interaction() {
        let alaya = Alaya::open_in_memory().unwrap();
        let interaction = crate::Interaction {
            text: "I prefer dark themes.".to_string(),
            role: crate::Role::User,
            session_id: "s1".to_string(),
            timestamp: 1000,
            context: crate::EpisodeContext::default(),
        };
        let report = alaya
            .lifecycle()
            .dream(&crate::NoOpProvider, Some(&interaction))
            .unwrap();
        assert!(report.perfuming.is_some());
    }

    #[test]
    fn reconcile_on_empty_db() {
        let alaya = Alaya::open_in_memory().unwrap();
        let report = alaya.lifecycle().reconcile().unwrap();
        assert_eq!(report.conflicts_detected, 0);
    }

    #[test]
    fn conflicts_empty_initially() {
        let alaya = Alaya::open_in_memory().unwrap();
        let conflicts = alaya.lifecycle().conflicts().unwrap();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn consolidate_batch_with_no_op() {
        let alaya = Alaya::open_in_memory().unwrap();
        let report = alaya
            .lifecycle()
            .consolidate_batch(&crate::NoOpProvider, 5)
            .unwrap();
        assert_eq!(report.nodes_created, 0);
    }

    #[test]
    fn consolidate_batch_zero_is_noop() {
        let alaya = Alaya::open_in_memory().unwrap();
        let report = alaya
            .lifecycle()
            .consolidate_batch(&crate::NoOpProvider, 0)
            .unwrap();
        assert_eq!(report.episodes_processed, 0);
        assert_eq!(report.nodes_created, 0);
    }

    #[test]
    fn auto_consolidate_batch_requires_provider() {
        let alaya = Alaya::open_in_memory().unwrap();
        let result = alaya.lifecycle().auto_consolidate_batch(5);
        assert!(result.is_err());
    }

    #[test]
    fn auto_consolidate_batch_zero_is_noop() {
        let alaya = Alaya::open_in_memory().unwrap();
        // batch_size=0 returns immediately without checking for a provider
        let report = alaya.lifecycle().auto_consolidate_batch(0).unwrap();
        assert_eq!(report.nodes_created, 0);
    }

    #[test]
    fn auto_consolidate_batch_with_mock() {
        let mut alaya = Alaya::open_in_memory().unwrap();
        alaya.set_extraction_provider(Box::new(crate::MockExtractionProvider::empty()));
        let report = alaya.lifecycle().auto_consolidate_batch(5).unwrap();
        assert_eq!(report.nodes_created, 0);
    }
}
