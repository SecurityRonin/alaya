use super::Admin;
use crate::db;
use crate::error::{AlayaError, Result};
use crate::types::*;
use crate::{graph, store};

/// Truncate a string to at most `max_chars` characters, appending "..." if truncated.
fn truncate_label(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{truncated}...")
    }
}

impl Admin<'_> {
    /// Counts across all stores.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn status(&self) -> Result<MemoryStatus> {
        Ok(MemoryStatus {
            episode_count: store::episodic::count_episodes(self.conn)?,
            semantic_node_count: store::semantic::count_nodes(self.conn)?,
            preference_count: store::implicit::count_preferences(self.conn)?,
            impression_count: store::implicit::count_impressions(self.conn)?,
            link_count: graph::links::count_links(self.conn)?,
            embedding_count: store::embeddings::count_embeddings(self.conn)?,
            category_count: store::categories::count_categories(self.conn)?,
        })
    }

    /// Purge data matching the filter.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn purge(&self, filter: PurgeFilter) -> Result<PurgeReport> {
        db::transact(self.conn, |tx| {
            let mut report = PurgeReport::default();
            match filter {
                PurgeFilter::Session(ref session_id) => {
                    let eps = store::episodic::get_episodes_by_session(tx, session_id)?;
                    let ids: Vec<EpisodeId> = eps.iter().map(|e| e.id).collect();
                    report.episodes_deleted =
                        store::episodic::delete_episodes(tx, &ids)? as u32;
                }
                PurgeFilter::OlderThan(ts) => {
                    report.episodes_deleted =
                        tx.execute("DELETE FROM episodes WHERE timestamp < ?1", [ts])? as u32;
                }
                PurgeFilter::All => {
                    tx.execute_batch(
                        "DELETE FROM episodes;
                         DELETE FROM impressions;
                         DELETE FROM preferences;
                         DELETE FROM embeddings;
                         DELETE FROM links;
                         DELETE FROM node_strengths;
                         UPDATE semantic_nodes SET category_id = NULL;
                         DELETE FROM categories;
                         DELETE FROM semantic_nodes;",
                    )?;
                }
            }
            Ok(report)
        })
    }

    /// Get crystallized preferences, optionally filtered by domain.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn preferences(&self, domain: Option<&str>) -> Result<Vec<Preference>> {
        store::implicit::get_preferences(self.conn, domain)
    }

    /// List emergent categories, optionally filtered by minimum stability.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn categories(&self, min_stability: Option<f32>) -> Result<Vec<Category>> {
        store::categories::list_categories(self.conn, min_stability)
    }

    /// Get direct child categories of a parent category.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn subcategories(&self, parent_id: CategoryId) -> Result<Vec<Category>> {
        store::categories::get_subcategories(self.conn, parent_id)
    }

    /// Get the category for a semantic node, if assigned.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn node_category(&self, node_id: NodeId) -> Result<Option<Category>> {
        store::categories::get_node_category(self.conn, node_id)
    }

    /// Resolve a `NodeRef` to a human-readable content string (first 30 chars).
    ///
    /// Returns `None` if the referenced node no longer exists.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn node_content(&self, node: NodeRef) -> Result<Option<String>> {
        fn not_found_to_none(result: Result<String>) -> Result<Option<String>> {
            match result {
                Ok(s) => Ok(Some(s)),
                Err(AlayaError::NotFound(_)) => Ok(None),
                Err(e) => Err(e),
            }
        }

        match node {
            NodeRef::Episode(id) => not_found_to_none(
                store::episodic::get_episode(self.conn, id)
                    .map(|ep| truncate_label(&ep.content, 30)),
            ),
            NodeRef::Semantic(id) => not_found_to_none(
                store::semantic::get_semantic_node(self.conn, id)
                    .map(|n| truncate_label(&n.content, 30)),
            ),
            NodeRef::Category(id) => not_found_to_none(
                store::categories::get_category(self.conn, id)
                    .map(|c| truncate_label(&c.label, 30)),
            ),
            _ => Ok(Some(format!("{}#{}", node.type_str(), node.id()))),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::testutil::fixtures::*;
    use crate::types::*;
    use crate::Alaya;

    #[test]
    fn status_empty_db() {
        let alaya = Alaya::open_in_memory().unwrap();
        let status = alaya.admin().status().unwrap();
        assert_eq!(status.episode_count, 0);
        assert_eq!(status.semantic_node_count, 0);
        assert_eq!(status.preference_count, 0);
        assert_eq!(status.link_count, 0);
    }

    #[test]
    fn status_counts_episodes() {
        let alaya = Alaya::open_in_memory().unwrap();
        alaya.episodes().store(&episode("msg 1")).unwrap();
        alaya.episodes().store(&episode("msg 2")).unwrap();
        let status = alaya.admin().status().unwrap();
        assert_eq!(status.episode_count, 2);
    }

    #[test]
    fn purge_all_clears_everything() {
        let alaya = Alaya::open_in_memory().unwrap();
        alaya.episodes().store(&episode("msg 1")).unwrap();
        insert_semantic_node(alaya.raw_conn(), "fact", 0.9);

        alaya.admin().purge(PurgeFilter::All).unwrap();
        let status = alaya.admin().status().unwrap();
        assert_eq!(status.episode_count, 0);
        assert_eq!(status.semantic_node_count, 0);
    }

    #[test]
    fn purge_session() {
        let alaya = Alaya::open_in_memory().unwrap();
        alaya.episodes().store(&episode("msg 1")).unwrap();
        alaya
            .admin()
            .purge(PurgeFilter::Session("test-session".to_string()))
            .unwrap();
        let status = alaya.admin().status().unwrap();
        assert_eq!(status.episode_count, 0);
    }

    #[test]
    fn purge_older_than() {
        let alaya = Alaya::open_in_memory().unwrap();
        // episode() creates with timestamp 1000
        alaya.episodes().store(&episode("old msg")).unwrap();
        alaya.admin().purge(PurgeFilter::OlderThan(2000)).unwrap();
        let status = alaya.admin().status().unwrap();
        assert_eq!(status.episode_count, 0);
    }

    #[test]
    fn preferences_empty() {
        let alaya = Alaya::open_in_memory().unwrap();
        let prefs = alaya.admin().preferences(None).unwrap();
        assert!(prefs.is_empty());
    }

    #[test]
    fn categories_empty() {
        let alaya = Alaya::open_in_memory().unwrap();
        let cats = alaya.admin().categories(None).unwrap();
        assert!(cats.is_empty());
    }

    #[test]
    fn subcategories_empty() {
        let alaya = Alaya::open_in_memory().unwrap();
        let subs = alaya.admin().subcategories(CategoryId(1)).unwrap();
        assert!(subs.is_empty());
    }

    #[test]
    fn node_category_none_for_unknown() {
        let alaya = Alaya::open_in_memory().unwrap();
        let cat = alaya.admin().node_category(NodeId(999)).unwrap();
        assert!(cat.is_none());
    }

    #[test]
    fn node_content_none_for_missing() {
        let alaya = Alaya::open_in_memory().unwrap();
        let content = alaya
            .admin()
            .node_content(NodeRef::Episode(EpisodeId(999)))
            .unwrap();
        assert!(content.is_none());
    }

    #[test]
    fn node_content_truncates() {
        let alaya = Alaya::open_in_memory().unwrap();
        let long_content = "a".repeat(50);
        let id = alaya
            .episodes()
            .store(&episode(&long_content))
            .unwrap();
        let content = alaya
            .admin()
            .node_content(NodeRef::Episode(id))
            .unwrap();
        assert!(content.is_some());
        let text = content.unwrap();
        assert!(text.ends_with("..."));
        assert!(text.len() <= 33 + 3); // 30 chars + "..."
    }

    #[test]
    fn node_content_for_semantic_node() {
        let alaya = Alaya::open_in_memory().unwrap();
        let nid = insert_semantic_node(alaya.raw_conn(), "test fact content", 0.9);
        let content = alaya
            .admin()
            .node_content(NodeRef::Semantic(nid))
            .unwrap();
        assert_eq!(content, Some("test fact content".to_string()));
    }

    #[test]
    fn node_content_for_preference_fallback() {
        let alaya = Alaya::open_in_memory().unwrap();
        let content = alaya
            .admin()
            .node_content(NodeRef::Preference(PreferenceId(42)))
            .unwrap();
        assert_eq!(content, Some("preference#42".to_string()));
    }
}
