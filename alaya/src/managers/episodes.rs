use super::Episodes;
use crate::db;
use crate::error::{AlayaError, Result};
use crate::types::*;
use crate::{graph, store};

impl Episodes<'_> {
    /// Store a conversation episode with full context.
    ///
    /// ```
    /// use alaya::{Alaya, NewEpisode, Role, EpisodeContext};
    ///
    /// let alaya = Alaya::open_in_memory().unwrap();
    /// let id = alaya.episodes().store(&NewEpisode {
    ///     content: "Rust has zero-cost abstractions.".to_string(),
    ///     role: Role::User,
    ///     session_id: "session-1".to_string(),
    ///     timestamp: 1700000000,
    ///     context: EpisodeContext::default(),
    ///     embedding: None,
    /// }).unwrap();
    /// assert!(id.0 > 0);
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn store(&self, episode: &NewEpisode) -> Result<EpisodeId> {
        if episode.content.trim().is_empty() {
            return Err(AlayaError::InvalidInput(
                "episode content must not be empty".into(),
            ));
        }
        if episode.session_id.trim().is_empty() {
            return Err(AlayaError::InvalidInput(
                "session_id must not be empty".into(),
            ));
        }

        const MAX_CONTENT_BYTES: usize = 100 * 1024; // 100KB
        if episode.content.len() > MAX_CONTENT_BYTES {
            return Err(AlayaError::InvalidInput(format!(
                "episode content exceeds 100KB limit ({} bytes)",
                episode.content.len()
            )));
        }
        if let Some(ref emb) = episode.embedding {
            if emb.iter().any(|v| !v.is_finite()) {
                return Err(AlayaError::InvalidInput(
                    "embedding contains NaN or infinity values".into(),
                ));
            }
        }

        db::transact(self.conn, |tx| {
            let id = store::episodic::store_episode(tx, episode)?;

            let effective_embedding = match &episode.embedding {
                Some(emb) => Some(emb.clone()),
                None => self
                    .embedding_provider
                    .and_then(|p| p.embed(&episode.content).ok()),
            };
            if let Some(ref emb) = effective_embedding {
                store::embeddings::store_embedding(tx, "episode", id.0, emb, "")?;
            }

            store::strengths::init_strength(tx, NodeRef::Episode(id))?;

            if let Some(prev) = episode.context.preceding_episode {
                graph::links::create_link(
                    tx,
                    NodeRef::Episode(prev),
                    NodeRef::Episode(id),
                    LinkType::Temporal,
                    0.5,
                )?;
            }

            Ok(id)
        })
    }

    /// Return all episodes belonging to the given session.
    pub fn by_session(&self, session_id: &str) -> Result<Vec<Episode>> {
        store::episodic::get_episodes_by_session(self.conn, session_id)
    }

    /// Return unconsolidated episodes (not yet linked to any semantic node).
    pub fn unconsolidated(&self, limit: u32) -> Result<Vec<Episode>> {
        store::episodic::get_unconsolidated_episodes(self.conn, limit)
    }
}

#[cfg(test)]
mod tests {
    use crate::testutil::fixtures::*;
    use crate::Alaya;

    #[test]
    fn store_and_retrieve_episode() {
        let alaya = Alaya::open_in_memory().unwrap();
        let id = alaya.episodes().store(&episode("test content")).unwrap();
        assert!(id.0 > 0);

        let eps = alaya.episodes().by_session("test-session").unwrap();
        assert_eq!(eps.len(), 1);
        assert_eq!(eps[0].content, "test content");
    }

    #[test]
    fn store_rejects_empty_content() {
        let alaya = Alaya::open_in_memory().unwrap();
        let result = alaya.episodes().store(&episode(""));
        assert!(result.is_err());
    }

    #[test]
    fn store_rejects_empty_session_id() {
        let alaya = Alaya::open_in_memory().unwrap();
        let mut ep = episode("content");
        ep.session_id = "".to_string();
        assert!(alaya.episodes().store(&ep).is_err());
    }

    #[test]
    fn store_rejects_oversized_content() {
        let alaya = Alaya::open_in_memory().unwrap();
        let mut ep = episode("x");
        ep.content = "x".repeat(100 * 1024 + 1);
        let result = alaya.episodes().store(&ep);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("100KB"), "error should mention limit: {err}");
    }

    #[test]
    fn store_rejects_nan_embedding() {
        let alaya = Alaya::open_in_memory().unwrap();
        let mut ep = episode("valid content");
        ep.embedding = Some(vec![1.0, f32::NAN, 3.0]);
        let result = alaya.episodes().store(&ep);
        assert!(result.is_err());
    }

    #[test]
    fn store_rejects_infinity_embedding() {
        let alaya = Alaya::open_in_memory().unwrap();
        let mut ep = episode("valid content");
        ep.embedding = Some(vec![1.0, f32::INFINITY, 3.0]);
        let result = alaya.episodes().store(&ep);
        assert!(result.is_err());
    }

    #[test]
    fn store_accepts_valid_embedding() {
        let alaya = Alaya::open_in_memory().unwrap();
        let mut ep = episode("valid content");
        ep.embedding = Some(vec![0.1, 0.2, 0.3]);
        let result = alaya.episodes().store(&ep);
        assert!(result.is_ok());
    }

    #[test]
    fn unconsolidated_returns_new_episodes() {
        let alaya = Alaya::open_in_memory().unwrap();
        alaya.episodes().store(&episode("msg 1")).unwrap();
        alaya.episodes().store(&episode("msg 2")).unwrap();
        let uncons = alaya.episodes().unconsolidated(100).unwrap();
        assert_eq!(uncons.len(), 2);
    }
}
