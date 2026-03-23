use alaya::{Alaya, EpisodeContext, NewEpisode, NewSemanticNode, Role, SemanticType};

/// Create an empty in-memory store for testing.
pub fn empty_store() -> Alaya {
    Alaya::open_in_memory().unwrap()
}

/// Create a store populated with sample episodes.
#[allow(dead_code)]
pub fn populated_store() -> Alaya {
    let store = empty_store();
    for i in 0..5 {
        store
            .episodes()
            .store(&make_episode(
                &format!("Test message {i}"),
                "user",
                "session-1",
                1700000000 + i * 60,
            ))
            .unwrap();
    }
    store
}

/// Create a store with a specific number of episodes.
#[allow(dead_code)]
pub fn store_with_episodes(n: i64) -> Alaya {
    let store = empty_store();
    for i in 0..n {
        store
            .episodes()
            .store(&make_episode(
                &format!("Episode {i}"),
                "user",
                "session-1",
                1700000000 + i * 60,
            ))
            .unwrap();
    }
    store
}

/// Build a NewEpisode with sensible defaults.
#[allow(dead_code)]
pub fn make_episode(content: &str, role: &str, session_id: &str, timestamp: i64) -> NewEpisode {
    let role = match role {
        "assistant" => Role::Assistant,
        "system" => Role::System,
        _ => Role::User,
    };
    NewEpisode {
        content: content.to_string(),
        role,
        session_id: session_id.to_string(),
        timestamp,
        context: EpisodeContext::default(),
        embedding: None,
    }
}

/// Build a NewSemanticNode with sensible defaults.
#[allow(dead_code)]
pub fn make_semantic_node(content: &str, node_type: SemanticType) -> NewSemanticNode {
    NewSemanticNode {
        content: content.to_string(),
        node_type,
        confidence: 0.95,
        source_episodes: vec![],
        embedding: None,
    }
}
