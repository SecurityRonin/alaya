//! LLM-based extraction provider for OpenAI-compatible APIs.

use crate::{AlayaError, Episode, EpisodeId, NewSemanticNode, SemanticType};

const SYSTEM_PROMPT: &str = "\
You extract structured knowledge from conversation episodes.

Given conversation episodes, extract key facts, relationships, events, and concepts.
Return a JSON array of objects, each with:
- \"content\": a concise factual statement
- \"node_type\": one of \"fact\", \"relationship\", \"event\", \"concept\"
- \"confidence\": 0.0 to 1.0, how certain this is stated (not inferred)

Rules:
- Only extract what is explicitly stated or strongly implied
- Prefer precision over recall (skip uncertain items)
- Merge duplicate information into single entries
- Use \"relationship\" for connections between entities (\"Alice manages auth team\")
- Use \"event\" for time-bound occurrences (\"migrated to Postgres on Monday\")
- Use \"concept\" for technical terms or domain knowledge explained
- Use \"fact\" for everything else

Respond with ONLY a JSON array. No markdown, no explanation.";

/// Format episodes into the user message for the LLM.
pub(crate) fn build_user_prompt(episodes: &[Episode]) -> String {
    let mut prompt = String::from("Extract knowledge from these conversation episodes:\n\n");
    for ep in episodes {
        prompt.push_str(&format!(
            "[{}] {}: {}\n",
            ep.id.0,
            ep.role.as_str(),
            ep.content
        ));
    }
    prompt
}

/// Intermediate struct for parsing LLM JSON output.
#[derive(serde::Deserialize)]
struct ExtractedFact {
    content: String,
    node_type: String,
    #[serde(default = "default_confidence")]
    confidence: f32,
}

fn default_confidence() -> f32 {
    0.8
}

/// Parse the LLM's JSON response text into NewSemanticNodes.
/// Handles both raw JSON arrays and JSON wrapped in markdown code fences.
pub(crate) fn parse_extraction_response(
    response_text: &str,
    source_episodes: &[EpisodeId],
) -> crate::Result<Vec<NewSemanticNode>> {
    // Strip markdown code fences if present
    let text = response_text.trim();
    let text = if text.starts_with("```") {
        let inner = text.trim_start_matches("```json").trim_start_matches("```");
        inner.trim_end_matches("```").trim()
    } else {
        text
    };

    let facts: Vec<ExtractedFact> = serde_json::from_str(text).map_err(|e| {
        crate::AlayaError::InvalidInput(format!("failed to parse extraction response: {e}"))
    })?;

    let nodes = facts
        .into_iter()
        .map(|f| {
            let node_type = match f.node_type.to_lowercase().as_str() {
                "relationship" => SemanticType::Relationship,
                "event" => SemanticType::Event,
                "concept" => SemanticType::Concept,
                _ => SemanticType::Fact, // default to Fact for unknown types
            };
            let confidence = f.confidence.clamp(0.0, 1.0);
            NewSemanticNode {
                content: f.content,
                node_type,
                confidence,
                source_episodes: source_episodes.to_vec(),
                embedding: None,
            }
        })
        .collect();

    Ok(nodes)
}

/// Extract the assistant's message content from an OpenAI-compatible API response.
pub(crate) fn parse_api_response(body: &str) -> crate::Result<String> {
    let json: serde_json::Value = serde_json::from_str(body).map_err(|e| {
        crate::AlayaError::InvalidInput(format!("failed to parse API response: {e}"))
    })?;

    json["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| {
            crate::AlayaError::InvalidInput(
                "API response missing choices[0].message.content".into(),
            )
        })
}

/// Concrete ExtractionProvider that calls an OpenAI-compatible LLM API.
///
/// Works with: OpenAI, Anthropic (via OpenRouter), Ollama, Groq, Together, etc.
///
/// # Example
/// ```no_run
/// use alaya::extraction::LlmExtractionProvider;
///
/// let provider = LlmExtractionProvider::builder()
///     .api_key("sk-...")
///     .model("gpt-4o-mini")
///     .build()
///     .unwrap();
/// ```
pub struct LlmExtractionProvider {
    agent: ureq::Agent,
    pub(crate) api_url: String,
    pub(crate) api_key: String,
    pub(crate) model: String,
}

impl std::fmt::Debug for LlmExtractionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmExtractionProvider")
            .field("api_url", &self.api_url)
            .field("model", &self.model)
            .field("api_key", &"[redacted]")
            .finish()
    }
}

/// Builder for LlmExtractionProvider.
pub struct LlmExtractionProviderBuilder {
    api_url: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
}

impl LlmExtractionProvider {
    pub fn builder() -> LlmExtractionProviderBuilder {
        LlmExtractionProviderBuilder {
            api_url: None,
            api_key: None,
            model: None,
        }
    }
}

impl LlmExtractionProviderBuilder {
    /// Set the API endpoint URL.
    /// Default: "https://api.openai.com/v1/chat/completions"
    pub fn api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = Some(url.into());
        self
    }

    /// Set the API key (required).
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the model name.
    /// Default: "gpt-4o-mini"
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Build the provider. Requires api_key to be set.
    pub fn build(self) -> crate::Result<LlmExtractionProvider> {
        let api_key = self.api_key.ok_or_else(|| {
            AlayaError::InvalidInput("api_key is required for LlmExtractionProvider".into())
        })?;

        Ok(LlmExtractionProvider {
            agent: ureq::Agent::new(),
            api_url: self
                .api_url
                .unwrap_or_else(|| "https://api.openai.com/v1/chat/completions".into()),
            api_key,
            model: self.model.unwrap_or_else(|| "gpt-4o-mini".into()),
        })
    }
}

impl crate::ExtractionProvider for LlmExtractionProvider {
    fn extract(&self, episodes: &[Episode]) -> crate::Result<Vec<NewSemanticNode>> {
        if episodes.is_empty() {
            return Ok(vec![]);
        }

        let source_ids: Vec<EpisodeId> = episodes.iter().map(|e| e.id).collect();
        let user_prompt = build_user_prompt(episodes);

        let request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        });

        let response = self
            .agent
            .post(&self.api_url)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(&request_body)
            .map_err(|e| AlayaError::InvalidInput(format!("LLM API request failed: {e}")))?;

        let body = response
            .into_string()
            .map_err(|e| AlayaError::InvalidInput(format!("failed to read API response: {e}")))?;

        let content = parse_api_response(&body)?;
        parse_extraction_response(&content, &source_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EpisodeContext, EpisodeId, ExtractionProvider, Role};

    fn make_episode(id: i64, role: Role, content: &str) -> Episode {
        Episode {
            id: EpisodeId(id),
            content: content.to_string(),
            role,
            session_id: "test-session".to_string(),
            timestamp: 1000 + id,
            context: EpisodeContext::default(),
        }
    }

    // --- build_user_prompt tests ---

    #[test]
    fn build_prompt_includes_all_episodes() {
        let episodes = vec![
            make_episode(1, Role::User, "I use Rust for backend"),
            make_episode(2, Role::Assistant, "Rust is great for performance"),
        ];
        let prompt = build_user_prompt(&episodes);
        assert!(prompt.contains("[1] user: I use Rust for backend"));
        assert!(prompt.contains("[2] assistant: Rust is great for performance"));
    }

    #[test]
    fn build_prompt_empty_episodes() {
        let prompt = build_user_prompt(&[]);
        assert!(prompt.contains("Extract knowledge"));
        assert!(!prompt.contains("["));
    }

    // --- parse_extraction_response tests ---

    #[test]
    fn parse_valid_json_array() {
        let json = r#"[
            {"content": "User prefers Rust", "node_type": "fact", "confidence": 0.9},
            {"content": "Alice manages auth", "node_type": "relationship", "confidence": 0.85}
        ]"#;
        let sources = vec![EpisodeId(1), EpisodeId(2)];
        let nodes = parse_extraction_response(json, &sources).unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].content, "User prefers Rust");
        assert!(matches!(nodes[0].node_type, SemanticType::Fact));
        assert_eq!(nodes[0].confidence, 0.9);
        assert_eq!(nodes[0].source_episodes.len(), 2);
        assert_eq!(nodes[1].content, "Alice manages auth");
        assert!(matches!(nodes[1].node_type, SemanticType::Relationship));
    }

    #[test]
    fn parse_with_markdown_code_fences() {
        let json = "```json\n[{\"content\": \"test fact\", \"node_type\": \"fact\", \"confidence\": 0.8}]\n```";
        let nodes = parse_extraction_response(json, &[]).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].content, "test fact");
    }

    #[test]
    fn parse_with_bare_code_fences() {
        let json = "```\n[{\"content\": \"test\", \"node_type\": \"fact\"}]\n```";
        let nodes = parse_extraction_response(json, &[]).unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn parse_missing_confidence_defaults() {
        let json = r#"[{"content": "no confidence", "node_type": "fact"}]"#;
        let nodes = parse_extraction_response(json, &[]).unwrap();
        assert_eq!(nodes[0].confidence, 0.8);
    }

    #[test]
    fn parse_unknown_node_type_defaults_to_fact() {
        let json = r#"[{"content": "test", "node_type": "unknown_type", "confidence": 0.5}]"#;
        let nodes = parse_extraction_response(json, &[]).unwrap();
        assert!(matches!(nodes[0].node_type, SemanticType::Fact));
    }

    #[test]
    fn parse_all_node_types() {
        let json = r#"[
            {"content": "a", "node_type": "fact"},
            {"content": "b", "node_type": "relationship"},
            {"content": "c", "node_type": "event"},
            {"content": "d", "node_type": "concept"}
        ]"#;
        let nodes = parse_extraction_response(json, &[]).unwrap();
        assert!(matches!(nodes[0].node_type, SemanticType::Fact));
        assert!(matches!(nodes[1].node_type, SemanticType::Relationship));
        assert!(matches!(nodes[2].node_type, SemanticType::Event));
        assert!(matches!(nodes[3].node_type, SemanticType::Concept));
    }

    #[test]
    fn parse_confidence_clamped() {
        let json = r#"[
            {"content": "a", "node_type": "fact", "confidence": 1.5},
            {"content": "b", "node_type": "fact", "confidence": -0.3}
        ]"#;
        let nodes = parse_extraction_response(json, &[]).unwrap();
        assert_eq!(nodes[0].confidence, 1.0);
        assert_eq!(nodes[1].confidence, 0.0);
    }

    #[test]
    fn parse_empty_array() {
        let nodes = parse_extraction_response("[]", &[]).unwrap();
        assert!(nodes.is_empty());
    }

    #[test]
    fn parse_invalid_json_errors() {
        let result = parse_extraction_response("not json at all", &[]);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("parse extraction response"));
    }

    // --- parse_api_response tests ---

    #[test]
    fn parse_api_response_valid() {
        let body = r#"{"choices":[{"message":{"content":"[{\"content\":\"test\"}]"}}]}"#;
        let content = parse_api_response(body).unwrap();
        assert_eq!(content, r#"[{"content":"test"}]"#);
    }

    #[test]
    fn parse_api_response_missing_choices() {
        let body = r#"{"error": "bad request"}"#;
        let result = parse_api_response(body);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("missing choices"));
    }

    #[test]
    fn parse_api_response_invalid_json() {
        let result = parse_api_response("not json");
        assert!(result.is_err());
    }

    // --- Builder tests ---

    #[test]
    fn builder_requires_api_key() {
        let result = LlmExtractionProvider::builder().build();
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("api_key"));
    }

    #[test]
    fn builder_with_defaults() {
        let provider = LlmExtractionProvider::builder()
            .api_key("test-key")
            .build()
            .unwrap();
        assert_eq!(
            provider.api_url,
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(provider.model, "gpt-4o-mini");
    }

    #[test]
    fn builder_with_custom_values() {
        let provider = LlmExtractionProvider::builder()
            .api_key("sk-test")
            .api_url("http://localhost:11434/v1/chat/completions")
            .model("llama3.2")
            .build()
            .unwrap();
        assert_eq!(
            provider.api_url,
            "http://localhost:11434/v1/chat/completions"
        );
        assert_eq!(provider.model, "llama3.2");
        assert_eq!(provider.api_key, "sk-test");
    }

    #[test]
    fn provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LlmExtractionProvider>();
    }

    // --- extract() with empty input ---

    #[test]
    fn extract_empty_episodes_returns_empty() {
        let provider = LlmExtractionProvider::builder()
            .api_key("test")
            .build()
            .unwrap();
        let result = provider.extract(&[]).unwrap();
        assert!(result.is_empty());
    }
}
