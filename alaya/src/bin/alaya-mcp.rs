//! Alaya MCP Server — expose memory operations over Model Context Protocol.
//!
//! Usage:
//!   cargo build --release --features mcp
//!   ./target/release/alaya-mcp
//!
//! Environment:
//!   ALAYA_DB           — path to SQLite database (default: ~/.alaya/memory.db)
//!   ALAYA_LLM_API_KEY  — API key for auto-consolidation (enables ExtractionProvider)
//!   ALAYA_LLM_API_URL  — API endpoint (default: https://api.openai.com/v1/chat/completions)
//!   ALAYA_LLM_MODEL    — model name (default: gpt-4o-mini)

use std::path::PathBuf;

use alaya::mcp::AlayaMcp;
#[allow(deprecated)]
use alaya::AlayaStore;
use rmcp::ServiceExt;
use tokio::io::{stdin, stdout};

#[cfg(not(tarpaulin_include))]
fn resolve_db_path() -> PathBuf {
    if let Ok(path) = std::env::var("ALAYA_DB") {
        return PathBuf::from(path);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let dir = PathBuf::from(home).join(".alaya");
    std::fs::create_dir_all(&dir).ok();
    dir.join("memory.db")
}

/// Configure LLM extraction provider from environment variables.
/// Returns silently if ALAYA_LLM_API_KEY is not set (auto-consolidation disabled).
#[cfg(feature = "llm")]
#[cfg(not(tarpaulin_include))]
fn configure_extraction(store: &mut AlayaStore) {
    let api_key = match std::env::var("ALAYA_LLM_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => return, // No key = no auto-consolidation, silent
    };

    let mut builder = alaya::LlmExtractionProvider::builder().api_key(api_key);

    if let Ok(url) = std::env::var("ALAYA_LLM_API_URL") {
        if !url.is_empty() {
            builder = builder.api_url(url);
        }
    }
    if let Ok(model) = std::env::var("ALAYA_LLM_MODEL") {
        if !model.is_empty() {
            builder = builder.model(model);
        }
    }

    match builder.build() {
        Ok(provider) => {
            eprintln!(
                "alaya-mcp: auto-consolidation enabled (model: {})",
                std::env::var("ALAYA_LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into())
            );
            store.set_extraction_provider(Box::new(provider));
        }
        Err(e) => {
            eprintln!("alaya-mcp: failed to configure extraction provider: {e}");
        }
    }
}

#[tokio::main]
#[cfg(not(tarpaulin_include))]
async fn main() -> anyhow::Result<()> {
    #[cfg(feature = "tracing")]
    {
        tracing_subscriber::fmt()
            .with_writer(std::io::stderr)
            .init();
    }

    let db_path = resolve_db_path();
    eprintln!("alaya-mcp: opening database at {}", db_path.display());

    let mut store = AlayaStore::open(&db_path)?;

    #[cfg(feature = "llm")]
    configure_extraction(&mut store);

    let server = AlayaMcp::new(store);

    let transport = (stdin(), stdout());
    let service = server.serve(transport).await?;
    service.waiting().await?;

    Ok(())
}
