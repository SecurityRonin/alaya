use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use alaya::{Alaya, EpisodeContext, NewEpisode, NewSemanticNode, NoOpProvider, Query, Role, SemanticType};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn populate_episodes(alaya: &Alaya, count: usize) {
    for i in 0..count {
        let episode = NewEpisode {
            content: format!("Episode number {} about topic {}", i, i % 10),
            role: if i % 2 == 0 { Role::User } else { Role::Assistant },
            session_id: format!("session-{}", i % 5),
            timestamp: 1000 + i as i64,
            context: EpisodeContext::default(),
            embedding: None,
        };
        alaya.episodes().store(&episode).unwrap();
    }
}

fn populate_semantic_nodes(alaya: &Alaya, count: usize) {
    let nodes: Vec<NewSemanticNode> = (0..count)
        .map(|i| NewSemanticNode {
            content: format!("Semantic fact number {} about domain {}", i, i % 7),
            node_type: SemanticType::Fact,
            confidence: 0.8,
            source_episodes: vec![],
            embedding: None,
        })
        .collect();
    alaya.knowledge().learn(nodes).unwrap();
}

/// Generate a deterministic pseudo-random f32 vector of the given dimension.
/// Uses a simple linear congruential generator seeded by `seed`.
fn pseudo_random_vec(dim: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    (0..dim)
        .map(|_| {
            // LCG parameters from Numerical Recipes
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            // Map to [-1, 1]
            (state as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_store_episode(c: &mut Criterion) {
    c.bench_function("store_episode", |b| {
        let alaya = Alaya::open_in_memory().unwrap();
        let mut i: usize = 0;
        b.iter(|| {
            let episode = NewEpisode {
                content: format!("Benchmark episode {} about concept {}", i, i % 10),
                role: if i % 2 == 0 { Role::User } else { Role::Assistant },
                session_id: format!("bench-session-{}", i % 3),
                timestamp: 2000 + i as i64,
                context: EpisodeContext::default(),
                embedding: None,
            };
            alaya.episodes().store(&episode).unwrap();
            i += 1;
        });
    });
}

fn bench_query_bm25(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_bm25");
    for size in [100, 500, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let alaya = Alaya::open_in_memory().unwrap();
            populate_episodes(&alaya, size);
            let query = Query::simple("topic about episode");
            b.iter(|| {
                alaya.knowledge().query(&query).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_consolidate(c: &mut Criterion) {
    c.bench_function("consolidate_100_episodes", |b| {
        let alaya = Alaya::open_in_memory().unwrap();
        populate_episodes(&alaya, 100);
        let provider = NoOpProvider;
        b.iter(|| {
            alaya.lifecycle().consolidate(&provider).unwrap();
        });
    });
}

fn bench_transform(c: &mut Criterion) {
    c.bench_function("transform_100_nodes", |b| {
        let alaya = Alaya::open_in_memory().unwrap();
        populate_episodes(&alaya, 50);
        populate_semantic_nodes(&alaya, 100);
        b.iter(|| {
            alaya.lifecycle().transform().unwrap();
        });
    });
}

fn bench_cosine_similarity(c: &mut Criterion) {
    c.bench_function("cosine_similarity_384d", |b| {
        let a = pseudo_random_vec(384, 42);
        let b_vec = pseudo_random_vec(384, 99);
        b.iter(|| {
            std::hint::black_box(cosine_similarity(
                std::hint::black_box(&a),
                std::hint::black_box(&b_vec),
            ));
        });
    });
}

fn bench_forget(c: &mut Criterion) {
    c.bench_function("forget_100_episodes", |b| {
        let alaya = Alaya::open_in_memory().unwrap();
        populate_episodes(&alaya, 100);
        populate_semantic_nodes(&alaya, 50);
        b.iter(|| {
            alaya.lifecycle().forget().unwrap();
        });
    });
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_store_episode,
    bench_query_bm25,
    bench_consolidate,
    bench_transform,
    bench_cosine_similarity,
    bench_forget,
);
criterion_main!(benches);
