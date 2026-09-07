// Run with: cargo run -p hnsw --release --example benchmark
//
// This is a Rust "example" — files under examples/ are separate small
// binaries that can use the crate's public API, useful for demos and
// manual benchmarks without cluttering the library itself.

use hnsw::HnswIndex;
use rand::Rng;
use std::time::Instant;

fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn brute_force_search(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, squared_euclidean(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn main() {
    let dim = 64;
    let n = 50_000;
    let k = 10;
    let num_queries = 200;
    let mut rng = rand::thread_rng();

    println!("Building {n} random {dim}-dim vectors...");
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    println!("Indexing into HNSW...");
    let build_start = Instant::now();
    let mut index = HnswIndex::new(16, 100);
    for v in &vectors {
        index.insert(v.clone());
    }
    println!("Build time: {:.2?}\n", build_start.elapsed());

    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    let brute_start = Instant::now();
    for q in &queries {
        std::hint::black_box(brute_force_search(&vectors, q, k));
    }
    let brute_elapsed = brute_start.elapsed();

    let hnsw_start = Instant::now();
    for q in &queries {
        std::hint::black_box(index.search(q, k, 50));
    }
    let hnsw_elapsed = hnsw_start.elapsed();

    println!("{num_queries} queries over {n} vectors:");
    println!(
        "  Brute force: {:.2?}  ({:.2?}/query)",
        brute_elapsed,
        brute_elapsed / num_queries as u32
    );
    println!(
        "  HNSW:        {:.2?}  ({:.2?}/query)",
        hnsw_elapsed,
        hnsw_elapsed / num_queries as u32
    );
    println!(
        "  Speedup: {:.1}x",
        brute_elapsed.as_secs_f64() / hnsw_elapsed.as_secs_f64()
    );
}
