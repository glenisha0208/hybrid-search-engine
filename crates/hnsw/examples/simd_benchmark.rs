// Run with: cargo run -p hnsw --release --example simd_benchmark
//
// Directly compares the SIMD distance function (used inside the real
// index) against a plain, one-number-at-a-time version, isolated from
// everything else -- this is the cleanest possible measurement of what
// SIMD alone bought us.

use rand::Rng;
use std::time::Instant;

fn scalar_squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn simd_squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    use wide::f32x8;
    let len = a.len();
    let chunks = len / 8;
    let mut sum = f32x8::ZERO;
    for i in 0..chunks {
        let base = i * 8;
        let va = f32x8::from(<[f32; 8]>::try_from(&a[base..base + 8]).unwrap());
        let vb = f32x8::from(<[f32; 8]>::try_from(&b[base..base + 8]).unwrap());
        let diff = va - vb;
        sum += diff * diff;
    }
    let mut total: f32 = sum.to_array().iter().sum();
    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        total += diff * diff;
    }
    total
}

fn main() {
    let dim = 384; // matches real sentence-embedding size
    let iterations = 2_000_000;
    let mut rng = rand::thread_rng();

    let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Sanity check: both methods must agree (within floating-point
    // rounding) before we trust any speed comparison.
    let scalar_result = scalar_squared_euclidean(&a, &b);
    let simd_result = simd_squared_euclidean(&a, &b);
    assert!(
        (scalar_result - simd_result).abs() < 1e-3,
        "results disagree! scalar={scalar_result} simd={simd_result}"
    );
    println!("Correctness check passed (scalar={scalar_result:.4}, simd={simd_result:.4})\n");

    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(scalar_squared_euclidean(&a, &b));
    }
    let scalar_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(simd_squared_euclidean(&a, &b));
    }
    let simd_time = start.elapsed();

    println!("{iterations} distance calculations on {dim}-dim vectors:");
    println!("  Scalar (one number at a time): {scalar_time:.2?}");
    println!("  SIMD   (8 numbers at a time):  {simd_time:.2?}");
    println!(
        "  Speedup: {:.2}x",
        scalar_time.as_secs_f64() / simd_time.as_secs_f64()
    );
}
