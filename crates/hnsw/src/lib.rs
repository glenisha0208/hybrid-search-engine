//! A from-scratch HNSW (Hierarchical Navigable Small World) index for
//! approximate nearest-neighbor search over vectors.
//!
//! See the conversation for the conceptual explanation. In short: we
//! maintain a graph of vectors split across multiple layers. Layer 0 has
//! every vector; higher layers have exponentially fewer, acting as
//! "express lanes" for fast traversal. Search greedily descends from the
//! top layer to layer 0, narrowing in on the true nearest neighbors.

use rand::Rng;
use serde::{Serialize, Deserialize};
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashSet;

/// Squared Euclidean distance. We use the *squared* distance (skip the
/// sqrt) everywhere internally: sqrt is monotonic, so it never changes
/// which of two candidates is closer — we only need relative ordering,
/// not the true distance value, so we save the (relatively expensive)
/// sqrt call entirely.
///
/// This is SIMD-accelerated: instead of subtracting and squaring one
/// number at a time, `wide::f32x8` packs 8 numbers into a single CPU
/// register and does the subtract-square-add on all 8 in roughly the
/// time a normal loop takes to do 1. This function runs millions of
/// times during a single search (it's the innermost operation of both
/// insertion and search), so it's exactly the kind of hot, simple,
/// repetitive math SIMD is built for.
fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    use wide::f32x8;

    let len = a.len();
    let chunks = len / 8; // how many full groups of 8 we can process at once
    let mut sum = f32x8::ZERO;

    for i in 0..chunks {
        let base = i * 8;
        // Load 8 numbers from each vector into one SIMD register.
        let va = f32x8::from(<[f32; 8]>::try_from(&a[base..base + 8]).unwrap());
        let vb = f32x8::from(<[f32; 8]>::try_from(&b[base..base + 8]).unwrap());
        let diff = va - vb;
        sum += diff * diff; // squares and adds all 8 lanes simultaneously
    }

    // `sum` now holds 8 partial sums (one per SIMD lane) — add them
    // together into a single number.
    let mut total: f32 = sum.to_array().iter().sum();

    // Vector lengths aren't always a multiple of 8 (e.g. a 384-dim
    // sentence embedding is; a toy 3-dim test vector isn't) — handle
    // whatever's left over the plain, scalar way.
    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        total += diff * diff;
    }

    total
}

/// A (node id, distance) pair used in our search priority queues.
///
/// f32 doesn't implement `Ord` (NaN has no defined order), so we can't put
/// raw f32s in a `BinaryHeap` directly. We wrap them in this struct and
/// implement `Ord` ourselves, asserting there are no NaNs in practice
/// (our distances are always real numbers, never NaN, so `.unwrap()` on
/// `partial_cmp` is safe here).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    id: usize,
    dist: f32,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap()
    }
}

/// One node's graph structure: its neighbor list at each layer it
/// participates in. `layers[0]` is always present (layer 0 has every
/// node); `layers[1]`, `layers[2]`, ... exist only if this node was
/// randomly promoted to higher layers.
#[derive(Serialize, Deserialize)]
struct Node {
    layers: Vec<Vec<usize>>,
}

#[derive(Serialize, Deserialize)]
pub struct HnswIndex {
    vectors: Vec<Vec<f32>>,
    nodes: Vec<Node>,
    entry_point: Option<usize>,
    top_layer: usize,
    // Max neighbors per node at layers above 0.
    m: usize,
    // Max neighbors per node at layer 0 (conventionally 2*m — layer 0
    // needs to stay well-connected since it carries the final, precise
    // search).
    m0: usize,
    // Candidate list size used while building the graph. Bigger =
    // better-quality graph (higher recall later) but slower to build.
    ef_construction: usize,
    // Controls how quickly the number of nodes shrinks per layer.
    // Standard choice: 1 / ln(m).
    level_mult: f64,
}

impl HnswIndex {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        HnswIndex {
            vectors: Vec::new(),
            nodes: Vec::new(),
            entry_point: None,
            top_layer: 0,
            m,
            m0: m * 2,
            ef_construction,
            level_mult: 1.0 / (m as f64).ln(),
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Randomly assign a max layer for a new node. Most nodes land at
    /// layer 0 only; progressively fewer reach higher layers. This
    /// exponential decay is what makes higher layers sparse "highways"
    /// rather than just smaller copies of layer 0.
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen_range(0.0..1.0);
        (-r.ln() * self.level_mult).floor() as usize
    }

    /// Greedy search within a single layer, starting from `entry_points`,
    /// keeping track of the `ef` best candidates found. This is the
    /// workhorse used by both insertion and search.
    ///
    /// Returns candidates sorted closest-first.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited: HashSet<usize> = entry_points.iter().copied().collect();

        // `candidates` is a min-heap (via Reverse) of nodes still worth
        // exploring, ordered closest-first.
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        // `results` is a max-heap of the best candidates found so far,
        // capped at size `ef`. Keeping it as a max-heap means the
        // *worst* of our current best is always at the top — cheap to
        // check "should this new candidate bump something out?" and
        // cheap to evict it with `.pop()`.
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        for &ep in entry_points {
            let dist = squared_euclidean(query, &self.vectors[ep]);
            candidates.push(Reverse(Candidate { id: ep, dist }));
            results.push(Candidate { id: ep, dist });
        }

        while let Some(Reverse(current)) = candidates.pop() {
            // If the closest thing left to explore is already farther
            // than our current worst kept result, nothing further out
            // can improve the result set — stop early.
            if let Some(worst) = results.peek() {
                if current.dist > worst.dist && results.len() >= ef {
                    break;
                }
            }

            let Some(neighbors) = self.nodes[current.id].layers.get(layer) else {
                continue; // this node doesn't exist at this layer
            };

            for &neighbor_id in neighbors {
                if !visited.insert(neighbor_id) {
                    continue; // already seen
                }
                let dist = squared_euclidean(query, &self.vectors[neighbor_id]);
                let worst_dist = results.peek().map(|c| c.dist).unwrap_or(f32::INFINITY);

                if results.len() < ef || dist < worst_dist {
                    candidates.push(Reverse(Candidate { id: neighbor_id, dist }));
                    results.push(Candidate { id: neighbor_id, dist });
                    if results.len() > ef {
                        results.pop(); // evict the current worst
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = results.into_vec();
        out.sort(); // ascending by dist, thanks to our Ord impl
        out
    }

    /// Insert a new vector into the index.
    pub fn insert(&mut self, vector: Vec<f32>) -> usize {
        let id = self.vectors.len();
        let node_level = self.random_level();

        self.vectors.push(vector);
        self.nodes.push(Node {
            layers: vec![Vec::new(); node_level + 1],
        });

        // First node: it becomes the entry point, nothing to connect.
        let Some(entry_point) = self.entry_point else {
            self.entry_point = Some(id);
            self.top_layer = node_level;
            return id;
        };

        let query = self.vectors[id].clone();
        let mut current = entry_point;

        // Phase 1: descend from the top layer down to node_level + 1
        // using pure greedy search (ef=1) — we just want the single
        // closest node to use as our entry point once we reach a layer
        // we'll actually connect at.
        for layer in (node_level + 1..=self.top_layer).rev() {
            let nearest = self.search_layer(&query, &[current], 1, layer);
            if let Some(best) = nearest.first() {
                current = best.id;
            }
        }

        // Phase 2: from node_level down to 0, do a wider search
        // (ef_construction) and connect the new node to its M nearest
        // neighbors at each layer.
        for layer in (0..=node_level.min(self.top_layer)).rev() {
            let candidates = self.search_layer(&query, &[current], self.ef_construction, layer);
            let max_conn = if layer == 0 { self.m0 } else { self.m };
            let selected: Vec<usize> = candidates.iter().take(max_conn).map(|c| c.id).collect();

            if let Some(best) = candidates.first() {
                current = best.id;
            }

            // Connect new node -> selected neighbors.
            self.nodes[id].layers[layer] = selected.clone();

            // Connect selected neighbors -> new node (bidirectional),
            // pruning each neighbor's list back down to max_conn if it
            // grew too large.
            for &neighbor_id in &selected {
                let neighbor_layers = &mut self.nodes[neighbor_id].layers;
                if layer >= neighbor_layers.len() {
                    continue; // neighbor doesn't exist at this layer (shouldn't happen, but be safe)
                }
                neighbor_layers[layer].push(id);

                if neighbor_layers[layer].len() > max_conn {
                    // Keep only the max_conn closest neighbors. We need
                    // the neighbor's own vector to measure distance —
                    // read it before taking the mutable borrow above to
                    // avoid borrowing `self.vectors` and `self.nodes`
                    // mutably at the same time.
                    let neighbor_vec = &self.vectors[neighbor_id];
                    let mut ranked: Vec<Candidate> = self.nodes[neighbor_id].layers[layer]
                        .iter()
                        .map(|&other_id| Candidate {
                            id: other_id,
                            dist: squared_euclidean(neighbor_vec, &self.vectors[other_id]),
                        })
                        .collect();
                    ranked.sort();
                    ranked.truncate(max_conn);
                    self.nodes[neighbor_id].layers[layer] = ranked.into_iter().map(|c| c.id).collect();
                }
            }
        }

        // If this node reached a higher layer than anything before it,
        // it becomes the new entry point.
        if node_level > self.top_layer {
            self.top_layer = node_level;
            self.entry_point = Some(id);
        }

        id
    }

    /// Find the `k` approximate nearest neighbors to `query`.
    /// Returns (id, squared_distance) pairs, closest first.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(usize, f32)> {
        let Some(entry_point) = self.entry_point else {
            return Vec::new();
        };

        let mut current = entry_point;

        // Greedy descent through the upper layers, same as insertion.
        for layer in (1..=self.top_layer).rev() {
            let nearest = self.search_layer(query, &[current], 1, layer);
            if let Some(best) = nearest.first() {
                current = best.id;
            }
        }

        // Wide search at layer 0 for the final, precise result.
        let ef = ef_search.max(k);
        let mut results = self.search_layer(query, &[current], ef, 0);
        results.truncate(k);
        results.into_iter().map(|c| (c.id, c.dist)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_vector(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
        (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    /// Exact brute-force k-NN, used as ground truth to measure HNSW's
    /// recall against.
    fn brute_force_knn(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, squared_euclidean(query, v)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn finds_exact_match() {
        let mut index = HnswIndex::new(16, 100);
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            index.insert(random_vector(32, &mut rng));
        }
        let target = random_vector(32, &mut rng);
        let target_id = index.insert(target.clone());

        let results = index.search(&target, 1, 50);
        assert_eq!(results[0].0, target_id);
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn recall_against_brute_force_is_high() {
        let dim = 32;
        let n = 1000;
        let k = 10;
        let mut rng = rand::thread_rng();

        let mut index = HnswIndex::new(16, 200);
        let mut all_vectors = Vec::new();
        for _ in 0..n {
            let v = random_vector(dim, &mut rng);
            all_vectors.push(v.clone());
            index.insert(v);
        }

        let num_queries = 20;
        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let query = random_vector(dim, &mut rng);
            let approx: HashSet<usize> = index
                .search(&query, k, 100)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let exact: HashSet<usize> = brute_force_knn(&all_vectors, &query, k)
                .into_iter()
                .collect();

            let overlap = approx.intersection(&exact).count();
            total_recall += overlap as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        println!("Average recall@{k}: {avg_recall:.2}");
        // HNSW is approximate by design, but with these parameters on
        // random data it should still find the great majority of true
        // nearest neighbors.
        assert!(avg_recall > 0.8, "recall too low: {avg_recall}");
    }
}
