//! Combines the keyword engine (bm25) and the meaning engine (hnsw) into
//! one search that returns a single, merged ranking.
//!
//! This is the "hybrid" in "hybrid search engine" — everything before this
//! crate was one algorithm or the other on its own.
//!
//! Two ways to build this index:
//!   - `HybridIndex::new()` — uses a simple placeholder for "meaning
//!     vectors" (word hashing). No AI model needed, works everywhere,
//!     good for quick tests. NOT real semantic understanding.
//!   - `HybridIndex::with_model(model_path, tokenizer_path)` — uses a
//!     real pretrained AI model (via the `embeddings` crate) for genuine
//!     semantic search.

use bm25::Bm25Index;
use embeddings::SentenceEmbedder;
use hnsw::HnswIndex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Placeholder "meaning vector": hashes each word into one of `dim`
/// buckets and counts how often each bucket is hit. Only used when no
/// real AI model is loaded.
fn placeholder_vector(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    for word in text.to_lowercase().split_whitespace() {
        let mut hash: u64 = 0;
        for byte in word.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        v[(hash as usize) % dim] += 1.0;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

pub struct HybridIndex {
    bm25: Bm25Index,
    hnsw: HnswIndex,
    documents: Vec<String>,
    embedder: Option<SentenceEmbedder>,
    placeholder_dim: usize,
}

impl HybridIndex {
    /// Build an index using the simple placeholder vectors (no AI model
    /// required). Good for quick testing.
    pub fn new() -> Self {
        HybridIndex {
            bm25: Bm25Index::new(),
            hnsw: HnswIndex::new(16, 100),
            documents: Vec::new(),
            embedder: None,
            placeholder_dim: 64,
        }
    }

    /// Build an index using a real pretrained sentence-embedding model —
    /// genuine semantic search instead of the placeholder.
    pub fn with_model(model_path: &str, tokenizer_path: &str) -> anyhow::Result<Self> {
        let embedder = SentenceEmbedder::from_files(model_path, tokenizer_path)?;
        Ok(HybridIndex {
            bm25: Bm25Index::new(),
            hnsw: HnswIndex::new(16, 100),
            documents: Vec::new(),
            embedder: Some(embedder),
            placeholder_dim: 64,
        })
    }

    fn text_to_vector(&self, text: &str) -> Vec<f32> {
        match &self.embedder {
            Some(embedder) => embedder
                .embed(&[text])
                .expect("embedding inference failed")
                .into_iter()
                .next()
                .expect("embedder returned no vector"),
            None => placeholder_vector(text, self.placeholder_dim),
        }
    }

    /// Save the index (keyword postings, vector graph, and the original
    /// document text) to a single file. Note this does NOT save the AI
    /// model itself — that stays in its own .onnx file, since it's shared,
    /// reusable weights rather than something that changes per-index.
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        #[derive(Serialize)]
        struct Persisted<'a> {
            bm25: &'a Bm25Index,
            hnsw: &'a HnswIndex,
            documents: &'a Vec<String>,
        }
        let persisted = Persisted {
            bm25: &self.bm25,
            hnsw: &self.hnsw,
            documents: &self.documents,
        };
        let bytes = bincode::serialize(&persisted)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a previously saved index back from disk.
    ///
    /// This is the "mmap" part: instead of reading the whole file into a
    /// freshly allocated buffer first (`std::fs::read`, which copies every
    /// byte through a syscall before you can even start parsing), we ask
    /// the operating system to map the file's bytes directly into our
    /// process's address space. The OS then pages the data in from disk
    /// lazily, on demand, using its normal (already-optimized) virtual
    /// memory and page-cache machinery — so loading a huge index doesn't
    /// require a huge up-front read, and the OS can share those same
    /// pages across multiple processes reading the same file.
    ///
    /// The loaded index can immediately do keyword search and search over
    /// its existing vectors. To embed NEW text (add documents or run new
    /// queries with real AI understanding), call `attach_model` afterward.
    pub fn load_readonly(path: &str) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        // `unsafe` here isn't about memory-unsafety in our code — it's
        // because mmap hands us a view into a file that another process
        // could, in principle, modify or truncate out from under us while
        // we're reading it. We're accepting that risk (reasonable for a
        // local, single-writer index file like this one).
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        #[derive(Deserialize)]
        struct Persisted {
            bm25: Bm25Index,
            hnsw: HnswIndex,
            documents: Vec<String>,
        }
        let persisted: Persisted = bincode::deserialize(&mmap[..])?;

        Ok(HybridIndex {
            bm25: persisted.bm25,
            hnsw: persisted.hnsw,
            documents: persisted.documents,
            embedder: None,
            placeholder_dim: 64,
        })
    }

    /// Attach (or replace) the AI model on an index — needed after
    /// `load_readonly` if you want to embed new text with real semantic
    /// understanding, rather than falling back to the placeholder vectors.
    pub fn attach_model(&mut self, model_path: &str, tokenizer_path: &str) -> anyhow::Result<()> {
        self.embedder = Some(SentenceEmbedder::from_files(model_path, tokenizer_path)?);
        Ok(())
    }

    /// Add a document. The same id is used across both engines (it's just
    /// this document's position in `documents`), so results from each
    /// engine can be lined back up to the same underlying document.
    pub fn add_document(&mut self, text: &str) -> u32 {
        let id = self.documents.len() as u32;
        self.documents.push(text.to_string());

        self.bm25.add_document(id, text);
        let vector = self.text_to_vector(text);
        self.hnsw.insert(vector);

        id
    }

    /// Search both engines and merge their rankings with Reciprocal Rank
    /// Fusion (RRF): a document's fused score is the sum, across every
    /// engine that ranked it, of 1 / (k + its rank in that engine).
    ///
    /// This only needs each engine's RANK ORDER, not its raw scores —
    /// which matters because BM25 scores and vector distances live on
    /// completely different, incomparable scales. `k` (60 is the standard
    /// default from the original RRF paper) softens how much the very top
    /// rank dominates.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(u32, f32, String)> {
        const K: f32 = 60.0;
        let candidate_pool = (top_k * 5).max(20);

        let keyword_results = self.bm25.search(query, candidate_pool);
        let query_vector = self.text_to_vector(query);
        let semantic_results = self.hnsw.search(&query_vector, candidate_pool, candidate_pool);

        let mut fused_scores: HashMap<u32, f32> = HashMap::new();

        for (rank, (doc_id, _score)) in keyword_results.iter().enumerate() {
            *fused_scores.entry(*doc_id).or_insert(0.0) += 1.0 / (K + rank as f32 + 1.0);
        }
        for (rank, (doc_id, _dist)) in semantic_results.iter().enumerate() {
            let doc_id = *doc_id as u32;
            *fused_scores.entry(doc_id).or_insert(0.0) += 1.0 / (K + rank as f32 + 1.0);
        }

        let mut results: Vec<(u32, f32)> = fused_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);

        results
            .into_iter()
            .map(|(id, score)| (id, score, self.documents[id as usize].clone()))
            .collect()
    }
}

impl Default for HybridIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyword_match_still_surfaces_in_fused_results() {
        let mut index = HybridIndex::new();
        index.add_document("The quick brown fox jumps over the lazy dog");
        index.add_document("A budget laptop is a cheap computer for everyday tasks");
        index.add_document("Completely unrelated content about deep sea fish");

        let results = index.search("cheap computer", 3);
        assert_eq!(results[0].0, 1); // the laptop document should win
    }

    #[test]
    fn results_are_sorted_descending_by_score() {
        let mut index = HybridIndex::new();
        index.add_document("apples and oranges");
        index.add_document("bananas and grapes");
        index.add_document("cars and trucks");

        let results = index.search("apples", 3);
        for pair in results.windows(2) {
            assert!(pair[0].1 >= pair[1].1);
        }
    }

    #[test]
    fn save_and_load_preserves_search_results() {
        let mut index = HybridIndex::new();
        index.add_document("The quick brown fox jumps over the lazy dog");
        index.add_document("A budget laptop is a cheap computer for everyday tasks");
        index.add_document("Completely unrelated content about deep sea fish");

        let path = std::env::temp_dir().join("hybrid_test_index.bin");
        let path_str = path.to_str().unwrap();

        index.save(path_str).expect("save should succeed");
        let loaded = HybridIndex::load_readonly(path_str).expect("load should succeed");

        let original_results = index.search("cheap computer", 3);
        let loaded_results = loaded.search("cheap computer", 3);

        assert_eq!(original_results.len(), loaded_results.len());
        for (orig, reloaded) in original_results.iter().zip(loaded_results.iter()) {
            assert_eq!(orig.0, reloaded.0); // same document ids, same order
        }

        std::fs::remove_file(path_str).ok();
    }
}
