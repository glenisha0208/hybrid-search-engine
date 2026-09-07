//! A from-scratch BM25 keyword index.
//!
//! BM25 scores how relevant a document is to a query based on:
//!   - how often query terms appear in the document (term frequency)
//!   - how rare those terms are across the whole collection (inverse document frequency)
//!   - document length, normalized against the average document length
//!
//! We store an "inverted index": instead of documents pointing to words,
//! each word points to the list of documents it appears in. That's what
//! makes keyword search fast — to answer "find docs with word X", you do
//! a single hash lookup instead of scanning every document.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// One entry in a term's posting list: which document, and how many times
/// the term appeared in it.
///
/// `derive(Debug, Clone)` auto-generates code so we can `{:?}`-print this
/// struct and `.clone()` it. Rust makes you opt in to these — nothing is
/// free by default, unlike Python/JS where objects are always printable/copyable.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Posting {
    doc_id: u32,
    term_freq: u32,
}

/// The BM25 index itself.
#[derive(Serialize, Deserialize)]
pub struct Bm25Index {
    /// term -> list of (doc, count) pairs. This is the inverted index.
    postings: HashMap<String, Vec<Posting>>,
    /// doc_id -> length of that document in tokens (needed for length normalization)
    doc_lengths: HashMap<u32, usize>,
    total_tokens: usize,
    doc_count: usize,
    // BM25 tuning constants. k1 controls term-frequency saturation (how much
    // repeating a word keeps helping); b controls how strongly document
    // length is penalized. These are the standard defaults used in most
    // production BM25 implementations (Lucene, Elasticsearch).
    k1: f32,
    b: f32,
}

impl Bm25Index {
    pub fn new() -> Self {
        // `Self` here just means `Bm25Index` — a shorthand you can use
        // inside the impl block.
        Bm25Index {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_tokens: 0,
            doc_count: 0,
            k1: 1.2,
            b: 0.75,
        }
    }

    /// Lowercase + split into alphanumeric tokens. Real search engines use
    /// far more sophisticated tokenizers (stemming, stopword removal,
    /// unicode handling) — we'll keep this simple for now and can swap it
    /// out later without touching the indexing logic. That's the point of
    /// keeping it as its own function.
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    /// Add a document to the index.
    ///
    /// `&mut self` means this method needs exclusive, mutable access to
    /// the index (it's changing it). Rust's borrow checker enforces at
    /// compile time that nothing else can read or write the index while
    /// this call is happening — that's what prevents whole classes of
    /// data races, even in single-threaded code where it just means
    /// "no aliased mutation bugs."
    pub fn add_document(&mut self, doc_id: u32, text: &str) {
        let tokens = Self::tokenize(text);
        let doc_len = tokens.len();

        // Count term frequencies within this one document first,
        // so "the the the" contributes term_freq=3 for "the", not
        // three separate postings entries.
        let mut term_counts: HashMap<String, u32> = HashMap::new();
        for token in tokens {
            *term_counts.entry(token).or_insert(0) += 1;
        }

        for (term, freq) in term_counts {
            self.postings
                .entry(term)
                .or_insert_with(Vec::new)
                .push(Posting { doc_id, term_freq: freq });
        }

        self.doc_lengths.insert(doc_id, doc_len);
        self.total_tokens += doc_len;
        self.doc_count += 1;
    }

    fn avg_doc_len(&self) -> f32 {
        if self.doc_count == 0 {
            0.0
        } else {
            self.total_tokens as f32 / self.doc_count as f32
        }
    }

    /// IDF (inverse document frequency): terms that appear in fewer
    /// documents are more informative and get a higher weight.
    fn idf(&self, term: &str) -> f32 {
        let n = self.doc_count as f32;
        let n_t = self
            .postings
            .get(term)
            .map(|p| p.len())
            .unwrap_or(0) as f32;
        // The +0.5 / +1 smoothing keeps this well-behaved (never negative,
        // never divides by zero) even for terms in almost every document.
        ((n - n_t + 0.5) / (n_t + 0.5) + 1.0).ln()
    }

    /// Search the index and return the top_k (doc_id, score) pairs,
    /// highest score first.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(u32, f32)> {
        let query_terms = Self::tokenize(query);
        let avgdl = self.avg_doc_len();

        // doc_id -> accumulated score
        let mut scores: HashMap<u32, f32> = HashMap::new();

        for term in &query_terms {
            let Some(postings) = self.postings.get(term) else {
                continue; // term not in index at all -> contributes nothing
            };
            let idf = self.idf(term);

            for posting in postings {
                let doc_len = *self.doc_lengths.get(&posting.doc_id).unwrap_or(&0) as f32;
                let tf = posting.term_freq as f32;

                // The core BM25 formula: term-frequency saturation (numerator)
                // divided by length-normalized denominator.
                let numerator = tf * (self.k1 + 1.0);
                let denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / avgdl);
                let score = idf * (numerator / denominator);

                *scores.entry(posting.doc_id).or_insert(0.0) += score;
            }
        }

        // Turn the HashMap into a sorted Vec. `partial_cmp` is needed
        // (instead of `cmp`) because f32 doesn't have a total ordering
        // (NaN complicates things) — Rust makes you handle that explicitly
        // rather than silently doing something questionable with NaN.
        let mut results: Vec<(u32, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);
        results
    }

    pub fn doc_count(&self) -> usize {
        self.doc_count
    }
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

// Unit tests live in the same file, in a `mod tests` block gated by
// `#[cfg(test)]` so they're compiled only when running `cargo test`,
// never in the release binary.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_scores_higher_than_no_match() {
        let mut idx = Bm25Index::new();
        idx.add_document(1, "the quick brown fox jumps over the lazy dog");
        idx.add_document(2, "a completely unrelated sentence about oceans");

        let results = idx.search("quick fox", 10);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn rarer_terms_weight_more() {
        let mut idx = Bm25Index::new();
        idx.add_document(1, "common common common word");
        idx.add_document(2, "common common common common");
        idx.add_document(3, "common word appears here");
        idx.add_document(4, "rare word appears here only");

        // "rare" appears in fewer docs than "common", so it should carry
        // more weight per occurrence.
        let idf_common = idx.idf("common");
        let idf_rare = idx.idf("rare");
        assert!(idf_rare > idf_common);
    }

    #[test]
    fn empty_query_returns_no_results() {
        let mut idx = Bm25Index::new();
        idx.add_document(1, "some content here");
        let results = idx.search("zzz_not_present", 10);
        assert!(results.is_empty());
    }
}
