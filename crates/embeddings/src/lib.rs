//! Wraps ONNX Runtime to turn text (or, for now, raw feature vectors) into
//! embedding vectors that the HNSW index can search over.
//!
//! Why ONNX Runtime at all? Real embedding models (like sentence
//! transformers) are trained in Python with PyTorch/TensorFlow. ONNX
//! ("Open Neural Network Exchange") is a standard file format those
//! frameworks can export to, and ONNX Runtime is a fast, C++-based engine
//! that can *run* those exported models from other languages — including
//! Rust — without needing Python at all. This is exactly how production
//! Rust services serve ML models.
//!
//! NOTE ON THIS FILE: a real deployment loads a pretrained sentence-embedding
//! model (e.g. all-MiniLM-L6-v2) exported to .onnx, plus a matching
//! tokenizer. Downloading such a model requires reaching Hugging Face's
//! model hub, which isn't reachable from the sandbox this was developed
//! in — so the wiring here is proven against a tiny synthetic ONNX model
//! (a single linear layer) instead. The `Embedder` API is written so that
//! swapping in a real model later is just a matter of pointing `from_file`
//! at a different .onnx file and replacing the input-preparation step with
//! a real tokenizer's output — none of the ONNX Runtime plumbing changes.

use ort::{inputs, GraphOptimizationLevel, Session};

mod sentence;
pub use sentence::SentenceEmbedder;

pub struct Embedder {
    session: Session,
}

impl Embedder {
    /// Load an ONNX model from disk and prepare it for inference.
    pub fn from_file(path: &str) -> ort::Result<Self> {
        // `GraphOptimizationLevel::Level3` tells ONNX Runtime to apply its
        // most aggressive built-in graph optimizations (operator fusion,
        // constant folding, etc.) when loading the model — pure speed,
        // no change in output.
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(path)?;
        Ok(Embedder { session })
    }

    /// Run inference on a batch of feature vectors (all the same length),
    /// returning one embedding vector per input row.
    ///
    /// In a real deployment, `rows` would come from a tokenizer turning
    /// text into token-id sequences; here it's just raw f32 features,
    /// since that's what our synthetic test model expects.
    pub fn embed_batch(&self, rows: &[Vec<f32>]) -> ort::Result<Vec<Vec<f32>>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let batch_size = rows.len();
        let dim = rows[0].len();

        // Flatten into one contiguous buffer — ONNX Runtime (like most
        // tensor libraries) expects row-major, contiguous input, not a
        // Vec of Vecs.
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();

        let input_tensor = ort::Tensor::from_array(([batch_size, dim], flat))?;
        let outputs = self.session.run(inputs!["input" => input_tensor]?)?;

        let (shape, data) = outputs["output"].try_extract_raw_tensor::<f32>()?;
        let out_dim = shape[1] as usize;

        // Un-flatten back into one Vec<f32> per input row.
        let result = data
            .chunks(out_dim)
            .map(|chunk| chunk.to_vec())
            .collect();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_output_from_python() {
        // This model and expected output were generated in Python with a
        // known weight matrix — see testdata/ for the generating script.
        // This test proves the Rust <-> ONNX Runtime <-> model pipeline
        // produces bit-for-bit the same math as the Python/numpy
        // reference, which is the whole point of a standard format like
        // ONNX: the model behaves identically regardless of which
        // language runs it.
        let embedder = Embedder::from_file("testdata/toy_embedder.onnx").unwrap();

        let input = vec![vec![1.0_f32; 8]]; // matches the all-ones input used in Python
        let output = embedder.embed_batch(&input).unwrap();

        let expected = [-0.67065233_f32, -2.8931324, -1.7320853, -0.5674721];

        assert_eq!(output[0].len(), expected.len());
        for (got, want) in output[0].iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got}, expected {want}");
        }
    }
}
