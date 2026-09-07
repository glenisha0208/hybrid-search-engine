//! Real sentence embeddings: tokenize text the way a BERT-style model
//! expects, run it through an ONNX-exported transformer, and pool the
//! output into one fixed-size vector per sentence.
//!
//! This is the piece that actually turns text into something the HNSW
//! index can search over semantically. It's written against models like
//! `all-MiniLM-L6-v2` — small (~90MB), fast, and the most common choice
//! for exactly this kind of project.
//!
//! IMPORTANT: this module needs two files that aren't included in this
//! repo (they're too large / require Hugging Face, which the sandbox this
//! was built in couldn't reach — see the setup instructions that came with
//! this code):
//!   - a `model.onnx` file (the exported transformer weights)
//!   - a `tokenizer.json` file (the matching WordPiece vocabulary + rules)
//! Both come from the same Hugging Face model repo, so they're guaranteed
//! to match each other as long as you download them together.

use ort::{inputs, GraphOptimizationLevel, Session};
use tokenizers::Tokenizer;

pub struct SentenceEmbedder {
    tokenizer: Tokenizer,
    session: Session,
}

impl SentenceEmbedder {
    pub fn from_files(model_path: &str, tokenizer_path: &str) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        Ok(SentenceEmbedder { tokenizer, session })
    }

    /// Embed a batch of sentences, returning one L2-normalized vector per
    /// sentence. Normalizing here means that later, comparing two
    /// embeddings with a plain dot product gives you cosine similarity for
    /// free — this is the standard convention sentence-transformer models
    /// are trained around.
    pub fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        // `encode_batch` tokenizes every sentence and, with `true` as the
        // second argument, adds the special [CLS]/[SEP] tokens BERT-style
        // models expect at the start/end of each sequence.
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        // Sentences are different lengths, but a tensor needs a single
        // rectangular shape — so we pad every sequence up to the length
        // of the longest one in this batch. `attention_mask` records
        // which positions are real tokens (1) vs. padding (0), so the
        // model (and our pooling step below) can ignore the padding.
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        let batch_size = encodings.len();

        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];
        let mut token_type_ids = vec![0i64; batch_size * max_len];

        for (row, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            for col in 0..ids.len() {
                input_ids[row * max_len + col] = ids[col] as i64;
                attention_mask[row * max_len + col] = mask[col] as i64;
                // token_type_ids distinguishes sentence A vs sentence B in
                // tasks like "does sentence B answer sentence A?". We only
                // ever pass one sentence at a time, so this stays all
                // zeros — but the model still expects the input to exist.
            }
        }

        let input_ids_tensor = ort::Tensor::from_array(([batch_size, max_len], input_ids))?;
        let attention_mask_tensor =
            ort::Tensor::from_array(([batch_size, max_len], attention_mask.clone()))?;
        let token_type_ids_tensor =
            ort::Tensor::from_array(([batch_size, max_len], token_type_ids))?;

        let outputs = self.session.run(inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ]?)?;

        // BERT-style models output one vector per input token:
        // shape [batch, seq_len, hidden_dim]. "last_hidden_state" is the
        // standard output name for Hugging Face transformer exports.
        let (shape, data) = outputs["last_hidden_state"].try_extract_raw_tensor::<f32>()?;
        let hidden_dim = shape[2] as usize;

        let mut results = Vec::with_capacity(batch_size);
        for row in 0..batch_size {
            // Mean pooling: average the per-token vectors into one
            // sentence vector, but only over real (non-padding) tokens —
            // that's what the attention mask is for here.
            let mut pooled = vec![0f32; hidden_dim];
            let mut valid_tokens = 0f32;

            for col in 0..max_len {
                if attention_mask[row * max_len + col] == 0 {
                    continue; // padding, skip
                }
                valid_tokens += 1.0;
                let base = (row * max_len + col) * hidden_dim;
                for d in 0..hidden_dim {
                    pooled[d] += data[base + d];
                }
            }
            for v in pooled.iter_mut() {
                *v /= valid_tokens.max(1.0);
            }

            // L2-normalize so a plain dot product later equals cosine
            // similarity.
            let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in pooled.iter_mut() {
                    *v /= norm;
                }
            }

            results.push(pooled);
        }

        Ok(results)
    }
}
