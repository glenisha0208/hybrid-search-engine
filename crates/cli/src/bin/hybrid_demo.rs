// Run with: cargo run -p search-cli --bin hybrid_demo
//
// Demonstrates the combined (hybrid) search: BM25 keyword matching and
// real AI semantic search, merged into one ranked list via RRF.
//
// Needs model/model.onnx and model/tokenizer.json to exist (see project
// README / chat instructions for where to download these from).

use fusion::HybridIndex;

fn main() {
    let model_path = "model/model.onnx";
    let tokenizer_path = "model/tokenizer.json";

    println!("Loading AI model from {model_path} ...");
    let mut index = match HybridIndex::with_model(model_path, tokenizer_path) {
        Ok(index) => index,
        Err(e) => {
            eprintln!("Could not load the AI model: {e}");
            eprintln!(
                "Make sure model/model.onnx and model/tokenizer.json exist, \
                 and that ORT_DYLIB_PATH points at your onnxruntime.dll."
            );
            std::process::exit(1);
        }
    };
    println!("Model loaded.\n");

    let documents = [
        "The quick brown fox jumps over the lazy dog",
        "A budget laptop is a cheap computer for everyday tasks",
        "Rust is a systems programming language focused on safety and speed",
        "Search engines combine keyword matching with semantic understanding",
        "The lazy dog slept all afternoon in the warm sun",
        "Vector embeddings capture the meaning of text as numbers",
    ];

    for text in documents {
        index.add_document(text);
    }

    println!("Indexed {} documents into the hybrid engine.\n", documents.len());

    // Note this query says "affordable notebook", NOT "cheap laptop" —
    // if the AI model is working, it should still find the laptop
    // document even though none of those exact words match. That's the
    // whole point of real semantic search, versus the placeholder from
    // before.
    let queries = [
        "lazy dog",
        "affordable notebook computer",
        "rust programming",
        "meaning of text",
    ];

    for query in queries {
        println!("Query: \"{query}\"");
        let results = index.search(query, 3);
        for (doc_id, score, text) in results {
            println!("  doc {doc_id}  fused_score={score:.4}  \"{text}\"");
        }
        println!();
    }
}
