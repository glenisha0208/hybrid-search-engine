// Run with: cargo run -p search-cli --bin interactive_search
//
// Lets you type any search query yourself and see real results, instead
// of the fixed example queries in hybrid_demo.rs.

use fusion::HybridIndex;
use std::io::{self, Write};

fn main() {
    println!("Loading AI model...");
    let mut index = match HybridIndex::with_model("model/model.onnx", "model/tokenizer.json") {
        Ok(index) => index,
        Err(e) => {
            eprintln!("Could not load the AI model: {e}");
            eprintln!("Make sure model/model.onnx and model/tokenizer.json exist.");
            std::process::exit(1);
        }
    };
    println!("Model loaded.\n");

    // Same starter documents as the other demo -- feel free to add your
    // own below, or change these entirely.
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
    println!("Indexed {} documents.\n", documents.len());
    println!("Type a search query and press Enter. Type 'quit' to exit.\n");

    let stdin = io::stdin();
    loop {
        print!("Search> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if stdin.read_line(&mut input).is_err() {
            break;
        }
        let query = input.trim();

        if query.is_empty() {
            continue;
        }
        if query.eq_ignore_ascii_case("quit") {
            break;
        }

        let results = index.search(query, 3);
        if results.is_empty() {
            println!("  (no matches)\n");
            continue;
        }
        for (doc_id, score, text) in results {
            println!("  doc {doc_id}  score={score:.4}  \"{text}\"");
        }
        println!();
    }

    println!("Goodbye!");
}
