// Run with: cargo run -p search-cli --bin persistence_demo
//
// Demonstrates saving a hybrid index to disk and loading it back via
// memory-mapping, without needing to rebuild anything from scratch or
// re-run the AI model on already-indexed documents.

use fusion::HybridIndex;

fn main() {
    let save_path = "saved_index.bin";

    println!("Building a fresh index and adding documents...");
    let mut index = HybridIndex::new(); // placeholder vectors -- no AI model needed for this demo
    index.add_document("The quick brown fox jumps over the lazy dog");
    index.add_document("A budget laptop is a cheap computer for everyday tasks");
    index.add_document("Rust is a systems programming language focused on safety and speed");

    println!("Searching before saving:");
    for (id, score, text) in index.search("cheap computer", 2) {
        println!("  doc {id}  score={score:.4}  \"{text}\"");
    }

    index.save(save_path).expect("save failed");
    println!("\nSaved index to {save_path}.\n");

    println!("Loading the index back from disk (via mmap, not rebuilding)...");
    let loaded = HybridIndex::load_readonly(save_path).expect("load failed");

    println!("Searching after loading -- should be identical:");
    for (id, score, text) in loaded.search("cheap computer", 2) {
        println!("  doc {id}  score={score:.4}  \"{text}\"");
    }
}
