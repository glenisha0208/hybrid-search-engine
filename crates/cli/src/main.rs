// `use` brings items from other crates/modules into scope. `bm25::Bm25Index`
// is the public struct we defined in crates/bm25/src/lib.rs — `pub` on a
// struct/fn is what makes it visible outside its own crate.
use bm25::Bm25Index;

fn main() {
    let mut index = Bm25Index::new();

    // A tiny toy corpus. In later phases this will come from real files.
    let documents = [
        (1, "The quick brown fox jumps over the lazy dog"),
        (2, "A budget laptop is a cheap computer for everyday tasks"),
        (3, "Rust is a systems programming language focused on safety and speed"),
        (4, "Search engines combine keyword matching with semantic understanding"),
        (5, "The lazy dog slept all afternoon in the warm sun"),
        (6, "Vector embeddings capture the meaning of text as numbers"),
    ];

    for (id, text) in documents {
        index.add_document(id, text);
    }

    println!("Indexed {} documents.\n", index.doc_count());

    let queries = ["lazy dog", "cheap computer", "rust programming", "meaning of text"];

    for query in queries {
        println!("Query: \"{query}\"");
        let results = index.search(query, 3);
        if results.is_empty() {
            println!("  (no matches)");
        }
        for (doc_id, score) in results {
            let text = documents.iter().find(|(id, _)| *id == doc_id).unwrap().1;
            println!("  doc {doc_id}  score={score:.3}  \"{text}\"");
        }
        println!();
    }
}
