// Run with: cargo run -p server --bin client_demo
//
// A separate program that connects to the running server over the
// network and asks it to search — proving the server and client are
// genuinely two independent processes talking over a real (if local)
// network connection, not just two functions in the same program.

use server::hybridsearch::hybrid_search_client::HybridSearchClient;
use server::hybridsearch::{AddDocumentRequest, SearchRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Connecting to server at http://127.0.0.1:50051 ...");
    let mut client = HybridSearchClient::connect("http://127.0.0.1:50051").await?;
    println!("Connected.\n");

    // Add one more document at runtime, over the network, to show the
    // server's index can grow live while it's running.
    let add_response = client
        .add_document(AddDocumentRequest {
            text: "Vector embeddings capture the meaning of text as numbers".to_string(),
        })
        .await?;
    println!(
        "Added a new document over the network, got back doc id {}\n",
        add_response.into_inner().doc_id
    );

    let queries = ["affordable notebook computer", "lazy dog", "meaning of text"];

    for query in queries {
        let response = client
            .search(SearchRequest { query: query.to_string(), top_k: 3 })
            .await?;

        println!("Query: \"{query}\"");
        for result in response.into_inner().results {
            println!(
                "  doc {}  score={:.4}  \"{}\"",
                result.doc_id, result.score, result.text
            );
        }
        println!();
    }

    Ok(())
}
