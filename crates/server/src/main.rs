// Run with: cargo run -p server
//
// Starts a network service. Other programs (including the client_demo
// binary in this same crate, running as a totally separate process) can
// connect to it over the network and ask it to search or add documents —
// this is the same basic pattern real search backends use, where the
// search engine runs once, continuously, and many callers talk to it.

use fusion::HybridIndex;
use server::hybridsearch::hybrid_search_server::{HybridSearch, HybridSearchServer};
use server::hybridsearch::{
    AddDocumentRequest, AddDocumentResponse, SearchRequest, SearchResponse, SearchResult,
};
use std::sync::Mutex;
use tonic::{transport::Server, Request, Response, Status};

/// Wraps our HybridIndex so it can be shared safely across many
/// simultaneous network requests. `Mutex` ensures only one request at a
/// time can actually touch the index (needed because `add_document`
/// mutates it) — the gRPC framework handles many connections concurrently,
/// but our shared data still needs this one-at-a-time guard.
struct SearchService {
    index: Mutex<HybridIndex>,
}

#[tonic::async_trait]
impl HybridSearch for SearchService {
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let index = self.index.lock().unwrap();
        let results = index.search(&req.query, req.top_k.max(1) as usize);

        let results = results
            .into_iter()
            .map(|(doc_id, score, text)| SearchResult { doc_id, score, text })
            .collect();

        Ok(Response::new(SearchResponse { results }))
    }

    async fn add_document(
        &self,
        request: Request<AddDocumentRequest>,
    ) -> Result<Response<AddDocumentResponse>, Status> {
        let req = request.into_inner();
        let mut index = self.index.lock().unwrap();
        let doc_id = index.add_document(&req.text);
        Ok(Response::new(AddDocumentResponse { doc_id }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "127.0.0.1:50051".parse()?;

    println!("Loading AI model...");
    let mut index = HybridIndex::with_model("model/model.onnx", "model/tokenizer.json")?;

    println!("Seeding a few starter documents...");
    index.add_document("The quick brown fox jumps over the lazy dog");
    index.add_document("A budget laptop is a cheap computer for everyday tasks");
    index.add_document("Rust is a systems programming language focused on safety and speed");
    index.add_document("Search engines combine keyword matching with semantic understanding");

    let service = SearchService { index: Mutex::new(index) };

    println!("Server listening on {addr}. Leave this window open.");
    println!("Run `cargo run -p server --bin client_demo` in ANOTHER window to talk to it.");

    Server::builder()
        .add_service(HybridSearchServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
