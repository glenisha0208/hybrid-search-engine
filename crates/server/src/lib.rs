//! The generated code from proto/search.proto lives here so both the
//! server binary and the client binary can share it — a Rust convention:
//! put shared code in the library part of a crate, and let separate
//! programs (in src/bin/) depend on that library.
pub mod hybridsearch {
    tonic::include_proto!("hybridsearch");
}
