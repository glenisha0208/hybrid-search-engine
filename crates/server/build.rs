fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Relies on a `protoc` compiler being available -- either on your
    // PATH, or pointed to directly via the PROTOC environment variable.
    // See chat instructions for downloading a ready-made protoc.exe.
    tonic_build::compile_protos("proto/search.proto")?;
    Ok(())
}
