//! `.cap` manifest types (see docs/CAPABILITY_FORMAT.md).
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Manifest {
    pub format_version: String,
    pub package_id: String,
    pub capability_id: String,
    pub version: String,
    pub model: ModelSpec,
    pub io: IoSpec,
    pub requires: Requires,
    #[serde(default)]
    pub dependencies: Vec<String>,
    pub resources: Resources,
    #[serde(default)]
    pub provenance: serde_json::Value,
    pub tests: TestsSpec,
    pub routing_file: String,
    pub created: String,
    pub signer: Signer,
    /// relative path -> sha256 hex, for every file except manifest.json/signature.json
    pub contents: BTreeMap<String, String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelSpec {
    /// e.g. "onnx"; declared so other formats can be added without a package redesign.
    pub format: String,
    pub variant: String,
    pub file: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IoSpec {
    pub input: InputSpec,
    pub output: OutputSpec,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct InputSpec {
    pub dtype: String,
    pub dim: usize,
    #[serde(default)]
    pub description: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OutputSpec {
    pub kind: String,
    pub labels: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Requires {
    pub runtime_min: String,
    #[serde(default)]
    pub device_caps: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Resources {
    pub params: u64,
    pub model_bytes: u64,
    pub min_ram_mb: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TestsSpec {
    pub file: String,
    pub count: usize,
    pub min_accuracy: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Signer {
    pub key_id: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SignatureFile {
    pub alg: String,
    pub key_id: String,
    pub sig_b64: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RoutingHints {
    pub description: String,
    pub keywords: Vec<String>,
}

#[derive(Deserialize, Debug)]
pub struct TestCase {
    pub input: Vec<f32>,
    pub expected: usize,
}
