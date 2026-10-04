//! `.cap` reading and the verification half of the activation sequence:
//! signature -> hashes -> compatibility. (Sandbox load, tests and resource
//! checks happen in `runtime`.)
use crate::capability::*;
use crate::device::DeviceProfile;
use crate::trust::TrustStore;
use crate::{RUNTIME_VERSION, SUPPORTED_FORMATS};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum PackageError {
    #[error("malformed package: {0}")]
    Malformed(String),
    #[error("signature check failed: {0}")]
    Signature(String),
    #[error("hash mismatch / corruption: {0}")]
    Hash(String),
    #[error("incompatible: {0}")]
    Incompatible(String),
}

pub struct CapPackage {
    pub manifest_bytes: Vec<u8>,
    pub manifest: Manifest,
    pub signature: SignatureFile,
    pub files: BTreeMap<String, Vec<u8>>,
}

const MAX_PACKAGE_FILE: u64 = 256 << 20;

impl CapPackage {
    pub fn open(path: &Path) -> Result<Self, PackageError> {
        let f = std::fs::File::open(path).map_err(|e| PackageError::Malformed(e.to_string()))?;
        let mut zip = zip::ZipArchive::new(f).map_err(|e| PackageError::Malformed(format!("not a zip: {e}")))?;
        let mut files = BTreeMap::new();
        for i in 0..zip.len() {
            let mut entry = zip.by_index(i).map_err(|e| PackageError::Malformed(e.to_string()))?;
            if entry.is_dir() { continue; }
            let name = entry.name().to_string();
            if name.starts_with('/') || name.contains("..") || name.contains('\\') {
                return Err(PackageError::Malformed(format!("unsafe path '{name}'")));
            }
            if entry.size() > MAX_PACKAGE_FILE {
                return Err(PackageError::Malformed(format!("file too large: {name}")));
            }
            let mut buf = Vec::new();
            (&mut entry).take(MAX_PACKAGE_FILE).read_to_end(&mut buf)
                .map_err(|e| PackageError::Malformed(format!("{name}: {e}")))?;
            files.insert(name, buf);
        }
        let manifest_bytes = files.remove("manifest.json")
            .ok_or_else(|| PackageError::Malformed("missing manifest.json".into()))?;
        let sig_bytes = files.remove("signature.json")
            .ok_or_else(|| PackageError::Malformed("missing signature.json".into()))?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| PackageError::Malformed(format!("manifest: {e}")))?;
        let signature: SignatureFile = serde_json::from_slice(&sig_bytes)
            .map_err(|e| PackageError::Malformed(format!("signature.json: {e}")))?;
        Ok(Self { manifest_bytes, manifest, signature, files })
    }

    /// Step 1: signature over the exact manifest bytes, by a trusted key matching the declared signer.
    pub fn verify_signature(&self, trust: &TrustStore) -> Result<(), PackageError> {
        if self.signature.alg != "ed25519" {
            return Err(PackageError::Signature(format!("unsupported alg {}", self.signature.alg)));
        }
        if self.signature.key_id != self.manifest.signer.key_id {
            return Err(PackageError::Signature("signature key_id != manifest signer".into()));
        }
        trust.verify(&self.signature.key_id, &self.manifest_bytes, &self.signature.sig_b64)
            .map_err(|e| PackageError::Signature(e.to_string()))
    }

    /// Step 2: every declared file present with the right hash, and no undeclared files.
    pub fn verify_hashes(&self) -> Result<(), PackageError> {
        for (path, want) in &self.manifest.contents {
            let data = self.files.get(path).ok_or_else(|| PackageError::Hash(format!("missing file {path}")))?;
            let got = hex::encode(Sha256::digest(data));
            if &got != want {
                return Err(PackageError::Hash(format!("{path}: expected {want}, got {got}")));
            }
        }
        for path in self.files.keys() {
            if !self.manifest.contents.contains_key(path) {
                return Err(PackageError::Hash(format!("undeclared file {path}")));
            }
        }
        for needed in [&self.manifest.model.file, &self.manifest.tests.file, &self.manifest.routing_file] {
            if !self.manifest.contents.contains_key(needed) {
                return Err(PackageError::Malformed(format!("manifest references undeclared file {needed}")));
            }
        }
        Ok(())
    }

    /// Step 3: format, runtime version, device capabilities, and declared resource needs.
    pub fn check_compat(&self, dev: &DeviceProfile, supported_model_formats: &[&str]) -> Result<(), PackageError> {
        let m = &self.manifest;
        if !SUPPORTED_FORMATS.contains(&m.format_version.as_str()) {
            return Err(PackageError::Incompatible(format!("package format {} unsupported", m.format_version)));
        }
        let need = semver::Version::parse(&m.requires.runtime_min)
            .map_err(|e| PackageError::Malformed(format!("runtime_min: {e}")))?;
        let have = semver::Version::parse(RUNTIME_VERSION).unwrap();
        if need > have {
            return Err(PackageError::Incompatible(format!("needs runtime >= {need}, have {have}")));
        }
        if !supported_model_formats.contains(&m.model.format.as_str()) {
            return Err(PackageError::Incompatible(format!("no backend for model format '{}'", m.model.format)));
        }
        for c in &m.requires.device_caps {
            if !dev.caps.contains(c) {
                return Err(PackageError::Incompatible(format!("device '{}' lacks capability '{c}'", dev.name)));
            }
        }
        if m.io.input.dtype != "f32" {
            return Err(PackageError::Incompatible(format!("input dtype {} unsupported", m.io.input.dtype)));
        }
        semver::Version::parse(&m.version).map_err(|e| PackageError::Malformed(format!("version: {e}")))?;
        if m.resources.min_ram_mb > dev.max_ram_mb {
            return Err(PackageError::Incompatible(format!("needs {} MB RAM, device has {}", m.resources.min_ram_mb, dev.max_ram_mb)));
        }
        if m.resources.model_bytes > dev.max_model_bytes {
            return Err(PackageError::Incompatible(format!("model {} B exceeds device limit {} B", m.resources.model_bytes, dev.max_model_bytes)));
        }
        Ok(())
    }

    pub fn model_bytes(&self) -> &[u8] { &self.files[&self.manifest.model.file] }

    pub fn tests(&self) -> Result<Vec<TestCase>, PackageError> {
        let raw = &self.files[&self.manifest.tests.file];
        let mut out = Vec::new();
        for line in String::from_utf8_lossy(raw).lines().filter(|l| !l.trim().is_empty()) {
            out.push(serde_json::from_str(line).map_err(|e| PackageError::Malformed(format!("tests: {e}")))?);
        }
        Ok(out)
    }

    pub fn hints(&self) -> Result<RoutingHints, PackageError> {
        serde_json::from_slice(&self.files[&self.manifest.routing_file])
            .map_err(|e| PackageError::Malformed(format!("routing hints: {e}")))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn semver_ordering() {
        assert!(semver::Version::parse("0.2.0").unwrap() > semver::Version::parse("0.1.0").unwrap());
    }
}
