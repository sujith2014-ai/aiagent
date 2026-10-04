//! Explicit trust roots: key_id -> Ed25519 public key. Unknown signers are rejected.
use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Default, Clone, Debug)]
pub struct TrustStore {
    keys: BTreeMap<String, VerifyingKey>,
    /// key ids that must no longer be accepted, even for packages that were valid when installed (re-checked on every load)
    revoked: std::collections::BTreeSet<String>,
}

impl TrustStore {
    pub fn from_file(p: &Path) -> Result<Self> { Self::from_json(&std::fs::read(p)?) }

    /// Trust roots as a JSON object `{"key_id": "<base64 ed25519 public key>"}` (e.g. from app assets/secure storage on a phone).
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        // `{"key_id": "<b64 pubkey>", ..., "__revoked__": ["key_id", ...]}`
        let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(bytes)?;
        let (mut keys, mut revoked) = (BTreeMap::new(), std::collections::BTreeSet::new());
        for (id, v) in raw {
            if id == "__revoked__" {
                for r in v.as_array().ok_or_else(|| anyhow!("__revoked__ must be a list"))? { revoked.insert(r.as_str().ok_or_else(|| anyhow!("revoked key ids must be strings"))?.to_string()); }
                continue;
            }
            let b64 = v.as_str().ok_or_else(|| anyhow!("key {id} must be a base64 string"))?;
            let bytes = B64.decode(b64)?;
            let arr: [u8; 32] = bytes.try_into().map_err(|_| anyhow!("bad key length for {id}"))?;
            keys.insert(id, VerifyingKey::from_bytes(&arr)?);
        }
        Ok(Self { keys, revoked })
    }
    /// Trust a key for this runtime only (e.g. the device's own signing key). Other runtimes still do not trust it.
    pub fn add_key(&mut self, key_id: &str, public: &[u8; 32]) -> Result<()> { self.keys.insert(key_id.to_string(), VerifyingKey::from_bytes(public)?); Ok(()) }

    pub fn verify(&self, key_id: &str, msg: &[u8], sig_b64: &str) -> Result<()> {
        if self.revoked.contains(key_id) { return Err(anyhow!("signer key '{key_id}' has been revoked")); }
        let key = self.keys.get(key_id).ok_or_else(|| anyhow!("untrusted signer key_id '{key_id}'"))?;
        let sig_bytes = B64.decode(sig_b64)?;
        let sig = Signature::from_slice(&sig_bytes).map_err(|e| anyhow!("malformed signature: {e}"))?;
        key.verify(msg, &sig).map_err(|_| anyhow!("signature verification failed"))
    }
    pub fn len(&self) -> usize { self.keys.len() }
    pub fn is_empty(&self) -> bool { self.keys.is_empty() }
}
