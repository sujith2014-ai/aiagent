//! Explicit trust roots: key_id -> Ed25519 public key. Unknown signers are rejected.
use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Default, Clone, Debug)]
pub struct TrustStore {
    keys: BTreeMap<String, VerifyingKey>,
}

impl TrustStore {
    pub fn from_file(p: &Path) -> Result<Self> {
        let raw: BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(p)?)?;
        let mut keys = BTreeMap::new();
        for (id, b64) in raw {
            let bytes = B64.decode(b64)?;
            let arr: [u8; 32] = bytes.try_into().map_err(|_| anyhow!("bad key length for {id}"))?;
            keys.insert(id, VerifyingKey::from_bytes(&arr)?);
        }
        Ok(Self { keys })
    }
    pub fn verify(&self, key_id: &str, msg: &[u8], sig_b64: &str) -> Result<()> {
        let key = self.keys.get(key_id).ok_or_else(|| anyhow!("untrusted signer key_id '{key_id}'"))?;
        let sig_bytes = B64.decode(sig_b64)?;
        let sig = Signature::from_slice(&sig_bytes).map_err(|e| anyhow!("malformed signature: {e}"))?;
        key.verify(msg, &sig).map_err(|_| anyhow!("signature verification failed"))
    }
    pub fn len(&self) -> usize { self.keys.len() }
    pub fn is_empty(&self) -> bool { self.keys.is_empty() }
}
