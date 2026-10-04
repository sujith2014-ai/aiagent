//! Persistent capability registry. Routing never switches on capability names;
//! it scores records by their declared metadata (see `router`).
use crate::capability::{Calibration, InputStats};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VersionRecord {
    pub version: String,
    pub package_id: String,
    pub store_file: String,
    pub params: u64,
    pub model_bytes: u64,
    pub test_accuracy: f64,
    pub activated_at: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Stats {
    pub calls: u64,
    pub success: u64,
    pub failure: u64,
    pub total_latency_us: u64,
    pub max_latency_us: u64,
    pub last_used: u64,
    pub last_trained: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CapabilityRecord {
    pub capability_id: String,
    pub active_version: String,
    pub versions: Vec<VersionRecord>,
    pub description: String,
    pub keywords: Vec<String>,
    pub input_dim: usize,
    pub labels: Vec<String>,
    pub device_caps: Vec<String>,
    pub dependencies: Vec<String>,
    pub provenance: serde_json::Value,
    pub stats: Stats,
    #[serde(default)]
    pub input_stats: Option<InputStats>,
    #[serde(default)]
    pub calibration: Option<Calibration>,
}

impl CapabilityRecord {
    pub fn active(&self) -> &VersionRecord {
        self.versions.iter().find(|v| v.version == self.active_version).expect("active version present")
    }
}

#[derive(Serialize, Deserialize, Default)]
struct Disk {
    capabilities: BTreeMap<String, CapabilityRecord>,
}

pub struct Registry {
    path: PathBuf,
    disk: Disk,
    /// usage statistics changed in memory but not yet written (structural changes always save immediately)
    stats_dirty: bool,
}

impl Registry {
    pub fn open(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root)?;
        let path = root.join("registry.json");
        let disk = if path.exists() { serde_json::from_slice(&std::fs::read(&path)?)? } else { Disk::default() };
        Ok(Self { path, disk, stats_dirty: false })
    }
    fn save(&self) -> Result<()> {
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&self.disk)?)?;
        std::fs::rename(tmp, &self.path)?; // atomic replace
        Ok(())
    }
    pub fn get(&self, id: &str) -> Option<&CapabilityRecord> { self.disk.capabilities.get(id) }
    pub fn all(&self) -> impl Iterator<Item = &CapabilityRecord> { self.disk.capabilities.values() }
    pub fn len(&self) -> usize { self.disk.capabilities.len() }
    pub fn is_empty(&self) -> bool { self.disk.capabilities.is_empty() }

    /// Insert a new capability or append+activate a new version (previous versions are kept for rollback).
    pub fn upsert(&mut self, mut rec: CapabilityRecord, ver: VersionRecord) -> Result<()> {
        match self.disk.capabilities.get_mut(&rec.capability_id) {
            Some(existing) => {
                existing.versions.push(ver.clone());
                existing.active_version = ver.version.clone();
                existing.description = rec.description;
                existing.keywords = rec.keywords;
                existing.input_dim = rec.input_dim;
                existing.labels = rec.labels;
                existing.device_caps = rec.device_caps;
                existing.dependencies = rec.dependencies;
                existing.provenance = rec.provenance;
                existing.input_stats = rec.input_stats;
                existing.calibration = rec.calibration;
                existing.stats.last_trained = now_secs();
            }
            None => {
                rec.versions = vec![ver.clone()];
                rec.active_version = ver.version;
                rec.stats.last_trained = now_secs();
                self.disk.capabilities.insert(rec.capability_id.clone(), rec);
            }
        }
        self.save()
    }

    pub fn rollback(&mut self, id: &str) -> Result<String> {
        let rec = self.disk.capabilities.get_mut(id).ok_or_else(|| anyhow::anyhow!("unknown capability {id}"))?;
        let idx = rec.versions.iter().position(|v| v.version == rec.active_version).unwrap();
        if idx == 0 { anyhow::bail!("no earlier version of {id} to roll back to"); }
        rec.active_version = rec.versions[idx - 1].version.clone();
        let v = rec.active_version.clone();
        self.save()?;
        Ok(v)
    }

    /// Stats are buffered in memory (writing the registry on every inference cost ~10x the inference itself,
    /// see docs/FINDINGS.md); call `flush` to persist. Also flushed on drop.
    pub fn record_call(&mut self, id: &str, latency_us: u64) -> Result<()> {
        if let Some(r) = self.disk.capabilities.get_mut(id) {
            r.stats.calls += 1;
            r.stats.total_latency_us += latency_us;
            r.stats.max_latency_us = r.stats.max_latency_us.max(latency_us);
            r.stats.last_used = now_secs();
            self.stats_dirty = true;
        }
        Ok(())
    }

    pub fn record_outcome(&mut self, id: &str, success: bool) -> Result<()> {
        if let Some(r) = self.disk.capabilities.get_mut(id) {
            if success { r.stats.success += 1 } else { r.stats.failure += 1 }
            self.stats_dirty = true;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if self.stats_dirty { self.save()?; self.stats_dirty = false; }
        Ok(())
    }
}

impl Drop for Registry {
    fn drop(&mut self) { let _ = self.flush(); }
}
