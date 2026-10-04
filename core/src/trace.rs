//! Append-only JSONL trace of every task and import event.
use anyhow::Result;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct Trace { path: PathBuf }

impl Trace {
    pub fn open(root: &Path) -> Self { Self { path: root.join("traces.jsonl") } }
    pub fn append(&self, event: &serde_json::Value) -> Result<()> {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        writeln!(f, "{}", serde_json::to_string(event)?)?;
        Ok(())
    }
}
