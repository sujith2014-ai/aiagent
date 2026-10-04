//! What a device offers. This is the only place platform differences enter the core.
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DeviceProfile {
    pub name: String,
    /// Abstract device capabilities, e.g. "camera", "browser", "fs.read".
    pub caps: Vec<String>,
    pub max_ram_mb: u64,
    pub max_model_bytes: u64,
    pub max_active_modules: usize,
}

impl DeviceProfile {
    pub fn pc_full() -> Self {
        Self { name: "PC_FULL".into(), caps: vec!["fs.read".into(), "browser".into()],
               max_ram_mb: 16384, max_model_bytes: 512 << 20, max_active_modules: 1024 }
    }
    /// Android-like limits (simulation only; real hardware validation is pending).
    pub fn pc_constrained() -> Self {
        Self { name: "PC_CONSTRAINED".into(), caps: vec!["fs.read".into()],
               max_ram_mb: 512, max_model_bytes: 8 << 20, max_active_modules: 4 }
    }
    pub fn by_name(n: &str) -> Option<Self> {
        match n { "PC_FULL" | "full" => Some(Self::pc_full()),
                  "PC_CONSTRAINED" | "constrained" => Some(Self::pc_constrained()), _ => None }
    }
}
