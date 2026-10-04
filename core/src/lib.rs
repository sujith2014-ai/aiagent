//! Platform-neutral core. No OS/Android-specific logic lives here: platform
//! differences are expressed only through `DeviceProfile` (what the device
//! provides) and the `Backend` trait (how a model format is executed).

pub mod backend;
pub mod capability;
pub mod device;
pub mod graph;
pub mod package;
pub mod registry;
pub mod router;
pub mod runtime;
pub mod trace;
pub mod trust;
pub mod workspace;

/// Version of this core runtime; packages declare a minimum.
pub const RUNTIME_VERSION: &str = "0.1.0";
/// `.cap` package-format versions this core can read.
pub const SUPPORTED_FORMATS: &[&str] = &["cap/1"];
