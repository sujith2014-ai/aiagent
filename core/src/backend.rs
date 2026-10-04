//! Model execution backends. Core logic only sees `Backend`/`Model`; formats
//! are declared in the manifest. First backend: ONNX via `tract` (pure Rust, so
//! the same code builds for Android without native runtime binaries).
use anyhow::{anyhow, Result};
#[cfg(feature = "backend-tract")]
use tract_onnx::prelude::*;

pub trait Model: Send {
    fn run(&self, input: &[f32]) -> Result<Vec<f32>>;
    /// which backend actually runs this model
    fn backend(&self) -> &'static str { "unknown" }
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn formats(&self) -> &[&'static str];
    fn load(&self, format: &str, bytes: &[u8], input_dim: usize) -> Result<Box<dyn Model>>;
}

/// Tries each backend in order and keeps the first that can load the model (a backend rejects what it cannot prove it understands).
pub struct ChainBackend { backends: Vec<Box<dyn Backend>>, formats: Vec<&'static str> }

impl ChainBackend {
    pub fn new(backends: Vec<Box<dyn Backend>>) -> Self {
        let mut formats: Vec<&'static str> = vec![];
        for b in &backends { for f in b.formats() { if !formats.contains(f) { formats.push(*f); } } }
        Self { backends, formats }
    }
}

impl Backend for ChainBackend {
    fn name(&self) -> &str { "chain" }
    fn formats(&self) -> &[&'static str] { &self.formats }
    fn load(&self, format: &str, bytes: &[u8], input_dim: usize) -> Result<Box<dyn Model>> {
        let mut errs = vec![];
        for b in &self.backends {
            if !b.formats().contains(&format) { continue; }
            // a third-party parser panicking on a malformed (even validly signed) model must be a rejected import, never a crash
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| b.load(format, bytes, input_dim)));
            match r {
                Ok(Ok(m)) => return Ok(Box::new(GuardedModel(m))),
                Ok(Err(e)) => errs.push(format!("{}: {e}", b.name())),
                Err(_) => errs.push(format!("{}: panicked while loading the model", b.name())),
            }
        }
        Err(anyhow!("no backend could load this model ({})", errs.join("; ")))
    }
}

/// Turns a panic inside a backend's `run` into an error.
struct GuardedModel(Box<dyn Model>);

impl Model for GuardedModel {
    fn run(&self, input: &[f32]) -> Result<Vec<f32>> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.0.run(input))).unwrap_or_else(|_| Err(anyhow!("backend panicked during inference")))
    }
    fn backend(&self) -> &'static str { self.0.backend() }
}

/// Preferred order: the tiny pure-Rust executor first, the general ONNX runtime (if compiled in) as fallback.
pub fn default_backend(kind: &str) -> Result<Box<dyn Backend>> {
    let mut v: Vec<Box<dyn Backend>> = vec![];
    match kind {
        "auto" => { v.push(Box::new(crate::onnx_lite::MlpLiteBackend)); #[cfg(feature = "backend-tract")] v.push(Box::new(OnnxBackend)); }
        "mlp" => v.push(Box::new(crate::onnx_lite::MlpLiteBackend)),
        #[cfg(feature = "backend-tract")]
        "tract" => v.push(Box::new(OnnxBackend)),
        other => return Err(anyhow!("backend '{other}' is not available in this build")),
    }
    Ok(Box::new(ChainBackend::new(v)))
}

#[cfg(feature = "backend-tract")]
pub struct OnnxBackend;

#[cfg(feature = "backend-tract")]
struct OnnxModel {
    plan: TypedRunnableModel<TypedModel>,
    dim: usize,
}

#[cfg(feature = "backend-tract")]
impl Model for OnnxModel {
    fn run(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.dim {
            return Err(anyhow!("input has {} values, model expects {}", input.len(), self.dim));
        }
        let t: Tensor = tract_ndarray::Array2::from_shape_vec((1, self.dim), input.to_vec())?.into();
        let out = self.plan.run(tvec!(t.into()))?;
        let view = out[0].to_array_view::<f32>()?;
        Ok(view.iter().copied().collect())
    }
    fn backend(&self) -> &'static str { "onnx-tract" }
}

#[cfg(feature = "backend-tract")]
impl Backend for OnnxBackend {
    fn name(&self) -> &str { "onnx-tract" }
    fn formats(&self) -> &[&'static str] { &["onnx"] }
    fn load(&self, format: &str, bytes: &[u8], input_dim: usize) -> Result<Box<dyn Model>> {
        if format != "onnx" {
            return Err(anyhow!("OnnxBackend cannot load '{format}'"));
        }
        let plan = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(bytes))?
            .with_input_fact(0, f32::fact([1, input_dim]).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(Box::new(OnnxModel { plan, dim: input_dim }))
    }
}
