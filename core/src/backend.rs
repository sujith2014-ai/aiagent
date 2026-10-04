//! Model execution backends. Core logic only sees `Backend`/`Model`; formats
//! are declared in the manifest. First backend: ONNX via `tract` (pure Rust, so
//! the same code builds for Android without native runtime binaries).
use anyhow::{anyhow, Result};
use tract_onnx::prelude::*;

pub trait Model: Send {
    fn run(&self, input: &[f32]) -> Result<Vec<f32>>;
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn formats(&self) -> &[&'static str];
    fn load(&self, format: &str, bytes: &[u8], input_dim: usize) -> Result<Box<dyn Model>>;
}

pub struct OnnxBackend;

struct OnnxModel {
    plan: TypedRunnableModel<TypedModel>,
    dim: usize,
}

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
}

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
