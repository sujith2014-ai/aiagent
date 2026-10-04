//! "mlp-lite": a tiny pure-Rust executor for ONNX graphs made only of Gemm / Relu / Sub / Div / Mul / Add / Identity over float32 initializers,
//! i.e. exactly what our exported MLP modules contain. No dependencies (a minimal protobuf wire reader is included), no native code,
//! so it builds for any Rust target without a C toolchain. Anything it cannot prove it understands is rejected at load time (the runtime then
//! falls back to the general backend if one is compiled in); it never guesses.
use crate::backend::{Backend, Model};
use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;

// ---- minimal protobuf wire reader -------------------------------------------------------------------------------------------
struct Rd<'a> { b: &'a [u8], i: usize }

impl<'a> Rd<'a> {
    fn new(b: &'a [u8]) -> Self { Self { b, i: 0 } }
    fn done(&self) -> bool { self.i >= self.b.len() }
    fn varint(&mut self) -> Result<u64> {
        let (mut v, mut shift) = (0u64, 0u32);
        loop {
            let byte = *self.b.get(self.i).ok_or_else(|| anyhow!("truncated varint"))?; self.i += 1;
            v |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 { return Ok(v); }
            shift += 7; if shift > 63 { bail!("varint too long"); }
        }
    }
    fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.varint()? as usize;
        let s = self.b.get(self.i..self.i.checked_add(n).ok_or_else(|| anyhow!("length overflow"))?).ok_or_else(|| anyhow!("truncated field"))?;
        self.i += n; Ok(s)
    }
    /// next (field number, wire type)
    fn tag(&mut self) -> Result<(u32, u8)> { let t = self.varint()?; Ok(((t >> 3) as u32, (t & 7) as u8)) }
    fn skip(&mut self, wt: u8) -> Result<()> {
        match wt { 0 => { self.varint()?; } 1 => self.i += 8, 2 => { self.bytes()?; } 5 => self.i += 4, _ => bail!("unsupported wire type {wt}") }
        if self.i > self.b.len() { bail!("truncated field"); }
        Ok(())
    }
    fn fixed32(&mut self) -> Result<u32> {
        let s = self.b.get(self.i..self.i + 4).ok_or_else(|| anyhow!("truncated fixed32"))?; self.i += 4;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
}

#[derive(Default, Debug)]
struct Attr { name: String, f: Option<f32>, i: Option<i64> }
#[derive(Default, Debug)]
struct NodeP { op: String, inputs: Vec<String>, outputs: Vec<String>, attrs: Vec<Attr> }
#[derive(Default, Debug)]
struct TensorP { name: String, dims: Vec<i64>, dtype: i32, raw: Vec<u8>, floats: Vec<f32> }
#[derive(Default, Debug)]
struct ValueP { name: String, dims: Vec<i64> }

fn s(b: &[u8]) -> Result<String> { Ok(String::from_utf8(b.to_vec()).map_err(|_| anyhow!("non-utf8 string"))?) }

fn attr(b: &[u8]) -> Result<Attr> {
    let mut r = Rd::new(b); let mut a = Attr::default();
    while !r.done() { match r.tag()? { (1, 2) => a.name = s(r.bytes()?)?, (2, 5) => a.f = Some(f32::from_bits(r.fixed32()?)), (3, 0) => a.i = Some(r.varint()? as i64), (_, wt) => r.skip(wt)? } }
    Ok(a)
}
fn node(b: &[u8]) -> Result<NodeP> {
    let mut r = Rd::new(b); let mut n = NodeP::default();
    while !r.done() { match r.tag()? { (1, 2) => n.inputs.push(s(r.bytes()?)?), (2, 2) => n.outputs.push(s(r.bytes()?)?), (4, 2) => n.op = s(r.bytes()?)?, (5, 2) => n.attrs.push(attr(r.bytes()?)?), (_, wt) => r.skip(wt)? } }
    Ok(n)
}
fn tensor(b: &[u8]) -> Result<TensorP> {
    let mut r = Rd::new(b); let mut t = TensorP::default();
    while !r.done() {
        match r.tag()? {
            (1, 0) => t.dims.push(r.varint()? as i64),
            (1, 2) => { let mut p = Rd::new(r.bytes()?); while !p.done() { t.dims.push(p.varint()? as i64); } }
            (2, 0) => t.dtype = r.varint()? as i32,
            (4, 2) => { let raw = r.bytes()?; if raw.len() % 4 != 0 { bail!("bad packed float_data"); } t.floats.extend(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))); }
            (4, 5) => t.floats.push(f32::from_bits(r.fixed32()?)),
            (8, 2) => t.name = s(r.bytes()?)?,
            (9, 2) => t.raw = r.bytes()?.to_vec(),
            (_, wt) => r.skip(wt)?,
        }
    }
    Ok(t)
}
fn value_info(b: &[u8]) -> Result<ValueP> {
    let mut r = Rd::new(b); let mut v = ValueP::default();
    while !r.done() {
        match r.tag()? {
            (1, 2) => v.name = s(r.bytes()?)?,
            (2, 2) => { // TypeProto -> tensor_type(1) -> shape(2) -> dim(1) -> dim_value(1)
                let mut tp = Rd::new(r.bytes()?);
                while !tp.done() { match tp.tag()? { (1, 2) => { let mut tt = Rd::new(tp.bytes()?);
                    while !tt.done() { match tt.tag()? { (2, 2) => { let mut sh = Rd::new(tt.bytes()?);
                        while !sh.done() { match sh.tag()? { (1, 2) => { let mut d = Rd::new(sh.bytes()?); let mut val = -1i64;
                            while !d.done() { match d.tag()? { (1, 0) => val = d.varint()? as i64, (_, wt) => d.skip(wt)? } } v.dims.push(val); } (_, wt) => sh.skip(wt)? } } } (_, wt) => tt.skip(wt)? } } } (_, wt) => tp.skip(wt)? } }
            }
            (_, wt) => r.skip(wt)?,
        }
    }
    Ok(v)
}

// ---- model --------------------------------------------------------------------------------------------------------------------
enum Step {
    Gemm { w: Vec<f32>, m: usize, k: usize, trans_b: bool, bias: Option<Vec<f32>>, alpha: f32, beta: f32, src: usize, dst: usize },
    Relu { src: usize, dst: usize },
    Ew { op: char, c: Vec<f32>, src: usize, dst: usize },      // elementwise with a constant: '-', '/', '*', '+'
    Copy { src: usize, dst: usize },
}

pub struct MlpLiteModel { dim: usize, steps: Vec<Step>, slots: usize, out_slot: usize, out_len: usize }

fn f32s(t: &TensorP) -> Result<Vec<f32>> {
    if t.dtype != 1 { bail!("initializer '{}' is not float32 (dtype {})", t.name, t.dtype); }
    if t.dims.iter().any(|d| *d < 0) { bail!("initializer '{}' has a negative dimension", t.name); }
    let n: i64 = t.dims.iter().try_fold(1i64, |a, d| a.checked_mul(*d)).ok_or_else(|| anyhow!("initializer '{}' dimensions overflow", t.name))?;
    let v: Vec<f32> = if !t.raw.is_empty() { if t.raw.len() % 4 != 0 { bail!("bad raw_data size"); } t.raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() } else { t.floats.clone() };
    if n < 0 || v.len() as i64 != n { bail!("initializer '{}' size mismatch ({} values for dims {:?})", t.name, v.len(), t.dims); }
    Ok(v)
}

pub fn parse(bytes: &[u8], input_dim: usize) -> Result<MlpLiteModel> {
    let (mut r, mut graph) = (Rd::new(bytes), None);
    while !r.done() { match r.tag()? { (7, 2) => graph = Some(r.bytes()?), (_, wt) => r.skip(wt)? } }
    let mut g = Rd::new(graph.ok_or_else(|| anyhow!("model has no graph"))?);
    let (mut nodes, mut inits, mut inputs, mut outputs) = (vec![], HashMap::new(), vec![], vec![]);
    while !g.done() {
        match g.tag()? {
            (1, 2) => nodes.push(node(g.bytes()?)?),
            (5, 2) => { let t = tensor(g.bytes()?)?; inits.insert(t.name.clone(), t); }
            (11, 2) => inputs.push(value_info(g.bytes()?)?),
            (12, 2) => outputs.push(value_info(g.bytes()?)?),
            (_, wt) => g.skip(wt)?,
        }
    }
    let real_inputs: Vec<&ValueP> = inputs.iter().filter(|v| !inits.contains_key(&v.name)).collect();
    if real_inputs.len() != 1 || outputs.len() != 1 { bail!("expected exactly one graph input and one output"); }
    let inp = real_inputs[0];
    if inp.dims.iter().any(|d| *d < 0 || *d > 1 << 31) || inp.dims.iter().try_fold(1usize, |a, d| a.checked_mul(*d as usize)) != Some(input_dim) || inp.dims.first() != Some(&1) { bail!("input shape {:?} is not [1, {input_dim}]", inp.dims); }
    let mut slot_of: HashMap<String, usize> = HashMap::new(); slot_of.insert(inp.name.clone(), 0);
    let mut len_of: Vec<usize> = vec![input_dim]; let mut steps = vec![];
    for n in &nodes {
        let src = |i: usize| -> Result<usize> { slot_of.get(n.inputs.get(i).ok_or_else(|| anyhow!("{}: missing input {i}", n.op))?).copied().ok_or_else(|| anyhow!("{}: input '{}' is not computed from the graph input", n.op, n.inputs[i])) };
        let konst = |i: usize| -> Result<&TensorP> { inits.get(n.inputs.get(i).ok_or_else(|| anyhow!("{}: missing input {i}", n.op))?).ok_or_else(|| anyhow!("{}: operand '{}' is not a constant initializer", n.op, n.inputs[i])) };
        if n.outputs.len() != 1 { bail!("{}: expected one output", n.op); }
        let dst = len_of.len();
        let step = match n.op.as_str() {
            "Relu" => { let s0 = src(0)?; len_of.push(len_of[s0]); Step::Relu { src: s0, dst } }
            "Identity" => { let s0 = src(0)?; len_of.push(len_of[s0]); Step::Copy { src: s0, dst } }
            "Sub" | "Div" | "Mul" | "Add" => {
                let s0 = src(0)?; let c = f32s(konst(1)?)?;
                if c.len() != len_of[s0] && c.len() != 1 { bail!("{}: constant of {} values does not broadcast over {}", n.op, c.len(), len_of[s0]); }
                let op = match n.op.as_str() { "Sub" => '-', "Div" => '/', "Mul" => '*', _ => '+' };
                len_of.push(len_of[s0]); Step::Ew { op, c, src: s0, dst }
            }
            "Gemm" => {
                let s0 = src(0)?; let wt = konst(1)?; let (mut alpha, mut beta, mut trans_a, mut trans_b) = (1.0f32, 1.0f32, 0i64, 0i64);
                for a in &n.attrs { match a.name.as_str() { "alpha" => alpha = a.f.unwrap_or(1.0), "beta" => beta = a.f.unwrap_or(1.0), "transA" => trans_a = a.i.unwrap_or(0), "transB" => trans_b = a.i.unwrap_or(0), o => bail!("Gemm: unsupported attribute '{o}'") } }
                if trans_a != 0 || wt.dims.len() != 2 { bail!("Gemm: only transA=0 with a 2-D weight is supported"); }
                let (d0, d1) = (wt.dims[0] as usize, wt.dims[1] as usize);
                let (k, m) = if trans_b == 1 { (d1, d0) } else { (d0, d1) };
                if k != len_of[s0] { bail!("Gemm: weight expects {k} inputs, got {}", len_of[s0]); }
                let bias = if n.inputs.len() > 2 && !n.inputs[2].is_empty() { let b = f32s(konst(2)?)?; if b.len() != m { bail!("Gemm: bias has {} values, expected {m}", b.len()); } Some(b) } else { None };
                len_of.push(m); Step::Gemm { w: f32s(wt)?, m, k, trans_b: trans_b == 1, bias, alpha, beta, src: s0, dst }
            }
            o => bail!("unsupported operator '{o}'"),
        };
        slot_of.insert(n.outputs[0].clone(), dst); steps.push(step);
    }
    let out_slot = *slot_of.get(&outputs[0].name).ok_or_else(|| anyhow!("graph output is not produced by any node"))?;
    Ok(MlpLiteModel { dim: input_dim, slots: len_of.len(), out_len: len_of[out_slot], out_slot, steps })
}

impl MlpLiteModel {
    /// Recover plain-MLP structure ([Sub, Div]? then Gemm(transB)/Relu stack) so an installed capability can be fine-tuned on-device.
    pub fn to_mlp(&self) -> Option<crate::train_lite::Mlp> {
        use crate::train_lite::{Layer, Mlp};
        let mut norm: Option<(Vec<f32>, Vec<f32>)> = None; let mut layers: Vec<Layer> = vec![]; let mut i = 0;
        if let (Some(Step::Ew { op: '-', c: mean, .. }), Some(Step::Ew { op: '/', c: std, .. })) = (self.steps.get(0), self.steps.get(1)) {
            if mean.len() != self.dim || std.len() != self.dim { return None; }
            norm = Some((mean.clone(), std.clone())); i = 2;
        }
        while i < self.steps.len() {
            match &self.steps[i] { Step::Gemm { w, m, k, trans_b: true, bias: Some(b), alpha, beta, .. } if *alpha == 1.0 && *beta == 1.0 => layers.push(Layer { w: w.clone(), b: b.clone(), inp: *k, out: *m }), _ => return None }
            i += 1;
            if i < self.steps.len() { if matches!(self.steps[i], Step::Relu { .. }) { i += 1; } else { return None; } }   // a Relu after every layer but the last
        }
        if layers.is_empty() || layers[0].inp != self.dim { return None; }
        Some(Mlp { in_dim: self.dim, layers, norm })
    }
}

impl Model for MlpLiteModel {
    fn run(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.dim { return Err(anyhow!("input has {} values, model expects {}", input.len(), self.dim)); }
        let mut v: Vec<Vec<f32>> = vec![vec![]; self.slots]; v[0] = input.to_vec();
        for st in &self.steps {
            match st {
                Step::Relu { src, dst } => v[*dst] = v[*src].iter().map(|x| if *x > 0.0 { *x } else { 0.0 }).collect(),
                Step::Copy { src, dst } => v[*dst] = v[*src].clone(),
                Step::Ew { op, c, src, dst } => v[*dst] = v[*src].iter().enumerate().map(|(i, x)| { let k = if c.len() == 1 { c[0] } else { c[i] };
                    match op { '-' => x - k, '/' => x / k, '*' => x * k, _ => x + k } }).collect(),
                Step::Gemm { w, m, k, trans_b, bias, alpha, beta, src, dst } => {
                    let a = &v[*src]; let mut out = vec![0f32; *m];
                    for j in 0..*m {
                        let mut acc = 0f32;
                        for i in 0..*k { acc += a[i] * if *trans_b { w[j * k + i] } else { w[i * m + j] }; }
                        out[j] = alpha * acc + bias.as_ref().map(|b| beta * b[j]).unwrap_or(0.0);
                    }
                    v[*dst] = out;
                }
            }
        }
        Ok(v[self.out_slot].clone())
    }
    fn backend(&self) -> &'static str { "onnx-mlp-lite" }
}

pub struct MlpLiteBackend;
impl Backend for MlpLiteBackend {
    fn name(&self) -> &str { "onnx-mlp-lite" }
    fn formats(&self) -> &[&'static str] { &["onnx"] }
    fn load(&self, format: &str, bytes: &[u8], input_dim: usize) -> Result<Box<dyn Model>> {
        if format != "onnx" { bail!("mlp-lite cannot load '{format}'"); }
        let m = parse(bytes, input_dim)?; let _ = m.out_len; Ok(Box::new(m))
    }
}
