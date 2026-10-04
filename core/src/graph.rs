//! Execution graphs ("plans"): data describing how capabilities and glue ops are combined.
//! Supports sequencing, module reuse, branching (`if`), bounded loops (`repeat`), convergence
//! (nodes reading several earlier outputs) and capability resolution by id *or* by intent
//! (through the router). Plans are validated statically; execution has a hard step budget.
//! Independent nodes are reported as parallelizable levels but are executed sequentially.
use crate::router::Status;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MAX_STEPS: usize = 10_000;
pub const MAX_REPEAT: usize = 1_000;

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(untagged)]
pub enum Ref {
    Const { #[serde(rename = "const")] value: f32 },
    Slot { slot: String, #[serde(default)] index: Option<usize> },
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(tag = "op")]
pub enum Node {
    /// Run a capability on the concatenated args; writes the predicted label index to `out` and its confidence to `<out>.conf`.
    #[serde(rename = "cap")]
    Cap { id: String, #[serde(default)] capability: Option<String>, #[serde(default)] intent: Option<String>, args: Vec<Ref>, out: String },
    /// out = options[index]
    #[serde(rename = "select")]
    Select { id: String, index: Ref, options: Vec<Ref>, out: String },
    /// out = (x,y) or (y,x) if flag label in `swap_if`
    #[serde(rename = "cond_swap")]
    CondSwap { id: String, pair: Vec<Ref>, flag: Ref, swap_if: Vec<usize>, out: String },
    /// out = scale * sum(args) + offset
    #[serde(rename = "affine")]
    Affine { id: String, args: Vec<Ref>, scale: f32, offset: f32, out: String },
    #[serde(rename = "gather")]
    Gather { id: String, args: Vec<Ref>, out: String },
    #[serde(rename = "if")]
    If { id: String, cond: Ref, in_set: Vec<usize>, then: Vec<Node>, #[serde(rename = "else")] otherwise: Vec<Node> },
    #[serde(rename = "repeat")]
    Repeat { id: String, times: usize, body: Vec<Node> },
}

impl Node {
    pub fn id(&self) -> &str {
        match self { Node::Cap { id, .. } | Node::Select { id, .. } | Node::CondSwap { id, .. } | Node::Affine { id, .. }
            | Node::Gather { id, .. } | Node::If { id, .. } | Node::Repeat { id, .. } => id }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Plan {
    pub id: String,
    pub inputs: BTreeMap<String, usize>,
    pub nodes: Vec<Node>,
    pub outputs: Vec<Ref>,
}

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("invalid plan: {0}")]
    Invalid(String),
    #[error("step budget exceeded ({0})")]
    Budget(usize),
    #[error("NEEDS_HELP at node {node}: {reason}")]
    NeedsHelp { node: String, reason: String },
    #[error("execution error at node {node}: {reason}")]
    Exec { node: String, reason: String },
}

fn refs_slots(rs: &[Ref]) -> Vec<&str> {
    rs.iter().filter_map(|r| if let Ref::Slot { slot, .. } = r { Some(slot.as_str()) } else { None }).collect()
}

fn check_nodes(nodes: &[Node], defined: &mut BTreeSet<String>, ids: &mut BTreeSet<String>) -> Result<(), PlanError> {
    let need = |rs: &[Ref], d: &BTreeSet<String>, id: &str| -> Result<(), PlanError> {
        for s in refs_slots(rs) { if !d.contains(s) { return Err(PlanError::Invalid(format!("node {id} reads undefined slot '{s}'"))); } }
        Ok(())
    };
    for n in nodes {
        if !ids.insert(n.id().to_string()) { return Err(PlanError::Invalid(format!("duplicate node id {}", n.id()))); }
        match n {
            Node::Cap { id, capability, intent, args, out } => {
                if capability.is_none() == intent.is_none() { return Err(PlanError::Invalid(format!("node {id}: give exactly one of capability/intent"))); }
                need(args, defined, id)?; defined.insert(out.clone()); defined.insert(format!("{out}.conf"));
            }
            Node::Select { id, index, options, out } => { need(std::slice::from_ref(index), defined, id)?; need(options, defined, id)?; if options.is_empty() { return Err(PlanError::Invalid(format!("node {id}: no options"))); } defined.insert(out.clone()); }
            Node::CondSwap { id, pair, flag, out, .. } => { if pair.len() != 2 { return Err(PlanError::Invalid(format!("node {id}: pair needs 2 refs"))); } need(pair, defined, id)?; need(std::slice::from_ref(flag), defined, id)?; defined.insert(out.clone()); }
            Node::Affine { id, args, out, .. } | Node::Gather { id, args, out } => { need(args, defined, id)?; defined.insert(out.clone()); }
            Node::If { id, cond, then, otherwise, .. } => {
                need(std::slice::from_ref(cond), defined, id)?;
                let mut d1 = defined.clone(); let mut d2 = defined.clone();
                check_nodes(then, &mut d1, ids)?; check_nodes(otherwise, &mut d2, ids)?;
                for s in d1.intersection(&d2) { defined.insert(s.clone()); } // only slots written by both branches are defined afterwards
            }
            Node::Repeat { id, times, body } => {
                if *times > MAX_REPEAT { return Err(PlanError::Invalid(format!("node {id}: times {times} > {MAX_REPEAT}"))); }
                check_nodes(body, defined, ids)?;
            }
        }
    }
    Ok(())
}

impl Plan {
    pub fn validate(&self) -> Result<(), PlanError> {
        let mut defined: BTreeSet<String> = self.inputs.keys().cloned().collect();
        let mut ids = BTreeSet::new();
        check_nodes(&self.nodes, &mut defined, &mut ids)?;
        for s in refs_slots(&self.outputs) { if !defined.contains(s) { return Err(PlanError::Invalid(format!("output reads undefined slot '{s}'"))); } }
        if self.outputs.is_empty() { return Err(PlanError::Invalid("no outputs".into())); }
        Ok(())
    }

    /// Dependency levels of top-level nodes (nodes in one level are independent => parallelizable).
    pub fn parallel_levels(&self) -> Vec<Vec<String>> {
        fn reads(n: &Node) -> Vec<String> {
            let v: Vec<&str> = match n {
                Node::Cap { args, .. } | Node::Affine { args, .. } | Node::Gather { args, .. } => refs_slots(args),
                Node::Select { index, options, .. } => { let mut v = refs_slots(options); v.extend(refs_slots(std::slice::from_ref(index))); v }
                Node::CondSwap { pair, flag, .. } => { let mut v = refs_slots(pair); v.extend(refs_slots(std::slice::from_ref(flag))); v }
                Node::If { cond, .. } => refs_slots(std::slice::from_ref(cond)),
                Node::Repeat { .. } => vec![],
            };
            v.into_iter().map(String::from).collect()
        }
        fn writes(n: &Node) -> Vec<String> {
            match n { Node::Cap { out, .. } => vec![out.clone(), format!("{out}.conf")],
                Node::Select { out, .. } | Node::CondSwap { out, .. } | Node::Affine { out, .. } | Node::Gather { out, .. } => vec![out.clone()],
                _ => vec![] }
        }
        let mut level_of: BTreeMap<String, usize> = BTreeMap::new();
        let mut levels: Vec<Vec<String>> = vec![];
        for n in &self.nodes {
            let l = reads(n).iter().filter_map(|s| level_of.get(s)).map(|l| l + 1).max().unwrap_or(0);
            if levels.len() <= l { levels.resize(l + 1, vec![]); }
            levels[l].push(n.id().to_string());
            for w in writes(n) { level_of.insert(w, l); } // later writers override (loop-carried state is sequential anyway)
        }
        levels
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct NodeRecord {
    pub node: String,
    pub op: &'static str,
    pub capability: Option<String>,
    pub version: Option<String>,
    pub label: Option<usize>,
    pub latency_us: u64,
}

#[derive(Serialize, Debug)]
pub struct PlanResult {
    pub plan_id: String,
    pub outputs: Vec<f32>,
    pub records: Vec<NodeRecord>,
    pub node_executions: usize,
    pub total_us: u64,
    pub module_us: u64,
    pub overhead_us: u64,
    pub parallel_levels: Vec<Vec<String>>,
    pub capability_calls: BTreeMap<String, usize>,
}

pub fn status_ok(s: &Status) -> bool { matches!(s, Status::Known) }

#[cfg(test)]
mod tests {
    use super::*;
    fn plan(json: &str) -> Plan { serde_json::from_str(json).unwrap() }

    #[test]
    fn rejects_undefined_slot_and_duplicates() {
        let p = plan(r#"{"id":"p","inputs":{"v":2},"nodes":[{"op":"gather","id":"a","args":[{"slot":"zz"}],"out":"o"}],"outputs":[{"slot":"o"}]}"#);
        assert!(p.validate().is_err());
        let p = plan(r#"{"id":"p","inputs":{"v":2},"nodes":[{"op":"gather","id":"a","args":[{"slot":"v"}],"out":"o"},{"op":"gather","id":"a","args":[{"slot":"v"}],"out":"o2"}],"outputs":[{"slot":"o"}]}"#);
        assert!(p.validate().is_err());
    }

    #[test]
    fn rejects_unbounded_repeat() {
        let p = plan(r#"{"id":"p","inputs":{"v":1},"nodes":[{"op":"repeat","id":"r","times":1000000,"body":[]}],"outputs":[{"slot":"v"}]}"#);
        assert!(matches!(p.validate(), Err(PlanError::Invalid(_))));
    }

    #[test]
    fn branch_outputs_defined_only_if_both_branches_write() {
        let p = plan(r#"{"id":"p","inputs":{"v":1},"nodes":[{"op":"if","id":"i","cond":{"slot":"v","index":0},"in_set":[1],
            "then":[{"op":"gather","id":"t","args":[{"const":1.0}],"out":"o"}],"else":[]}],"outputs":[{"slot":"o"}]}"#);
        assert!(p.validate().is_err());
    }

    #[test]
    fn independent_nodes_share_a_level() {
        let p = plan(r#"{"id":"p","inputs":{"a":1,"b":1},"nodes":[
            {"op":"gather","id":"x","args":[{"slot":"a"}],"out":"xa"},
            {"op":"gather","id":"y","args":[{"slot":"b"}],"out":"yb"},
            {"op":"gather","id":"z","args":[{"slot":"xa"},{"slot":"yb"}],"out":"zz"}],"outputs":[{"slot":"zz"}]}"#);
        p.validate().unwrap();
        let l = p.parallel_levels();
        assert_eq!(l[0], vec!["x", "y"]);
        assert_eq!(l[1], vec!["z"]);
    }
}
