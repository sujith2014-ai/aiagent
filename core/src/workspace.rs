//! Platform-independent working state for a task.
use serde::Serialize;

#[derive(Serialize, Clone, Debug)]
pub struct Step {
    pub capability_id: String,
    pub version: String,
    pub output_label: String,
    pub probs: Vec<f32>,
    pub latency_us: u64,
}

#[derive(Serialize, Default, Debug)]
pub struct Workspace {
    pub task_intent: String,
    pub observations: Vec<Vec<f32>>,
    pub history: Vec<Step>,
    pub uncertainty: Option<f32>,
    pub tool_results: Vec<serde_json::Value>,
}

impl Workspace {
    pub fn new(intent: &str, input: &[f32]) -> Self {
        Self { task_intent: intent.into(), observations: vec![input.to_vec()], ..Default::default() }
    }
}
