//! Baseline deterministic router: scores registry records from their declared
//! metadata only (keyword overlap + input shape). No capability-name logic.
//! A learned-router interface (`Router` trait) allows replacement later.
use crate::registry::{CapabilityRecord, Registry};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Task {
    pub intent: String,
    pub input: Vec<f32>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Status { #[serde(rename = "KNOWN")] Known, #[serde(rename = "UNCERTAIN")] Uncertain, #[serde(rename = "UNKNOWN")] Unknown }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Candidate {
    pub capability_id: String,
    /// fraction of the intent's tokens explained by the capability's declared keywords
    pub score: f64,
    /// tie-break only (Dice coefficient)
    pub dice: f64,
    pub shape_ok: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Decision {
    pub status: Status,
    pub chosen: Option<String>,
    pub candidates: Vec<Candidate>,
}

pub trait Router {
    fn route(&self, reg: &Registry, task: &Task) -> Decision;
}

pub struct KeywordRouter {
    pub known_threshold: f64,
    pub uncertain_threshold: f64,
    /// minimum score lead over the runner-up (shape-compatible) capability for KNOWN
    pub margin: f64,
}

impl Default for KeywordRouter {
    fn default() -> Self { Self { known_threshold: 0.50, uncertain_threshold: 0.25, margin: 0.10 } }
}

const STOP: &[&str] = &["a", "an", "the", "of", "to", "and", "or", "is", "are", "two", "this", "that", "please", "for", "in", "on", "these", "those", "with", "from", "which", "what", "given", "me", "it", "my"];

pub fn tokens(s: &str) -> Vec<String> {
    let mut v: Vec<String> = s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty() && !STOP.contains(t))
        .map(|t| t.to_string()).collect();
    v.sort(); v.dedup(); v
}

fn score(rec: &CapabilityRecord, toks: &[String]) -> (f64, f64) {
    let kws: Vec<String> = rec.keywords.iter().flat_map(|k| tokens(k)).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    if toks.is_empty() || kws.is_empty() { return (0.0, 0.0); }
    let m = toks.iter().filter(|t| kws.contains(t)).count() as f64;
    (m / toks.len() as f64, 2.0 * m / (toks.len() + kws.len()) as f64)
}

impl Router for KeywordRouter {
    fn route(&self, reg: &Registry, task: &Task) -> Decision {
        let toks = tokens(&task.intent);
        let mut cands: Vec<Candidate> = reg.all().map(|r| {
            let (score, dice) = score(r, &toks);
            Candidate { capability_id: r.capability_id.clone(), score, dice, shape_ok: r.input_dim == task.input.len() }
        }).collect();
        cands.sort_by(|a, b| b.shape_ok.cmp(&a.shape_ok)
            .then(b.score.partial_cmp(&a.score).unwrap())
            .then(b.dice.partial_cmp(&a.dice).unwrap())
            .then(a.capability_id.cmp(&b.capability_id)));
        let best = cands.first().filter(|c| c.shape_ok);
        let runner_up = cands.get(1).filter(|c| c.shape_ok).map(|c| c.score).unwrap_or(0.0);
        let (status, chosen) = match best {
            Some(c) if c.score >= self.known_threshold && c.score - runner_up >= self.margin => (Status::Known, Some(c.capability_id.clone())),
            Some(c) if c.score >= self.known_threshold => (Status::Uncertain, Some(c.capability_id.clone())), // ambiguous
            Some(c) if c.score >= self.uncertain_threshold => (Status::Uncertain, Some(c.capability_id.clone())),
            _ => (Status::Unknown, None),
        };
        Decision { status, chosen, candidates: cands }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tokenizer_drops_stopwords_and_dedups() {
        assert_eq!(tokens("Compare the numbers, compare!"), vec!["compare", "numbers"]);
    }
}
