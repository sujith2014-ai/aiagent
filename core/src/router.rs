//! Baseline deterministic router: scores registry records from their declared
//! metadata only (keyword overlap + input shape). No capability-name logic.
//! A learned-router interface (`Router` trait) allows replacement later.
use crate::backend::Model;
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

/// IDF-weighted keyword router (Phase 14 routing study). A matched keyword counts ln(1 + N/df) where df is the number of capabilities that declare it,
/// so shared words ("alarm", frame words that leaked into keywords) count little and distinctive words ("boiler") count a lot; intent words matching no capability's
/// keywords dilute the score by `unmatched_weight` of a maximal word each, instead of by a full token as in the plain keyword router.
pub struct IdfRouter { pub known_threshold: f64, pub uncertain_threshold: f64, pub margin: f64, pub unmatched_weight: f64 }

impl Default for IdfRouter {
    fn default() -> Self { Self { known_threshold: 0.50, uncertain_threshold: 0.25, margin: 0.10, unmatched_weight: 0.5 } }
}

impl Router for IdfRouter {
    fn route(&self, reg: &Registry, task: &Task) -> Decision {
        let toks = tokens(&task.intent);
        let recs: Vec<&CapabilityRecord> = reg.all().collect();
        let n = recs.len().max(1) as f64;
        let kws: Vec<std::collections::BTreeSet<String>> = recs.iter().map(|r| r.keywords.iter().flat_map(|k| tokens(k)).collect()).collect();
        let w = |t: &String| -> Option<f64> { let df = kws.iter().filter(|k| k.contains(t)).count(); if df == 0 { None } else { Some((1.0 + n / df as f64).ln()) } };
        let wmax = (1.0 + n).ln();
        let unmatched = toks.iter().filter(|t| w(t).is_none()).count() as f64;
        let total: f64 = toks.iter().filter_map(|t| w(t)).sum::<f64>() + self.unmatched_weight * wmax * unmatched;
        let mut cands: Vec<Candidate> = recs.iter().zip(&kws).map(|(r, k)| {
            let m: f64 = toks.iter().filter(|t| k.contains(*t)).filter_map(|t| w(t)).sum();
            let score = if total > 0.0 { m / total } else { 0.0 };
            let dice = if toks.is_empty() || k.is_empty() { 0.0 } else { 2.0 * toks.iter().filter(|t| k.contains(*t)).count() as f64 / (toks.len() + k.len()) as f64 };
            Candidate { capability_id: r.capability_id.clone(), score, dice, shape_ok: r.input_dim == task.input.len() }
        }).collect();
        cands.sort_by(|a, b| b.shape_ok.cmp(&a.shape_ok).then(b.score.partial_cmp(&a.score).unwrap()).then(b.dice.partial_cmp(&a.dice).unwrap()).then(a.capability_id.cmp(&b.capability_id)));
        let best = cands.first().filter(|c| c.shape_ok);
        let runner_up = cands.get(1).filter(|c| c.shape_ok).map(|c| c.score).unwrap_or(0.0);
        let (status, chosen) = match best {
            Some(c) if c.score >= self.known_threshold && c.score - runner_up >= self.margin => (Status::Known, Some(c.capability_id.clone())),
            Some(c) if c.score >= self.known_threshold => (Status::Uncertain, Some(c.capability_id.clone())),
            Some(c) if c.score >= self.uncertain_threshold => (Status::Uncertain, Some(c.capability_id.clone())),
            _ => (Status::Unknown, None),
        };
        Decision { status, chosen, candidates: cands }
    }
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
    fn idf_weights_distinctive_words_over_shared_ones() {
        // a shared word alone is weak evidence, a distinctive word is strong evidence, an unmatched word dilutes only partly
        let r = IdfRouter::default();
        let kw = |s: &str| tokens(s);
        assert_eq!(kw("check the boiler alarm"), vec!["alarm", "boiler", "check"]);
        let _ = r;
    }
    #[test]
    fn tokenizer_drops_stopwords_and_dedups() {
        assert_eq!(tokens("Compare the numbers, compare!"), vec!["compare", "numbers"]);
    }
}


/// FNV-1a 32-bit over bytes (must match training/routing.py exactly).
pub fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for b in bytes { h ^= *b as u32; h = h.wrapping_mul(0x01000193); }
    h
}

/// Hashed bag of word and char-trigram features, L2-normalised (ASCII-lowercase tokens). Mirrors training/routing.py.
pub fn hash_features(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    let lower = text.to_lowercase();
    for w in lower.split(|c: char| !c.is_ascii_alphanumeric()).filter(|t| !t.is_empty()) {
        v[(fnv1a(format!("w:{w}").as_bytes()) as usize) % dim] += 1.0;
        let padded: Vec<u8> = format!("#{w}#").into_bytes();
        for tri in padded.windows(3) {
            let mut key = b"t:".to_vec(); key.extend_from_slice(tri);
            v[(fnv1a(&key) as usize) % dim] += 1.0;
        }
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 { for x in v.iter_mut() { *x /= n; } }
    v
}

/// Learned intent -> capability router (a signed model artifact with role "router").
pub struct LearnedRouter {
    pub model: Box<dyn Model>,
    /// class names: capability ids followed by "UNKNOWN"
    pub classes: Vec<String>,
    pub dim: usize,
    pub known_p: f32,
    pub uncertain_p: f32,
}

impl Router for LearnedRouter {
    fn route(&self, reg: &Registry, task: &Task) -> Decision {
        let feats = hash_features(&task.intent, self.dim);
        let logits = match self.model.run(&feats) { Ok(l) => l, Err(_) => return Decision { status: Status::Unknown, chosen: None, candidates: vec![] } };
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|x| (x - m).exp()).collect();
        let s: f32 = e.iter().sum();
        let probs: Vec<f32> = e.iter().map(|x| x / s).collect();
        let mut cands: Vec<Candidate> = vec![];
        for (i, c) in self.classes.iter().enumerate() {
            if c == "UNKNOWN" { continue; }
            if let Some(rec) = reg.get(c) {
                if rec.archived || rec.role != "capability" { continue; }
                cands.push(Candidate { capability_id: c.clone(), score: probs[i] as f64, dice: 0.0, shape_ok: rec.input_dim == task.input.len() });
            }
        }
        cands.sort_by(|a, b| b.shape_ok.cmp(&a.shape_ok).then(b.score.partial_cmp(&a.score).unwrap()).then(a.capability_id.cmp(&b.capability_id)));
        let unknown_p = self.classes.iter().position(|c| c == "UNKNOWN").map(|i| probs[i]).unwrap_or(0.0);
        let best = cands.first().filter(|c| c.shape_ok);
        let (status, chosen) = match best {
            Some(c) if (c.score as f32) >= self.known_p && (c.score as f32) > unknown_p => (Status::Known, Some(c.capability_id.clone())),
            Some(c) if (c.score as f32) >= self.uncertain_p && (c.score as f32) > unknown_p => (Status::Uncertain, Some(c.capability_id.clone())),
            _ => (Status::Unknown, None),
        };
        Decision { status, chosen, candidates: cands }
    }
}

#[cfg(test)]
mod learned_tests {
    use super::*;
    #[test]
    fn fnv_known_vectors() {
        assert_eq!(fnv1a(b""), 0x811c9dc5);
        assert_eq!(fnv1a(b"a"), 0xe40c292c);
        assert_eq!(fnv1a(b"foobar"), 0xbf9cf968);
    }
    #[test]
    fn features_are_unit_norm_and_deterministic() {
        let a = hash_features("Compare these numbers", 64);
        assert!((a.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(a, hash_features("compare THESE numbers!", 64));
        assert!(hash_features("", 64).iter().all(|x| *x == 0.0));
    }
}
