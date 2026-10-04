//! Deterministic permission policy for external actions. The AI proposes an action; this module decides
//! allow / deny / require_approval from an operator-owned policy file. It has no side effects and no model in the loop.
//! Hard guards (credentials in URLs, non-http(s) schemes, private/loopback hosts, secret-looking text) apply to every
//! rule and cannot be relaxed by a rule except `allow_private_hosts`, which the operator must set explicitly.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::net::IpAddr;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Effect { Allow, Deny, RequireApproval }

#[derive(Deserialize, Clone, Debug)]
pub struct Rule {
    pub id: String,
    /// exact action id ("web.search") or prefix wildcard ("shell.*")
    pub action: String,
    pub effect: Effect,
    #[serde(default)] pub domains: Vec<String>,
    #[serde(default)] pub schemes: Vec<String>,
    #[serde(default)] pub max_per_task: Option<u32>,
    #[serde(default)] pub max_query_len: Option<usize>,
    #[serde(default)] pub deny_patterns: Vec<String>,
    #[serde(default)] pub allow_private_hosts: bool,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Policy {
    pub version: u32,
    pub default: Effect,
    pub rules: Vec<Rule>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Request {
    pub action: String,
    #[serde(default)] pub params: BTreeMap<String, serde_json::Value>,
    #[serde(default)] pub task_id: String,
    /// number of times each action was already taken for this task
    #[serde(default)] pub counts: BTreeMap<String, u32>,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct Decision { pub effect: Effect, pub rule_id: String, pub reason: String }

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Approval { pub request_hash: String, pub approver: String, pub approved_at: u64, pub expires_at: u64 }

const SECRET_MARKERS: &[&str] = &["password", "passwd", "secret", "api_key", "apikey", "api-key", "authorization:", "bearer ", "-----begin", "sk-", "ghp_", "xoxb-", "token="];

pub fn request_hash(req: &Request) -> String {
    // canonical: action + sorted params (BTreeMap) as JSON; task id and counters are deliberately excluded
    let canon = serde_json::json!({"action": req.action, "params": req.params});
    hex::encode(Sha256::digest(serde_json::to_vec(&canon).unwrap()))
}

fn deny(rule: &str, reason: impl Into<String>) -> Decision { Decision { effect: Effect::Deny, rule_id: rule.into(), reason: reason.into() } }

fn action_matches(pattern: &str, action: &str) -> bool {
    match pattern.strip_suffix(".*") { Some(prefix) => action.starts_with(&format!("{prefix}.")), None => pattern == action }
}

fn host_matches(pattern: &str, host: &str) -> bool {
    let host = host.to_ascii_lowercase();
    match pattern.strip_prefix("*.") { Some(suffix) => host.ends_with(&format!(".{}", suffix.to_ascii_lowercase())), None => host == pattern.to_ascii_lowercase() }
}

fn private_host(host: &str) -> bool {
    let h = host.trim_matches(|c| c == '[' || c == ']').to_ascii_lowercase();
    if h == "localhost" || h.ends_with(".localhost") || h.ends_with(".local") || h.ends_with(".internal") || h.ends_with(".lan") || !h.contains('.') && h.parse::<IpAddr>().is_err() { return true; }
    match h.parse::<IpAddr>() {
        Ok(IpAddr::V4(v)) => v.is_loopback() || v.is_private() || v.is_link_local() || v.is_unspecified() || v.is_broadcast() || v.octets()[0] == 100 && (v.octets()[1] & 0xC0) == 64,
        Ok(IpAddr::V6(v)) => v.is_loopback() || v.is_unspecified() || v.is_unique_local() || v.is_unicast_link_local() || v.to_ipv4_mapped().map(|m| m.is_loopback() || m.is_private() || m.is_link_local()).unwrap_or(false),
        Err(_) => false,
    }
}

fn contains_secret(text: &str, extra: &[String]) -> Option<String> {
    let t = text.to_ascii_lowercase();
    for m in SECRET_MARKERS.iter().map(|s| s.to_string()).chain(extra.iter().map(|s| s.to_ascii_lowercase())) {
        if t.contains(&m) { return Some(m); }
    }
    // long unbroken base64/hex-looking runs
    let mut run = 0usize;
    for c in text.chars() { if c.is_ascii_alphanumeric() || "+/_-=".contains(c) { run += 1; if run >= 32 { return Some("long token-like string".into()); } } else { run = 0; } }
    None
}

fn check_params(rule: &Rule, req: &Request) -> Result<(), String> {
    let mut texts: Vec<String> = vec![];
    if let Some(q) = req.params.get("query").and_then(|v| v.as_str()) {
        if q.len() > rule.max_query_len.unwrap_or(200) { return Err(format!("query longer than {} bytes", rule.max_query_len.unwrap_or(200))); }
        texts.push(q.into());
    }
    if let Some(u) = req.params.get("url").and_then(|v| v.as_str()) {
        let url = url::Url::parse(u).map_err(|e| format!("unparseable url: {e}"))?;
        if !url.username().is_empty() || url.password().is_some() { return Err("credentials embedded in url".into()); }
        let scheme = url.scheme().to_string();
        if scheme != "http" && scheme != "https" { return Err(format!("scheme '{scheme}' not allowed")); }
        let allowed: Vec<String> = if rule.schemes.is_empty() { vec!["https".into()] } else { rule.schemes.clone() };
        if !allowed.contains(&scheme) { return Err(format!("scheme '{scheme}' not permitted by rule")); }
        let host = url.host_str().ok_or("url has no host")?.to_string();
        if private_host(&host) && !rule.allow_private_hosts { return Err(format!("private/loopback host '{host}'")); }
        if !rule.domains.is_empty() && !rule.domains.iter().any(|d| host_matches(d, &host)) { return Err(format!("host '{host}' not in allowlist")); }
        texts.push(url.query().unwrap_or("").into()); texts.push(url.path().into());
    }
    for (k, v) in &req.params {
        if k != "query" && k != "url" { if let Some(s) = v.as_str() {
            // a value under a `*_sha256` key that is exactly 64 hex digits is a content digest, not a credential (tool approvals bind to code digests)
            if k.ends_with("_sha256") && s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit()) { continue; }
            texts.push(s.into());
        } }
    }
    for t in texts { if let Some(m) = contains_secret(&t, &rule.deny_patterns) { return Err(format!("secret-looking content ({m})")); } }
    Ok(())
}

/// Evaluate a request. First matching rule wins; if its constraints fail the request is denied (never falls through to a later allow).
pub fn evaluate(policy: &Policy, req: &Request, approvals: &[Approval], now: u64) -> Decision {
    if policy.version != 1 { return deny("policy", format!("unsupported policy version {}", policy.version)); }
    for rule in &policy.rules {
        if !action_matches(&rule.action, &req.action) { continue; }
        if rule.effect == Effect::Deny { return deny(&rule.id, "denied by rule"); }
        if let Some(max) = rule.max_per_task {
            if req.counts.get(&req.action).copied().unwrap_or(0) >= max { return deny(&rule.id, format!("per-task limit {max} reached")); }
        }
        if let Err(e) = check_params(rule, req) { return deny(&rule.id, e); }
        return match rule.effect {
            Effect::Allow => Decision { effect: Effect::Allow, rule_id: rule.id.clone(), reason: "allowed".into() },
            Effect::RequireApproval => {
                let h = request_hash(req);
                if approvals.iter().any(|a| a.request_hash == h && a.expires_at > now) {
                    Decision { effect: Effect::Allow, rule_id: rule.id.clone(), reason: "approved by operator".into() }
                } else { Decision { effect: Effect::RequireApproval, rule_id: rule.id.clone(), reason: format!("operator approval required (request {h})") } }
            }
            Effect::Deny => unreachable!(),
        };
    }
    Decision { effect: policy.default, rule_id: "default".into(), reason: "no rule matched".into() }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pol() -> Policy {
        serde_json::from_str(r#"{"version":1,"default":"deny","rules":[
          {"id":"search","action":"web.search","effect":"allow","max_per_task":2,"max_query_len":80},
          {"id":"fetch","action":"web.fetch","effect":"allow","domains":["*.wikipedia.org","docs.python.org"],"max_per_task":2},
          {"id":"files","action":"fs.read","effect":"require_approval"},
          {"id":"shell","action":"shell.*","effect":"deny"}]}"#).unwrap()
    }
    fn req(a: &str, k: &str, v: &str) -> Request { Request { action: a.into(), params: [(k.to_string(), serde_json::json!(v))].into(), ..Default::default() } }
    fn eff(p: &Policy, r: &Request) -> Effect { evaluate(p, r, &[], 0).effect }

    #[test] fn default_deny_and_shell_deny() {
        assert_eq!(eff(&pol(), &req("browser.click", "x", "y")), Effect::Deny);
        assert_eq!(eff(&pol(), &req("shell.exec", "cmd", "ls")), Effect::Deny);
        assert_eq!(eff(&pol(), &req("shell.anything.else", "cmd", "ls")), Effect::Deny);
    }
    #[test] fn search_allowed_but_limited_and_clean() {
        assert_eq!(eff(&pol(), &req("web.search", "query", "number ordering rules")), Effect::Allow);
        assert_eq!(eff(&pol(), &req("web.search", "query", &"a ".repeat(100))), Effect::Deny);
        assert_eq!(eff(&pol(), &req("web.search", "query", "my password=hunter2")), Effect::Deny);
        let mut r = req("web.search", "query", "ok"); r.counts.insert("web.search".into(), 2);
        assert_eq!(eff(&pol(), &r), Effect::Deny);
    }
    #[test] fn fetch_guards() {
        for (u, want) in [("https://en.wikipedia.org/wiki/X", Effect::Allow), ("https://docs.python.org/3/", Effect::Allow), ("https://wikipedia.org/", Effect::Deny),
                          ("https://evil.example.com/", Effect::Deny), ("http://en.wikipedia.org/", Effect::Deny), ("https://user:pw@en.wikipedia.org/", Effect::Deny),
                          ("file:///etc/passwd", Effect::Deny), ("https://en.wikipedia.org.evil.com/", Effect::Deny), ("https://en.wikipedia.org/?token=abc", Effect::Deny)] {
            assert_eq!(eff(&pol(), &req("web.fetch", "url", u)), want, "{u}");
        }
    }
    #[test] fn private_hosts_blocked_even_with_empty_allowlist() {
        let p: Policy = serde_json::from_str(r#"{"version":1,"default":"deny","rules":[{"id":"f","action":"web.fetch","effect":"allow"}]}"#).unwrap();
        for u in ["https://127.0.0.1/", "https://localhost/", "https://10.0.0.5/", "https://192.168.1.1/", "https://169.254.169.254/latest/meta-data", "https://[::1]/", "https://[fd00::1]/", "https://intranet/", "https://printer.local/", "https://100.64.0.1/"] {
            assert_eq!(eff(&p, &req("web.fetch", "url", u)), Effect::Deny, "{u}");
        }
        assert_eq!(eff(&p, &req("web.fetch", "url", "https://example.com/")), Effect::Allow);
    }
    #[test] fn approval_binds_to_the_exact_request_and_expires() {
        let r = req("fs.read", "path", "/home/me/a.txt");
        assert_eq!(evaluate(&pol(), &r, &[], 10).effect, Effect::RequireApproval);
        let ap = Approval { request_hash: request_hash(&r), approver: "me".into(), approved_at: 1, expires_at: 100 };
        assert_eq!(evaluate(&pol(), &r, &[ap.clone()], 10).effect, Effect::Allow);
        assert_eq!(evaluate(&pol(), &r, &[ap.clone()], 100).effect, Effect::RequireApproval);                       // expired
        assert_eq!(evaluate(&pol(), &req("fs.read", "path", "/home/me/b.txt"), &[ap], 10).effect, Effect::RequireApproval);   // different request
    }
    #[test] fn rule_constraint_failure_never_falls_through_to_a_later_allow() {
        let p: Policy = serde_json::from_str(r#"{"version":1,"default":"deny","rules":[
          {"id":"a","action":"web.fetch","effect":"allow","domains":["good.com"]},{"id":"b","action":"web.fetch","effect":"allow"}]}"#).unwrap();
        assert_eq!(eff(&p, &req("web.fetch", "url", "https://other.com/")), Effect::Deny);
    }
    #[test] fn content_digests_are_not_mistaken_for_credentials_but_other_long_tokens_still_are() {
        let p: Policy = serde_json::from_str(r#"{"version":1,"default":"deny","rules":[{"id":"t","action":"tool.install","effect":"allow"}]}"#).unwrap();
        let h = "a".repeat(64);
        assert_eq!(eff(&p, &req("tool.install", "code_sha256", &h)), Effect::Allow);
        assert_eq!(eff(&p, &req("tool.install", "code_sha256", &"a".repeat(63))), Effect::Deny);       // not a full digest
        assert_eq!(eff(&p, &req("tool.install", "note", &h)), Effect::Deny);                          // same string under another key
        assert_eq!(eff(&p, &req("tool.install", "code_sha256", &"g".repeat(64))), Effect::Deny);       // not hex
    }
    #[test] fn unknown_policy_version_denies_everything() {
        let mut p = pol(); p.version = 2;
        assert_eq!(eff(&p, &req("web.search", "query", "x")), Effect::Deny);
    }
}
