//! JNI bridge. Thin by design: marshals strings/float arrays, keeps runtimes in a handle table, and returns JSON strings.
//! All logic (registry, validation, routing, execution, policy) stays in the platform-neutral `aicore` crate.
//! Errors never cross the boundary as exceptions or panics: every call returns a JSON document, `{"error": "..."}` on failure.
use aicore::device::DeviceProfile;
use aicore::graph::Plan;
use aicore::router::Task;
use aicore::runtime::Runtime;
use aicore::trust::TrustStore;
use jni::objects::{JClass, JFloatArray, JString};
use jni::sys::{jlong, jstring};
use jni::JNIEnv;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

static NEXT: AtomicI64 = AtomicI64::new(1);
type Table = Mutex<HashMap<i64, Arc<Mutex<Runtime>>>>;
fn table() -> &'static Table { static T: OnceLock<Table> = OnceLock::new(); T.get_or_init(|| Mutex::new(HashMap::new())) }

fn get(h: i64) -> Result<Arc<Mutex<Runtime>>, String> {
    table().lock().map_err(|_| "handle table poisoned".to_string())?.get(&h).cloned().ok_or_else(|| format!("invalid or closed handle {h}"))
}

fn jstr(env: &mut JNIEnv, s: &JString) -> Result<String, String> { env.get_string(s).map(|x| x.into()).map_err(|e| format!("bad string argument: {e}")) }

fn out(env: &mut JNIEnv, r: std::thread::Result<Result<Value, String>>) -> jstring {
    let v = match r {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => json!({"error": e}),
        Err(_) => json!({"error": "internal panic in native core"}),
    };
    env.new_string(v.to_string()).map(|s| s.into_raw()).unwrap_or(std::ptr::null_mut())
}

fn open_impl(root: String, trust_json: String, device_json: String) -> Result<Value, String> {
    let trust = TrustStore::from_json(trust_json.as_bytes()).map_err(|e| format!("trust roots: {e}"))?;
    let device: DeviceProfile = serde_json::from_str(&device_json).map_err(|e| format!("device profile: {e}"))?;
    let rt = Runtime::open(&PathBuf::from(root), device, trust).map_err(|e| format!("open: {e}"))?;
    let h = NEXT.fetch_add(1, Ordering::SeqCst);
    table().lock().map_err(|_| "handle table poisoned".to_string())?.insert(h, Arc::new(Mutex::new(rt)));
    Ok(json!({"handle": h}))
}

#[no_mangle]
pub extern "system" fn Java_com_aiagent_core_NativeCore_nativeOpen(mut env: JNIEnv, _c: JClass, root: JString, trust: JString, device: JString) -> jstring {
    let args = (jstr(&mut env, &root), jstr(&mut env, &trust), jstr(&mut env, &device));
    let r = catch_unwind(AssertUnwindSafe(|| match args { (Ok(a), Ok(b), Ok(c)) => open_impl(a, b, c), (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => Err(e) }));
    out(&mut env, r)
}

#[no_mangle]
pub extern "system" fn Java_com_aiagent_core_NativeCore_nativeClose(mut env: JNIEnv, _c: JClass, handle: jlong) -> jstring {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let removed = table().lock().map_err(|_| "handle table poisoned".to_string())?.remove(&handle).is_some();
        Ok(json!({"closed": removed}))
    }));
    out(&mut env, r)
}

#[no_mangle]
pub extern "system" fn Java_com_aiagent_core_NativeCore_nativeImport(mut env: JNIEnv, _c: JClass, handle: jlong, path: JString) -> jstring {
    let p = jstr(&mut env, &path);
    let r = catch_unwind(AssertUnwindSafe(|| {
        let rt = get(handle)?; let p = p?;
        let mut g = rt.lock().map_err(|_| "runtime poisoned".to_string())?;
        serde_json::to_value(g.import(&PathBuf::from(p))).map_err(|e| e.to_string())
    }));
    out(&mut env, r)
}

#[no_mangle]
pub extern "system" fn Java_com_aiagent_core_NativeCore_nativeSolve(mut env: JNIEnv, _c: JClass, handle: jlong, intent: JString, input: JFloatArray) -> jstring {
    let it = jstr(&mut env, &intent);
    let n = env.get_array_length(&input).unwrap_or(0) as usize;
    let mut buf = vec![0f32; n];
    let got = env.get_float_array_region(&input, 0, &mut buf).map_err(|e| format!("bad input array: {e}"));
    let r = catch_unwind(AssertUnwindSafe(|| {
        let rt = get(handle)?; got?;
        let mut g = rt.lock().map_err(|_| "runtime poisoned".to_string())?;
        let o = g.solve(&Task { intent: it?, input: buf }).map_err(|e| e.to_string())?;
        serde_json::to_value(o).map_err(|e| e.to_string())
    }));
    out(&mut env, r)
}

#[no_mangle]
pub extern "system" fn Java_com_aiagent_core_NativeCore_nativeList(mut env: JNIEnv, _c: JClass, handle: jlong) -> jstring {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let rt = get(handle)?; let g = rt.lock().map_err(|_| "runtime poisoned".to_string())?;
        Ok(Value::Array(g.registry.all().map(|r| json!({"capability_id": r.capability_id, "active_version": r.active_version, "params": r.active().params,
            "model_bytes": r.active().model_bytes, "input_dim": r.input_dim, "labels": r.labels, "calls": r.stats.calls})).collect()))
    }));
    out(&mut env, r)
}

#[no_mangle]
pub extern "system" fn Java_com_aiagent_core_NativeCore_nativeRunPlan(mut env: JNIEnv, _c: JClass, handle: jlong, plan_json: JString, inputs_json: JString) -> jstring {
    let (p, i) = (jstr(&mut env, &plan_json), jstr(&mut env, &inputs_json));
    let r = catch_unwind(AssertUnwindSafe(|| {
        let rt = get(handle)?;
        let plan: Plan = serde_json::from_str(&p?).map_err(|e| format!("plan: {e}"))?;
        let inputs: BTreeMap<String, Vec<f32>> = serde_json::from_str(&i?).map_err(|e| format!("inputs: {e}"))?;
        let mut g = rt.lock().map_err(|_| "runtime poisoned".to_string())?;
        match g.run_plan(&plan, &inputs) {
            Ok(res) => serde_json::to_value(res).map_err(|e| e.to_string()),
            Err(aicore::graph::PlanError::NeedsHelp { node, reason }) => Ok(json!({"needs_help": true, "node": node, "reason": reason})),
            Err(e) => Err(e.to_string()),
        }
    }));
    out(&mut env, r)
}

#[no_mangle]
pub extern "system" fn Java_com_aiagent_core_NativeCore_nativePolicyCheck(mut env: JNIEnv, _c: JClass, policy_json: JString, request_json: JString, approvals_json: JString) -> jstring {
    let (p, q, a) = (jstr(&mut env, &policy_json), jstr(&mut env, &request_json), jstr(&mut env, &approvals_json));
    let r = catch_unwind(AssertUnwindSafe(|| {
        let policy: aicore::policy::Policy = serde_json::from_str(&p?).map_err(|e| format!("policy: {e}"))?;
        let req: aicore::policy::Request = serde_json::from_str(&q?).map_err(|e| format!("request: {e}"))?;
        let approvals: Vec<aicore::policy::Approval> = serde_json::from_str(&a?).map_err(|e| format!("approvals: {e}"))?;
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let mut d = serde_json::to_value(aicore::policy::evaluate(&policy, &req, &approvals, now)).map_err(|e| e.to_string())?;
        d["request_hash"] = json!(aicore::policy::request_hash(&req));
        Ok(d)
    }));
    out(&mut env, r)
}

#[no_mangle]
pub extern "system" fn Java_com_aiagent_core_NativeCore_nativeVersion(mut env: JNIEnv, _c: JClass) -> jstring {
    let mut backends = vec!["onnx-mlp-lite"];
    if cfg!(feature = "tract") { backends.push("onnx-tract"); }
    out(&mut env, Ok(Ok(json!({"core": aicore::RUNTIME_VERSION, "package_formats": aicore::SUPPORTED_FORMATS, "backends": backends}))))
}
