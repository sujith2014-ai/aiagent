# `.cap` format, version `cap/1`

A `.cap` is a zip archive:

```
manifest.json        # signed; exact bytes are what the signature covers
signature.json       # {"alg":"ed25519","key_id":...,"sig_b64":...}
model/model.<fmt>    # model artifact (format declared in manifest)
tests/cases.jsonl    # bundled tests: {"input":[...],"expected":<label index>}
routing/hints.json   # {"description":..., "keywords":[...]}
```

## Manifest fields
`format_version`, `package_id`, `capability_id`, `version` (semver), `model{format,variant,file}`,
`io{input{dtype,dim,description}, output{kind,labels}}`, `requires{runtime_min, device_caps[]}`,
`dependencies[]` (capability ids that must already be installed), `resources{params,model_bytes,min_ram_mb}`,
`provenance{...}`, `tests{file,count,min_accuracy}`, `routing_file`, `created`, `signer{key_id}`,
`contents{path: sha256}` for every file except `manifest.json`/`signature.json`.

## Activation sequence (implemented in `core/src/runtime.rs`)
parse -> signature -> hashes -> compatibility -> (version policy, dependencies) -> sandbox_load -> bundled_tests -> resource_check -> activate.
Any failure stops the import; the registry and store are unchanged (tested). Older versions stay in the registry for rollback.

## Rules
- No undeclared files; paths with `..`, absolute paths and backslashes are rejected.
- `device_caps` are abstract names (`camera`, `browser`, `fs.read`); the device profile lists which it provides.
- New model formats need a new `Backend`; the package structure does not change.
- Only dtype `f32` input and `classification` output exist in `cap/1` (a deliberate Phase 1 limitation).
