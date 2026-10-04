"""Export an installed tool as an OpenClaw SKILL.md. The skill only teaches the agent to call the guarded host; it grants nothing and contains no tool code."""
from __future__ import annotations
import json
from pathlib import Path


def skill_name(tool_id: str) -> str: return tool_id.replace("_", "-")


def export(registry, tool_id: str, out_dir: Path, aitool: Path, registry_root: Path) -> Path:
    rec = registry.record(tool_id, registry.active_version(tool_id))
    d = Path(out_dir) / skill_name(tool_id); d.mkdir(parents=True, exist_ok=True)
    desc = rec["description"][:150].replace("\n", " ")
    meta = json.dumps({"openclaw": {"requires": {"bins": ["python3"]}}}, separators=(",", ":"))
    body = f"""---
name: {skill_name(tool_id)}
description: {desc}
metadata: {meta}
---

# {skill_name(tool_id)}

{rec['description']}. Installed tool `{tool_id}` version {rec['version']} (code sha256 `{rec['code_sha256'][:16]}...`), approved by the operator.

Run it ONLY through the guarded host, passing one JSON object; never read, copy or execute the tool's source yourself:

```bash
python3 {aitool} --root {registry_root} run {tool_id} --input '<JSON object>'
```

The host replies with one JSON line: `{{"ok": true, "output": {{...}}}}` or `{{"ok": false, "refused": "<reason>"}}`. A refusal is final: do not retry with modified input to get around it,
and do not try to install, edit or approve tools. Tools get no files, network or processes.
"""
    (d / "SKILL.md").write_text(body)
    return d
