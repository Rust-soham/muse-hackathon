"""Check serving/patch.diff against the LIVE patch rules (GET /api/v1/serving). Exit 1 on any violation.

    python tools/serving/validate_patch.py [patch.diff]

Scans the whole diff, context lines included, for forbidden words (a hook placed near stock code can pull one in).
"""
import json
import pathlib
import re
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
patch = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "serving" / "patch.diff"
req = urllib.request.Request("https://openkernelchallenge.dev/api/v1/serving", headers={"User-Agent": "curl/8.5.0"})
with urllib.request.urlopen(req, timeout=30) as r:
    api = json.load(r)
rules = api["rules"]
raw = patch.read_bytes()
text = raw.decode("utf-8")
bad = []
if len(raw) > rules["max_patch_bytes"]:
    bad.append(f"size {len(raw)} > {rules['max_patch_bytes']}")
files = re.findall(r"^diff --git a/(\S+) b/(\S+)$", text, re.M)
for a, b in files:
    if a != b:
        bad.append(f"rename {a} -> {b}")
    if not b.startswith(rules["allowed_prefix"]):
        bad.append(f"outside prefix: {b}")
    if any(b.startswith(p) for p in rules["forbidden_prefixes"]):
        bad.append(f"forbidden prefix: {b}")
    if not any(b.endswith(s) for s in rules["allowed_suffixes"]):
        bad.append(f"suffix not allowed: {b}")
for marker in ("GIT binary patch", "old mode", "new mode", "rename from", "similarity index"):
    if re.search(rf"^{marker}", text, re.M):
        bad.append(f"diff contains '{marker}'")
for n, line in enumerate(text.splitlines(), 1):
    if line.startswith("+") and not line.startswith("+++") and len(line) - 1 > rules["max_line_chars"]:
        bad.append(f"line {n}: added line of {len(line) - 1} chars")
    for w in rules["forbidden_words"]:
        if w in line:
            bad.append(f"line {n}: forbidden word {w!r} ({'added' if line.startswith('+') else 'context/removed'})")
base = api["base"]
print(f"patch {patch} | {len(raw)} B of {rules['max_patch_bytes']} | {len(files)} files | base {base['commit'][:12]}")
print(f"serving opens {api['tracks']['serving']['opens']} | check deadline {api['check']['deadline_min']} min, C{api['check']['rungs'][0]['concurrency']}")
print("\n".join("VIOLATION " + b for b in bad) if bad else "OK: no rule violations")
sys.exit(1 if bad else 0)
