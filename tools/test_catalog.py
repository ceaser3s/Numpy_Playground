#!/usr/bin/env python3
"""Runs EVERY catalog snippet (same runner the website uses) (same runner the website uses) and reports problems.

    python tools/test_catalog.py                  # everything
    python tools/test_catalog.py --cat strings    # one category
    python tools/test_catalog.py --name read_csv  # entries whose name contains this
    python tools/test_catalog.py -v               # also print every OK entry

Checks per snippet: runs without error | no pandas warnings | produces visible output | not slow.
"""
import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "py"), str(ROOT / "tools")]
os.chdir(ROOT)

import runner  # noqa: E402
from catalog_src import CATEGORIES, ENTRIES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat")
    ap.add_argument("--name")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--slow", type=float, default=4000)
    args = ap.parse_args()

    info = json.loads(runner.init())
    print(f"numpy {info['numpy']} | datasets {info['shapes']}")

    bad, warns, total, skipped = [], [], 0, 0
    per_cat = {c["id"]: [0, 0] for c in CATEGORIES}
    for e in ENTRIES:
        if args.cat and e["cat"] != args.cat:
            continue
        if args.name and args.name not in e["name"]:
            continue
        total += 1
        ast.parse(e["code"])  # syntax check always
        per_cat[e["cat"]][0] += 1
        if e["local"]:
            skipped += 1
            continue
        t0 = time.perf_counter()
        r = json.loads(runner.run(e["code"], "t:" + e["id"]))
        ms = (time.perf_counter() - t0) * 1000
        tag = f"{e['cat']}:{e['name']}"
        problem = None
        if not r["ok"]:
            err = r["error"]
            problem = f"ERROR {err['type']}: {err['message'][:200]} (line {err['line']})"
        elif r.get("warnings"):
            warns.append((tag, r["warnings"]))
        if not problem:
            res = r.get("result", {})
            if res.get("kind") == "none" and not r["stdout"].strip() and not r["figures"]:
                problem = "NO OUTPUT (snippet returns nothing visible)"
            elif res.get("kind") == "array" and res.get("facts", {}).get("size", 1) == 0 and not e.get("allow_empty"):
                problem = "EMPTY RESULT (size 0)"
            elif ms > args.slow:
                problem = f"SLOW {ms:.0f} ms"
        if problem:
            bad.append((tag, problem))
        else:
            per_cat[e["cat"]][1] += 1
            if args.v:
                print(f"  ok   {tag:48} {ms:7.0f} ms  {res.get('kind')}")
    print()
    for t, p in bad:
        print(f"FAIL {t}\n     {p}")
    for t, w in warns:
        print(f"WARN {t}\n     " + "\n     ".join(w))
    print(f"\n{total} entries | {total - skipped - len(bad)} ok | {len(bad)} failing | {skipped} browser-local (syntax-checked only) | {len(warns)} with warnings")
    for c in CATEGORIES:
        n, ok = per_cat[c["id"]]
        if n:
            print(f"  {c['id']:12} {n:3d} entries")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
