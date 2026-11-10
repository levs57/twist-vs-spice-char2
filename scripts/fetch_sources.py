#!/usr/bin/env python3
"""Fetch the audited source revisions, or verify existing checkouts with --check."""
import argparse
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def git(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check pins and tracked changes without fetching")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "audit/sources.lock.json").read_text())
    for repo in manifest["repositories"]:
        dest = ROOT / "sources" / repo["name"]
        revision = repo["revision"]
        if not dest.exists():
            if args.check:
                raise SystemExit(f"Missing checkout: {dest}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            git("clone", "--filter=blob:none", "--no-checkout", repo["url"], str(dest))
            git("checkout", "--detach", revision, cwd=dest)
        actual = git("rev-parse", "HEAD", cwd=dest)
        if actual != revision:
            raise SystemExit(f"{repo['name']}: expected {revision}, found {actual}; existing checkout left untouched")
        if git("status", "--porcelain", "--untracked-files=no", cwd=dest):
            raise SystemExit(f"{repo['name']}: tracked source changes present; checkout left untouched")
        print(f"{repo['name']}: {actual} (clean)")


if __name__ == "__main__":
    main()
