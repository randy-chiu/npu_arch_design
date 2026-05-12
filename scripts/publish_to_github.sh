#!/usr/bin/env bash
set -euo pipefail

repo_name="${1:-npu_arch_design}"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh CLI is not installed. Install it and run: gh auth login" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: gh CLI is not authenticated. Run: gh auth login" >&2
  exit 1
fi

git branch -M main

if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main
else
  gh repo create "${repo_name}" --private --source=. --remote=origin --push
fi

