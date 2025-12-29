#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-masuoka}"
REMOTE="${REMOTE:-origin}"

echo "=== git safe rebase & push ==="
echo "repo   : $(pwd)"
echo "remote : ${REMOTE}"
echo "branch : ${BRANCH}"
echo

# ---- preflight ----
git rev-parse --is-inside-work-tree >/dev/null

current_branch="$(git branch --show-current)"
if [[ "${current_branch}" != "${BRANCH}" ]]; then
  echo "ERROR: current branch is '${current_branch}', expected '${BRANCH}'."
  echo "Run: git checkout ${BRANCH}"
  exit 1
fi

# サブモジュールがある場合の安全策（不要なら消してOK）
# git submodule update --init --recursive

echo "[1/6] Fetch from ${REMOTE} ..."
git fetch --prune "${REMOTE}"

# ローカルに未コミット変更があるとrebaseが止まりやすいので検知
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo
  echo "ERROR: You have uncommitted changes."
  echo "Please commit or stash them before running this script."
  echo
  git status --porcelain
  exit 1
fi

echo "[2/6] Ensure ${REMOTE}/${BRANCH} exists ..."
git show-ref --verify --quiet "refs/remotes/${REMOTE}/${BRANCH}" || {
  echo "ERROR: remote branch ${REMOTE}/${BRANCH} not found."
  exit 1
}

echo "[3/6] Rebase ${BRANCH} onto ${REMOTE}/${BRANCH} ..."
# 直前に途中rebaseが残っていたら止める
if [[ -d .git/rebase-apply || -d .git/rebase-merge ]]; then
  echo "ERROR: A rebase is already in progress."
  echo "Fix it then run again:"
  echo "  git rebase --continue   OR   git rebase --abort"
  exit 1
fi

git rebase "${REMOTE}/${BRANCH}"

echo "[4/6] Show status ..."
git status -sb
echo

echo "[5/6] Push with --force-with-lease (safe for rebased history) ..."
git push --force-with-lease "${REMOTE}" "${BRANCH}:${BRANCH}"

echo "[6/6] Done."
echo "Latest commit:"
git log --oneline -1
