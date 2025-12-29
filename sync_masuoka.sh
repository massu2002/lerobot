#!/usr/bin/env bash
set -euo pipefail

# ====== settings (override by env/args) ======
BRANCH="${1:-masuoka}"              # working branch (train/dev)
REMOTE="${REMOTE:-origin}"
EVAL_WORKTREE_REL="${EVAL_WORKTREE_REL:-../lerobot_eval}"  # relative to repo root
TAG_PREFIX="${TAG_PREFIX:-eval}"    # "eval" -> eval-YYYYMMDD
DATE_YYYYMMDD="${DATE_YYYYMMDD:-$(date +%Y%m%d)}"
TAG_NAME="${TAG_NAME:-${TAG_PREFIX}-${DATE_YYYYMMDD}}"
TAG_MESSAGE="${TAG_MESSAGE:-Evaluation snapshot ${DATE_YYYYMMDD}}"
# ============================================

echo "=== sync + tag + eval-worktree ==="
echo "repo           : $(pwd)"
echo "remote         : ${REMOTE}"
echo "branch         : ${BRANCH}"
echo "eval worktree  : ${EVAL_WORKTREE_REL}"
echo "tag            : ${TAG_NAME}"
echo

# ---- preflight ----
git rev-parse --is-inside-work-tree >/dev/null

current_branch="$(git branch --show-current)"
if [[ "${current_branch}" != "${BRANCH}" ]]; then
  echo "ERROR: current branch is '${current_branch}', expected '${BRANCH}'."
  echo "Run: git checkout ${BRANCH}"
  exit 1
fi

# Rebase途中が残っていたら止める
if [[ -d .git/rebase-apply || -d .git/rebase-merge ]]; then
  echo "ERROR: A rebase is already in progress."
  echo "Fix it then run again:"
  echo "  git rebase --continue   OR   git rebase --abort"
  exit 1
fi

# 未コミット変更があると事故りやすいので止める（必要ならstash運用に変えられます）
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: You have uncommitted changes. Commit or stash first."
  git status --porcelain
  exit 1
fi

echo "[1/7] Fetch from ${REMOTE} ..."
git fetch --prune "${REMOTE}" --tags

echo "[2/7] Ensure ${REMOTE}/${BRANCH} exists ..."
git show-ref --verify --quiet "refs/remotes/${REMOTE}/${BRANCH}" || {
  echo "ERROR: remote branch ${REMOTE}/${BRANCH} not found."
  exit 1
}

echo "[3/7] Rebase ${BRANCH} onto ${REMOTE}/${BRANCH} ..."
git rebase "${REMOTE}/${BRANCH}"

echo "[4/7] Push ${BRANCH} (rebased) with --force-with-lease ..."
git push --force-with-lease "${REMOTE}" "${BRANCH}:${BRANCH}"

echo "[5/7] Create or update tag: ${TAG_NAME} ..."
# 既にタグがあったら安全のため止める（上書き運用にしたいなら後述）
if git show-ref --tags --verify --quiet "refs/tags/${TAG_NAME}"; then
  echo "ERROR: tag '${TAG_NAME}' already exists."
  echo "If you want a new snapshot, change DATE_YYYYMMDD or TAG_NAME."
  exit 1
fi

git tag -a "${TAG_NAME}" -m "${TAG_MESSAGE}"

echo "[6/7] Push tag ${TAG_NAME} ..."
git push "${REMOTE}" "${TAG_NAME}"

echo "[7/7] Ensure eval worktree exists and checkout tag ..."
# worktree の絶対パスを算出
REPO_ROOT="$(git rev-parse --show-toplevel)"
EVAL_WORKTREE_PATH="$(cd "${REPO_ROOT}" && cd "$(dirname "${EVAL_WORKTREE_REL}")" && pwd)/$(basename "${EVAL_WORKTREE_REL}")"

# worktree が存在するか
if [[ -d "${EVAL_WORKTREE_PATH}/.git" || -f "${EVAL_WORKTREE_PATH}/.git" ]]; then
  echo " - eval worktree exists: ${EVAL_WORKTREE_PATH}"
else
  echo " - creating eval worktree: ${EVAL_WORKTREE_PATH}"
  git worktree add "${EVAL_WORKTREE_PATH}" "${TAG_NAME}"
fi

# 反映（タグ checkout）
(
  cd "${EVAL_WORKTREE_PATH}"
  git fetch --tags "${REMOTE}" >/dev/null 2>&1 || true
  git checkout -f "${TAG_NAME}"
  echo " - eval worktree now at: $(git describe --tags --always --dirty)"
)

echo
echo "DONE ✅"
echo "Train repo : ${REPO_ROOT} (${BRANCH})"
echo "Eval repo  : ${EVAL_WORKTREE_PATH} (tag ${TAG_NAME})"
echo "Tip: run inference from '${EVAL_WORKTREE_PATH}'"


# Example usage:
# TAG_NAME=eval-20260101 ./sync_masuoka.sh