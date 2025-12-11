#!/usr/bin/env bash
# ==========================================================
# Hugging Face community datasets 一括ダウンロード（安全版）
#  - 429対策: Retry-After + 指数バックオフ + 内部並列抑制
#  - HfHubHTTPError 後方互換 import（古い hub でもOK）
#  - 404/403 をスキップ継続、接続系は自動リトライ
#  - ALLOW_PATTERNS / IGNORE_PATTERNS で対象絞り込み可
# ==========================================================

set -euo pipefail

# ---- 設定（必要に応じて変更）------------------------------
HF_TOKEN=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
IDS_FILE="${IDS_FILE:-dataset_ids.txt}"   # 取得元リスト
OUT_DIR="${OUT_DIR:-./datasets}"          # 保存先ルート
MAX_JOBS="${MAX_JOBS:-1}"                 # 並列数（1=逐次, >1 で並列）
RETRIES="${RETRIES:-5}"                   # 失敗時リトライ回数（429/接続系）
REPO_TYPE="dataset"                       # 固定：データセット
FLATTEN_PATHS="${FLATTEN_PATHS:-0}"       # パスを "user__repo" にフラット化
# 取得対象を絞る（例: "**/*.mp4,**/*.json"）※空なら無効
ALLOW_PATTERNS="${ALLOW_PATTERNS:-}"
IGNORE_PATTERNS="${IGNORE_PATTERNS:-*.lock}"

# ダウンロード間に入れるジッター（秒, 叩き過ぎ防止に微睡眠）
PER_ID_SLEEP="${PER_ID_SLEEP:-0.4}"

# Python 実行ファイル（サブシェルでも同じ Python を使わせる）
HFPY="${HFPY:-$(python3 -c 'import sys; print(sys.executable)')}"
export HFPY
# -----------------------------------------------------------

usage() {
  echo "Usage: $0 [IDS_FILE] [OUT_DIR]"
  echo "  env: IDS_FILE, OUT_DIR, MAX_JOBS, RETRIES, FLATTEN_PATHS,"
  echo "       ALLOW_PATTERNS, IGNORE_PATTERNS, HFPY, PER_ID_SLEEP"
  exit 1
}

# 引数（任意）
if [[ $# -ge 1 ]]; then IDS_FILE="$1"; fi
if [[ $# -ge 2 ]]; then OUT_DIR="$2"; fi

[[ -f "$IDS_FILE" ]] || { echo "❌ IDS_FILE が見つかりません: $IDS_FILE"; usage; }
mkdir -p "$OUT_DIR"

# あると速い（大容量最適化）。未インストールでもOK。
if "$HFPY" -c "import hf_transfer" >/dev/null 2>&1; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi

# 認証チェック（匿名は制限が厳しい）
if [[ -z "${HF_TOKEN:-}" ]]; then
  if command -v huggingface-cli >/dev/null 2>&1; then
    if ! huggingface-cli whoami >/dev/null 2>&1; then
      echo "⚠ 未認証です（匿名アクセス）。'huggingface-cli login' または 'export HF_TOKEN=...' を推奨。"
    fi
  else
    echo "⚠ 'huggingface-cli' が未検出。HF_TOKEN が無い場合は匿名アクセスになります。"
  fi
fi

# ===== HFコマンド決定（3段フォールバック） =====
if command -v huggingface-cli >/dev/null 2>&1; then
  HFCMD_TYPE="cli"
elif "$HFPY" -c "import huggingface_hub" >/dev/null 2>&1; then
  HFCMD_TYPE="module"
else
  echo "❌ huggingface_hub が見つかりません。例:  $HFPY -m pip install 'huggingface_hub[cli]'"
  exit 2
fi

# 文字列→Pythonリスト化のためのヘルパ（空なら None）
_py_list_expr() {
  local s="$1"
  if [[ -z "$s" ]]; then
    printf "None"
  else
    # カンマ区切り→['a','b']（前後空白除去）
    python3 - <<PY
s = "$s"
items = [x.strip() for x in s.split(",") if x.strip()]
print(repr(items) if items else "None")
PY
  fi
}

hf_download() {
  local repo_id="$1"
  local dst="$2"
  local repo_type="$3"

  # 1) CLI（あれば）
  if command -v huggingface-cli >/dev/null 2>&1; then
    local cli_args=(download "$repo_id" --repo-type "$repo_type" --local-dir "$dst" --resume-download)
    [[ -n "$ALLOW_PATTERNS" ]] && cli_args+=(--allow "$ALLOW_PATTERNS")
    [[ -n "$IGNORE_PATTERNS" ]] && cli_args+=(--ignore "$IGNORE_PATTERNS")
    if HF_TOKEN="${HF_TOKEN:-}" huggingface-cli "${cli_args[@]}"; then
      return 0
    fi
  fi

  # 2) python -m huggingface_hub.cli がある場合のみ使う（__main__ は使わない）
  if "$HFPY" - >/dev/null 2>&1 <<'PY'
import importlib, sys
sys.exit(0 if importlib.util.find_spec("huggingface_hub.cli") else 1)
PY
  then
    if "$HFPY" -m huggingface_hub.cli download "$repo_id" --repo-type "$repo_type" --local-dir "$dst" --resume-download \
      ${ALLOW_PATTERNS:+--allow "$ALLOW_PATTERNS"} \
      ${IGNORE_PATTERNS:+--ignore "$IGNORE_PATTERNS"}; then
      return 0
    fi
  fi

  # 3) API snapshot_download（Retry-After + 指数バックオフ）
  local allow_py; allow_py=$(_py_list_expr "$ALLOW_PATTERNS")
  local ignore_py; ignore_py=$(_py_list_expr "$IGNORE_PATTERNS")

  HF_TOKEN="${HF_TOKEN:-}" "$HFPY" - "$repo_id" "$dst" "$repo_type" "$RETRIES" "$allow_py" "$ignore_py" <<'PY'
import sys, os, time, random, socket
from typing import Optional
# 後方互換 import
try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print(f"[ERR] cannot import snapshot_download: {e}", file=sys.stderr)
    sys.exit(1)
HfHubHTTPError = None
for _mod in ("huggingface_hub.errors", "huggingface_hub", "huggingface_hub.utils._errors"):
    try:
        mod = __import__(_mod, fromlist=["HfHubHTTPError"])
        if hasattr(mod, "HfHubHTTPError"):
            HfHubHTTPError = getattr(mod, "HfHubHTTPError"); break
    except Exception:
        pass
if HfHubHTTPError is None:
    class HfHubHTTPError(Exception):
        def __init__(self, *args, response=None, **kwargs):
            super().__init__(*args); self.response = response

repo_id, dst, repo_type = sys.argv[1], sys.argv[2], sys.argv[3]
max_retries = int(sys.argv[4])
allow_patterns = eval(sys.argv[5])  # None or list[str]
ignore_patterns = eval(sys.argv[6])  # None or list[str]
os.makedirs(dst, exist_ok=True)

def _retry_after_seconds(e: Exception) -> Optional[float]:
    resp = getattr(e, "response", None)
    if resp is None: return None
    for k in ("retry-after","Retry-After"):
        if hasattr(resp, "headers") and k in resp.headers:
            try: return float(resp.headers[k])
            except Exception: pass
    return None

base_sleep, cap_sleep = 4.0, 120.0
for attempt in range(max_retries + 1):
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=dst,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=3,
            token=os.environ.get("HF_TOKEN") or True,
        )
        sys.exit(0)
    except (HfHubHTTPError, OSError, ConnectionError, TimeoutError, socket.timeout) as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        transient = (status == 429) or (status is not None and 500 <= status < 600) or isinstance(e, (OSError, ConnectionError, TimeoutError, socket.timeout))
        if transient and attempt < max_retries:
            ra = _retry_after_seconds(e)
            sleep_s = min(cap_sleep, (ra if ra is not None else (base_sleep * (2 ** attempt))) + random.uniform(0.0, 1.5))
            print(f"[retry] attempt {attempt+1}/{max_retries} after {sleep_s:.1f}s (status={status})", flush=True)
            time.sleep(sleep_s); continue
        if status in (401,):  # 認証エラーは明示
            print("[auth] 401 Unauthorized. Set valid HF_TOKEN or permission.", flush=True)
            sys.exit(1)
        if status in (403, 404):
            print(f"[skip] repo={repo_id} status={status}", flush=True)
            sys.exit(0)
        raise
PY
  return $?
}

export -f hf_download
export OUT_DIR FLATTEN_PATHS RETRIES REPO_TYPE HFPY HFCMD_TYPE ALLOW_PATTERNS IGNORE_PATTERNS PER_ID_SLEEP

download_one() {
  local id="$1"
  [[ -n "$id" ]] || return 0

  local subpath
  if [[ "$FLATTEN_PATHS" == "1" ]]; then
    subpath="${id//\//__}"
  else
    subpath="$id"
  fi
  local dst="${OUT_DIR}/${subpath}"

  if [[ -d "$dst" ]] && [[ -n "$(ls -A "$dst" 2>/dev/null || true)" ]]; then
    echo "✔ Skip (exists): $id"
    return 0
  fi

  mkdir -p "$dst"

  local i=0
  while :; do
    echo "↓ Downloading: $id -> $dst"
    if hf_download "$id" "$dst" "$REPO_TYPE"; then
      echo "✔ Done: $id"
      # 叩き過ぎ防止の微睡眠
      awk "BEGIN {s=$PER_ID_SLEEP + (rand()*0.3); if (s>0) {print \"...sleep \" s \"s\"; system(\"sleep \" s)} }" >/dev/null
      return 0
    fi
    if (( i >= RETRIES )); then
      echo "✗ Failed: $id (after ${RETRIES} retries)"
      return 1
    fi
    i=$((i+1))
    echo "… retrying ${i}/${RETRIES} in 3s"
    sleep 3
  done
}

export -f download_one

clean_ids() {
  sed -e 's/\r$//' -e 's/^[[:space:]]\+//' -e 's/[[:space:]]\+$//' "$IDS_FILE" \
  | grep -v -E '^\s*$' \
  | grep -v -E '^\s*#'
}

echo "📝 読み込み: $IDS_FILE"
echo "📁 保存先:  $OUT_DIR"
echo "⚙ 並列数:   $MAX_JOBS  | リトライ: $RETRIES  | フラット保存: $FLATTEN_PATHS"
echo "🔧 実行モード: ${HFCMD_TYPE}  | Python: ${HFPY}"
[[ -n "$ALLOW_PATTERNS" ]] && echo "🎯 ALLOW_PATTERNS: $ALLOW_PATTERNS"
[[ -n "$IGNORE_PATTERNS" ]] && echo "🚫 IGNORE_PATTERNS: $IGNORE_PATTERNS"

if [[ "$MAX_JOBS" -le 1 ]]; then
  while IFS= read -r id; do
    download_one "$id" || true
  done < <(clean_ids)
else
  clean_ids | xargs -P "$MAX_JOBS" -I{} bash -lc 'download_one "$@"' _ "{}" || true
fi

echo "🎉 完了。保存先: $(realpath "$OUT_DIR")"
