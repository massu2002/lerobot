#!/usr/bin/env bash
# ==========================================================
# dataset_list.txt に列挙された各データセットについて
#  - meta/info.json から observation.images.* のカメラキーを自動検出
#  - 各カメラで CoTracker 動的領域検出を実行
#  - 途中で失敗しても続行し、最後にサマリー表示
#  - jq 不在でも Python でフォールバック
#  - JSONを"実行"しない（必ずjq/pyでパース）
# ----------------------------------------------------------
# 使い方:
#   ./cotracker_generate.sh dataset_list.txt
# ==========================================================

# 途中で失敗しても続けるため -e は付けない
set -u -o pipefail

HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/home/data01/smolvla}"
LIST_FILE="${1:-dataset_list.txt}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LC_ALL=C  # 安定ソート

# 入力ファイル存在チェック
if [[ ! -f "$LIST_FILE" ]]; then
  echo "ERROR: list file not found: $LIST_FILE" >&2
  exit 2
fi

# --- 成否カウント/記録 ---
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_LIST=()

# --- カメラキー検出（jq→Pythonフォールバック、かつ権限チェック）---
detect_cameras() {
    local repo_rel="$1"                              # 例: 00ri/so100_battery
    local repo_root="${HF_LEROBOT_HOME}/${repo_rel}"
    local info_path="${repo_root}/meta/info.json"
    local tasks_path="${repo_root}/meta/tasks.jsonl"
    local cams=()

    # info.json から抽出
    if [[ -f "$info_path" ]]; then
        if [[ ! -r "$info_path" ]]; then
            echo "  [warn] not readable: $info_path (permission?)" >&2
        else
            if command -v jq >/dev/null 2>&1; then
                mapfile -t cams < <(jq -r '
                    .features
                    | objects
                    | to_entries
                    | map(select(.key | startswith("observation.images.")))
                    | map(select(.value.dtype=="video" or (.value.info|type=="object" and (.value.info|keys|map(startswith("video."))|any))))
                    | .[].key
                ' "$info_path" 2>/dev/null | sort -u || true)
            fi

            if [[ ${#cams[@]} -eq 0 ]]; then
                mapfile -t cams < <("$PYTHON" - <<'PYCODE' "$info_path" 2>/dev/null || true
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
out = []
try:
    with p.open("r", encoding="utf-8") as f:
        j = json.load(f)
    feats = j.get("features", {})
    if isinstance(feats, dict):
        for k, v in feats.items():
            if isinstance(k, str) and k.startswith("observation.images."):
                dtype_ok = isinstance(v, dict) and v.get("dtype") == "video"
                info = v.get("info", {}) if isinstance(v, dict) else {}
                is_videoish = isinstance(info, dict) and any(
                    kk.startswith("video.") for kk in info.keys()
                )
                if dtype_ok or is_videoish:
                    out.append(k)
except Exception:
    pass
for k in sorted(set(out)):
    print(k)
PYCODE
)
            fi
        fi
    fi

    # info.json で見つからなければ tasks.jsonl を走査
    if [[ ${#cams[@]} -eq 0 && -f "$tasks_path" ]]; then
        if [[ ! -r "$tasks_path" ]]; then
            echo "  [warn] not readable: $tasks_path (permission?)" >&2
        else
            mapfile -t cams < <("$PYTHON" - <<'PYCODE' "$tasks_path" 2>/dev/null || true
import json, sys, pathlib, re
p = pathlib.Path(sys.argv[1])
found = set()
try:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # まず正規表現で拾う（壊れた行にも効く）
            for m in re.finditer(r'"observation\.images\.[^"]+"', line):
                found.add(m.group(0).strip('"'))
            # JSON としても試す
            try:
                obj = json.loads(line)
            except Exception:
                continue
            def walk(x):
                if isinstance(x, dict):
                    for k, v in x.items():
                        if isinstance(k, str) and k.startswith("observation.images."):
                            found.add(k)
                        walk(v)
                elif isinstance(x, list):
                    for it in x:
                        walk(it)
                elif isinstance(x, str):
                    if x.startswith("observation.images."):
                        found.add(x)
            walk(obj)
except Exception:
    pass
for k in sorted(found):
    print(k)
PYCODE
)
        fi
    fi

    # デフォルト（見つからない場合）
    if [[ ${#cams[@]} -eq 0 ]]; then
        cams=("observation.images.front" "observation.images.wrist")
    fi

    printf "%s\n" "${cams[@]}"
}

# --- 1ジョブ実行（Python呼び出し） ---
run_dynamic() {
    local repo_id="$1"
    local camera_key="$2"
    local disp_thresh

    case "$camera_key" in
        observation.images.front) disp_thresh=1.25 ;;
        observation.images.wrist) disp_thresh=1.5 ;;
        *) disp_thresh=1.0 ;;
    esac

    echo "=== Running CoTracker for ${repo_id} / ${camera_key} (disp_thresh=${disp_thresh}) ==="

    REPO_ID="$repo_id" CAMERA_KEY="$camera_key" DISP_THRESH="$disp_thresh" \
    "$PYTHON" - <<'PYCODE'
from tool.cotracker import process_all_cotracker_dynamic
import os, sys, traceback

repo_id = os.environ["REPO_ID"]
camera_key = os.environ["CAMERA_KEY"]
disp_thresh = float(os.environ["DISP_THRESH"])

try:
    process_all_cotracker_dynamic(
        repo_id=repo_id,
        camera_key=camera_key,
        device="cuda",
        grid_size=40,
        chunk_len=128,
        overlap=16,
        t_stride=1,         # 0は実装により解釈が揺れるため1に統一
        normalized=False,
        mode="mask",
        disp_thresh=disp_thresh,
        fps=30.0,
        new_repo_suffix="_dynamic",
    )
except Exception:
    traceback.print_exc()
    sys.exit(1)
PYCODE

    local py_status=$?
    if [[ $py_status -ne 0 ]]; then
        echo "❌ FAILED: ${repo_id} / ${camera_key}"
        return 1
    else
        echo "✅ SUCCESS: ${repo_id} / ${camera_key}"
        return 0
    fi
}

# --- メイン：リストを1行ずつ処理 ---
while IFS= read -r repo_id || [[ -n "${repo_id:-}" ]]; do
    # 空行/コメント行スキップ（空白のみの行も除外）
    [[ -z "${repo_id// }" || "$repo_id" =~ ^# ]] && continue

    echo "__ Start: ${repo_id}"

    # 参照ルートの存在／権限チェック
    target_root="${HF_LEROBOT_HOME}/${repo_id}"
    if [[ ! -d "$target_root" ]]; then
        echo "  [warn] not found: $target_root" >&2
        ((FAIL_COUNT++))
        FAILED_LIST+=("${repo_id} :: <root-missing>")
        continue
    fi
    if [[ ! -r "$target_root" ]]; then
        echo "  [warn] not readable: $target_root (permission?)" >&2
        ((FAIL_COUNT++))
        FAILED_LIST+=("${repo_id} :: <root-unreadable>")
        continue
    fi

    # カメラ検出
    mapfile -t CAMERA_KEYS < <(detect_cameras "$repo_id")
    echo "  _ Detected cameras: ${CAMERA_KEYS[*]}"

    # 各カメラを実行
    for cam in "${CAMERA_KEYS[@]}"; do
        if run_dynamic "$repo_id" "$cam"; then
            ((SUCCESS_COUNT++))
        else
            ((FAIL_COUNT++))
            FAILED_LIST+=("${repo_id} :: ${cam}")
            continue
        fi
    done
done < "$LIST_FILE"

# --- サマリー ---
echo
echo "================= SUMMARY ================="
echo "  ✅ Success: ${SUCCESS_COUNT}"
echo "  ❌ Failed : ${FAIL_COUNT}"
if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
    echo "-------------------------------------------"
    echo "  Failed jobs:"
    for item in "${FAILED_LIST[@]}"; do
        echo "   - ${item}"
    done
fi
echo "==========================================="