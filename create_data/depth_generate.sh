#!/usr/bin/env bash
# ==========================================================
# dataset_list.txt に列挙された各データセットについて
#  - meta/info.json から observation.images.* のカメラキーを自動検出
#  - 各カメラで Depth Anything 推論 + 可視化を実行
#  - 途中で失敗しても続行し、最後にサマリー表示
#  - jq 不在でも Python でフォールバック（JSONは"実行"しない）
# ----------------------------------------------------------
# 使い方:
#   ./depth_generate.sh dataset_list.txt
#   環境変数で一部調整可:
#     HF_LEROBOT_HOME=/home/data01/smolvla \
#     MODEL_ID="depth-anything/Depth-Anything-V2-Base-hf" \
#     BATCH_SIZE=8 T_STRIDE=0 OUT_DTYPE=float16 \
#     python=python ./depth_generate_continue.sh list.txt
# ==========================================================

# 途中で失敗しても続けるため -e は付けない
set -u -o pipefail

HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/home/data01/smolvla}"
LIST_FILE="${1:-dataset_list.txt}"
PYTHON="${python:-${PYTHON:-python}}"
MODEL_ID="${MODEL_ID:-depth-anything/Depth-Anything-V2-Base-hf}"
BATCH_SIZE="${BATCH_SIZE:-4}"
T_STRIDE="${T_STRIDE:-0}"         # 0/未設定なら後で 1 に矯正
OUT_DTYPE="${OUT_DTYPE:-float16}" # float16 | float32
NORMALIZE="${NORMALIZE:-global}"  # none | frame | global
SCALE="${SCALE:-global}"          # global | per_frame | percentile
FPS="${FPS:-30.0}"
PERCENTILES="${PERCENTILES:-1.0,99.0}"
NEW_SUFFIX="${NEW_SUFFIX:-_depth}"
LC_ALL=C

# 入力ファイル存在チェック
if [[ ! -f "$LIST_FILE" ]]; then
  echo "ERROR: list file not found: $LIST_FILE" >&2
  exit 2
fi

# --- 成否カウント/記録 ---
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_LIST=()

# --- カメラキー検出（jq→Python フォールバック、権限チェック付き） ---
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
                mapfile -t cams < <( "$PYTHON" - "$info_path" <<'PYCODE' 2>/dev/null || true
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
            mapfile -t cams < <( "$PYTHON" - "$tasks_path" <<'PYCODE' 2>/dev/null || true
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
run_depth() {
    local repo_id="$1"
    local camera_key="$2"

    # t_stride を安全値に矯正
    local t_stride_safe="$T_STRIDE"
    if [[ -z "${t_stride_safe:-}" || "$t_stride_safe" -le 0 ]]; then
        t_stride_safe=1
    fi

    echo "=== Running DepthAnything for ${repo_id} / ${camera_key} ==="

    REPO_ID="$repo_id" CAMERA_KEY="$camera_key" \
    MODEL_ID="$MODEL_ID" BATCH_SIZE="$BATCH_SIZE" \
    T_STRIDE="$t_stride_safe" OUT_DTYPE="$OUT_DTYPE" \
    NORMALIZE="$NORMALIZE" SCALE="$SCALE" FPS="$FPS" \
    PERCENTILES="$PERCENTILES" NEW_SUFFIX="$NEW_SUFFIX" \
    "$PYTHON" - <<'PYCODE'
from tool.depthanything import process_all_depths
import os, sys, traceback

def _coerce_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def _coerce_tuple_floats(csv: str):
    try:
        parts = [float(x.strip()) for x in csv.split(',')]
        if len(parts) == 2:
            return (parts[0], parts[1])
    except Exception:
        pass
    return (1.0, 99.0)

repo_id    = os.environ["REPO_ID"]
camera_key = os.environ["CAMERA_KEY"]
model_id   = os.environ.get("MODEL_ID", "depth-anything/Depth-Anything-V2-Base-hf")
batch_size = _coerce_int_env("BATCH_SIZE", 4)
t_stride   = _coerce_int_env("T_STRIDE", 1)   # ここではすでに 1 以上に矯正済
out_dtype  = os.environ.get("OUT_DTYPE", "float16")
normalize  = os.environ.get("NORMALIZE", "global")
scale      = os.environ.get("SCALE", "global")
fps        = float(os.environ.get("FPS", "30.0"))
percentiles= _coerce_tuple_floats(os.environ.get("PERCENTILES", "1.0,99.0"))
new_suffix = os.environ.get("NEW_SUFFIX", "_depth")

try:
    process_all_depths(
        repo_id=repo_id,
        camera_key=camera_key,
        model_id=model_id,
        device="cuda",
        batch_size=batch_size,
        t_stride=t_stride,         # 実装が 0 許容でも 1 を渡して安定運用
        normalize=normalize,       # "none" | "frame" | "global"
        out_dtype=out_dtype,       # "float16" | "float32"
        fps=fps,
        colormap="magma",
        scale=scale,               # "global" | "percentile" | "per_frame"
        percentiles=percentiles,
        new_repo_suffix=new_suffix,
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

    echo "📂 Start: ${repo_id}"

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
    echo "  ↳ Detected cameras: ${CAMERA_KEYS[*]}"

    # 各カメラを実行
    for cam in "${CAMERA_KEYS[@]}"; do
        if run_depth "$repo_id" "$cam"; then
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
