#!/usr/bin/env bash

# ==========================================================
# 全chunk/fileに対してDepth Anythingで深度推定＋可視化を実行するスクリプト
# 対象: observation.images.front / observation.images.wrist
# ----------------------------------------------------------
# 実行例:
#   ./upgrade_lerobot_datasets.sh ./create_data/dataset_ids.txt /home/data01/smolvla
# ==========================================================

set -euo pipefail

# ---------------- 設定 ----------------
# 第1引数: dataset_ids.txt のパス
IDS_FILE="${1:-dataset_ids.txt}"

# 第2引数: ローカルのデータセットのルート
#  例) /home/data01/smolvla/00ri/so100_battery/meta/info.json
#        ↑ DATA_ROOT=/home/data01/smolvla
DATA_ROOT="${2:-/home/data01/smolvla}"

# 使用する Python コマンド（必要なら python3 等に変更）
PYTHON="${PYTHON:-python}"

# lerobot の convert スクリプトへのパス（環境に合わせて必要なら修正）
V20_TO_V21_SCRIPT="src/lerobot/datasets/v21/convert_dataset_v20_to_v21.py"
V21_TO_V30_SCRIPT="src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py"

# jq が必要です（なければ: sudo apt-get install jq）
command -v jq >/dev/null 2>&1 || {
  echo "ERROR: jq が見つかりません。sudo apt-get install jq などでインストールしてください。" >&2
  exit 1
}

# --------------------------------------
echo "IDS_FILE : ${IDS_FILE}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo

while IFS= read -r REPO_ID; do
  # 空行とコメント行はスキップ
  [[ -z "$REPO_ID" ]] && continue
  [[ "$REPO_ID" =~ ^# ]] && continue

  echo "==============================================="
  echo "Dataset: ${REPO_ID}"

  # ローカルディレクトリ（例: /home/data01/smolvla/0x00raghu/toffee_blue）
  LOCAL_DIR="${DATA_ROOT}/${REPO_ID}"
  INFO_PATH="${LOCAL_DIR}/meta/info.json"

  if [[ ! -f "$INFO_PATH" ]]; then
    echo "  [WARN] info.json が見つかりません: ${INFO_PATH}"
    echo "        このデータセットはスキップします"
    continue
  fi

  # codebase_version を取得
  CODEBASE_VERSION="$(jq -r '.codebase_version // "unknown"' "$INFO_PATH" 2>/dev/null || echo "unknown")"
  echo "  現在の version: ${CODEBASE_VERSION}"

  case "$CODEBASE_VERSION" in
    "v3.0")
      echo "  → すでに v3.0 なのでスキップ"
      ;;

    "v2.1")
      echo "  → v2.1 → v3.0 に変換します"
      set +e
      "${PYTHON}" "${V21_TO_V30_SCRIPT}" \
        --repo-id="${REPO_ID}" \
        --root="${DATA_ROOT}" \
        --push-to-hub=false
      status=$?
      set -e

      if [[ $status -ne 0 ]]; then
        echo "  [ERROR] v2.1→v3.0 変換に失敗しました (repo_id=${REPO_ID})" >&2
      else
        echo "  ✓ v3.0 への変換完了 (repo_id=${REPO_ID})"
      fi
      ;;

    "v2.0")
      echo "  → v2.0 → v2.1 → v3.0 に変換します"

      # まず v2.0 → v2.1
      echo "  [step1] v2.0 → v2.1"
      set +e
      "${PYTHON}" "${V20_TO_V21_SCRIPT}" \
        --repo-id="${REPO_ID}" \
        --root="${LOCAL_DIR}"
      status1=$?
      set -e

      if [[ $status1 -ne 0 ]]; then
        echo "  [ERROR] v2.0→v2.1 変換に失敗しました (repo_id=${REPO_ID})" >&2
        echo "         v3.0 への変換はスキップします"
        continue
      fi

      # 次に v2.1 → v3.0
      echo "  [step2] v2.1 → v3.0"
      set +e
      "${PYTHON}" "${V21_TO_V30_SCRIPT}" \
        --repo-id="${REPO_ID}" \
        --root="${DATA_ROOT}" \
        --push-to-hub=false
      status2=$?
      set -e

      if [[ $status2 -ne 0 ]]; then
        echo "  [ERROR] v2.1→v3.0 変換に失敗しました (repo_id=${REPO_ID})" >&2
      else
        echo "  ✓ v3.0 への変換完了 (repo_id=${REPO_ID})"
      fi
      ;;

    *)
      echo "  [WARN] codebase_version が 'v2.0', 'v2.1', 'v3.0' のいずれでもありません: ${CODEBASE_VERSION}"
      echo "        このデータセットはスキップします"
      ;;
  esac

  echo
done < "$IDS_FILE"

echo "=== 全データセットの処理が終了しました ==="

find /home/data01/smolvla -type f -path "*/meta/info.json" \
  -exec sh -c 'echo -n "{}: "; jq -r .codebase_version {}' \;