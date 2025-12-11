#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/data01/smolvla"

# 代表スキーマとして使う .parquet （必要に応じて変更）
REP_PARQUET="/home/masuoka/.cache/huggingface/lerobot/katoken/box_black_plate/data/chunk-000/file-001.parquet"

multi_task=$(
ROOT_DIR="$ROOT_DIR" REP_PARQUET="$REP_PARQUET" python - << 'PY'
import os
import sys
from pathlib import Path
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path(os.environ["ROOT_DIR"]).expanduser().resolve()
rep_parquet = Path(os.environ["REP_PARQUET"]).expanduser().resolve()

# 1) 代表スキーマの読み込み
pf = pq.ParquetFile(rep_parquet)
rep_schema = pf.schema_arrow
EXPECTED_ORDER = [f.name for f in rep_schema]
EXPECTED_TYPES = {f.name: str(f.type) for f in rep_schema}


def log(*args, **kwargs):
    """ログは stderr に出す"""
    print(*args, file=sys.stderr, **kwargs)


def dataset_has_no_mismatch(ds_dir: Path) -> bool:
    """
    ds_dir: /home/data01/smolvla/{owner}/{ds} を想定
    その中の data/chunk-*/file-*.parquet が
    代表スキーマ(EXPECTED_ORDER/EXPECTED_TYPES)と
    1つも MISMATCH しなければ True を返す。
    """
    data_dir = ds_dir / "data"
    if not data_dir.is_dir():
        log(f"[WARN] {ds_dir} has no data/ directory, skip")
        return False

    chunk_dirs = sorted([p for p in data_dir.glob("chunk-*") if p.is_dir()])
    if not chunk_dirs:
        log(f"[WARN] {data_dir} has no chunk-* directories, skip")
        return False

    for chunk_dir in chunk_dirs:
        parquet_paths = sorted(chunk_dir.glob("*.parquet"))
        if not parquet_paths:
            log(f"[WARN] {chunk_dir} has no .parquet files, skip this chunk")
            continue

        for pq_path in parquet_paths:
            try:
                pf = pq.ParquetFile(pq_path)
                schema = pf.schema_arrow
            except Exception as e:
                log(f"=== ERROR reading {pq_path}: {e!r}")
                return False

            cols = [f.name for f in schema]
            types = {f.name: str(f.type) for f in schema}

            missing = [c for c in EXPECTED_ORDER if c not in cols]
            extra   = [c for c in cols if c not in EXPECTED_ORDER]

            type_mismatch = []
            for name in EXPECTED_ORDER:
                if name in types:
                    t = types[name]
                    et = EXPECTED_TYPES[name]
                    if t != et:
                        type_mismatch.append((name, t, et))

            order_differs = (cols != EXPECTED_ORDER)

            if missing or extra or type_mismatch or order_differs:
                # 1つでも MISMATCH があればこの dataset は NG
                log(f"[MISMATCH] {pq_path}")
                if missing:
                    log("  missing:", missing)
                if extra:
                    log("  extra  :", extra)
                if type_mismatch:
                    log("  type mismatch:", type_mismatch)
                if order_differs:
                    log("  NOTE: column order differs from EXPECTED_ORDER")
                return False

    # ここまで来たらこの dataset は「クリーン」
    return True


good_repo_ids: list[str] = []

for owner in root.iterdir():
    if not owner.is_dir():
        continue
    if owner.name == "multi_task":
        # 出力先フォルダは除外
        continue

    for ds in owner.iterdir():
        if not ds.is_dir():
            continue

        # 1) owner と ds が同名 (Chojins/Chojins) はスキップ
        if owner.name == ds.name:
            continue

        # 2) LeRobot データセット判定: meta/info.json があるか
        info_json = ds / "meta" / "info.json"
        if not info_json.exists():
            continue

        # 3) episodes がちゃんとあるか
        episodes_dir = ds / "meta" / "episodes"
        parquets = list(episodes_dir.rglob("*.parquet")) if episodes_dir.is_dir() else []
        if not parquets:
            log(f"[SKIP] {owner.name}/{ds.name}: no episodes parquet under {episodes_dir}")
            continue

        # 4) data/ 側のスキーマ MISMATCH が無いかチェック
        if not dataset_has_no_mismatch(ds):
            repo_id = f"{owner.name}/{ds.name}"
            log(f"[NG]   {repo_id}: schema mismatch")
            continue

        repo_id = f"{owner.name}/{ds.name}"

        # ここまで通ったものだけ「OK」とみなす
        log(f"[OK]   {repo_id}")
        good_repo_ids.append(repo_id)

# ★ 最後に stdout に repo_id リストを返す（bash 側の $(...) で受け取る用）
print(repr(good_repo_ids))
PY
)

echo "[INFO] Selected clean repo_ids:"
echo "$multi_task"

out_repo_id="merged/pretraindata_v1"

lerobot-edit-dataset \
  --repo_id "${out_repo_id}" \
  --root "/home/data01/smolvla/${out_repo_id}" \
  --operation.type merge \
  --operation.repo_ids "${multi_task}"
