from pathlib import Path
import pyarrow.parquet as pq

# 代表スキーマを入れておくグローバル
EXPECTED_ORDER: list[str] = []
EXPECTED_TYPES: dict[str, str] = {}

def set_expected_schema(rep_parquet: Path) -> None:
    """
    代表となる .parquet ファイルからスキーマを読み込み、
    EXPECTED_ORDER / EXPECTED_TYPES を更新する
    """
    global EXPECTED_ORDER, EXPECTED_TYPES

    rep_parquet = rep_parquet.expanduser().resolve()
    pf = pq.ParquetFile(rep_parquet)
    schema = pf.schema_arrow

    EXPECTED_ORDER = [f.name for f in schema]
    EXPECTED_TYPES = {f.name: str(f.type) for f in schema}

    print("[INFO] representative schema loaded from:", rep_parquet)
    print("       EXPECTED_ORDER:", EXPECTED_ORDER)

def list_dataset_dirs_two_levels(parent: Path):
    """
    parent 直下と、その1つ奥のディレクトリのうち、
    'data' ディレクトリを持つものを dataset とみなして列挙
    """
    dataset_dirs = []

    for child in parent.iterdir():
        if not child.is_dir():
            continue

        # 1階層目が dataset の場合
        if (child / "data").is_dir():
            dataset_dirs.append(child)

        # 2階層目も探索
        for gchild in child.iterdir():
            if gchild.is_dir() and (gchild / "data").is_dir():
                dataset_dirs.append(gchild)

    # 重複を避ける
    return sorted(set(dataset_dirs))


def check_parquet_schemas(parent_dir: Path):
    dataset_dirs = list_dataset_dirs_two_levels(parent_dir)

    print("[INFO] 対象のデータセットディレクトリ:")
    for d in dataset_dirs:
        print("  -", d)
    print()

    mismatch_count = 0

    for ds in dataset_dirs:
        data_dir = ds / "data"
        if not data_dir.is_dir():
            print(f"[WARN] {data_dir} not found, skip")
            continue

        chunk_dirs = sorted([p for p in data_dir.glob("chunk-*") if p.is_dir()])
        if not chunk_dirs:
            print(f"[WARN] {data_dir} has no chunk-* directories")
            continue

        for chunk_dir in chunk_dirs:
            parquet_paths = sorted(chunk_dir.glob("*.parquet"))
            if not parquet_paths:
                print(f"[WARN] {chunk_dir} has no .parquet files")
                continue

            for pq_path in parquet_paths:
                try:
                    pf = pq.ParquetFile(pq_path)
                    schema = pf.schema_arrow
                except Exception as e:
                    print(f"=== ERROR 読み込み失敗: {pq_path}")
                    print("    ", repr(e))
                    continue

                cols = [f.name for f in schema]
                types = {f.name: str(f.type) for f in schema}

                # 列セットの差分
                missing = [c for c in EXPECTED_ORDER if c not in cols]
                extra = [c for c in cols if c not in EXPECTED_ORDER]

                # 型の差分
                type_mismatch = []
                for name in EXPECTED_ORDER:
                    if name in types and name in EXPECTED_TYPES:
                        t = types[name]
                        et = EXPECTED_TYPES[name]
                        if t != et:
                            type_mismatch.append((name, t, et))

                # 列順もチェックしたければここも見る
                order_differs = cols != EXPECTED_ORDER

                # 何か1つでも違えば「代表スキーマと違うファイル」として出力
                if missing or extra or type_mismatch or order_differs:
                    mismatch_count += 1
                    print(f"=== MISMATCH: {pq_path}")
                    print("  columns :", cols)
                    if missing:
                        print("  missing :", missing)
                    if extra:
                        print("  extra   :", extra)
                    if type_mismatch:
                        print("  type mismatch:")
                        for n, t, et in type_mismatch:
                            print(f"    - {n}: {t} (expected {et})")
                    if order_differs:
                        print("  NOTE: column order differs from EXPECTED_ORDER")
                    print()

    print(f"[INFO] not match .parquet is {mismatch_count} files")
    
# EXPECTED_ORDER: ['action', 'observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']

def main():

    parent_dir = "/home/data01/smolvla"
    rep_parquet = "/home/masuoka/.cache/huggingface/lerobot/real_data/box_black_plate/data/chunk-000/file-000.parquet"

    parent = Path(parent_dir).expanduser().resolve()
    rep_pq = Path(rep_parquet).expanduser().resolve()
    
    if not parent.is_dir():
        raise SystemExit(f"[ERROR] {parent} is not a directory")
    if not rep_pq.is_file():
        raise SystemExit(f"[ERROR] {rep_pq} is not a file")

    # 代表スキーマをロード
    set_expected_schema(rep_pq)

    # 差分チェック
    check_parquet_schemas(parent)


if __name__ == "__main__":
    main()