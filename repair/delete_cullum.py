import pyarrow.parquet as pq
from pathlib import Path
import pyarrow as pa

# 代表スキーマ情報
EXPECTED_ORDER: list[str] = []
EXPECTED_TYPES: dict[str, str] = {}
EXPECTED_METADATA: dict[bytes, bytes] | None = None

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

def set_expected_schema(rep_parquet: Path) -> None:
    """
    代表となる .parquet ファイルからスキーマを読み込み、
    EXPECTED_ORDER / EXPECTED_TYPES / EXPECTED_METADATA を更新する
    """
    global EXPECTED_ORDER, EXPECTED_TYPES, EXPECTED_METADATA

    rep_parquet = rep_parquet.expanduser().resolve()
    pf = pq.ParquetFile(rep_parquet)
    schema = pf.schema_arrow  # pa.Schema

    EXPECTED_ORDER = [f.name for f in schema]
    EXPECTED_TYPES = {f.name: str(f.type) for f in schema}
    EXPECTED_METADATA = schema.metadata  # HFのhuggingface/pandasメタデータも含まれる

    print("[INFO] representative schema loaded from:", rep_parquet)
    print("       EXPECTED_ORDER:", EXPECTED_ORDER)

def fix_parquet_to_expected(path: Path) -> bool:
    """
    1つの .parquet ファイルを「代表スキーマ」に合わせて修正する。

    - EXPECTED_ORDER に含まれない列は drop
    - 列順は EXPECTED_ORDER に並べ替え
    - スキーマ metadata も EXPECTED_METADATA に統一

    何か変更した場合 True, 何も変えていない場合 False を返す
    """
    if not EXPECTED_ORDER:
        raise RuntimeError("EXPECTED_ORDER is empty. Call set_expected_schema() first.")

    pf = pq.ParquetFile(path)
    table = pf.read()
    schema = table.schema

    current_cols = schema.names
    current_set = set(current_cols)
    expected_set = set(EXPECTED_ORDER)

    # 代表にない列（余計な列）
    extra = sorted(list(current_set - expected_set))
    # 必須だが欠けている列（あれば警告）
    missing = sorted(list(expected_set - current_set))

    if not extra and not missing and current_cols == EXPECTED_ORDER:
        # 完全一致なら何もしない
        return False

    print(f"[FIX] {path}")
    if extra:
        print("  - drop extra columns:", extra)
    if missing:
        print("  - WARNING: missing expected columns:", missing)

    # 代表にある列だけを、期待された順番で並べる
    new_arrays = []
    new_fields = []
    for name in EXPECTED_ORDER:
        if name not in current_set:
            # 欠けている列がある場合は、とりあえずスキップ（必要なら後で追加ロジックを書く）
            print(f"    * skip missing column '{name}' (not present in {path.name})")
            continue
        col = table[name]
        new_arrays.append(col)
        new_fields.append(pa.field(name, col.type))

    # 新しい schema を作る（metadata は代表のものを使う）
    new_schema = pa.schema(new_fields, metadata=EXPECTED_METADATA or schema.metadata)
    new_table = pa.Table.from_arrays(new_arrays, schema=new_schema)

    pq.write_table(new_table, path)
    return True

def fix_all_parquet_to_expected(parent_dir: Path):
    dataset_dirs = list_dataset_dirs_two_levels(parent_dir)

    print("[INFO] 対象のデータセットディレクトリ:")
    for d in dataset_dirs:
        print("  -", d)
    print()

    total_fixed = 0
    total_files = 0

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
                total_files += 1
                try:
                    changed = fix_parquet_to_expected(pq_path)
                    if changed:
                        total_fixed += 1
                except Exception as e:
                    print(f"=== ERROR fixing: {pq_path}")
                    print("    ", repr(e))

    print(f"[INFO] scanned {total_files} parquet files")
    print(f"[INFO] fixed  {total_fixed} files to match representative schema")

if __name__ == "__main__":

    parent_dir = "/home/data01/smolvla"
    rep_parquet = "/home/masuoka/.cache/huggingface/lerobot/real_data/box_black_plate/data/chunk-000/file-000.parquet"

    parent = Path(parent_dir).expanduser().resolve()
    rep_pq = Path(rep_parquet).expanduser().resolve()

    if not parent.is_dir():
        raise SystemExit(f"[ERROR] {parent} is not a directory")
    if not rep_pq.is_file():
        raise SystemExit(f"[ERROR] {rep_pq} is not a file")

    # 1) 代表スキーマ読み込み
    set_expected_schema(rep_pq)

    # 2) 全 parquet を代表スキーマに揃える
    fix_all_parquet_to_expected(parent)
