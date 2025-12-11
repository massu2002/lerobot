from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa


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


def list_to_fixed_size(list_arr: pa.Array, expected_list_size: int | None = None) -> pa.FixedSizeListArray:
    """
    可変長 list<...>（ChunkedArray を含む）を fixed_size_list<...>[N] に変換する。
    expected_list_size が指定されていれば、その長さ以外があれば例外を投げる。
    """
    # 1) ChunkedArray の場合は 1 本にまとめる
    if isinstance(list_arr, pa.ChunkedArray):
        list_arr = list_arr.combine_chunks()

    # 2) list 型かチェック
    if not pa.types.is_list(list_arr.type):
        raise TypeError(f"Expected ListArray, got {list_arr.type}")

    # 3) offsets から各行の長さを調べる
    offsets = list_arr.offsets.to_pylist()
    lengths = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
    unique_lengths = set(lengths)

    if len(unique_lengths) != 1:
        raise ValueError(f"List lengths are not uniform: {unique_lengths}")

    list_size = unique_lengths.pop()

    if expected_list_size is not None and list_size != expected_list_size:
        raise ValueError(f"List size {list_size} != expected {expected_list_size}")

    values = list_arr.values  # 連結された中身

    fs_arr = pa.FixedSizeListArray.from_arrays(values, list_size=list_size)
    return fs_arr


def fix_parquet_file(path: Path, expected_list_size: int = 6) -> bool:
    pf = pq.ParquetFile(path)
    table = pf.read()
    schema = table.schema

    fixed = False
    new_columns = []
    new_fields = []

    for field, col in zip(schema, table.columns):
        name = field.name

        if name in ["action", "observation.state"] and pa.types.is_list(col.type):
            print(f"  [FIX] {path.name}: convert {name} ({col.type}) -> fixed_size_list[{expected_list_size}]")
            fs_arr = list_to_fixed_size(col, expected_list_size=expected_list_size)
            new_columns.append(fs_arr)
            new_fields.append(pa.field(name, fs_arr.type, nullable=field.nullable))
            fixed = True
        else:
            new_columns.append(col)
            new_fields.append(field)

    if not fixed:
        return False

    new_schema = pa.schema(new_fields, metadata=schema.metadata)
    new_table = pa.Table.from_arrays(new_columns, schema=new_schema)
    pq.write_table(new_table, path)
    return True


def fix_all_under_parent(parent_dir: Path, expected_list_size: int = 6):
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
                    changed = fix_parquet_file(pq_path, expected_list_size=expected_list_size)
                    if changed:
                        total_fixed += 1
                except Exception as e:
                    print(f"=== ERROR fixing: {pq_path}")
                    print("    ", repr(e))

    print(f"[INFO] scanned {total_files} parquet files")
    print(f"[INFO] fixed  {total_fixed} files (list -> fixed_size_list[{expected_list_size}])")
    
def main():

    parent_dir = "/home/data01/smolvla"
    rep_parquet = "/home/masuoka/.cache/huggingface/lerobot/real_data/box_black_plate/data/chunk-000/file-000.parquet"
    size=6

    parent = Path(parent_dir).expanduser().resolve()
    rep_pq = Path(rep_parquet).expanduser().resolve()
    
    if not parent.is_dir():
        raise SystemExit(f"[ERROR] {parent} is not a directory")
    if not rep_pq.is_file():
        raise SystemExit(f"[ERROR] {rep_pq} is not a file")

    fix_all_under_parent(parent, expected_list_size=size)


if __name__ == "__main__":
    main()