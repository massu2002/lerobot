# import pyarrow.parquet as pq
# from pathlib import Path

# root = Path("/home/data01/smolvla/roboticshack/team2-guess_who_less_ligth/data/chunk-000/")
# for path in sorted(root.rglob("*.parquet")):
#     table = pq.read_table(path)
#     print("===", path)
#     print(table.schema)

import pandas as pd
from pathlib import Path

root = Path("/home/data01/smolvla/merged/pretraindata_v1")

# 1) episodes と data を読み込む
episodes_path = root / "meta/episodes/chunk-000/file-000.parquet"
data_path     = root / "data/chunk-000/file-000.parquet"

episodes_df = pd.read_parquet(episodes_path)
data_df     = pd.read_parquet(data_path)

print("=== EPISODES HEAD ===")
print(episodes_df[[
    "episode_index",
    "dataset_from_index", "dataset_to_index",
    "data/chunk_index", "data/file_index",
    "videos/observation.images.front/chunk_index",
    "videos/observation.images.front/file_index",
    "videos/observation.images.wrist/chunk_index",
    "videos/observation.images.wrist/file_index",
]].head(10))

print("\n=== DATA HEAD ===")
print(data_df[[
    "episode_index",
    "index",
]].head(10))

    




    
    