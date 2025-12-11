#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import shutil
from pathlib import Path

import pandas as pd
import tqdm
from typing import Dict, Any, Optional, Set

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    get_file_size_in_mb,
    get_parquet_file_size_in_mb,
    to_parquet_with_hf_images,
    update_chunk_file_indices,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s
import warnings
import copy

TARGET_REPO = "katoken/box_black_plate"

CAMERA_COMPAT_PAIRS = {
    "observation.images.wrist": [
        "observation.images.wrist",
        "observation.images.phone",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
        "observation.images.left",
        "observation.images.realsense_side",
        "observation.images.side",
        "observation.images.hand",
        "observation.images.gripper",
        "observation.images.realsensergb",
        "observation.images.logitech",
        "observation.images.wrist_right",
        "observation.images.Lwebcam",
        "observation.images.right",
        "observation.images.webcam",
        "observation.images.right_follower",
        "observation.images.body",
        "observation.images.cam_right",
        "observation.images.left_follower",
        "observation.images.Right_follower",
        "observation.images.cam_left",
        "observation.images.robor",
        "observation.images.left_arm_cam",
        "observation.images.laptop2",
        "observation.images.right_arm_cam",
        "observation.images.webcam_2",
        "observation.images.third",
        "observation.images.macwebcam",
        "observation.images.static_right",
        "observation.images.secondary_cam",
        "observation.images.right_follower",
        "observation.images.Right_follower",
        "observation.images.Logic_camera",
        "wrist",
        "phone",
        "wrist_left",
        "wrist_right",
        "left",
        "realsense_side",
        "side",
        "hand",
        "gripper",
        "realsensergb",
        "logitech",
        "wrist_right",
        "Lwebcam",
        "right",
        "webcam",
        "right_follower",
        "body",
        "cam_right",
        "left_follower",
        "Right_follower",
        "cam_left",
        "robor",
        "left_arm_cam",
        "laptop2",
        "right_arm_cam",
        "webcam_2",
        "third",
        "macwebcam",
        "static_right",
        "secondary_cam",
        "right_follower",
        "Right_follower",
        "Logic_camera",
    ],
    "observation.images.front": [
        "observation.images.front",
        "observation.images.laptop",
        "observation.images.top",
        "observation.images.back",
        "observation.images.up",
        "observation.images.realsense_top",
        "observation.images.overhead",
        "observation.images.realsense",
        "observation.images.s_left",
        "observation.images.back",
        "observation.images.phone2",
        "observation.images.main",
        "observation.images.center",
        "observation.images.overhead",
        "observation.images.robot",
        "observation.images.cam_middle",
        "observation.images.center",
        "observation.images.Logic_camera",
        "observation.images.base_cam",
        "observation.images.laptop1",
        "observation.images.top_cam",
        "observation.images.webcam_1",
        "observation.images.static_left",
        "observation.images.main_cam",
        "observation.images.left_follower",
        "observation.images.Left_follower",
        "observation.images.overview2",
        "observation.images.s_right",
        "front",
        "laptop",
        "top",
        "back",
        "up",
        "realsense_top",
        "overhead",
        "realsense",
        "s_left",
        "back",
        "phone2",
        "main",
        "center",
        "overhead",
        "robot",
        "cam_middle",
        "center",
        "Logic_camera",
        "base_cam",
        "laptop1",
        "top_cam",
        "webcam_1",
        "static_left",
        "main_cam",
        "left_follower",
        "Left_follower",
        "overview2",
        "s_right",
    ],
}

def _iter_camera_aliases(logical_key: str) -> list[str]:
    """
    logical_key に対して、互換カメラキー一覧を返す。
    - logical_key が CAMERA_COMPAT_PAIRS のキーなら、その alias 群を返す
    - logical_key が alias 側なら、その基準キーグループを返す
    - どちらでもなければ logical_key 単独で返す
    """
    # logical_key が基準キーだった場合
    if logical_key in CAMERA_COMPAT_PAIRS:
        return [logical_key] + CAMERA_COMPAT_PAIRS[logical_key]

    # logical_key が alias 側の場合：逆引き
    for base_key, aliases in CAMERA_COMPAT_PAIRS.items():
        if logical_key in aliases:
            return [base_key] + aliases

    # グループに属さない → 自分自身だけ
    return [logical_key]

def _find_compatible_video_key_in_src(
    src_meta,
    logical_key: str,
) -> list[str]:
    """
    src_meta 内で logical_key と互換な video キーを探す。
    - CAMERA_COMPAT_PAIRS に基づき alias を列挙
    - episodes のカラム名を元に存在チェック
    """

    # episodes 側の全カラム名（Dataset なので column_names を使う）
    episode_cols = set(src_meta.episodes.column_names)
    
    # stats 側に登録されている video キー一覧
    video_stats_keys = set(src_meta.stats.get("videos", {}).keys())

    def _names_to_try(candidate: str) -> list[str]:
        """
        1つの candidate から、実際に episodes に存在しそうな名前の候補を列挙する。
        例:
          "observation.images.wrist" -> ["observation.images.wrist", "wrist"]
          "wrist"                    -> ["wrist", "observation.images.wrist"]
        """
        names = [candidate]
        prefix = "observation.images."

        if candidate.startswith(prefix):
            bare = candidate[len(prefix):]  # "wrist"
            names.append(bare)
        else:
            names.append(prefix + candidate)

        # 重複排除
        seen = set()
        uniq = []
        for n in names:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        return uniq
    
    found: list[str] = []

    # logical_key から alias 群を取得して順に試す
    for candidate in _iter_camera_aliases(logical_key):
        for name in _names_to_try(candidate):
            chunk_field = f"videos/{name}/chunk_index"
            file_field  = f"videos/{name}/file_index"
            
            has_stats    = name in video_stats_keys
            has_episodes = (chunk_field in episode_cols) and (file_field in episode_cols)

            if has_stats or has_episodes:
                found.append(name)

    # 重複除去しつつ順序維持
    uniq: list[str] = []
    seen = set()
    for n in found:
        if n not in seen:
            seen.add(n)
            uniq.append(n)

    # if not uniq:
        # print("video_stats_keys:", sorted(video_stats_keys))
        # print("episode_cols:", sorted(episode_cols))
        # print(
        #     f"[WARN] no compatible video key found in src for '{logical_key}' "
        #     f"(checked aliases: {_iter_camera_aliases(logical_key)})"
        # )
    return uniq

def validate_all_metadata(
    all_metadata: list["LeRobotDatasetMetadata"],
    baseline_metadata: "LeRobotDatasetMetadata",
):
    """
    baseline_metadata を基準に、fps / robot_type / features の互換性をチェックし、
    互換 OK な metadata だけ valid_metadata に入れて返す。

    - fps: baseline と違えば SKIP
    - robot_type: 違っていても WARN だけ（baseline の robot_type で解釈される前提）
    - features: 上の互換ルールで判定し、互換 NG なら SKIP
    """

    fps_baseline        = baseline_metadata.fps
    robot_type_baseline = baseline_metadata.robot_type
    features_baseline   = baseline_metadata.features

    valid_metadata: list["LeRobotDatasetMetadata"] = []
    skip_count = 0

    for meta in tqdm.tqdm(all_metadata, desc="Validate all metadata"):
        skip = False

        # ---- fps mismatch → 無条件で SKIP ----
        if meta.fps != fps_baseline:
            skip = True
            skip_count += 1

        if not skip:
            valid_metadata.append(meta)
            
    print(f"[INFO] {skip_count} metadata skipped due to incompatibility.")

    # マージに使う fps / robot_type / features は baseline の値を返す
    return fps_baseline, robot_type_baseline, features_baseline, valid_metadata


def update_data_df(df, src_meta, dst_meta):
    """Updates a data DataFrame with new indices and task mappings for aggregation.

    Adjusts episode indices, frame indices, and task indices to account for
    previously aggregated data in the destination dataset.

    Args:
        df: DataFrame containing the data to be updated.
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.

    Returns:
        pd.DataFrame: Updated DataFrame with adjusted indices.
    """

    df["episode_index"] = df["episode_index"] + dst_meta.info["total_episodes"]
    df["index"] = df["index"] + dst_meta.info["total_frames"]

    src_task_names = src_meta.tasks.index.take(df["task_index"].to_numpy())
    df["task_index"] = dst_meta.tasks.loc[src_task_names, "task_index"].to_numpy()

    return df

def update_meta_data(
    df: pd.DataFrame,
    dst_meta,
    meta_idx: dict,
    data_idx: dict,
    videos_idx: dict,
):
    """
    episodes メタを「集約後のインデックス」に書き換える。

    - meta/episodes/chunk_index,file_index は単純に上書き（オフセット加算はしない）
    - data/chunk_index,file_index は data_idx["src_to_mapping"] で 1:1 マップ
    - videos/.../chunk_index,file_index も videos_idx[*]["src_to_mapping"] で 1:1 マップ
      - NaN 行はスキップ
      - mapping に無い src_key は「動画なし」とみなして NaN にする
    - dataset_*_index, episode_index は dst_meta.info のトータル分だけオフセット
    """

    # 1) episodes 自身の meta/episodes/chunk_index, file_index を上書き
    df["meta/episodes/chunk_index"] = meta_idx["chunk"]
    df["meta/episodes/file_index"]  = meta_idx["file"]

    # 2) data 側の (chunk,file) を src_to_mapping で更新
    data_chunk_col = "data/chunk_index"
    data_file_col  = "data/file_index"

    data_mapping = data_idx.get("src_to_mapping", None)
    if (
        data_mapping is not None
        and data_chunk_col in df.columns
        and data_file_col in df.columns
    ):
        orig_chunk = df[data_chunk_col].to_numpy()
        orig_file  = df[data_file_col].to_numpy()
        new_chunk  = orig_chunk.copy()
        new_file   = orig_file.copy()

        for i in range(len(df)):
            ck = orig_chunk[i]
            fi = orig_file[i]

            # NaN → data 自体が無い行なので、そのまま残す
            if pd.isna(ck) or pd.isna(fi):
                continue

            src_key = (int(ck), int(fi))
            info = data_mapping.get(src_key)

            if info is None:
                # この data ファイルは集約時にコピーしていない
                # → 「データ無し」とみなして NaN に落とす
                new_chunk[i] = np.nan
                new_file[i]  = np.nan
            else:
                new_chunk[i] = info["chunk"]
                new_file[i]  = info["file"]

        df[data_chunk_col] = new_chunk
        df[data_file_col]  = new_file
    # （もし data_mapping が無ければ、そのままにしておく）

    # 3) 動画メタ（videos/...）も同様に src_to_mapping で更新
    for key, video_idx in videos_idx.items():
        chunk_col = f"videos/{key}/chunk_index"
        file_col  = f"videos/{key}/file_index"

        if chunk_col not in df.columns or file_col not in df.columns:
            continue

        mapping = video_idx.get("src_to_mapping", None)
        if mapping is None:
            # このカメラは一切コピーしていない
            continue

        orig_chunk = df[chunk_col].to_numpy()
        orig_file  = df[file_col].to_numpy()
        new_chunk  = orig_chunk.copy()
        new_file   = orig_file.copy()

        # timestamp カラム（存在すれば）も握っておく
        from_col = f"videos/{key}/from_timestamp"
        to_col   = f"videos/{key}/to_timestamp"
        has_ts = from_col in df.columns and to_col in df.columns

        for i in range(len(df)):
            ck = orig_chunk[i]
            fi = orig_file[i]

            # NaN → そもそもこのカメラの動画が無い行
            if pd.isna(ck) or pd.isna(fi):
                continue

            src_key = (int(ck), int(fi))
            info = mapping.get(src_key)

            if info is None:
                # この src (chunk,file) の動画はコピーしていない
                # → 「この行ではこのカメラの動画は無し」とみなして NaN にする
                new_chunk[i] = np.nan
                new_file[i]  = np.nan
                if has_ts:
                    df.at[i, from_col] = np.nan
                    df.at[i, to_col]   = np.nan
            else:
                new_chunk[i] = info["chunk"]
                new_file[i]  = info["file"]

        df[chunk_col] = new_chunk
        df[file_col]  = new_file
        # timestamp は 1 src = 1 dst かつ fps 不変ならそのままで OK

    # 4) データセット全体での index / episode_index をずらす
    offset_frames   = dst_meta.info["total_frames"]
    offset_episodes = dst_meta.info["total_episodes"]

    if "dataset_from_index" in df.columns:
        df["dataset_from_index"] += offset_frames
    if "dataset_to_index" in df.columns:
        df["dataset_to_index"]   += offset_frames
    if "episode_index" in df.columns:
        df["episode_index"]      += offset_episodes

    return df

from typing import Dict, Set, Tuple
import numpy as np
from pathlib import Path
import math
from moviepy.video.io.VideoFileClip import VideoFileClip


def validate_video_timestamps(
    episodes_df,
    dst_root: Path,
    logical_key: str,
    *,
    eps: float = 1e-2,          # 許容誤差 0.01 秒（絶対値）
    max_rel_err: float = 0.03,  # 許容相対誤差（3%）
):
    """
    episodes_df 内の timestamps が、
    実 mp4 ファイルの長さと整合しているか検証し、
    軽微なズレはクリップ、大きなズレは「動画なし」として NaN 化する。

    - 対象カメラの chunk/file/from/to が NaN の行は「そもそも動画なし」とみなしスキップ
    - from <= to（これを満たさない場合も補正 or NaN 化）
    - to <= video_duration + eps
    - from/to >= 0
    """

    import math
    import numpy as np

    chunk_col = f"videos/{logical_key}/chunk_index"
    file_col  = f"videos/{logical_key}/file_index"
    from_col  = f"videos/{logical_key}/from_timestamp"
    to_col    = f"videos/{logical_key}/to_timestamp"

    for col in (chunk_col, file_col, from_col, to_col):
        if col not in episodes_df.columns:
            print(f"[timestamp-check] {logical_key}: column {col} not found, skip")
            return

    # NaN 行を除外（そのエピソードにはこのカメラの動画が無い）
    mask = (
        episodes_df[chunk_col].notna()
        & episodes_df[file_col].notna()
        & episodes_df[from_col].notna()
        & episodes_df[to_col].notna()
    )

    sub = episodes_df.loc[mask]  # ★ reset_index しない

    if len(sub) == 0:
        print(f"[timestamp-check] {logical_key}: no valid rows (all NaN), skip")
        return

    video_length_cache: dict[Path, float] = {}

    print(f"\n=== TIMESTAMP VALIDATION for {logical_key} (rows={len(sub)}) ===")

    n_clipped = 0
    n_dropped = 0

    for idx, row in sub.iterrows():
        ck_raw = row[chunk_col]
        fi_raw = row[file_col]
        ft_raw = row[from_col]
        tt_raw = row[to_col]

        # 念のためこの時点でも NaN をチェック
        if any(math.isnan(float(x)) for x in (ck_raw, fi_raw, ft_raw, tt_raw)):
            # マッピング的には「動画なし」扱いで良いのでスキップ
            continue

        ck = int(ck_raw)
        fi = int(fi_raw)
        ft = float(ft_raw)
        tt = float(tt_raw)

        video_path = dst_root / f"videos/{logical_key}/chunk-{ck:03d}/file-{fi:03d}.mp4"

        if not video_path.exists():
            raise FileNotFoundError(f"[timestamp-check] Missing file: {video_path}")

        if video_path not in video_length_cache:
            clip = VideoFileClip(str(video_path))
            video_length_cache[video_path] = clip.duration
            clip.close()

        dur = video_length_cache[video_path]

        # --- チェック1: from <= to ---
        if ft > tt + 1e-9:
            # ここも「直す or ドロップ」の2択にする
            print(
                f"[timestamp-check][WARN] {logical_key} idx={idx}: "
                f"from={ft} > to={tt}, drop this camera for this row"
            )
            # この行のこのカメラを NaN にする
            episodes_df.at[idx, chunk_col] = np.nan
            episodes_df.at[idx, file_col]  = np.nan
            episodes_df.at[idx, from_col]  = np.nan
            episodes_df.at[idx, to_col]    = np.nan
            n_dropped += 1
            continue

        # --- チェック2: 負の値 ---
        if ft < -eps or tt < -eps:
            print(
                f"[timestamp-check][WARN] {logical_key} idx={idx}: "
                f"negative timestamp from={ft}, to={tt}, drop this camera for this row"
            )
            episodes_df.at[idx, chunk_col] = np.nan
            episodes_df.at[idx, file_col]  = np.nan
            episodes_df.at[idx, from_col]  = np.nan
            episodes_df.at[idx, to_col]    = np.nan
            n_dropped += 1
            continue

        # --- チェック3: to > dur ---
        if tt > dur + eps:
            delta = tt - dur
            rel_err = delta / max(dur, 1e-6)

            if rel_err <= max_rel_err:
                # 軽微なズレ → クリップして許容
                print(
                    f"[timestamp-check][CLIP] {logical_key} idx={idx}: "
                    f"to={tt} > dur={dur} (delta={delta:.3f}, rel={rel_err:.3%}), "
                    f"clamp to {dur}"
                )
                episodes_df.at[idx, to_col] = dur
                n_clipped += 1
            else:
                # 明らかにおかしい → この行のこのカメラを無かったことにする
                print(
                    f"[timestamp-check][DROP] {logical_key} idx={idx}: "
                    f"to={tt} >> dur={dur} (delta={delta:.3f}, rel={rel_err:.3%}), "
                    f"drop this camera for this row"
                )
                episodes_df.at[idx, chunk_col] = np.nan
                episodes_df.at[idx, file_col]  = np.nan
                episodes_df.at[idx, from_col]  = np.nan
                episodes_df.at[idx, to_col]    = np.nan
                n_dropped += 1

    print(
        f"[OK-ish] timestamps checked for key={logical_key} "
        f"(clipped={n_clipped}, dropped_rows={n_dropped})"
    )

def validate_video_mapping(
    src_meta,
    videos_idx: Dict[str, dict],
    logical_to_src_key: Dict[str, str],
) -> None:
    """
    aggregate_videos_aligned の結果が src_meta と整合しているか検証する。

    - src_meta.episodes の videos/{src_key}/chunk_index, file_index から
      一意な (chunk,file) を列挙
    - videos_idx[logical_key]["src_to_mapping"] のキー集合と一致しているかチェック
    """

    for logical_key, src_key in logical_to_src_key.items():
        # 1) src 側の (chunk,file) を episodes から列挙
        chunk_col = f"videos/{src_key}/chunk_index"
        file_col  = f"videos/{src_key}/file_index"

        if chunk_col not in src_meta.episodes.column_names or \
           file_col not in src_meta.episodes.column_names:
            raise AssertionError(
                f"[validate_video_mapping] src episodes missing columns "
                f"{chunk_col} / {file_col}"
            )

        src_chunks = np.array(src_meta.episodes[chunk_col])
        src_files  = np.array(src_meta.episodes[file_col])

        src_pairs: Set[Tuple[int, int]] = set(
            (int(c), int(f)) for c, f in zip(src_chunks, src_files, strict=False)
        )

        # 2) mapping 側のキー集合
        mapping = videos_idx[logical_key].get("src_to_mapping", {})
        map_pairs: Set[Tuple[int, int]] = set(mapping.keys())

        if src_pairs != map_pairs:
            # 差分を見やすく出す
            missing = src_pairs - map_pairs
            extra   = map_pairs - src_pairs
            raise AssertionError(
                f"[validate_video_mapping] mismatch for logical_key={logical_key}\n"
                f"  missing in mapping: {sorted(missing)}\n"
                f"  extra in mapping  : {sorted(extra)}"
            )


def validate_data_mapping(
    src_meta,
    data_idx: dict,
) -> None:
    """
    aggregate_data の結果が src_meta と整合しているか検証する。

    - src_meta.episodes の data/chunk_index, data/file_index から
      一意な (chunk,file) を列挙
    - data_idx['src_to_mapping'] のキー集合と一致しているかチェック
    """

    if "src_to_mapping" not in data_idx:
        raise AssertionError("[validate_data_mapping] data_idx has no 'src_to_mapping'")

    chunk_col = "data/chunk_index"
    file_col  = "data/file_index"

    if chunk_col not in src_meta.episodes.column_names or \
       file_col not in src_meta.episodes.column_names:
        raise AssertionError(
            f"[validate_data_mapping] src episodes missing columns "
            f"{chunk_col} / {file_col}"
        )

    src_chunks = np.array(src_meta.episodes[chunk_col])
    src_files  = np.array(src_meta.episodes[file_col])

    src_pairs: Set[Tuple[int, int]] = set(
        (int(c), int(f)) for c, f in zip(src_chunks, src_files, strict=False)
    )

    mapping = data_idx["src_to_mapping"]
    map_pairs: Set[Tuple[int, int]] = set(mapping.keys())

    if src_pairs != map_pairs:
        missing = src_pairs - map_pairs
        extra   = map_pairs - src_pairs
        raise AssertionError(
            f"[validate_data_mapping] mismatch\n"
            f"  missing in mapping: {sorted(missing)}\n"
            f"  extra in mapping  : {sorted(extra)}"
        )

def aggregate_datasets(
    repo_ids: list[str],
    aggr_repo_id: str,
    roots: list[Path] | None = None,
    aggr_root: Path | None = None,
    data_files_size_in_mb: float | None = None,
    video_files_size_in_mb: float | None = None,
    chunk_size: int | None = None,
):
    """Aggregates multiple LeRobot datasets into a single unified dataset.

    This is the main function that orchestrates the aggregation process by:
    1. Loading and validating all source dataset metadata
    2. Creating a new destination dataset with unified tasks
    3. Aggregating videos, data, and metadata from all source datasets
    4. Finalizing the aggregated dataset with proper statistics

    Args:
        repo_ids: List of repository IDs for the datasets to aggregate.
        aggr_repo_id: Repository ID for the aggregated output dataset.
        roots: Optional list of root paths for the source datasets.
        aggr_root: Optional root path for the aggregated dataset.
        data_files_size_in_mb: Maximum size for data files in MB (defaults to DEFAULT_DATA_FILE_SIZE_IN_MB)
        video_files_size_in_mb: Maximum size for video files in MB (defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB)
        chunk_size: Maximum number of files per chunk (defaults to DEFAULT_CHUNK_SIZE)
    """
    logging.info("Start aggregate_datasets")

    if data_files_size_in_mb is None:
        data_files_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_files_size_in_mb is None:
        video_files_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE
        
    baseline_metadata = None
    all_metadata = []

    if roots is None:
        # repo_ids だけで metadata を作るパターン
        for rid in repo_ids:
            meta = LeRobotDatasetMetadata(rid)

            if rid == TARGET_REPO:
                baseline_metadata = meta   # ★ これが基準 metadata
            all_metadata.append(meta)

    else:
        # repo_ids + roots 両方指定されるパターン
        for rid, root in zip(repo_ids, roots, strict=False):
            meta = LeRobotDatasetMetadata(rid, root=root)

            if rid == TARGET_REPO:
                baseline_metadata = meta   # ★ これが基準 metadata
            all_metadata.append(meta)
    
    fps, robot_type, features, valid_metadata = validate_all_metadata(all_metadata, baseline_metadata)
    metadata_to_use = valid_metadata
    video_keys = [key for key in features if features[key]["dtype"] == "video"]

    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=aggr_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=aggr_root,
        use_videos=len(video_keys) > 0,
        chunks_size=chunk_size,
        data_files_size_in_mb=data_files_size_in_mb,
        video_files_size_in_mb=video_files_size_in_mb,
    )

    logging.info("Find all tasks")
    unique_tasks = pd.concat([m.tasks for m in all_metadata]).index.unique()
    dst_meta.tasks = pd.DataFrame({"task_index": range(len(unique_tasks))}, index=unique_tasks)

    meta_idx = {"chunk": 0, "file": 0}
    data_idx = {"chunk": 0, "file": 0}
    videos_idx = {
        key: {"chunk": 0, "file": 0, "latest_duration": 0, "episode_duration": 0} for key in video_keys
    }

    dst_meta.episodes = {}
    
    # コピー対象も valid なものだけ
    for src_meta in tqdm.tqdm(metadata_to_use, desc="Copy data and videos"):
        
        episode_has_video = np.zeros(src_meta.total_episodes, dtype=bool)
        
        videos_idx, logical_to_src_key = aggregate_videos_aligned(
            src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size
        )
        
        # この src で1つもカメラが使われなかった場合は、data/meta も丸ごとスキップ
        if not logical_to_src_key:
            print("[SKIP] no compatible videos in this src → skip data & metadata")
            continue

        data_idx = aggregate_data(
            src_meta,
            dst_meta,
            data_idx,
            data_files_size_in_mb,
            chunk_size,
        )
        
        # 4) data マッピングが src_meta と整合しているかチェック
        validate_data_mapping(src_meta, data_idx)

        meta_idx = aggregate_metadata(
            src_meta,
            dst_meta,
            meta_idx,
            data_idx,
            videos_idx,
            logical_to_src_key,   # ★追加
        )

        dst_meta.info["total_episodes"] += src_meta.total_episodes
        dst_meta.info["total_frames"]   += src_meta.total_frames
        
        episodes_df = pd.read_parquet(
            dst_meta.root / "meta/episodes/chunk-000/file-000.parquet"
        )

        validate_video_timestamps(
            episodes_df=episodes_df,
            dst_root=dst_meta.root,
            logical_key="observation.images.front"
        )

        validate_video_timestamps(
            episodes_df=episodes_df,
            dst_root=dst_meta.root,
            logical_key="observation.images.wrist"
        )


    # ⑤ finalize_aggregation も valid なメタデータ一覧を渡す
    finalize_aggregation(dst_meta, metadata_to_use)
    logging.info("Aggregation complete.")

def aggregate_videos_aligned(
    src_meta,
    dst_meta,
    videos_idx,
    video_files_size_in_mb,  # 使わなくなるが引数としては残しておく
    chunk_size,
    keys=None,
):
    """
    全ての keys をロックステップで処理し、出力の (chunk_index, file_index) を揃える。
    ※ ファイル連結は行わず、「1 src ファイル = 1 dst ファイル」とすることで
       DTS エラーを回避する。
    """

    # 対象キーを決定
    if keys is None:
        keys = list(videos_idx.keys())

    # 1) logical_key → src_video_key の対応表を作る（互換キー探索）
    logical_to_src_key: dict[str, str] = {}
    
    front_lk = "observation.images.front"
    wrist_lk = "observation.images.wrist"
    
    # それぞれの logical_key について「マッチした候補全部」を取得
    front_candidates = _find_compatible_video_key_in_src(src_meta, front_lk)
    wrist_candidates = _find_compatible_video_key_in_src(src_meta, wrist_lk)

    # 1) front も wrist も普通にマッチした → 素直に最初の候補を使う
    if front_candidates and wrist_candidates:
        logical_to_src_key[front_lk] = front_candidates[0]
        logical_to_src_key[wrist_lk] = wrist_candidates[0]

    # 2) front だけ複数マッチ・wrist は0件
    elif front_candidates and not wrist_candidates:
        # 先頭の候補を front に
        logical_to_src_key[front_lk] = front_candidates[0]
        # 2つ以上あれば、2つ目を wrist に割り当てる
        if len(front_candidates) >= 2:
            logical_to_src_key[wrist_lk] = front_candidates[1]
        # それでも 1つしかない場合 → wrist には何も割り当てない(スキップ扱い)

    # 3) wrist だけ複数マッチ・front は0件（あまり無さそうだけど対称性のため）
    elif wrist_candidates and not front_candidates:
        logical_to_src_key[wrist_lk] = wrist_candidates[0]
        if len(wrist_candidates) >= 2:
            logical_to_src_key[front_lk] = wrist_candidates[1]

    # 4) どちらも 0件 → 何もせず（active_keys にも入らない）

    active_keys: Set[str] = set(logical_to_src_key.keys())
    
    # 互換キーゼロなら何もしない
    if not active_keys:
        return videos_idx, set()

    # 実際に処理するキーを互換キーありのものだけに絞る
    keys = list(active_keys)

    # 2) 初期化: 各キーでオフセットマップとエピソード長をリセット
    for k in keys:
        videos_idx[k]["episode_duration"] = 0.0
        videos_idx[k]["src_to_mapping"] = {} 

    # 3) 共有の出力インデックス（全キーで同じ値を使う）
    # 先頭キーから開始値を採用
    shared_chunk_idx = videos_idx[keys[0]]["chunk"]
    shared_file_idx  = videos_idx[keys[0]]["file"]

    # 4) 各キーの (src_chunk, src_file) ペアを収集（※ src 側の実キー名で）
    pairs_by_key = {}
    for logical_key in keys:
        src_key = logical_to_src_key[logical_key]
        pairs = {
            (c, f)
            for c, f in zip(
                src_meta.episodes[f"videos/{src_key}/chunk_index"],
                src_meta.episodes[f"videos/{src_key}/file_index"],
                strict=False,
            )
        }
        pairs_by_key[logical_key] = sorted(pairs)

    # 全キーのユニオンを時間順（chunk, file のタプルの昇順）で走査
    all_pairs = sorted(set().union(*pairs_by_key.values()))

    for src_chunk_idx, src_file_idx in all_pairs:
        # この (src_chunk_idx, src_file_idx) を持っている論理キーのみ処理対象
        active_keys = [
            logical_key
            for logical_key in keys
            if (src_chunk_idx, src_file_idx) in pairs_by_key[logical_key]
        ]
        if not active_keys:
            continue

        # 今回の (src_chunk, src_file) に割り当てる dst 側の index
        cur_chunk_idx = shared_chunk_idx
        cur_file_idx  = shared_file_idx

        for logical_key in active_keys:
            src_key = logical_to_src_key[logical_key]

            src_path = src_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=src_key,
                chunk_index=src_chunk_idx,
                file_index=src_file_idx,
            )
            dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=logical_key,
                chunk_index=cur_chunk_idx,
                file_index=cur_file_idx,
            )

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # 「1 src = 1 dst」として単純コピー（連結しない）
            shutil.copy(str(src_path), str(dst_path))

            # この dst ファイル内での開始オフセット（常に 0 でOK）
            videos_idx[logical_key]["src_to_mapping"][(src_chunk_idx, src_file_idx)] = {
                "chunk": cur_chunk_idx,
                "file": cur_file_idx,
                "offset": 0.0,  # 1 src = 1 dst なので 0 でよい
            }

            # 動画時間を episode_duration に積算
            dur = get_video_duration_in_s(src_path)
            videos_idx[logical_key]["episode_duration"] += dur

        # 次のファイル用に shared_* を更新
        shared_chunk_idx, shared_file_idx = update_chunk_file_indices(
            shared_chunk_idx, shared_file_idx, chunk_size
        )

    # 8) 処理後、全キーの出力インデックスを最後の値で揃える
    for logical_key in keys:
        videos_idx[logical_key]["chunk"] = shared_chunk_idx
        videos_idx[logical_key]["file"]  = shared_file_idx

    return videos_idx, logical_to_src_key

def aggregate_data(src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size):
    """Aggregates data chunks from a source dataset into the destination dataset.
    """

    # --- 初期化: src→dst マッピング用の dict を data_idx に持たせる ---
    if "src_to_mapping" not in data_idx:
        data_idx["src_to_mapping"] = {}

    seen: set[tuple[int, int]] = set()
    unique_chunk_file_ids: list[tuple[int, int]] = []

    for c, f in zip(
        src_meta.episodes["data/chunk_index"],
        src_meta.episodes["data/file_index"],
        strict=False,
    ):
        key = (int(c), int(f))
        if key in seen:
            continue
        seen.add(key)
        unique_chunk_file_ids.append(key)

    for src_chunk_idx, src_file_idx in unique_chunk_file_ids:
        src_path = src_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=src_chunk_idx,
            file_index=src_file_idx,
        )
        df = pd.read_parquet(src_path)

        # src → dst 用に index 等をずらす
        df = update_data_df(df, src_meta, dst_meta)

        # --- 書き込み前に「今から書く dst の (chunk,file)」をメモしておく ---
        out_chunk = data_idx["chunk"]
        out_file  = data_idx["file"]

        # df を append / rotate して data_idx を更新
        data_idx = append_or_create_parquet_file(
            df,
            src_path,
            data_idx,
            data_files_size_in_mb,
            chunk_size,
            DEFAULT_DATA_PATH,
            contains_images=len(dst_meta.image_keys) > 0,
            aggr_root=dst_meta.root,
        )

        # 「この src の (chunk,file) は dst の (out_chunk,out_file) に対応する」
        # というマッピングを記録
        data_idx["src_to_mapping"][(src_chunk_idx, src_file_idx)] = {
            "chunk": out_chunk,
            "file": out_file,
        }

    return data_idx

from typing import Dict

def aggregate_metadata(
    src_meta,
    dst_meta,
    meta_idx,
    data_idx,
    videos_idx,
    logical_to_src_key: Dict[str, str],
):
    """Aggregates metadata from a source dataset into the destination dataset.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        meta_idx: Dictionary tracking metadata chunk and file indices.
        data_idx: Dictionary tracking data chunk and file indices.
        videos_idx: Dictionary tracking video indices and timestamps.
        logical_to_src_key: logical_key -> src_video_key mapping
                            (aggregate_videos_aligned で確定したもの)
    """

    # 0) この src で扱える video キーが無ければ何もしない
    if not logical_to_src_key:
        return meta_idx

    # 1) この src で実際に扱える video キーだけを抽出
    active_keys = list(logical_to_src_key.keys())

    # update_meta_data に渡す videos_idx は、この src で有効なキーだけのサブセットにする
    videos_idx_for_src = {k: videos_idx[k] for k in active_keys}

    # 2) episodes の (chunk, file) ごとに Parquet を読み、必要ならカラム名を rename
    chunk_file_ids = {
        (c, f)
        for c, f in zip(
            src_meta.episodes["meta/episodes/chunk_index"],
            src_meta.episodes["meta/episodes/file_index"],
            strict=False,
        )
    }

    chunk_file_ids = sorted(chunk_file_ids)
    for chunk_idx, file_idx in chunk_file_ids:
        src_path = src_meta.root / DEFAULT_EPISODES_PATH.format(
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        df = pd.read_parquet(src_path)

        # --- 2-1) 動画カラムの rename（src 側のキー名 → 論理キー名） ---
        # 例:
        #   videos/observation.images.phone/chunk_index
        #     → videos/observation.images.wrist/chunk_index
        rename_map = {}
        for logical_key, src_key in logical_to_src_key.items():
            if src_key == logical_key:
                # すでに同じ名前なら rename 不要
                continue

            src_prefix = f"videos/{src_key}/"
            dst_prefix = f"videos/{logical_key}/"

            for col in df.columns:
                if col.startswith(src_prefix):
                    new_col = dst_prefix + col[len(src_prefix):]
                    rename_map[col] = new_col

        if rename_map:
            df = df.rename(columns=rename_map)

        # --- 2-2) インデックス・タイムスタンプなどを更新 ---
        # この src で有効な video キーだけを渡す
        df = update_meta_data(
            df,
            dst_meta,
            meta_idx,
            data_idx,
            videos_idx_for_src,
        )

        # --- 2-3) Parquet へ append / rotate ---
        meta_idx = append_or_create_parquet_file(
            df,
            src_path,
            meta_idx,
            DEFAULT_DATA_FILE_SIZE_IN_MB,
            DEFAULT_CHUNK_SIZE,
            DEFAULT_EPISODES_PATH,
            contains_images=False,
            aggr_root=dst_meta.root,
        )

    # 3) この src データセットから加算された duration を latest_duration に反映
    for k in active_keys:
        videos_idx[k]["latest_duration"] += videos_idx[k]["episode_duration"]

    return meta_idx

def append_or_create_parquet_file(
    df: pd.DataFrame,
    src_path: Path,
    idx: dict[str, int],
    max_mb: float,
    chunk_size: int,
    default_path: str,
    contains_images: bool = False,
    aggr_root: Path = None,
):
    """Appends data to an existing parquet file or creates a new one based on size constraints.

    Manages file rotation when size limits are exceeded to prevent individual files
    from becoming too large. Handles both regular parquet files and those containing images.

    Args:
        df: DataFrame to write to the parquet file.
        src_path: Path to the source file (used for size estimation).
        idx: Dictionary containing current 'chunk' and 'file' indices.
        max_mb: Maximum allowed file size in MB before rotation.
        chunk_size: Maximum number of files per chunk before incrementing chunk index.
        default_path: Format string for generating file paths.
        contains_images: Whether the data contains images requiring special handling.
        aggr_root: Root path for the aggregated dataset.

    Returns:
        dict: Updated index dictionary with current chunk and file indices.
    """
    dst_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])

    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if contains_images:
            to_parquet_with_hf_images(df, dst_path)
        else:
            df.to_parquet(dst_path)
        return idx

    src_size = get_parquet_file_size_in_mb(src_path)
    dst_size = get_parquet_file_size_in_mb(dst_path)

    if dst_size + src_size >= max_mb:
        idx["chunk"], idx["file"] = update_chunk_file_indices(idx["chunk"], idx["file"], chunk_size)
        new_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])
        new_path.parent.mkdir(parents=True, exist_ok=True)
        final_df = df
        target_path = new_path
    else:
        existing_df = pd.read_parquet(dst_path)
        final_df = pd.concat([existing_df, df], ignore_index=True)
        target_path = dst_path

    if contains_images:
        to_parquet_with_hf_images(final_df, target_path)
    else:
        final_df.to_parquet(target_path)

    return idx


def finalize_aggregation(aggr_meta, all_metadata):
    """Finalizes the dataset aggregation by writing summary files and statistics.

    Writes the tasks file, info file with total counts and splits, and
    aggregated statistics from all source datasets.

    Args:
        aggr_meta: Aggregated dataset metadata.
        all_metadata: List of all source dataset metadata objects.
    """
    logging.info("write tasks")
    write_tasks(aggr_meta.tasks, aggr_meta.root)

    logging.info("write info")
    aggr_meta.info.update(
        {
            "total_tasks": len(aggr_meta.tasks),
            "total_episodes": sum(m.total_episodes for m in all_metadata),
            "total_frames": sum(m.total_frames for m in all_metadata),
            "splits": {"train": f"0:{sum(m.total_episodes for m in all_metadata)}"},
        }
    )
    write_info(aggr_meta.info, aggr_meta.root)

    logging.info("write stats")
    aggr_meta.stats = aggregate_stats([m.stats for m in all_metadata])
    write_stats(aggr_meta.stats, aggr_meta.root)
