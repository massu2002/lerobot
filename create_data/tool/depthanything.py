"""
Depth Anythingを用いた深度推定
"""

from time import time
import numpy as np
import math, gc
import torch
import cv2
import imageio.v3 as iio
import imageio
from pathlib import Path
import re
from typing import Iterable, Tuple, Optional, Dict, Any, List, Literal
from tqdm import tqdm
from transformers import pipeline, AutoImageProcessor
from PIL import Image

from lerobot.datasets.dataset_tools import (
    add_features,
    delete_episodes,
    merge_datasets,
    modify_features,
    remove_feature,
    split_dataset,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

HF_LEROBOT_HOME = "/home/data01/smolvla"

def torch_nanmin(x):
    return torch.min(x[~torch.isnan(x)]) if torch.isnan(x).any() else torch.min(x)

def torch_nanmax(x):
    return torch.max(x[~torch.isnan(x)]) if torch.isnan(x).any() else torch.max(x)

# 深度マップ動画保存
def save_depth_video_streaming(
    video_path: str,
    depths: torch.Tensor,          # [T, H, W] (GPU/CPUどちらでも可)
    fps: float = 30.0,
    colormap: str = "magma",       # "magma" | "jet" | "turbo" | "gray"
    to_uint8: bool = True,
    scale: str = "global",         # "global" | "per_frame" | "percentile"
    percentiles: tuple = (1.0, 99.0),  # scale="percentile" のとき使用
) -> tuple[Path, torch.Tensor]:
    """
    超長尺TでもOOMしないストリーミング版。
    - scale="global": 全フレームで共通min/max（推奨: 見え方が安定）
    - scale="per_frame": 各フレームのmin/max（コントラスト最優先）
    - scale="percentile": 全体のパーセンタイルでロバスト正規化
    """
    p = Path(video_path)
    out_path = p.with_name(p.stem + "_depths.mp4")

    assert depths.ndim == 3, "depths must be [T, H, W]"
    T, H, W = depths.shape

    # --- カラーマップ ---
    cmap_dict = {
        "magma": cv2.COLORMAP_MAGMA,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "gray": cv2.COLORMAP_BONE,
    }
    cmap_code = cmap_dict.get(colormap, cv2.COLORMAP_MAGMA)

    # --- VideoWriter 準備 ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter. "
                           "ffmpeg/codec が必要な場合があります。")

    # --- スケーリング範囲を決める（global/percentile のときのみ一度計算）---
    if to_uint8:
        if scale == "global":
            d_min = torch_nanmin(depths).item()
            d_max = torch_nanmax(depths).item()
        elif scale == "percentile":
            lo, hi = torch.tensor(percentiles, device=depths.device, dtype=torch.float32) / 100.0
            q = torch.quantile(depths.float().view(-1), torch.stack([lo, hi]))
            d_min, d_max = float(q[0].item()), float(q[1].item())
        else:
            d_min = d_max = None
    else:
        d_min = d_max = None

    first_frame_tensor = None

    # --- フレーム単位でCPUへ移しながら書き出し ---
    for i in range(T):
        # GPU→CPUへ“そのフレームだけ”移す（non_blocking指定でオーバーラップも可）
        frame_t = depths[i]

        if to_uint8:
            if scale == "per_frame":
                fmin = torch.nanmin(frame_t).item()
                fmax = torch.nanmax(frame_t).item()
            else:
                fmin, fmax = d_min, d_max

            denom = max(fmax - fmin, 1e-8)
            frame_u8 = ((frame_t - fmin) / denom).clamp(0, 1)
            frame_u8 = (frame_u8 * 255).to(torch.uint8).cpu().numpy()
        else:
            # 既に [0,255] 前提で uint8 化する場合のみ
            frame_u8 = frame_t.to(torch.uint8).cpu().numpy()

        # OpenCVはBGR入力。applyColorMapは1ch uint8 -> 3ch BGR を返す
        frame_color = cv2.applyColorMap(frame_u8, cmap_code)

        if i == 0:
            # 返却用（RGBのtorch.Tensor）
            first_frame_tensor = torch.from_numpy(
                cv2.cvtColor(frame_color, cv2.COLOR_BGR2RGB)
            ).to(torch.uint8)

        writer.write(np.ascontiguousarray(frame_color))

    writer.release()
    print(f"[Depth] Saved full-length depth video → {out_path}")

    return out_path, first_frame_tensor

# モデルロード
@torch.inference_mode()
def load_depthanything_pipeline(model_id: str, device: str = "cuda"):
    image_processor = AutoImageProcessor.from_pretrained(
        model_id, use_fast=True  # ★ fast を強制
    )
    pipe = pipeline(
        task="depth-estimation",
        model=model_id,
        device=device if device != "cpu" else -1,
        image_processor=image_processor,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    try:
        pipe.model.eval()
        if device.startswith("cuda"):
            pipe.model.half()
    except Exception:
        pass
    # 可能なら半精度へ（失敗したら自動でfloat32にフォールバック）
    if device != "cpu":
        try:
            pipe.model.to(dtype=torch.float16)
        except Exception:
            try:
                pipe.model.to(dtype=torch.bfloat16)
            except Exception:
                pass
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return pipe

def _iter_frames_v3_or_v2(video_path):
    """v3.iter() が無ければ v2.get_reader に自動フォールバックしてフレームをyield"""
    try:
        r = iio.imopen(video_path, plugin="FFMPEG", io_mode="r")
        it = getattr(r, "iter", None)
        if callable(it):
            try:
                for fr in it():
                    yield fr
            finally:
                r.close()
            return
        else:
            r.close()
    except Exception:
        # v3失敗時はv2へ
        pass

    # v2 fallback
    r2 = imageio.get_reader(video_path, "ffmpeg")
    try:
        for fr in r2:
            yield fr
    finally:
        r2.close()

def _probe_hw_safe(video_path: str) -> Tuple[int, int]:
    """(H, W) を v3→v2 の順で安全に取得"""
    # v3: props.shape = (frames or inf, H, W, C)
    try:
        with iio.imopen(video_path, plugin="FFMPEG", io_mode="r") as r0:
            props = r0.properties()
            shp = getattr(props, "shape", None)
            if shp and len(shp) >= 3:
                return int(shp[1]), int(shp[2])
    except Exception:
        pass
    # v2: meta["size"] = (W, H)
    try:
        r2 = imageio.get_reader(video_path, "ffmpeg")
        try:
            meta = r2.get_meta_data()
            if "size" in meta:
                W, H = meta["size"]
                return int(H), int(W)
        finally:
            r2.close()
    except Exception:
        pass
    # フォールバック：先頭フレームから読む
    for fr0 in _iter_frames_v3_or_v2(video_path):
        return int(fr0.shape[0]), int(fr0.shape[1])
    raise ValueError("Video has no frames.")

# 動画をチャンクごとに深度推定
@torch.inference_mode()
def run_depthanything_chunked(
    video_path: str,
    *,
    pipe=None,                                   # 既に作成済みの pipeline を渡せる
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: str = "cuda",
    batch_size: int = 16,                        # chunk_len の代わりに実メモリで調整しやすい
    t_stride: int = 1,                           # 2や3で間引き
    normalize: Literal["none", "frame", "global"] = "none",
    out_dtype: Literal["float16", "float32"] = "float16",  # メモリ削減に効く
    progress: bool = True,
) -> torch.Tensor:
    """
    動画を逐次読みして小さなバッチでDepth Anything推論。
    ・入力フレームは常に逐次処理（全読みしない）
    ・出力は常に元動画サイズ (H, W)
    ・返り値: depths [T, H, W] torch.(float16/float32) on CPU
    """
    # ---------------- 1) 1st pass: フレーム数と元解像度 ----------------
    H, W = _probe_hw_safe(video_path)

    T = 0
    for idx, _ in enumerate(_iter_frames_v3_or_v2(video_path)):
        if t_stride <= 1 or idx % t_stride == 0:
            T += 1

    if T == 0:
        raise ValueError("No frames after applying t_stride (check the video and t_stride).")

    # ---------------- 2) 出力確保（CPU, 省メモリ） ----------------
    torch_dtype = torch.float16 if out_dtype == "float16" else torch.float32
    depths = torch.empty((T, H, W), dtype=torch_dtype, device="cpu")

    # ---------------- 3) パイプライン準備 ----------------
    if pipe is None:
        pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=device,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        )
        try:
            pipe.model.eval()
            if device.startswith("cuda"):
                pipe.model.half()  # 省VRAM
        except Exception:
            pass

    # ---------------- 4) 2nd pass: 逐次 + バッチ推論 ----------------
    frames_batch: List[np.ndarray] = []
    write_pos = 0

    # global正規化用
    gmin = float("inf")
    gmax = float("-inf")

    # v3/v2 どちらでも回せるフレームイテレータを使う（※ rdr=imopen は使わない）
    frames_iter = _iter_frames_v3_or_v2(video_path)

    # 進捗表示（保持フレーム数Tに基づく手動カウント）
    pbar = tqdm(total=T, desc="Depth (streamed/batched)", disable=not progress)
    kept = 0

    @torch.inference_mode()
    def process_batch(batch: List[np.ndarray], start_idx: int):
        nonlocal gmin, gmax
        if not batch:
            return

        # --- BGR→RGB (OpenCVで取り出した可能性に備える) ---
        pil_list = []
        for fr in batch:
            if fr.ndim == 3 and fr.shape[2] == 3:  # 3chのときだけチェック
                # frがBGR想定: RGBへ
                fr_rgb = fr[..., ::-1]
                pil_list.append(Image.fromarray(fr_rgb))
            else:
                pil_list.append(Image.fromarray(fr))

        # 推論
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                results = pipe(pil_list, batch_size=batch_size)
        else:
            results = pipe(pil_list, batch_size=batch_size)

        # --- 出力取り出し: predicted_depth / depth の両対応 ---
        for i, res in enumerate(results):
            d = None
            if isinstance(res, dict):
                d = res.get("predicted_depth", res.get("depth"))
            if d is None:
                raise KeyError("Depth result must contain 'predicted_depth' or 'depth'.")

            # nd -> numpy
            if isinstance(d, Image.Image):
                d_np = np.array(d)
            elif torch.is_tensor(d):
                d_np = d.squeeze().detach().cpu().float().numpy()
            else:
                d_np = np.asarray(d)

            # [H, W]化
            if d_np.ndim == 3:
                d_np = d_np[..., 0]
            # リサイズ（出力を必ず元サイズへ）
            if d_np.shape[0] != H or d_np.shape[1] != W:
                d_np = cv2.resize(d_np, (W, H), interpolation=cv2.INTER_LINEAR)

            # global用のmin/maxを「float32」で堅牢に更新
            if normalize == "global":
                _mn = float(np.nanmin(d_np))
                _mx = float(np.nanmax(d_np))
                gmin = min(gmin, _mn)
                gmax = max(gmax, _mx)

            depths[start_idx + i].copy_(
                torch.from_numpy(d_np).to(torch.float16 if out_dtype == "float16" else torch.float32)
            )

        # クリーンアップ
        del results, pil_list
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 全フレームを走査しつつ t_stride で間引き
    for idx, fr in enumerate(frames_iter):
        if idx % t_stride != 0:
            continue
        frames_batch.append(fr)
        kept += 1
        pbar.update(1)  # 保持フレーム分だけ進捗を進める

        if len(frames_batch) >= batch_size:
            process_batch(frames_batch, write_pos)
            write_pos += len(frames_batch)
            frames_batch.clear()

    # 端数
    if frames_batch:
        process_batch(frames_batch, write_pos)
        write_pos += len(frames_batch)
        frames_batch.clear()

    pbar.close()

    # ---------------- 5) 正規化（インプレースで最小メモリ） ----------------
    eps = 1e-12
    if normalize == "frame":
        for t in tqdm(range(T), disable=not progress, desc="Normalize (per-frame)"):
            d = depths[t].to(torch.float32)
            mn = torch.nanmin(d)
            mx = torch.nanmax(d)
            # 分散が極小のフレームはそのまま（色ベタ防止）
            if float(mx - mn) > 1e-6:
                d = (d - mn) / (mx - mn + eps)
            depths[t].copy_(d.to(torch_dtype))
            del d
    elif normalize == "global":
        # gmin/gmaxが未更新（異常）の場合の保険
        if not np.isfinite(gmin) or not np.isfinite(gmax) or (gmax - gmin) <= 1e-6:
            # 自動でフレーム別へフォールバック
            for t in tqdm(range(T), disable=not progress, desc="Normalize (fallback per-frame)"):
                d = depths[t].to(torch.float32)
                mn = torch.nanmin(d); mx = torch.nanmax(d)
                if float(mx - mn) > 1e-6:
                    d = (d - mn) / (mx - mn + eps)
                depths[t].copy_(d.to(torch_dtype))
                del d
        else:
            for t in tqdm(range(T), disable=not progress, desc="Normalize (global)"):
                d = depths[t].to(torch.float32)
                d = (d - gmin) / (gmax - gmin + eps)
                depths[t].copy_(d.to(torch_dtype))
                del d

    return depths  # [T, H, W] (cpu, float16/float32)

_num = re.compile(r'(\d+)')

def _nkey(name: str):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]

def iter_chunk_file_paths(root: Path, camera_key: str) -> Iterable[Tuple[int, int, Path]]:
    """
    例: {root}/videos/{camera_key}/chunk-000/file-000.mp4 を総なめ
    ※ 語尾に "_" があるファイル (例: file-000_.mp4) はスキップ
    """
    base = root / "videos" / camera_key
    if not base.exists():
        raise FileNotFoundError(f"not found: {base}")
    
    for chunk_dir in sorted((p for p in base.glob("chunk-*") if p.is_dir()), key=lambda p: _nkey(p.name)):
        m = re.search(r'chunk-(\d+)', chunk_dir.name)
        if not m:
            continue
        cidx = int(m.group(1))
        
        for fpath in sorted(chunk_dir.glob("file-*.mp4"), key=lambda p: _nkey(p.name)):
            # 語尾が "_" のものをスキップ
            if fpath.stem.endswith("_"):
                continue
            m2 = re.search(r'file-(\d+)', fpath.stem)
            if not m2:
                continue
            fidx = int(m2.group(1))
            yield cidx, fidx, fpath

def safe_t_stride(v: Optional[int]) -> int:
    return 1 if (v is None or v == 0) else v

def build_video_feature_meta(h: int, w: int, fps: float = 30.0) -> Dict[str, Any]:
    # codec/pix_fmt は環境により異なるため、最低限のメタだけ埋めます（必要なら後で拡張）
    return {
        "dtype": "video",
        "shape": [h, w, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": h,
            "video.width": w,
            "video.codec": "av1",       # 実コーデックに合わせて適宜変更可
            "video.pix_fmt": "yuv420p", # 同上
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }

def process_all_depths(
    repo_id: str,
    camera_key: str = "observation.images.front",
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: str = "cuda",
    batch_size: int = 8,
    t_stride: int = 1,                # 1=全フレーム、>1=間引き
    normalize: str = "global",
    out_dtype: str = "float16",
    fps: float = 30.0,
    colormap: str = "magma",
    scale: str = "global",
    percentiles=(1.0, 99.0),
    new_repo_suffix: str = "_with_depth",
):
    """
    データセット配下の {camera_key} の全動画に対して深度推定→可視化→features追記。
    既存の add_features パイプを動画単位で繰り返し呼び出して「積み増し」します。
    """
    # ルート（HF_LEROBOT_HOME は既存コードのグローバル想定）
    root = Path(f"{HF_LEROBOT_HOME}/{repo_id}")

    # t_stride ガード（0/None → 1）
    t_stride = safe_t_stride(t_stride)

    # 走査
    for cidx, fidx, video_path in iter_chunk_file_paths(root, camera_key):
        print(f"[depth] chunk={cidx:03d} file={fidx:03d} : {video_path}")

        try:
            # --- 深度推定（T,H,W）---
            depths = run_depthanything_chunked(
                str(video_path),
                model_id=model_id,
                device=device,
                batch_size=batch_size,
                t_stride=t_stride,
                normalize=normalize,
                out_dtype=out_dtype,
                progress=True,
            )
            print("  depths:", depths.shape)

            # --- 深度可視化を保存 ---
            out_path, out_frame = save_depth_video_streaming(
                video_path=str(video_path),
                depths=depths,
                fps=fps,
                colormap=colormap,
                to_uint8=True,
                scale=scale,
                percentiles=percentiles,
            )
            print("  depth vis saved ->", out_path)
            print("  out_frame:", out_frame.shape)

        finally:
            # --- 明示的キャッシュ解放 ---
            del depths, out_frame
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("  [cache cleared]\n")
            
            if (fidx + 1) % 5 == 0:
                print("  cooling GPU memory for stability...")
                time.sleep(3)

        # # --- features に追記 ---
        # # 動画単位でユニークになるキーにする（積み増しで衝突させない）
        # feature_key = f"{camera_key}.depth/chunk-{cidx:03d}/file-{fidx:03d}"
        # meta = build_video_feature_meta(h=out_frame.shape[1], w=out_frame.shape[2], fps=fps)

        # dataset = add_features(   # 戻り値（新データセット）を次ループの入力に更新＝積み増し
        #     dataset,
        #     features={
        #         feature_key: (out_frame, meta),
        #     },
        #     repo_id=out_repo_id,
        # )
        # print(f"  added feature: {feature_key}")