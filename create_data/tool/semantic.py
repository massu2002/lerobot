"""
Grounded-SAM を用いて対象物体のセマンティックを取得
"""

from time import time
import numpy as np
import pandas as pd
import math, gc
import torch
import cv2
import imageio.v3 as iio
import imageio
from pathlib import Path
import re
import spacy
from typing import Iterable, Tuple, Optional, Dict, Any, List, Literal
from tqdm import tqdm
from transformers import pipeline, AutoImageProcessor
from PIL import Image
import json, subprocess, shlex, tempfile, os
from pathlib import Path

from tool.Grounded_Segment_Anything.GroundingDINO.groundingdino.util.slconfig import SLConfig
from tool.Grounded_Segment_Anything.GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

from tool.Grounded_Segment_Anything.GroundingDINO.groundingdino.models import build_model

from segment_anything import (
    sam_model_registry, SamPredictor
)
from segment_anything.utils.transforms import ResizeLongestSide

import sys
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR  = THIS_DIR.parent.parent / "src"      # ../src
sys.path.insert(0, str(SRC_DIR))

from lerobot.datasets.dataset_tools import (
    add_features,
    delete_episodes,
    merge_datasets,
    modify_features,
    remove_feature,
    split_dataset,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import HF_LEROBOT_HOME

# ---------- 動画フレームイテレータ & プロパティ取得 ----------
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

# ---------- SAM ユーティリティ ----------
def _to_rgb_from_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

def _cxcywh_norm_to_xyxy_px(boxes_cxcywh_norm: torch.Tensor, W: int, H: int) -> torch.Tensor:
    scale = torch.tensor([W, H, W, H], dtype=boxes_cxcywh_norm.dtype, device=boxes_cxcywh_norm.device)
    b = boxes_cxcywh_norm * scale
    xy1 = b[:, :2] - b[:, 2:] / 2.0
    xy2 = b[:, :2] + b[:, 2:] / 2.0
    return torch.cat([xy1, xy2], dim=1)

def _preprocess_rgb_for_sam(rgb: np.ndarray, transform: ResizeLongestSide, img_size: int, device: torch.device, sam: SamPredictor) -> torch.Tensor:
    resized = transform.apply_image(rgb)                          # (H',W',3)
    t = torch.as_tensor(resized, device=device).permute(2,0,1)    # [3,H',W']
    h, w = t.shape[1:]
    pad_h, pad_w = img_size - h, img_size - w
    t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))
    mean = torch.tensor(sam.pixel_mean, device=device).view(-1,1,1)
    std  = torch.tensor(sam.pixel_std,  device=device).view(-1,1,1)
    return (t.float() - mean) / std                                # [3,1024,1024]

@torch.no_grad()
def get_grounding_output(model, image, caption, box_threshold, text_threshold,
                         with_logits=True, device="cpu"):
    """
    image: いずれかを受け付ける
      - np.ndarray (H,W,3) RGB/uint8
      - PIL.Image
      - torch.Tensor: (3,H,W) or (H,W,3) or (1,3,H,W) / (N,3,H,W)
    """

    # --------- caption 前処理 ---------
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    # --------- 画像を Tensor(N,3,H,W), float32, [0,1] に統一 ---------
    def to_tensor_batched(x) -> torch.Tensor:
        # PIL -> np
        if isinstance(x, Image.Image):
            x = np.array(x.convert("RGB"))  # HWC uint8

        # numpy (HWC, uint8/float)
        if isinstance(x, np.ndarray):
            assert x.ndim == 3 and x.shape[-1] == 3, f"numpy image must be HWC, got {x.shape}"
            if x.dtype != np.uint8:
                x = x.astype(np.uint8)
            t = torch.from_numpy(x)                 # HWC uint8
            t = t.permute(2, 0, 1).contiguous()     # CHW
            t = t.float().div_(255.0)               # [0,1]
            t = t.unsqueeze(0)                      # NCHW
            return t

        # torch.Tensor
        if isinstance(x, torch.Tensor):
            t = x
            if t.ndim == 3:
                # (H,W,3) or (3,H,W)
                if t.shape[-1] == 3:               # HWC
                    t = t.permute(2, 0, 1).contiguous()
                # ここで CHW
                t = t.unsqueeze(0)                  # NCHW
            elif t.ndim == 4:
                # NCHW or NHWC
                if t.shape[-1] == 3:                # NHWC
                    t = t.permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError(f"Unsupported tensor shape: {t.shape}")

            if t.dtype != torch.float32:
                t = t.float()
            # 値域が 0-255 っぽければ [0,1] に
            if t.max() > 1.0:
                t = t.div(255.0)
            return t

        raise TypeError(f"Unsupported image type: {type(x)}")

    image_t = to_tensor_batched(image).to(device, non_blocking=True)  # (1,3,H,W)

    # （注意）Transformers v5 以降は device 引数非推奨 → ここで to(device)
    model = model.to(device)
    model.eval()

    # --------- 推論 ---------
    # GroundingDINO 実装の多くは (images, captions=[...]) を受ける想定
    outputs = model(image_t, captions=[caption])

    # 返り値のキーは実装依存（一般的な辞書構造に合わせる）
    logits = outputs["pred_logits"].sigmoid()[0].cpu()  # (nq, vocab)
    boxes  = outputs["pred_boxes"][0].cpu()             # (nq, 4) [cx,cy,w,h] normalized

    # --------- 閾値でフィルタ ---------
    filt = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt]
    boxes_filt  = boxes[filt]

    # --------- フレーズ復元 ---------
    tokenizer = model.tokenizer  # 元コードの `tokenlizer` はtypo
    tokenized = tokenizer(caption)

    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        # get_phrases_from_posmap は既存ユーティリティをそのまま使用
        phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        if with_logits:
            pred_phrases.append(f"{phrase}({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(phrase)

    return boxes_filt, pred_phrases

# ---------- フレームバッチ → GroundingDINO → SAM(フルバッチ) ----------
@torch.no_grad()
def _sam_fullbatch_decode(
    predictor: "SamPredictor",
    rgbs: List[np.ndarray],                         # RGB np.uint8 HWC
    boxes_per_image_xyxy_px: List[torch.Tensor],    # each: [Mi, 4] (CPU, xyxy px)
    device: torch.device,
    multimask_output: bool = False,
) -> List[torch.Tensor]:
    """
    各フレームを SamPredictor にセットし、検出ボックスでマスク推論。
    返り値: List[Tensor] 各要素は [Mi, 1, H, W] の bool (Miはそのフレームの検出数)
    """
    out_masks: List[torch.Tensor] = []

    for rgb, boxes_xyxy in zip(rgbs, boxes_per_image_xyxy_px):
        H, W = rgb.shape[:2]
        if boxes_xyxy.numel() == 0:
            out_masks.append(torch.empty(0, 1, H, W, dtype=torch.bool))
            continue

        # 1) 画像をセット
        predictor.set_image(rgb)  # 内部で埋め込み生成

        # 2) ボックス座標をSAMの内部座標に変換
        #    注意: apply_boxes_torchは (xyxy) を入力に取り、(H, W) を渡す
        boxes_xyxy_t = boxes_xyxy.to(device=device, dtype=torch.float32)
        transformed = predictor.transform.apply_boxes_torch(boxes_xyxy_t, (H, W))

        # 3) 予測（ポイント指定なし、ボックス指定）
        masks, scores, logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=multimask_output,
        )

        # masks: [Mi, 1, H, W] bool (SAM実装によりfloat→閾値化が必要な場合あり)
        # 一部実装では float で返る場合があるため安全に bool 化
        if masks.dtype != torch.bool:
            masks = masks > 0.0

        # CPUに戻して格納（必要ならGPUのままでもOK）
        out_masks.append(masks.cpu())

    return out_masks

# ---------- メイン: 動画を逐次 + フレームバッチで GSA 実行 ----------
@torch.no_grad()
def run_gsa_video_chunked(
    video_path: str,
    *,
    dino_model,                                 # GroundingDINO モデル
    text_prompt: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    sam_checkpoint: str,
    sam_version: str = "vit_h",
    device: str = "cuda",
    batch_size_frames: int = 8,                 # フレームのバッチ
    t_stride: int = 1,                          # 0/1=全フレーム, 2や3で間引き
    multimask_output: bool = False,
    progress: bool = True,
    max_boxes_per_frame: Optional[int] = None,  # 大量検出時の上限
) -> Dict[str, List]:
    """
    動画を読み出し:
      1) フレームを t_stride で間引きつつ batch_size_frames ごとにまとめる
      2) GroundingDINO で各フレームのボックス/フレーズ取得（DINO入力はTensor化）
      3) SAM 低レベルAPIで『フレーム一括エンコード』『全ボックス一括デコード』
    戻り:
      {
        "masks":   List[Tensor]  各フレームの [Mi, C, H, W] (bool)
        "boxes":   List[Tensor]  各フレームの [Mi, 4] (xyxy px)
        "phrases": List[List[str]]
        "frame_indices": List[int] 元動画で採用したフレーム番号
      }
    """
    import gc
    import numpy as np
    import torch

    dev = torch.device(device)

    # --- SAM 準備 ---
    sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(dev)
    sam.eval()
    predictor = SamPredictor(sam)

    # --- 1st pass: 採用フレーム数見積り（t_stride<=1 は全フレーム） ---
    H, W = _probe_hw_safe(video_path)
    keep_every = 1 if t_stride <= 1 else int(t_stride)

    T = 0
    for idx, _ in enumerate(_iter_frames_v3_or_v2(video_path)):
        if idx % keep_every == 0:
            T += 1
    if T == 0:
        raise ValueError(
            f"No frames after applying t_stride={t_stride} (video may be empty or stride too large)."
        )

    # --- 出力バッファ ---
    all_masks:   List[torch.Tensor] = []
    all_boxes:   List[torch.Tensor] = []
    all_phrases: List[List[str]]    = []
    frame_indices: List[int]        = []

    frames_batch_bgr: List[np.ndarray] = []
    idx_batch: List[int] = []

    pbar = tqdm(total=T, desc="GSA (GroundingDINO→SAM)", disable=not progress)

    def _prepare_dino_input(rgb_np: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        rgb_np: np.uint8 [H,W,3], RGB
        return: torch.float32 [1,3,H,W], [0,1] on device
        """
        t = torch.from_numpy(rgb_np).to(device=device, non_blocking=True)
        if t.dtype != torch.uint8:
            t = t.to(torch.uint8)
        t = t.float().div_(255.0)                # [0,1]
        t = t.permute(2, 0, 1).contiguous()      # CHW
        t = t.unsqueeze(0)                       # NCHW
        return t

    def process_frames_batch(frames_bgr: List[np.ndarray], indices: List[int]):
        if not frames_bgr:
            return

        # --- OpenCV BGR → RGB (np.uint8, HWC) ---
        rgbs = [_to_rgb_from_bgr(fr) for fr in frames_bgr]

        boxes_xyxy_list: List[torch.Tensor] = []
        phrases_list:    List[List[str]]    = []

        # --- GroundingDINO per frame ---
        for rgb in rgbs:
            try:
                # DINO 入力をこの関数側で Tensor 化（NCHW, float32, [0,1], device）
                rgb_t = _prepare_dino_input(rgb, dev)
                # 例: get_grounding_output(dino_model, image_t, text, ...)
                boxes_cxcywh_norm, phrases = get_grounding_output(
                    dino_model,
                    rgb,                         # ← Tensor を渡す
                    text_prompt,
                    box_threshold,
                    text_threshold,
                    device=dev,                    # 内部で .to() があっても Tensor なので安全
                )

                # 0件対応
                if boxes_cxcywh_norm is None or boxes_cxcywh_norm.size(0) == 0:
                    boxes_xyxy_list.append(torch.empty(0, 4))
                    phrases_list.append([])
                    continue

                # 上限カット
                if isinstance(max_boxes_per_frame, int) and boxes_cxcywh_norm.size(0) > max_boxes_per_frame:
                    boxes_cxcywh_norm = boxes_cxcywh_norm[:max_boxes_per_frame]
                    phrases = phrases[:max_boxes_per_frame]

                # 元解像度の xyxy(px) に変換（CPU TensorでOK）
                b_xyxy = _cxcywh_norm_to_xyxy_px(
                    boxes_cxcywh_norm.to(torch.float32).cpu(),
                    W=rgb.shape[1],
                    H=rgb.shape[0],
                )
                boxes_xyxy_list.append(b_xyxy)
                phrases_list.append(phrases)

            except Exception as e:
                # 1フレーム失敗しても継続
                print(f"[WARN] DINO failed on a frame: {e}")
                boxes_xyxy_list.append(torch.empty(0, 4))
                phrases_list.append([])
                continue

        # --- SAM: フレーム一括エンコード→全ボックス一括デコード ---
        masks_list = _sam_fullbatch_decode(
            predictor=predictor,
            rgbs=rgbs,                                  # SAM は RGB np.uint8 HWC を想定
            boxes_per_image_xyxy_px=boxes_xyxy_list,    # CPU上の xyxy(px)
            device=dev,
            multimask_output=multimask_output,
        )

        # --- 書き出し ---
        all_masks.extend(masks_list)
        all_boxes.extend(boxes_xyxy_list)
        all_phrases.extend(phrases_list)
        frame_indices.extend(indices)

        # --- メモリ掃除 ---
        del rgbs, boxes_xyxy_list, phrases_list, masks_list
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- 2nd pass: 逐次 + フレームバッチ ---
    kept = 0
    for idx, fr_bgr in enumerate(_iter_frames_v3_or_v2(video_path)):
        if idx % keep_every != 0:
            continue
        frames_batch_bgr.append(fr_bgr)
        idx_batch.append(idx)
        kept += 1
        pbar.update(1)

        if len(frames_batch_bgr) >= batch_size_frames:
            process_frames_batch(frames_batch_bgr, idx_batch)
            frames_batch_bgr.clear()
            idx_batch.clear()

    # 端数
    if frames_batch_bgr:
        process_frames_batch(frames_batch_bgr, idx_batch)
        frames_batch_bgr.clear()
        idx_batch.clear()

    pbar.close()

    return {
        "masks": all_masks,              # List[ [Mi, C, H, W] bool ]
        "boxes": all_boxes,              # List[ [Mi, 4] xyxy(px) ]
        "phrases": all_phrases,          # List[ List[str] ]
        "frame_indices": frame_indices,  # 元動画のフレーム番号
    }

def get_prompts_from_repo_id(
    repo_id: str
) -> List[str]:
    """
    タスク記述からセマンティックセグメンテーションのプロンプトを抽出
    """
    ds_meta = LeRobotDatasetMetadata(repo_id)
    
    return ds_meta.tasks

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

def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def _safe_bool_or(m: torch.Tensor) -> torch.Tensor:
    """インスタンス/候補(C)をまとめてOR（bool） -> [H,W]"""
    # m: [Mi, C, H, W] bool（C=1 or 3）
    if m.numel() == 0:
        return None
    if m.ndim == 4:
        # (Mi,C,H,W) → (H,W)
        return m.any(dim=1).any(dim=0)  # まずC、次にMiでOR
    elif m.ndim == 3:
        # (Mi,H,W)
        return m.any(dim=0)
    else:
        raise ValueError(f"unexpected mask shape: {tuple(m.shape)}")

def run_semantic_chunked(
    video_path: str,
    *,
    dino_model,
    text_prompt: str,
    sam_checkpoint: str,
    sam_version: str = "vit_h",
    device: str = "cuda",
    batch_size_frames: int = 8,
    t_stride: int = 1,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    multimask_output: bool = False,
    progress: bool = True,
) -> torch.Tensor:
    """
    Grounded-Segment-Anything を使って動画全体のセマンティックマスクを作る。
    返り値: masks [T, H, W] (bool, CPU)
    T は t_stride 適用後のフレーム数。
    """
    out = run_gsa_video_chunked(
        video_path=video_path,
        dino_model=dino_model,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        sam_checkpoint=sam_checkpoint,
        sam_version=sam_version,
        device=device,
        batch_size_frames=batch_size_frames,
        t_stride=t_stride,
        multimask_output=multimask_output,
        progress=progress,
        max_boxes_per_frame=None,
    )

    # 各フレーム: [Mi, C, H, W] (bool) → セマンティック [H, W] (bool)
    sem_list: List[torch.Tensor] = []
    for m in out["masks"]:
        if m.numel() == 0:
            # 検出なし → Falseの2Dマスクを作るためにサイズが必要
            # ただしサイズは boxes が無いと分からないので、動画から推定する方が安全
            # → ここでは m が空の場合はスキップし、後で整形時に埋める
            sem_list.append(None)
        else:
            sem_list.append(_safe_bool_or(m))

    # 空のときもサイズを復元するため、最初に有効マスクがあるフレームから (H,W) を拾う
    H = W = None
    for t in range(len(sem_list)):
        if sem_list[t] is not None:
            H, W = sem_list[t].shape[-2], sem_list[t].shape[-1]
            break
    if H is None or W is None:
        # 全フレーム検出ゼロ → 動画解像度を直接取得
        cap = cv2.VideoCapture(str(video_path))
        ok, fr = cap.read()
        cap.release()
        if not ok or fr is None:
            raise RuntimeError("failed to probe a frame for empty semantic masks")
        H, W = fr.shape[0], fr.shape[1]

    T = len(sem_list)
    out_masks = torch.zeros((T, H, W), dtype=torch.bool, device="cpu")
    for t in range(T):
        if sem_list[t] is None:
            # 検出なし → 全False
            continue
        if sem_list[t].device.type != "cpu":
            out_masks[t].copy_(sem_list[t].to("cpu"))
        else:
            out_masks[t].copy_(sem_list[t])

    return out_masks  # [T, H, W] bool (cpu)


def _probe_codec(path: str) -> str | None:
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of json "{path}"'
    try:
        out = subprocess.check_output(shlex.split(cmd))
        info = json.loads(out)
        streams = info.get("streams", [])
        return streams[0]["codec_name"] if streams else None
    except Exception:
        return None

def _maybe_transcode_av1_to_h264(src_path: str) -> str:
    codec = _probe_codec(src_path)
    if codec and codec.lower() in {"av1"}:
        tmp = Path(tempfile.gettempdir()) / (Path(src_path).stem + "_h264.mp4")
        if not tmp.exists():  # 使い回し
            cmd = f'ffmpeg -y -hide_banner -loglevel error -i "{src_path}" -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 20 -c:a copy "{tmp}"'
            subprocess.check_call(shlex.split(cmd))
        return str(tmp)
    return src_path

def save_semantic_overlay_video(video_path: str, sem_masks: torch.Tensor, *, t_stride: int = 1,
                                alpha: float = 0.5, color: tuple = (0, 255, 0),
                                out_suffix: str = "_semantic_overlay") -> str:
    p = Path(video_path)
    out_path = str(p.with_name(p.stem + out_suffix + p.suffix))

    # ★ AV1なら自動でH.264に変換してから読み込む
    video_path = _maybe_transcode_av1_to_h264(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    H, W = sem_masks.shape[1:]
    writer = cv2.VideoWriter(out_path, fourcc, fps / max(1, t_stride), (W, H))

    keep_idx = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % t_stride != 0:
            idx += 1
            continue

        # 安全のため形状確認＆必要ならリサイズ
        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

        mask = sem_masks[keep_idx].numpy()  # bool
        overlay = frame.copy()
        overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
        writer.write(overlay)

        keep_idx += 1
        idx += 1
        if keep_idx >= sem_masks.shape[0]:
            break

    writer.release()
    cap.release()
    return out_path

# --- 全データセット処理のメイン関数 ---
def process_all_semantic(
    repo_id: str,
    camera_key: str = "observation.images.front",
    *,
    config_file: str = "tool/Grounded_Segment_Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    grounded_checkpoint: str = "tool/Grounded_Segment_Anything/groundingdino_swint_ogc.pth",
    sam_checkpoint: str = "tool/Grounded_Segment_Anything/sam_vit_h_4b8939.pth",
    sam_version: str = "vit_h",
    device: str = "cuda",
    batch_size: int = 4,         # ここでは「フレームのバッチ」数
    t_stride: int = 1,           # 1=全フレーム、>1=間引き
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    new_repo_suffix: str = "_semantic",
    overlay_alpha: float = 0.8,
    overlay_color: tuple = (0, 255, 0),
):
    """
    データセット配下の {camera_key} の全動画に対してセマンティックセグメンテーションを実行し、
    セマンティックマスクの可視化動画を保存（必要ならマスク自体も保存に拡張可）。
    """
    root = Path(f"{HF_LEROBOT_HOME}/{repo_id}")
    out_repo_id = f"{repo_id}{new_repo_suffix}"

    # 元データセット（必要に応じて）
    dataset = LeRobotDataset(repo_id)

    # t_stride ガード
    t_stride = safe_t_stride(t_stride)

    # GroundingDINO をロード
    bert_base_uncased_path = None
    dino_model = load_model(
        model_config_path=config_file,
        model_checkpoint_path=grounded_checkpoint,
        bert_base_uncased_path=bert_base_uncased_path,
        device=device,
    )
    
    # テキストを取得
    nlp = spacy.load("en_core_web_sm")
    prompts = get_prompts_from_repo_id(repo_id)
    texts = [str(i) for i in prompts.index] 
    print("Task texts:", texts)
    doc = nlp(texts[0])
    prompts_nouns = ", ".join(
        chunk.text for chunk in doc.noun_chunks if chunk.root.pos_ != "PRON"
    )
    print("Task nouns:", prompts_nouns)

    for cidx, fidx, video_path in iter_chunk_file_paths(root, camera_key):
        print(f"[semantic] chunk={cidx:03d} file={fidx:03d} : {video_path}")

        try:
            # ---- セマンティックマスク作成（T,H,W, bool）----
            sem_masks = run_semantic_chunked(
                str(video_path),
                dino_model=dino_model,
                text_prompt=prompts_nouns,
                sam_checkpoint=sam_checkpoint,
                sam_version=sam_version,
                device=device,
                batch_size_frames=batch_size,
                t_stride=t_stride,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                multimask_output=False,
                progress=True,
            )
            print("  semantic masks:", sem_masks.shape, sem_masks.dtype, sem_masks.device)

            # ---- 可視化動画の保存（半透明オーバーレイ）----
            out_vis_path = save_semantic_overlay_video(
                video_path=str(video_path),
                sem_masks=sem_masks,
                t_stride=t_stride,
                alpha=overlay_alpha,
                color=overlay_color,   # BGR
                out_suffix="_semantic",
            )
            print("  semantic overlay saved ->", out_vis_path)

        finally:
            # ---- キャッシュ解放 ----
            del sem_masks
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("  [cache cleared]\n")

            if (fidx + 1) % 5 == 0:
                print("  cooling GPU memory for stability...")
                time.sleep(3)
                
if __name__ == "__main__":
    process_all_semantic(repo_id="real_data/box_black_plate")