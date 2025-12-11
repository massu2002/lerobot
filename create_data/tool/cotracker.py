"""
CO-trackerによる動的領域の検出
"""

import os
import numpy as np
import torch
import cv2
import imageio.v3 as iio
import imageio
from pathlib import Path
import re
from typing import Iterable, Tuple, Optional, Dict, Any
from tqdm import tqdm

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

# 動的領域のみを描画した動画を保存する関数
def save_dynamic_regions_video(
    video_path: str,
    tracks: torch.Tensor,          # (1, T, N, 2)
    vis: torch.Tensor,             # (1, T, N)
    out_path: Optional[str] = None,
    t_stride: int = 1,             # run時と合わせる
    normalized: bool = False,      # Trueなら (x*=W, y*=H) して画素座標化
    disp_thresh: float = 1.25,      # 動的判定しきい値[px]
    point_radius: int = 12,        # ← 四角の半辺
    dilate_px: int = 9,
    blur_ksize: int = 3,
    mode: str = "mask",            # "overlay" or "mask"
    overlay_alpha: float = 0.6,
    keep_color_outside: bool = False,
    # --- 追加: 線分を四角で埋める設定 ---
    line_step_px: Optional[int] = None,  # スタンプ間隔(px)。Noneなら自動（= point_radius）
    include_start: bool = True,          # 始点も押す
    include_end: bool = True,            # 終点も押す
):
    """
    Returns:
        out_path_str (str): 書き出した動画ファイルへのパス（拡張子は実際に使われたもの）
        out_tensor (torch.Tensor): 書き出しに使ったフレーム列 (T, H, W, 3), dtype=uint8, RGB

    注意: 長尺動画では out_tensor を返すためにメモリを多く消費します。
    必要に応じて後段で .cpu() のまま保存したり、del/out_tensor で解放してください。
    """
    def _fill_square(mask: np.ndarray, x: float, y: float, half: int, value: int = 255):
        """(x,y)中心・半辺halfの正方形を塗り潰し。画面外クリップ込み。"""
        xi = int(round(x)); yi = int(round(y))
        x0 = max(0, xi - half); y0 = max(0, yi - half)
        x1 = min(mask.shape[1] - 1, xi + half); y1 = min(mask.shape[0] - 1, yi + half)
        if x0 <= x1 and y0 <= y1:
            cv2.rectangle(mask, (x0, y0), (x1, y1), value, thickness=-1)

    p = Path(video_path)
    if out_path is None:
        out_path = p.with_name(p.stem + "_dynamic.mp4")
    else:
        out_path = Path(out_path)
    if out_path.suffix.lower() not in [".mp4", ".mkv", ".avi"]:
        out_path = out_path.with_suffix(".mp4")

    # fps 取得（なければ 30）
    try:
        fps = iio.immeta(str(video_path), plugin="FFMPEG").get("fps", 30)
    except Exception:
        fps = 30

    # フレーム読み込み（RGB）
    frames = iio.imread(str(video_path), plugin="FFMPEG")  # [T,H,W,3] RGB uint8
    if t_stride > 1:
        frames = frames[::t_stride]
    T0, H, W, _ = frames.shape

    # tracks / vis を CPU numpy 化
    tr = tracks.detach().cpu().numpy()[0]   # [T,N,2]
    vs = vis.detach().cpu().numpy()[0]      # [T,N]
    T, N, _ = tr.shape
    if T != T0:
        raise ValueError(f"フレーム数が一致しません: video={T0}, tracks={T}. t_stride の整合を確認してください。")

    if normalized:
        tr[..., 0] *= W
        tr[..., 1] *= H

    # スタンプ間隔のデフォルト（四角の“半径”と同程度）
    if line_step_px is None or line_step_px <= 0:
        line_step_px = max(1, point_radius)

    # マスク後処理用
    ksz = max(1, dilate_px | 1)  # 奇数化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    bk = (blur_ksize | 1) if (blur_ksize and blur_ksize > 0) else 0

    # エンコーダ選択
    tried = []
    writer = None
    for codec, ext in [("libx264", ".mp4"), ("mpeg4", ".mp4"), ("mjpegb", ".avi")]:
        try:
            out_try = out_path.with_suffix(ext)
            writer = imageio.get_writer(
                str(out_try),
                fps=fps,
                codec=codec,
                quality=8,
                macro_block_size=None,
                ffmpeg_log_level="warning",
            )
            ok_codec = codec
            out_final = out_try
            break
        except Exception as e:
            tried.append((codec, str(e)))
            writer = None
    if writer is None:
        raise RuntimeError(f"動画エンコーダの初期化に失敗しました: {tried}")

    # ===== 出力フレームを貯めて最後にテンソル返す =====
    out_frame = None
    
    
    # ループ
    for t in range(T):
        frame_rgb = frames[t].copy()              # RGB uint8
        mask = np.zeros((H, W), dtype=np.uint8)

        if t > 0:
            prev_xy = tr[t-1]; curr_xy = tr[t]
            prev_vis = vs[t-1]; curr_vis = vs[t]
            valid = (
                (prev_vis > 0.5) & (curr_vis > 0.5)
                & (~np.isnan(prev_xy).any(axis=1))
                & (~np.isnan(curr_xy).any(axis=1))
            )
            if np.any(valid):
                dxy = curr_xy[valid] - prev_xy[valid]
                disp = np.linalg.norm(dxy, axis=1)
                moving_idx = np.where(disp >= disp_thresh)[0]
                if moving_idx.size > 0:
                    valid_ids = np.flatnonzero(valid)
                    for k in moving_idx:
                        n = valid_ids[k]
                        x0, y0 = prev_xy[n]; x1, y1 = curr_xy[n]

                        # 始点→終点の区間を四角でスタンプして埋める
                        dx = x1 - x0; dy = y1 - y0
                        seg_len = float(np.hypot(dx, dy))

                        if seg_len < 1e-6:
                            _fill_square(mask, x1 if include_end else x0, y1 if include_end else y0, point_radius, 255)
                        else:
                            n_interval = max(1, int(np.floor(seg_len / line_step_px)))
                            n_points = n_interval + 1

                            t0 = 0.0 if include_start else (1.0 / n_points)
                            t1 = 1.0 if include_end   else (1.0 - 1.0 / n_points)

                            if t1 < t0:
                                ts = [0.5]
                            else:
                                eff_len = max(1, int(round((t1 - t0) * n_points)))
                                ts = np.linspace(t0, t1, eff_len, endpoint=True)

                            for tt in ts:
                                xx = x0 + dx * float(tt)
                                yy = y0 + dy * float(tt)
                                _fill_square(mask, xx, yy, point_radius, 255)
        else:
            # 初回: 可視点のみ四角で点描
            v0 = vs[0]
            ok = (v0 > 0.5) & (~np.isnan(tr[0]).any(axis=1))
            for n in np.where(ok)[0]:
                x, y = tr[0, n]
                _fill_square(mask, x, y, point_radius, 255)

        # マスク後処理
        if bk > 0:
            mask = cv2.GaussianBlur(mask, (bk, bk), 0)
            _, mask = cv2.threshold(mask, 8, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # マスク面積が画像の75%以上なら無効化
        mask_ratio = np.count_nonzero(mask) / (H * W)
        if mask_ratio >= 0.75:
            mask[:] = 0  # 採用しない（真っ黒マスク）

        # 合成（RGB前提）
        if mode == "overlay":
            color = np.zeros_like(frame_rgb); color[..., 1] = 255  # 緑
            overlay = frame_rgb.copy()
            overlay[mask > 0] = color[mask > 0]
            out_rgb = cv2.addWeighted(frame_rgb, 1.0, overlay, overlay_alpha, 0.0)
        elif mode == "mask":
            if keep_color_outside:
                dark = (frame_rgb * 0.15).astype(np.uint8)
                out_rgb = dark
                out_rgb[mask > 0] = frame_rgb[mask > 0]
            else:
                out_rgb = np.zeros_like(frame_rgb)
                out_rgb[mask > 0] = frame_rgb[mask > 0]
        else:
            raise ValueError("mode は 'overlay' か 'mask' を指定してください。")

        writer.append_data(out_rgb)
        if out_frame is None:
            out_frame = out_rgb

    writer.close()

    out_path_str = str(out_final)
    print(f"[OK] 動的領域のみの動画を保存しました: {out_path_str}（codec={ok_codec}）")
    return out_path_str, out_frame

# 動画をチャンクごとに処理してCoTrackerを実行する関数
@torch.inference_mode()
def run_cotracker_chunked(
    video_path: str,
    device: str = "cuda",
    grid_size: int = 10,
    chunk_len: int = 256,
    overlap: int = 16,
    t_stride: int = 1,             # 例: 2や3で時間間引き
    use_fp16: bool = False,
    synchronize_cuda_memory: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CoTrackerを時間分割して実行し、(1, T, N, 2), (1, T, N) を返す。
    - pred_tracks_global:  正規化座標ではなく画素座標 (H, W) を想定
    - pred_visibility_global: bool(0/1)
    """
    # 1) 動画読み込み + 時間間引き
    frames = iio.imread(video_path, plugin="FFMPEG")       # [T,H,W,3]
    if t_stride > 1:
        frames = frames[::t_stride]
    T, H, W, C = frames.shape

    # 2) モデル読み込み
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    cotracker = cotracker.to(device)
    if use_fp16:
        cotracker = cotracker.half()

    # 3) 出力のための器（最初のNは不明なので、1チャンク目で確定）
    pred_tracks_global = None     # (1, T, N, 2)
    pred_vis_global    = None     # (1, T, N)

    # 4) チャンク分割のためのインデックス列
    #    窓: [s, e) （eは排他的）。オーバーラップしたまま前進
    starts = list(range(0, T, chunk_len - overlap))
    if starts and starts[-1] + chunk_len > T:
        # 最終窓がTを超えるなら末尾に合わせて再配置
        starts[-1] = max(0, T - chunk_len)
    starts = sorted(set(starts))  # 念のためユニーク＆ソート

    # 5) 最近傍対応付けヘルパ
    def _reindex_by_nn(
        prev_xy: torch.Tensor,  # [N_prev, 2]
        curr_xy: torch.Tensor,  # [N_curr, 2]
        max_dist: float = None  # 例: (H+W)/grid_size * 0.5 くらいの閾値を自動で
    ) -> torch.Tensor:
        """最近傍による対応付けでcurrのインデックスpermを返す"""
        Np = prev_xy.shape[0]
        Nc = curr_xy.shape[0]
        # 距離行列 [Np, Nc]
        D = torch.cdist(prev_xy.float(), curr_xy.float(), p=2)  # floatで安全側

        if max_dist is None:
            # 画像サイズとグリッドから自動閾値（大きめに）
            max_dist = (H + W) / max(grid_size, 2) * 0.75

        # グリーディ割当
        assigned_prev = set()
        assigned_curr = set()
        pairs = []

        # 各prev行について最良currを貪欲選択
        flat = D.view(-1)
        vals, idxs = torch.sort(flat)
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            if v > max_dist:
                break
            i = idx // Nc  # prev idx
            j = idx %  Nc  # curr idx
            if i in assigned_prev or j in assigned_curr:
                continue
            assigned_prev.add(i)
            assigned_curr.add(j)
            pairs.append((i, j))
            if len(assigned_prev) >= min(Np, Nc):
                break

        # permを構成：対応が見つかったcurrのjをprevのi順に並べる
        perm_matched = [j for (_, j) in sorted(pairs, key=lambda x: x[0])]
        unmatched = [j for j in range(Nc) if j not in assigned_curr]
        perm = perm_matched + unmatched
        return torch.tensor(perm, device=curr_xy.device, dtype=torch.long)

    # 6) メインループ
    last_end = 0
    global_N = None
    for si, s in tqdm(enumerate(starts), total=len(starts)):
        e = min(s + chunk_len, T)
        # チャンクのテンソル作成（1, t, 3, H, W）
        frames_chunk = torch.tensor(frames[s:e]).to(device, non_blocking=True)
        if use_fp16:
            video_chunk = frames_chunk.permute(0,3,1,2)[None].half()  # [1,t,3,H,W]
        else:
            video_chunk = frames_chunk.permute(0,3,1,2)[None].float()

        # 推論
        pred_tracks, pred_vis = cotracker(video_chunk, grid_size=grid_size)
        # 形状: pred_tracks [1, t, N, 2], pred_vis [1, t, N]
        _, t_local, N_local, _ = pred_tracks.shape

        # 初回：グローバル器を作る
        if pred_tracks_global is None:
            global_N = N_local
            pred_tracks_global = torch.full((1, T, global_N, 2), float("nan"), device=device, dtype=pred_tracks.dtype)
            pred_vis_global    = torch.zeros((1, T, global_N), device=device, dtype=pred_vis.dtype)

        # 2回目以降：オーバーラップ先頭でインデックス整列
        # （前窓の末尾フレーム = s、現窓の先頭フレーム = 0 を対応付け）
        if si > 0 and overlap > 0:
            # グローバル側：フレーム s（＝重なり開始のグローバル時刻）における直前トラックの座標
            prev_xy = pred_tracks_global[0, s, :, :]  # [N,2] (NaN含む)
            prev_vis= pred_vis_global[0, s, :]        # [N]
            # NaNまたは不可視は除外して対応付け作成
            valid_prev = (~torch.isnan(prev_xy).any(dim=-1)) & (prev_vis > 0.5)
            prev_xy_v = prev_xy[valid_prev]
            # 現在チャンク：先頭フレーム0の座標
            curr_xy_0 = pred_tracks[0, 0, :, :]       # [N_local, 2]
            # 対応付け
            if prev_xy_v.numel() > 0:
                perm = _reindex_by_nn(prev_xy_v, curr_xy_0)
                pred_tracks = pred_tracks[:, :, perm, :]
                pred_vis    = pred_vis[:, :, perm]

        # グローバルへ配置
        pred_tracks_global[0, s:e, :, :] = pred_tracks[0]
        pred_vis_global[0, s:e, :]      = pred_vis[0]

        # VRAM/RAMの掃除（逐次実行時のメモリ圧縮）
        del video_chunk, frames_chunk, pred_tracks, pred_vis
        if synchronize_cuda_memory and torch.cuda.is_available():
            torch.cuda.empty_cache()

        last_end = e

    return pred_tracks_global, pred_vis_global

def _nkey(s: str) -> Tuple[int, str]:
    # "chunk-000" / "file-012" などの数値でソートしたいとき用のキー
    m = re.search(r"(\d+)", s)
    return (int(m.group(1)) if m else -1, s)

def iter_chunk_file_paths(root: Path, camera_key: str) -> Iterable[Tuple[int, int, Path]]:
    """
    例: {root}/videos/{camera_key}/chunk-000/file-000.mp4 を総なめ
    ※ 語尾に "_" が含まれるファイル名 (例: file-000_.mp4) はスキップ
    """

    # --- 正規化（ありがちな混入を除去） ---
    # root は ~ を展開、camera_key は改行・CR・引用符・前後空白を除去
    root = Path(os.path.expanduser(str(root)))
    ck = (
        camera_key
        .replace("\r", "")
        .replace("\n", "")
        .strip()
        .strip('"')
        .strip("'")
        .strip("/")   # 末尾/が紛れ込んだ場合の保険
    )

    base = root / "videos" / ck

    # --- 存在チェック（診断情報を添えて） ---
    if not base.exists():
        # 近い候補（videos/ 直下にあるサブディレクトリ一覧）
        videos_dir = root / "videos"
        candidates = []
        if videos_dir.exists():
            for p in sorted([d for d in videos_dir.iterdir() if d.is_dir()]):
                candidates.append(p.name)
        msg = [
            f"not found: {base}",
            f"  - root        = {root}",
            f"  - camera_key  = {repr(camera_key)}  -> normalized: {repr(ck)}",
        ]
        if candidates:
            # camera_key に含まれるトークンで簡易フィルタ
            toks = [t for t in ck.split(".") if t]
            similar = [c for c in candidates if all(t in c for t in toks[-2:])] if len(toks)>=2 else []
            msg.append(f"  - videos/ 下の候補: {candidates}")
            if similar:
                msg.append(f"  - 類似候補: {similar}")
        else:
            msg.append(f"  - {videos_dir} 自体が存在しません")
        raise FileNotFoundError("\n".join(msg))

    # --- chunk-*/file-*.mp4 を列挙 ---
    for chunk_dir in sorted((p for p in base.glob("chunk-*") if p.is_dir()), key=lambda p: _nkey(p.name)):
        m = re.search(r'chunk-(\d+)', chunk_dir.name)
        if not m:
            continue
        cidx = int(m.group(1))

        for fpath in sorted(chunk_dir.glob("file-*.mp4"), key=lambda p: _nkey(p.name)):
            # "file-000_.mp4" のように "file-" 以降に "_" が含まれるものはスキップ
            stem_after = fpath.stem[len("file-"):] if fpath.stem.startswith("file-") else fpath.stem
            if "_" in stem_after:
                continue

            m2 = re.search(r'file-(\d+)', fpath.stem)
            if not m2:
                continue
            fidx = int(m2.group(1))
            yield cidx, fidx, fpath

def safe_t_stride(v: Optional[int]) -> int:
    # 0 / None → 1（全フレーム扱い）
    return 1 if (v is None or v == 0) else v

def build_video_feature_meta(h: int, w: int, fps: float = 30.0) -> Dict[str, Any]:
    # codec/pix_fmt は環境依存なので最低限の共通情報だけ埋める
    return {
        "dtype": "video",
        "shape": [h, w, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": h,
            "video.width": w,
            "video.codec": "av1",       # 実際の書き出しに合わせて必要なら変更
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }

# ====== ここから CoTracker 全動画処理ラッパー ======
def process_all_cotracker_dynamic(
    repo_id: str,
    camera_key: str = "observation.images.front",
    # --- CoTracker推論パラメータ ---
    device: str = "cuda",
    grid_size: int = 40,
    chunk_len: int = 128,
    overlap: int = 16,
    t_stride: int = 0,          # 0/Noneは内部で1に矯正（全フレーム）
    disp_thresh: float = 1.25,  # 動的判定しきい値[px]
    # --- 可視化・保存パラメータ ---
    normalized: bool = False,   # tracksが0-1正規化ならTrue
    mode: str = "mask",         # "mask" or "overlay"
    fps: float = 30.0,
    # --- 出力先 ---
    new_repo_suffix: str = "_with_dynamic",
):
    """
    データセット配下の {camera_key} の全動画に対して
      1) CoTracker推定（動的領域）
      2) 動的領域のみの動画を書き出し
    を順次実行する。
    """
    # ルート（HF_LEROBOT_HOME は既存のグローバル想定）
    root = Path(f"{HF_LEROBOT_HOME}/{repo_id}")

    # t_stride を安全化
    t_stride = safe_t_stride(t_stride)

    for cidx, fidx, video_path in iter_chunk_file_paths(root, camera_key):
        print(f"[cotracker] chunk={cidx:03d} file={fidx:03d} : {video_path}")

        # --- 1) CoTracker（動的領域検出）---
        tracks, vis = run_cotracker_chunked(
            str(video_path),
            device=device,
            grid_size=grid_size,
            chunk_len=chunk_len,
            overlap=overlap,
            t_stride=t_stride,  # 1=全フレーム、>1=間引き
        )
        print("  tracks/vis:", tracks.shape, vis.shape)  # (1, T', N, 2), (1, T', N)

        # --- 2) 動的領域のみを描画して保存 ---
        out_path, out_frame = save_dynamic_regions_video(
            str(video_path),
            tracks,
            vis,
            out_path=None,       # Noneで {file-XXX}_dynamic.mp4 を同じ場所へ
            t_stride=t_stride,   # run時と合わせる
            normalized=normalized,
            mode=mode,           # "mask" なら動的領域のみ残す
            disp_thresh=disp_thresh,
        )
        print("  dynamic vis saved ->", out_path)
        print("  out_frame:", out_frame.shape)  # (T, H, W, C)