#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import os
import csv
from collections.abc import Callable
from typing import Any
from collections import Counter

from lerobot.utils.utils import format_big_number


class AverageMeter:
    """
    Computes and stores the average and current value
    Adapted from https://github.com/pytorch/examples/blob/main/imagenet/main.py
    """

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class MetricsTracker:
    """
    A helper class to track and log metrics over time.

    Usage pattern:

    ```python
    # initialize, potentially with non-zero initial step (e.g. if resuming run)
    metrics = {"loss": AverageMeter("loss", ":.3f")}
    train_metrics = MetricsTracker(cfg, dataset, metrics, initial_step=step)

    # update metrics derived from step (samples, episodes, epochs) at each training step
    train_metrics.step()

    # update various metrics
    loss = policy.forward(batch)
    train_metrics.loss = loss

    # display current metrics
    logging.info(train_metrics)

    # export for wandb
    wandb.log(train_metrics.to_dict())

    # reset averages after logging
    train_metrics.reset_averages()
    ```
    """

    __keys__ = [
        "_batch_size",
        "_num_frames",
        "_avg_samples_per_ep",
        "metrics",
        "steps",
        "samples",
        "episodes",
        "epochs",
        "accelerator",
        "_csv_path",
        "_csv_interval",
        "_csv_enabled",
        "_num_episodes",
        "_episode_counter",
        "_csv_fieldnames",
        "_is_main_process",
        "_extract_episode_indices_fn",
        "_to_csv_cell_fn",
        "_jain_fairness_index_fn",
    ]

    def __init__(
        self,
        batch_size: int,
        num_frames: int,
        num_episodes: int,
        metrics: dict[str, AverageMeter],
        initial_step: int = 0,
        accelerator: Callable | None = None,
        csv_path: str | None = None,
        csv_interval: int = 0,
        is_main_process: bool = True,
        extract_episode_indices_fn: Callable[[Any], list[int]] | None = None,
        to_csv_cell_fn: Callable[[Any], Any] | None = None,
        jain_fairness_index_fn: Callable[[Counter, int], float] | None = None,
        stable_field_order: bool = True,
    ):
        self.__dict__.update(dict.fromkeys(self.__keys__))
        self._batch_size = batch_size
        self._num_frames = num_frames
        self._avg_samples_per_ep = num_frames / num_episodes
        self.metrics = metrics

        self.steps = initial_step
        # A sample is an (observation,action) pair, where observation and action
        # can be on multiple timestamps. In a batch, we have `batch_size` number of samples.
        self.samples = self.steps * self._batch_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames
        self.accelerator = accelerator
        
        self._is_main_process = is_main_process
        self._csv_path = csv_path
        self._csv_interval = int(csv_interval) if csv_interval else 0
        self._csv_enabled = (self._csv_path is not None) and (self._csv_interval > 0)
        self._num_episodes = int(num_episodes)
        self._episode_counter = Counter()

        self._extract_episode_indices_fn = extract_episode_indices_fn
        self._to_csv_cell_fn = to_csv_cell_fn
        self._jain_fairness_index_fn = jain_fairness_index_fn

        # 列順固定用（初回に確定して以降維持）
        self._csv_fieldnames = None if stable_field_order else []
        if self._csv_enabled:
            os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)

    def __getattr__(self, name: str) -> int | dict[str, AverageMeter] | AverageMeter | Any:
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.metrics:
            return self.metrics[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            super().__setattr__(name, value)
        elif name in self.metrics:
            self.metrics[name].update(value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def step(self) -> None:
        """
        Updates metrics that depend on 'step' for one step.
        """
        self.steps += 1
        self.samples += self._batch_size * (self.accelerator.num_processes if self.accelerator else 1)
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames
        
    def on_batch(self, batch: Any) -> None:
        """episode の出現回数をカウントする（Jain用）。"""
        if not self._csv_enabled:
            return
        if self._extract_episode_indices_fn is None:
            return
        for ep in self._extract_episode_indices_fn(batch):
            if 0 <= ep < self._num_episodes:
                self._episode_counter[ep] += 1
                
    def _build_row(self, output_dict: dict[str, Any], step: int) -> dict[str, Any]:
        to_cell = self._to_csv_cell_fn or (lambda x: x)
        row = {k: to_cell(v) for k, v in output_dict.items()}
        row["step"] = step

        if self._jain_fairness_index_fn is not None:
            row["jain_fairness_index"] = self._jain_fairness_index_fn(self._episode_counter, self._num_episodes)
        return row

    def maybe_write_csv(self, step: int, use_avg: bool = True) -> None:
        """interval条件を満たすときだけCSVに追記する（trackerのmetricsを保存）。"""
        if (not self._csv_enabled) or (not self._is_main_process):
            return
        if step % self._csv_interval != 0:
            return

        # 1) trackerの値をそのまま dict 化
        row = self.to_dict(use_avg=use_avg)  # steps/samples/episodes/epochs + metrics

        # 2) Jain を追加
        if self._jain_fairness_index_fn is not None:
            row["jain_fairness_index"] = self._jain_fairness_index_fn(self._episode_counter, self._num_episodes)

        # 3) CSVセルに変換
        to_cell = self._to_csv_cell_fn or (lambda x: x)
        row = {k: to_cell(v) for k, v in row.items()}

        write_header = not os.path.exists(self._csv_path)

        # 列順を初回に確定して固定
        if self._csv_fieldnames is None:
            # steps/samples/... を先に置きたければここで並べ替え
            core = ["steps", "samples", "episodes", "epochs"]
            rest = sorted([k for k in row.keys() if k not in core])
            self._csv_fieldnames = core + rest

        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def __str__(self) -> str:
        display_list = [
            f"step:{format_big_number(self.steps)}",
            # number of samples seen during training
            f"smpl:{format_big_number(self.samples)}",
            # number of episodes seen during training
            f"ep:{format_big_number(self.episodes)}",
            # number of time all unique samples are seen
            f"epch:{self.epochs:.2f}",
            *[str(m) for m in self.metrics.values()],
        ]
        return " ".join(display_list)

    def to_dict(self, use_avg: bool = True) -> dict[str, int | float]:
        """
        Returns the current metric values (or averages if `use_avg=True`) as a dict.
        """
        return {
            "steps": self.steps,
            "samples": self.samples,
            "episodes": self.episodes,
            "epochs": self.epochs,
            **{k: m.avg if use_avg else m.val for k, m in self.metrics.items()},
        }

    def reset_averages(self) -> None:
        """Resets average meters."""
        for m in self.metrics.values():
            m.reset()
            
    def update_metric(self, key: str, value: Any, n: int = 1) -> None:
        """key文字列で AverageMeter を更新する（'recon_loss/foo' みたいなキー対応）"""
        if key not in self.metrics:
            raise KeyError(f"Unknown metric key: {key}. Please register it in `metrics` first.")
        self.metrics[key].update(value, n=n)

    def update_metrics(self, values: dict[str, Any], n: int = 1, ignore_unknown: bool = True) -> None:
        """まとめて更新。未知キーは ignore_unknown=True なら無視。"""
        for k, v in values.items():
            if k in self.metrics:
                self.metrics[k].update(v, n=n)
            else:
                if not ignore_unknown:
                    raise KeyError(f"Unknown metric key: {k}")
