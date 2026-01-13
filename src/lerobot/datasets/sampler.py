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
import math
import torch
import torch.distributed as dist
from typing import Iterator, List, Optional, Sequence

class DistributedEpisodeAwareSampler(torch.utils.data.Sampler[int]):
    def __init__(
        self,
        dataset_from_indices: Sequence[int],
        dataset_to_indices: Sequence[int],
        episode_indices_to_use: Optional[Sequence[int]] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        horizon: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        if len(dataset_from_indices) != len(dataset_to_indices):
            raise ValueError("dataset_from_indices and dataset_to_indices must have same length")

        indices: List[int] = []
        use_set = set(int(x) for x in episode_indices_to_use) if episode_indices_to_use is not None else None

        for ep_idx, (start, end) in enumerate(zip(dataset_from_indices, dataset_to_indices, strict=True)):
            if use_set is not None and ep_idx not in use_set:
                continue
            start = int(start)
            end = int(end)

            eff_start = start + drop_n_first_frames
            eff_end = end - drop_n_last_frames  # exclusive
            last_start = eff_end - horizon - 1

            if last_start >= eff_start:
                indices.extend(range(eff_start, last_start + 1))

        self.indices = indices

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _ordered_indices(self) -> List[int]:
        if not self.shuffle:
            return list(self.indices)

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)   # ★全rank同一の並び
        perm = torch.randperm(len(self.indices), generator=g).tolist()
        return [self.indices[i] for i in perm]

    def __iter__(self) -> Iterator[int]:
        if len(self.indices) == 0:
            return iter(())

        ordered = self._ordered_indices()

        # ★ 全rankで同じサンプル数になるよう padding
        total = len(ordered)
        total_size = ((total + self.world_size - 1) // self.world_size) * self.world_size
        if total_size > total:
            ordered += ordered[: (total_size - total)]

        # ★ shard（全rank同数 = total_size/world_size）
        shard = ordered[self.rank:total_size:self.world_size]
        return iter(shard)

    def __len__(self) -> int:
        # ★ padding後に各rankへ配られる要素数
        total = len(self.indices)
        total_size = ((total + self.world_size - 1) // self.world_size) * self.world_size
        return total_size // self.world_size