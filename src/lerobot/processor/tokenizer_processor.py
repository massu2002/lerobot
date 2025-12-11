#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
This script defines a processor for tokenizing natural language instructions from an environment transition.

It uses a tokenizer from the Hugging Face `transformers` library to convert task descriptions (text) into
token IDs and attention masks, which are then added to the observation dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Dict

from transformers import CLIPTokenizer

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import _transformers_available

from .core import EnvTransition, TransitionKey
from .pipeline import ObservationProcessorStep, ProcessorStepRegistry

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import AutoTokenizer
else:
    AutoTokenizer = None


@dataclass
@ProcessorStepRegistry.register(name="tokenizer_processor")
class TokenizerProcessorStep(ObservationProcessorStep):
    """
    Processor step to tokenize a natural language task description.

    This step extracts a task string from the `complementary_data` of an `EnvTransition`,
    tokenizes it using a Hugging Face `transformers` tokenizer, and adds the resulting
    token IDs and attention mask to the `observation` dictionary.

    Requires the `transformers` library to be installed.

    Attributes:
        tokenizer_name: The name of a pretrained tokenizer from the Hugging Face Hub (e.g., "bert-base-uncased").
        tokenizer: A pre-initialized tokenizer object. If provided, `tokenizer_name` is ignored.
        max_length: The maximum length to pad or truncate sequences to.
        task_key: The key in `complementary_data` where the task string is stored.
        padding_side: The side to pad on ('left' or 'right').
        padding: The padding strategy ('max_length', 'longest', etc.).
        truncation: Whether to truncate sequences longer than `max_length`.
        input_tokenizer: The internal tokenizer instance, loaded during initialization.
    """

    tokenizer_name: str | None = None
    tokenizer: Any | None = None  # Use `Any` for compatibility without a hard dependency
    max_length: int = 512
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    # Internal tokenizer instance (not part of the config)
    input_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Initializes the tokenizer after the dataclass is created.

        It checks for the availability of the `transformers` library and loads the tokenizer
        either from a provided object or by name from the Hugging Face Hub.

        Raises:
            ImportError: If the `transformers` library is not installed.
            ValueError: If neither `tokenizer` nor `tokenizer_name` is provided.
        """
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'` to use TokenizerProcessorStep."
            )

        if self.tokenizer is not None:
            # Use provided tokenizer object directly
            self.input_tokenizer = self.tokenizer
        elif self.tokenizer_name is not None:
            if AutoTokenizer is None:
                raise ImportError("AutoTokenizer is not available")
            self.input_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self.input_tokenizer.pad_token is None:
                self.input_tokenizer.pad_token = self.input_tokenizer.eos_token
        else:
            raise ValueError(
                "Either 'tokenizer' or 'tokenizer_name' must be provided. "
                "Pass a tokenizer object directly or a tokenizer name to auto-load."
            )

    def get_task(self, transition: EnvTransition) -> list[str] | None:
        """
        Extracts the task description(s) from the transition's complementary data.

        Args:
            transition: The environment transition.

        Returns:
            A list of task strings, or None if the task key is not found or the value is None.
        """
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            raise ValueError("Complementary data is None so no task can be extracted from it")

        task = complementary_data[self.task_key]
        if task is None:
            raise ValueError("Task extracted from Complementary data is None")

        # Standardize to a list of strings for the tokenizer
        if isinstance(task, str):
            return [task]
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            return task

        return None

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Tokenizes the task description and adds it to the observation dictionary.

        This method retrieves the task, tokenizes it, moves the resulting tensors to the
        same device as other data in the transition, and updates the observation.

        Args:
            observation: The original observation dictionary.

        Returns:
            The updated observation dictionary including token IDs and an attention mask.
        """
        task = self.get_task(self.transition)
        if task is None:
            raise ValueError("Task cannot be None")

        # Tokenize the task (this will create CPU tensors)
        tokenized_prompt = self._tokenize_text(task)

        # Detect the device from existing tensors in the transition to ensure consistency
        target_device = self._detect_device(self.transition)

        # Move new tokenized tensors to the detected device
        if target_device is not None:
            tokenized_prompt = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized_prompt.items()
            }

        # Create a new observation dict to avoid modifying the original in place
        new_observation = dict(observation)

        # Add tokenized data to the observation
        new_observation[OBS_LANGUAGE_TOKENS] = tokenized_prompt["input_ids"]
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = tokenized_prompt["attention_mask"].to(dtype=torch.bool)

        return new_observation

    def _detect_device(self, transition: EnvTransition) -> torch.device | None:
        """
        Detects the torch.device from existing tensors in the transition.

        It checks tensors in the observation dictionary first, then the action tensor.

        Args:
            transition: The environment transition.

        Returns:
            The detected `torch.device`, or None if no tensors are found.
        """
        # Check observation tensors first (most likely place to find tensors)
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation:
            for value in observation.values():
                if isinstance(value, torch.Tensor):
                    return value.device

        # Fallback to checking the action tensor
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            return action.device

        return None  # No tensors found, default will be CPU

    def _tokenize_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        """
        A wrapper around the tokenizer call.

        Args:
            text: A string or list of strings to tokenize.

        Returns:
            A dictionary containing tokenized 'input_ids' and 'attention_mask' as PyTorch tensors.
        """
        return self.input_tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="pt",
        )

    def get_config(self) -> dict[str, Any]:
        """
        Returns the serializable configuration of the processor.

        Note: The tokenizer object itself is not serialized. If the processor was initialized
        with a tokenizer name, that name will be included in the config.

        Returns:
            A dictionary with the processor's configuration parameters.
        """
        config = {
            "max_length": self.max_length,
            "task_key": self.task_key,
            "padding_side": self.padding_side,
            "padding": self.padding,
            "truncation": self.truncation,
        }

        # Only save tokenizer_name if it was used to create the tokenizer
        if self.tokenizer_name is not None and self.tokenizer is None:
            config["tokenizer_name"] = self.tokenizer_name

        return config

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Adds feature definitions for the language tokens and attention mask.

        This updates the policy features dictionary to include the new data added to the
        observation, ensuring downstream components are aware of their shape and type.

        Args:
            features: The dictionary of existing policy features.

        Returns:
            The updated dictionary of policy features.
        """
        # Add a feature for the token IDs if it doesn't already exist
        if OBS_LANGUAGE_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        # Add a feature for the attention mask if it doesn't already exist
        if OBS_LANGUAGE_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        return features

@dataclass
class ClipTokenizerProcessorStep(ObservationProcessorStep):
    """
    CoTVLA / CLIP 用のトークナイズ ProcessorStep（Hugging Face 版）。

    - complementary_data['task'] からタスク文字列を取り出す
    - CLIPTokenizer で input_ids / attention_mask を生成
    - OBS_LANGUAGE_TOKENS / OBS_LANGUAGE_ATTENTION_MASK を observation に追加
    """

    task_key: str = "task"
    max_length: int = 77  # デフォルト。__post_init__ で tokenizer に合わせて上書き
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # 内部で使う tokenizer インスタンス
    _tokenizer: CLIPTokenizer = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Hugging Face の CLIPTokenizer をロード
        self._tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_name)

        # tokenizer 側の max_length に合わせる（通常 77）
        tokenizer_max_len = getattr(self._tokenizer, "model_max_length", None)
        if tokenizer_max_len is not None and tokenizer_max_len > 0:
            if self.max_length != tokenizer_max_len:
                print(
                    f"[ClipTokenizerProcessorStep] max_length={self.max_length} → "
                    f"tokenizer.model_max_length={tokenizer_max_len} に揃えます。"
                )
            self.max_length = tokenizer_max_len

    # ---- タスク文字列の取得 ----
    def get_task(self, transition: EnvTransition) -> List[str]:
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            raise ValueError("Complementary data is None so no task can be extracted from it")

        task = complementary_data.get(self.task_key, None)
        if task is None:
            raise ValueError(f"Task '{self.task_key}' is None in complementary data")

        # string or list[str] を許容
        if isinstance(task, str):
            return [task]
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            return task
        else:
            raise ValueError(f"Task must be str or list[str], got {type(task)}")

    # ---- device の検出（元実装と同じロジック） ----
    def _detect_device(self, transition: EnvTransition) -> torch.device | None:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation:
            for value in observation.values():
                if isinstance(value, torch.Tensor):
                    return value.device
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            return action.device
        return None

    # ---- CLIPTokenizer でのトークナイズ ----
    def _tokenize_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Hugging Face の CLIPTokenizer を使って
        - input_ids: [B, L]
        - attention_mask: [B, L]
        を生成する。
        """
        enc = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]          # [B, L], long
        attention_mask = enc["attention_mask"]  # [B, L], long (1: 有効, 0: pad)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    # ---- Pipeline から呼ばれるメイン処理 ----
    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        task 文字列を CLIP トークナイザで token 化し、
        OBS_LANGUAGE_TOKENS / OBS_LANGUAGE_ATTENTION_MASK を observation に追加。
        """
        texts = self.get_task(self.transition)  # List[str]

        tokenized = self._tokenize_text(texts)

        # 既存テンソルの device を見て、そこに合わせる
        target_device = self._detect_device(self.transition)
        if target_device is not None:
            tokenized = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized.items()
            }

        new_observation = dict(observation)
        new_observation[OBS_LANGUAGE_TOKENS] = tokenized["input_ids"]  # [B, L]
        # attention_mask は bool に変換しておく
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = tokenized["attention_mask"].to(dtype=torch.bool)  # [B, L]

        return new_observation

    # ---- 特徴量定義の更新（元 TokenizerProcessorStep の transform_features と同等） ----
    def transform_features(
        self, features: Dict[PipelineFeatureType, Dict[str, PolicyFeature]]
    ) -> Dict[PipelineFeatureType, Dict[str, PolicyFeature]]:
        if OBS_LANGUAGE_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        if OBS_LANGUAGE_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        return features