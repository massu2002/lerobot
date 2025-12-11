from typing import Any, Tuple

import torch

from lerobot.processor import (
    PolicyProcessorPipeline,
    PolicyAction,
    RenameObservationsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
    TokenizerProcessorStep,
    ClipTokenizerProcessorStep
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
)

# SmolVLA と同じ「言語末尾に改行を足す」ステップをそのまま使う想定
from lerobot.policies.smolvla.processor_smolvla import SmolVLANewLineProcessor

# あなたが定義した CoTVLAConfig
from lerobot.policies.cotvla.configuration_cotvla import CoTVLAConfig


def make_cotvla_pre_post_processors(
    config: CoTVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> Tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    CoTVLA ポリシー用の前処理 / 後処理パイプラインを構築する。

    前処理:
      1. 観測名を rename（今は no-op）。
      2. バッチ次元を追加。
      3. 言語タスク記述の末尾に改行を追加。
      4. 言語を tokenizer でトークナイズ。
      5. データを指定デバイスへ移動。
      6. dataset_stats に基づいて正規化。

    後処理:
      1. 出力 action を元スケールに unnormalize。
      2. CPU に戻す。
    """

    # CoTVLA で使う tokenizer 名：
    # - config.tokenizer_name があればそれを使う
    # - なければ gpt2 (CoTVLA の GPT-2 ベース) をデフォルトにする
    tokenizer_name = getattr(config, "tokenizer_name", None)
    if tokenizer_name is None:
        tokenizer_name = "gpt2-medium" # CoTVLA のデフォルト tokenizer

    input_steps = [
        # SmolVLA と同様、pretrained 設定と互換をとるための rename（今は空）
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        SmolVLANewLineProcessor(),
        ClipTokenizerProcessorStep(
            task_key="task",          # complementary_data["task"] を使う前提
            max_length=77,            # 一応指定するが、内部で 77 に揃える
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
