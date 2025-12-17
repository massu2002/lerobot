#!/bin/bash

export PATH=/mnt/qnap5/masuoka/conda_envs/cotvla/bin:\$PATH;
export LD_LIBRARY_PATH=/mnt/qnap5/masuoka/conda_envs/cotvla/lib:\$LD_LIBRARY_PATH;

export HF_LEROBOT_HOME="/data_ssd/robot_data"

# 事前学習データセット設定
dataset_name="pretraindata_v1"
repo_id="${HF_LEROBOT_HOME}/${dataset_name}"
job_name="cotvla_${dataset_name}"

# 学習設定
phase="pretrain"
optimizer_lr=1e-3
batch_size=8
steps=200000
save_freq=50000
obs_pred=true
if [ "$obs_pred" = true ]; then
    mode="visual_cot"
else
    mode="vanilla"
fi
img_recon_loss_weight=0.25
seed=42

# 実行（事前学習あり）
accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  $(command -v lerobot-train) \
  --policy.type=cotvla \
  --dataset.repo_id=${repo_id} \
  --rename_map='{
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2"
  }' \
  --dataset.video_backend=pyav \
  --output_dir=./outputs/cotvla/${dataset_name}_${mode}/weight_${img_recon_loss_weight}/seed_${seed} \
  --policy.phase=${phase} \
  --policy.obs_pred=${obs_pred} \
  --policy.img_recon_loss_weight=${img_recon_loss_weight} \
  --policy.optimizer_lr=${optimizer_lr} \
  --policy.push_to_hub=false \
  --steps=${steps} \
  --save_freq=${save_freq} \
  --job_name=${job_name}_${mode}_weight_${img_recon_loss_weight} \
  --batch_size=${batch_size} \
  --seed=${seed} \
  --wandb.enable=true