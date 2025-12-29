#!/usr/bin/bash

export PATH=/mnt/qnap5/masuoka/conda_envs/cotvla/bin:\$PATH;
export LD_LIBRARY_PATH=/mnt/qnap5/masuoka/conda_envs/cotvla/lib:\$LD_LIBRARY_PATH;

export HF_LEROBOT_HOME="/mnt/qnap5/masuoka/lerobot/finetune_dataset"

# 学習設定
phase="finetune"
optimizer_lr=7e-4
num_processes=4
GLOBAL_BATCH_SIZE=64
batch_size=$(( GLOBAL_BATCH_SIZE / num_processes ))
steps=40000
save_freq=20000
obs_pred=false
mode="vanilla"
seed=42

# 事前学習データセット設定
dataset_name="task_1"
pretiran_steps=200000
repo_id="${HF_LEROBOT_HOME}/${dataset_name}"
job_name="vanilla_vla_${dataset_name}_${optimizer_lr}"

# 実行（事前学習あり）
accelerate launch \
  --multi_gpu \
  --num_processes=${num_processes} \
  $(command -v lerobot-train) \
  --policy.path=../outputs/cotvla/pretraindata_v1_${mode}/seed_42/checkpoints/${pretiran_steps}/pretrained_model \
  --dataset.repo_id=${repo_id} \
  --dataset.video_backend=pyav \
  --output_dir=../outputs/vanilla_vla/${dataset_name}/${optimizer_lr}/seed_${seed} \
  --policy.phase=${phase} \
  --policy.obs_pred=${obs_pred} \
  --policy.optimizer_lr=${optimizer_lr} \
  --policy.push_to_hub=false \
  --steps=${steps} \
  --save_freq=${save_freq} \
  --job_name=${job_name} \
  --batch_size=${batch_size} \
  --seed=${seed} \
  --wandb.enable=true

# export HF_LEROBOT_HOME="/home/masuoka/lerobot/finetune_dataset"

# # 学習設定
# phase="finetune"
# optimizer_lr=2e-3
# batch_size=64
# steps=40000
# save_freq=20000
# obs_pred=false
# mode="vanilla"
# seed=42

# # 事前学習データセット設定
# dataset_name="task_1"
# pretiran_steps=200000
# repo_id="${HF_LEROBOT_HOME}/${dataset_name}"
# job_name="vanilla_vla_${dataset_name}_${optimizer_lr}"

# # 実行（事前学習あり）
# lerobot-train \
#   --policy.path=../outputs/cotvla/pretraindata_v1_${mode}/seed_42/checkpoints/${pretiran_steps}/pretrained_model \
#   --dataset.repo_id=${repo_id} \
#   --dataset.video_backend=pyav \
#   --output_dir=../outputs/vanilla_vla/${dataset_name}/${optimizer_lr}/seed_${seed} \
#   --policy.phase=${phase} \
#   --policy.obs_pred=${obs_pred} \
#   --policy.optimizer_lr=${optimizer_lr} \
#   --policy.push_to_hub=false \
#   --steps=${steps} \
#   --save_freq=${save_freq} \
#   --job_name=${job_name} \
#   --batch_size=${batch_size} \
#   --seed=${seed} \
#   --wandb.enable=true