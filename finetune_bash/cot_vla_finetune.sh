#!/usr/bin/bash

export PATH=/mnt/qnap5/masuoka/conda_envs/cotvla/bin:\$PATH;
export LD_LIBRARY_PATH=/mnt/qnap5/masuoka/conda_envs/cotvla/lib:\$LD_LIBRARY_PATH;

export HF_LEROBOT_HOME="/mnt/qnap5/masuoka/lerobot/finetune_dataset"

# 学習設定
phase="finetune"
steps=50000
save_freq=10000
optimizer_lr=1e-3

scheduler_warmup_steps=$(( steps * 3 / 100 ))
scheduler_decay_steps=$(( steps * 98 / 100 ))
scheduler_decay_lr=$(python -c "print(float('${optimizer_lr}')/500)")

optimizer_weight_decay=5e-5
optimizer_grad_clip_norm=1.0

num_processes=8
GLOBAL_BATCH_SIZE=64
batch_size=$(( GLOBAL_BATCH_SIZE / num_processes ))

obs_pred=true
mode="visual_cot"

mask_weights='{"block":0.0, "robot":0.0, "background":0.0}'
img_recon_loss_weight=0.25

seed=42
num_workers=4

# 学習データセット設定
dataset_name="task_1"
pretiran_steps=200000
repo_id="${HF_LEROBOT_HOME}/${dataset_name}"
job_name="cotvla_${dataset_name}_${img_recon_loss_weight}_${optimizer_lr}"

# 実行（事前学習あり）
accelerate launch \
  --multi_gpu \
  --num_processes=${num_processes} \
  $(command -v lerobot-train) \
  --policy.path=../outputs/cotvla/pretraindata_v1_${mode}/weight_${img_recon_loss_weight}/seed_42/checkpoints/${pretiran_steps}/pretrained_model \
  --dataset.repo_id=${repo_id} \
  --dataset.video_backend=pyav \
  --output_dir=../outputs/cotvla/${dataset_name}/${img_recon_loss_weight}/${optimizer_lr}/seed_${seed} \
  --policy.phase=${phase} \
  --policy.obs_pred=${obs_pred} \
  --policy.img_recon_loss_weight=${img_recon_loss_weight} \
  --policy.optimizer_lr=${optimizer_lr} \
  --policy.scheduler_warmup_steps=${scheduler_warmup_steps} \
  --policy.scheduler_decay_steps=${scheduler_decay_steps} \
  --policy.scheduler_decay_lr=${scheduler_decay_lr} \
  --policy.optimizer_weight_decay=${optimizer_weight_decay} \
  --policy.optimizer_grad_clip_norm=${optimizer_grad_clip_norm} \
  --policy.push_to_hub=false \
  --policy.mask_weights="${mask_weights}" \
  --steps=${steps} \
  --save_freq=${save_freq} \
  --job_name=${job_name} \
  --batch_size=${batch_size} \
  --num_workers=${num_workers} \
  --seed=${seed}