# !/bin/bash
export HF_LEROBOT_HOME="/home/masuoka/lerobot/finetune_dataset"

# ファインチューニング実行設定
steps=100000
save_freq=25000
optimizer_lr=3e-4
scheduler_warmup_steps=$(( steps / 100 ))
scheduler_decay_steps=$(( steps * 40 / 100 ))
scheduler_decay_lr=$(python -c "print(float('${optimizer_lr}')/10)")
optimizer_weight_decay=5e-4
optimizer_grad_clip_norm=5.0
batch_size=32
steps=100000
save_freq=25000
seed=42

# ファインチューニングデータセット設定
dataset_name="task_1"
repo_id="${HF_LEROBOT_HOME}/${dataset_name}"
job_name="pi0_${dataset_name}_${optimizer_lr}"

# 実行（事前学習済みモデルからファインチューニング）
lerobot-train \
  --policy.type=pi0 \
  --policy.pretrained_path=pepijn223/pi0_base \
  --dataset.repo_id=${repo_id} \
  --output_dir=/home/masuoka/lerobot/outputs/pi0/${dataset_name}/${optimizer_lr}/seed_${seed} \
  --job_name=${job_name} \
  --policy.optimizer_lr=${optimizer_lr} \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --steps=${steps} \
  --save_freq=${save_freq} \
  --batch_size=${batch_size} \
  --dataset.video_backend=pyav \
  --seed=${seed}