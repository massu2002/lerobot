
export HF_LEROBOT_HOME="/home/masuoka/lerobot/finetune_dataset"

# ファインチューニング実行設定
optimizer_lr=4e-4
batch_size=32
steps=40000
save_freq=10000
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
  --seed=${seed} \
  --wandb.enable=true \