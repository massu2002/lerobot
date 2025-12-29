
export HF_LEROBOT_HOME="/home/masuoka/lerobot/finetune_dataset"

# ファインチューニング実行設定
batch_size=64
steps=20000
save_freq=10000
seed=42

# ファインチューニングデータセット設定
dataset_name="task_2"
repo_id="${HF_LEROBOT_HOME}/${dataset_name}"
job_name="smolvla_${dataset_name}"

# 実行（事前学習済みモデルからファインチューニング）
lerobot-train \
  --policy.path=../smolvla_base \
  --dataset.repo_id=${repo_id} \
  --batch_size=${batch_size} \
  --steps=${steps} \
  --save_freq=${save_freq} \
  --dataset.video_backend=pyav \
  --output_dir=../outputs/smolvla/${dataset_name}/seed_${seed} \
  --policy.push_to_hub=false \
  --steps=${steps} \
  --save_freq=${save_freq} \
  --job_name=${job_name} \
  --wandb.enable=true