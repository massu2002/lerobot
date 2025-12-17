
export HF_LEROBOT_HOME="/home/masuoka/lerobot/finetune_dataset"

# ファインチューニングデータセット設定
dataset_name="task_1"
repo_id="${HF_LEROBOT_HOME}/${dataset_name}"
job_name="pi0_${dataset_name}"

# ファインチューニング実行設定
batch_size=32
steps=3000
save_freq=1000
seed=42

# 実行（事前学習済みモデルからファインチューニング）
lerobot-train \
  --policy.type=pi0 \
  --policy.pretrained_path=lerobot/pi0_base \
  --dataset.repo_id=${repo_id} \
  --rename_map='{
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2"
  }' \
  --output_dir=/home/masuoka/lerobot/outputs/pi0/${dataset_name}/seed_${seed} \
  --job_name=${job_name} \
  --policy.push_to_hub=false \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --steps=${steps} \
  --save_freq=${save_freq} \
  --batch_size=${batch_size} \
  --dataset.video_backend=pyav \
  --seed=${seed} \
  --wandb.enable=true