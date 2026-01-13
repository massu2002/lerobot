
export PATH=/mnt/qnap5/masuoka/conda_envs/cotvla/bin:\$PATH;
export LD_LIBRARY_PATH=/mnt/qnap5/masuoka/conda_envs/cotvla/lib:\$LD_LIBRARY_PATH;

export HF_LEROBOT_HOME="/data_ssd/robot_data"

# 学習設定
phase="pretrain"
steps=100000
save_freq=25000
optimizer_lr=2e-3

scheduler_warmup_steps=$(( steps * 5 / 100 ))
scheduler_decay_steps=$(( steps * 95 / 100 ))
scheduler_decay_lr=$(python -c "print(float('${optimizer_lr}')/50)")

optimizer_weight_decay=1e-4
optimizer_grad_clip_norm=1.0

num_processes=4
GLOBAL_BATCH_SIZE=64
batch_size=$(( GLOBAL_BATCH_SIZE / num_processes ))

obs_pred=false
mode="vanilla"

seed=42
num_workers=8

# 学習データセット設定
dataset_name="pretraindata_v1"
repo_id="${HF_LEROBOT_HOME}/${dataset_name}"
job_name="vanilla_vla_${dataset_name}"

# 実行（事前学習あり）
accelerate launch \
  --num_processes=${num_processes} \
  $(command -v lerobot-train) \
  --policy.type=cotvla \
  --dataset.repo_id=${repo_id} \
  --rename_map='{
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2"
  }' \
  --dataset.video_backend=pyav \
  --output_dir=../outputs/pretrain/${dataset_name}/vanilla_vla/seed_${seed} \
  --policy.phase=${phase} \
  --policy.obs_pred=${obs_pred} \
  --policy.optimizer_lr=${optimizer_lr} \
  --policy.optimizer_weight_decay=${optimizer_weight_decay} \
  --policy.optimizer_grad_clip_norm=${optimizer_grad_clip_norm} \
  --policy.scheduler_warmup_steps=${scheduler_warmup_steps} \
  --policy.scheduler_decay_steps=${scheduler_decay_steps} \
  --policy.scheduler_decay_lr=${scheduler_decay_lr} \
  --policy.push_to_hub=false \
  --steps=${steps} \
  --save_freq=${save_freq} \
  --job_name=${job_name}_${mode} \
  --batch_size=${batch_size} \
  --num_workers=${num_workers} \
  --seed=${seed}