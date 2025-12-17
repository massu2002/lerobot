
export HF_LEROBOT_HOME="/home/masuoka/lerobot/finetune_dataset"

# 事前学習データセット設定
dataset_name="task_1"
repo_id="${HF_LEROBOT_HOME}/${dataset_name}"
job_name="cotvla_${dataset_name}"

# 学習設定
phase="finetune"
optimizer_lr=1e-4
batch_size=64
steps=60000
save_freq=30000
obs_pred=true
if [ "$obs_pred" = true ]; then
    mode="visual_cot"
else
    mode="vanilla"
fi
img_recon_loss_weight=0.25
seed=42

# 実行（事前学習あり）
lerobot-train \
  --policy.path=./outputs/cotvla/pretraindata_v1_visual_cot/weight_${img_recon_loss_weight}/seed_42/checkpoints/050000/pretrained_model \
  --dataset.repo_id=${repo_id} \
  --dataset.video_backend=pyav \
  --output_dir=./outputs/cotvla_${img_recon_loss_weight}/${dataset_name}/seed_${seed} \
  --policy.phase=${phase} \
  --policy.obs_pred=${obs_pred} \
  --policy.img_recon_loss_weight=${img_recon_loss_weight} \
  --policy.optimizer_lr=${optimizer_lr} \
  --policy.push_to_hub=false \
  --steps=${steps} \
  --save_freq=${save_freq} \
  --job_name=${job_name} \
  --batch_size=${batch_size} \
  --seed=${seed} \
  --wandb.enable=true