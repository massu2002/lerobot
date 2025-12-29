HF_USER=atsuto-1

# Press Right Arrow (→): Early stop the current episode or reset time and move to the next.
# Press Left Arrow (←): Cancel the current episode and re-record it.
# Press Escape (ESC): Immediately stop the session, encode videos, and upload the dataset.


task_number=1
model_type=cotvla

declare -A TASKS=(
  [1]="put the green block in the white box"
  [2]="put the red block in the black box"
  [3]="put green block in white box, red block in black box"
  [4]="open drawer and put green block in"
  [5]="open drawer and put red block in"
)

TASK_PROMPT="${TASKS[$task_number]}"

if [ -z "$TASK_PROMPT" ]; then
  echo "Error: invalid task_number: $task_number"
  exit 1
fi

echo "TASK_PROMPT: $TASK_PROMPT"
echo "MODEL_NAME: $model_type"

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}, 
    front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=${HF_USER}/eval_task_${task_number}_${model_type} \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=40 \
    --dataset.single_task="$TASK_PROMPT" \
    --policy.path=../vla_models/task_${task_number}/${model_type}/pretrained_model \
    --model_type=${model_type} \
    --resume=True