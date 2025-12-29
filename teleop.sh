HF_USER=atsuto-1

# ポート確認
# ls /dev/ttyACM*

# カメラ確認
# lerobot-find-cameras opencv

# データを取らないデモ用のテレオペ
# フォロワーとリーダーのポート番号に注意
# キャッシュファイルにあるjsonのキャリブレイト設定を読み込んで実行
# lerobot-teleoperate \
#     --robot.type=so101_follower \
#     --robot.port=/dev/ttyACM1 \
#     --robot.id=my_awesome_follower_arm \
#     --robot.cameras="{front: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}, 
#     wrist: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
#     --teleop.type=so101_leader \
#     --teleop.port=/dev/ttyACM0 \
#     --teleop.id=my_awesome_leader_arm

# Press Right Arrow (→): Early stop the current episode or reset time and move to the next.
# Press Left Arrow (←): Cancel the current episode and re-record it.
# Press Escape (ESC): Immediately stop the session, encode videos, and upload the dataset.

# task_1 = "put the green block in the white box"
# task_2 = "put the red block in the black box"
# task_3 = "put green block in white box, red block in black box"
# task_4 = "open drawer and put green block in"
# task_5 = "open drawer and put red block in"

# データ取得用のテレオペ
# フォロワーとリーダーのポート番号に注意
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}, 
    front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --dataset.repo_id=${HF_USER}/task_2 \
    --dataset.num_episodes=20 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="put the red block in the black box" \
    --dataset.push_to_hub=False \
    --resume=True

