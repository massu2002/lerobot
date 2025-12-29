import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

config = OpenCVCameraConfig(
    index_or_path="/dev/video0",  # ★ ここを video0 に変更
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)

camera = OpenCVCamera(config)
camera.connect()

try:
    while True:
        frame = camera.async_read(timeout_ms=200)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Live Camera", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.disconnect()
    cv2.destroyAllWindows()
