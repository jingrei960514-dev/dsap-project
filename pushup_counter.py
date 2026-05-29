import cv2
import mediapipe as mp
import numpy as np

# ── 新增：引入攝影機選擇器 ──
from camera_selector import select_camera

# ──────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) \
            - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def get_landmark_coords(landmarks, index, w, h):
    lm = landmarks[index]
    return [lm.x * w, lm.y * h]


def check_body_alignment(landmarks, w, h):
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    hip      = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    ankle    = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    slope_upper = (hip.y - shoulder.y) / (abs(hip.x - shoulder.x) + 1e-6)
    slope_lower = (ankle.y - hip.y)    / (abs(ankle.x - hip.x)    + 1e-6)
    return abs(slope_upper - slope_lower) < 0.6


# ──────────────────────────────────────────────
# 顯示輔助函式
# ──────────────────────────────────────────────

def draw_rounded_rect(img, pt1, pt2, color, alpha=0.6, radius=10):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text_with_bg(img, text, pos, font_scale=0.7, color=(255, 255, 255),
                     bg_color=(40, 40, 40), thickness=2):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                         font_scale, thickness)
    x, y = pos
    draw_rounded_rect(img, (x - 6, y - th - 6), (x + tw + 6, y + baseline), bg_color)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── 伏地挺身狀態變數 ──
counter  = 0
stage    = None
feedback = ""

ANGLE_DOWN = 90
ANGLE_UP   = 160

# ────────────────────────────────────────────────────
# ★ 唯一的修改：用 select_camera() 取代 cv2.VideoCapture(0)
#    程式啟動時會顯示選單，讓使用者選電腦或手機鏡頭
# ────────────────────────────────────────────────────
cap = select_camera()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = pose.process(image_rgb)

    feedback_color = (0, 255, 0)

    if results.pose_landmarks:
        lms = results.pose_landmarks.landmark

        left_shoulder  = get_landmark_coords(lms, mp_pose.PoseLandmark.LEFT_SHOULDER.value,  w, h)
        left_elbow     = get_landmark_coords(lms, mp_pose.PoseLandmark.LEFT_ELBOW.value,     w, h)
        left_wrist     = get_landmark_coords(lms, mp_pose.PoseLandmark.LEFT_WRIST.value,     w, h)
        right_shoulder = get_landmark_coords(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
        right_elbow    = get_landmark_coords(lms, mp_pose.PoseLandmark.RIGHT_ELBOW.value,    w, h)
        right_wrist    = get_landmark_coords(lms, mp_pose.PoseLandmark.RIGHT_WRIST.value,    w, h)

        left_angle  = calculate_angle(left_shoulder,  left_elbow,  left_wrist)
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        avg_angle   = (left_angle + right_angle) / 2

        body_ok = check_body_alignment(lms, w, h)

        if avg_angle > ANGLE_UP:
            stage = "up"

        if avg_angle < ANGLE_DOWN and stage == "up":
            stage = "down"
            if body_ok:
                counter  += 1
                feedback  = "Good rep!"
                feedback_color = (0, 255, 0)
            else:
                feedback  = "Keep body straight!"
                feedback_color = (0, 100, 255)

        cv2.putText(frame, f"{int(left_angle)}",
                    tuple(np.int32(left_elbow)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"{int(right_angle)}",
                    tuple(np.int32(right_elbow)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,  66, 230), thickness=2, circle_radius=2)
        )

    draw_rounded_rect(frame, (0, 0), (220, 130), (30, 30, 30), alpha=0.55)
    put_text_with_bg(frame, "PUSH-UPS",         (10,  30), font_scale=0.8,
                     color=(100, 220, 255), bg_color=(0, 0, 0))
    put_text_with_bg(frame, f"Count : {counter}", (10,  65), font_scale=0.75)
    put_text_with_bg(frame, f"Stage : {stage  }", (10, 100), font_scale=0.75)

    if feedback:
        put_text_with_bg(frame, feedback,
                         (10, h - 20), font_scale=0.8,
                         color=feedback_color, bg_color=(20, 20, 20))

    cv2.imshow("Virtual Coach - Push-up Counter", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
