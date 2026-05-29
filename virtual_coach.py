"""
virtual_coach.py
────────────────
主程式：只負責攝影機、HUD、鍵盤，不含任何動作邏輯。

新增動作步驟：
  1. 在 exercises/ 建立新檔案，繼承 BaseExercise
  2. 在 exercises/__init__.py import 新 class
  3. 把新 class 加進下方 EXERCISES 清單，完成

快捷鍵：
  Tab  — 切換動作
  R    — 重置當前計數
  Q    — 離開
"""

import cv2
import mediapipe as mp
import numpy as np

from camera_selector import select_camera
from exercises       import PushUp, PullUp, Squat
from exercises.base_exercise import FormWarning
from voice_coach     import VoiceCoach, warnings_to_voice

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ── 語音語言設定：'zh-tw' 繁中 / 'zh-cn' 簡中 / 'en' 英文 ──
VOICE_LANG = "zh-tw"


# ── 在這裡登記所有動作，順序即 Tab 切換順序 ──
EXERCISES = [
    PushUp(),
    PullUp(),
    Squat(),
]


# ──────────────────────────────────────────────
# HUD 繪製（與動作邏輯完全解耦）
# ──────────────────────────────────────────────

def _draw_bg(img, pt1, pt2, color=(30, 30, 30), alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _put_text(img, text, pos, scale=0.7, color=(255, 255, 255),
              bg=(40, 40, 40), thick=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    _draw_bg(img, (x-6, y-th-6), (x+tw+6, y+bl), bg)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def draw_hud(frame, exercise, result, exercise_index: int, total: int,
             muted: bool = False):
    h, w = frame.shape[:2]

    # ── 左上：動作資訊面板 ──
    _draw_bg(frame, (0, 0), (240, 130), alpha=0.55)
    _put_text(frame, exercise.get_name(),
              (10, 30), scale=0.85,
              color=exercise.get_hud_color(), bg=(0, 0, 0))
    _put_text(frame, f"Count : {result.counter}", (10, 68),  scale=0.75)
    _put_text(frame, f"Stage : {result.stage  }", (10, 105), scale=0.75)

    # 引體向上：加顯示下巴過槓狀態
    chin_over = result.extra.get("chin_over_bar")
    if chin_over is not None:
        txt   = "Chin OVER bar" if chin_over else "Chin below bar"
        color = (0, 220, 100)   if chin_over else (80, 80, 255)
        _put_text(frame, txt, (10, 138), scale=0.58, color=color)

    # ── 右上：操作提示 + 靜音狀態 ──
    mute_icon = "🔇" if muted else "🔊"
    hint = f"[Tab]{exercise_index+1}/{total} [R]reset [M]{mute_icon} [Q]quit"
    _put_text(frame, hint,
              (w - 360, 28), scale=0.52,
              color=(200, 200, 200), bg=(20, 20, 20))

    # ── 底部：姿勢警告（最多 2 條，由嚴重到輕微） ──
    severity_order = {"error": 0, "warn": 1, "good": 2}
    sorted_warns   = sorted(result.warnings,
                            key=lambda x: severity_order.get(x.severity, 9))

    color_map = {
        "error": (0,  80, 255),
        "warn":  (0, 200, 255),
        "good":  (0, 220, 100),
    }

    y_base = h - 20
    for warn in sorted_warns[:2]:
        c = color_map.get(warn.severity, (200, 200, 200))
        _put_text(frame, f"  {warn.message}",
                  (10, y_base), scale=0.75, color=c, bg=(15, 15, 15))
        y_base -= 44


def draw_angle_labels(frame, lms, result, w, h):
    PL = mp_pose.PoseLandmark
    for idx, angle in [
        (PL.LEFT_ELBOW.value,  result.angle_left),
        (PL.RIGHT_ELBOW.value, result.angle_right),
    ]:
        p  = lms[idx]
        px = (int(p.x * w), int(p.y * h))
        cv2.putText(frame, f"{int(angle)}",
                    px, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2, cv2.LINE_AA)


# ──────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────

def main():
    pose = mp_pose.Pose(
        static_image_mode        = False,
        model_complexity         = 1,
        enable_segmentation      = False,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )

    cap = select_camera()

    idx      = 0
    exercise = EXERCISES[idx]
    total    = len(EXERCISES)

    # ── 語音教練初始化 ──
    coach = VoiceCoach(lang=VOICE_LANG)

    print(f"\n▶  開始：{exercise.get_name()}，按 Tab 切換動作，M 靜音，Q 離開。\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = pose.process(rgb)

        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark

            # 1. 動作偵測
            result = exercise.process(lms, w, h)

            # 2. 語音回饋
            warnings_to_voice(coach, result.warnings, lang=VOICE_LANG)

            # 2. 動作專屬視覺輔助
            exercise.draw_overlay(frame, lms, w, h)

            # 3. 角度數字
            draw_angle_labels(frame, lms, result, w, h)

            # 4. 骨架
            mp_drawing.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117,  66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,  66, 230), thickness=2, circle_radius=2),
            )

            # 5. HUD
            draw_hud(frame, exercise, result, idx, total, muted=coach.muted)

        else:
            _put_text(frame, "No pose detected",
                      (w//2 - 100, h//2), scale=0.8,
                      color=(80, 80, 255), bg=(20, 20, 20))

        cv2.imshow("Virtual Coach", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        elif key == 9:           # Tab — 切換動作
            idx      = (idx + 1) % total
            exercise = EXERCISES[idx]
            coach.clear()        # 清空舊動作的語音佇列
            print(f"[切換] {exercise.get_name()}")
        elif key == ord("r"):    # R — 重置
            exercise.reset()
            coach.clear()
            print(f"[重置] {exercise.get_name()} 計數歸零")
        elif key == ord("m"):    # M — 靜音切換
            coach.toggle_mute()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()