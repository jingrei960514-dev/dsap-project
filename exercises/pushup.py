"""
exercises/pushup.py
───────────────────
伏地挺身偵測器，繼承 BaseExercise。

偵測邏輯：
  - 左右肩-肘-腕角度平均值驅動狀態機
  - 身體水平對齊檢查（核心塌陷 / 臀部過高）
"""

from __future__ import annotations
import cv2
import mediapipe as mp

from .base_exercise import (
    BaseExercise, ExerciseResult, FormWarning,
    calc_angle, lm_px,
)

mp_pose = mp.solutions.pose
PL      = mp_pose.PoseLandmark


class PushUp(BaseExercise):

    # ── 角度閾值 ──
    ANGLE_UP   = 160   # 手臂伸直 → UP 狀態
    ANGLE_DOWN = 90    # 手肘彎曲 → DOWN 狀態（完成一次）

    # ── 身體對齊閾值 ──
    ALIGNMENT_TOLERANCE = 0.6   # 肩-髖 與 髖-踝 斜率差容忍值

    # ── HUD 設定 ──
    def get_name(self) -> str:
        return "PUSH-UPS"

    def get_hud_color(self) -> tuple[int, int, int]:
        return (255, 220, 100)   # 藍偏青

    # ── 主要偵測邏輯 ──
    def process(self, landmarks, w: int, h: int) -> ExerciseResult:
        lms = landmarks

        # 取得像素座標
        l_shoulder = lm_px(lms, PL.LEFT_SHOULDER.value,  w, h)
        l_elbow    = lm_px(lms, PL.LEFT_ELBOW.value,     w, h)
        l_wrist    = lm_px(lms, PL.LEFT_WRIST.value,     w, h)
        r_shoulder = lm_px(lms, PL.RIGHT_SHOULDER.value, w, h)
        r_elbow    = lm_px(lms, PL.RIGHT_ELBOW.value,    w, h)
        r_wrist    = lm_px(lms, PL.RIGHT_WRIST.value,    w, h)

        l_angle = calc_angle(l_shoulder, l_elbow, l_wrist)
        r_angle = calc_angle(r_shoulder, r_elbow, r_wrist)
        avg     = (l_angle + r_angle) / 2

        warnings = self._check_form(lms)

        # 狀態機
        if avg > self.ANGLE_UP:
            self.stage = "up"

        if avg < self.ANGLE_DOWN and self.stage == "up":
            self.stage = "down"
            body_ok = self._check_alignment(lms)
            if body_ok:
                self.counter += 1
                warnings.insert(0, FormWarning("good", "Good rep!", severity="good"))
            else:
                warnings.insert(0, FormWarning(
                    "body_align", "Keep body straight!", severity="error"
                ))

        return ExerciseResult(
            counter     = self.counter,
            stage       = self.stage,
            angle_left  = l_angle,
            angle_right = r_angle,
            warnings    = warnings,
        )

    # ── 姿勢檢查 ──
    def _check_alignment(self, lms) -> bool:
        """身體是否保持水平直線（肩-髖-踝斜率差）"""
        sh = lms[PL.LEFT_SHOULDER.value]
        hi = lms[PL.LEFT_HIP.value]
        an = lms[PL.LEFT_ANKLE.value]
        s1 = (hi.y - sh.y) / (abs(hi.x - sh.x) + 1e-6)
        s2 = (an.y - hi.y) / (abs(an.x - hi.x) + 1e-6)
        return abs(s1 - s2) < self.ALIGNMENT_TOLERANCE

    def _check_form(self, lms) -> list[FormWarning]:
        warnings: list[FormWarning] = []

        # 頭部位置：頭不能過度下垂（鼻子 y 比肩膀 y 大太多）
        nose = lms[PL.NOSE.value]
        sh_y = lms[PL.LEFT_SHOULDER.value].y
        if nose.y - sh_y > 0.15:
            warnings.append(FormWarning(
                "head_drop", "Keep head neutral", severity="warn"
            ))

        return warnings

    # ── 視覺輔助：在肩膀連線畫對齊指示 ──
    def draw_overlay(self, frame, landmarks, w: int, h: int) -> None:
        lms = landmarks
        sh  = lms[PL.LEFT_SHOULDER.value]
        hi  = lms[PL.LEFT_HIP.value]
        an  = lms[PL.LEFT_ANKLE.value]

        pts = [
            (int(sh.x * w), int(sh.y * h)),
            (int(hi.x * w), int(hi.y * h)),
            (int(an.x * w), int(an.y * h)),
        ]

        body_ok = self._check_alignment(lms)
        color   = (0, 220, 100) if body_ok else (0, 80, 255)

        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i+1], color, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 5, color, -1)
