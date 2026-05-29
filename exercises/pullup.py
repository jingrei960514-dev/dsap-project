"""
exercises/pullup.py
───────────────────
引體向上偵測器，繼承 BaseExercise。

偵測邏輯：
  - 肩-肘-腕角度狀態機（up / down 計次）
  - 肩膀對稱性（高低肩）：輕微 / 嚴重 兩級警報
  - 下巴過槓：只有鼻子超過肩膀中線才計入有效次數
  - 手肘過寬警告（Chicken Wing）
  - 身體搖晃警告
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


class PullUp(BaseExercise):

    # ── 角度閾值 ──
    ANGLE_UP   = 150   # 手臂伸直懸掛 → UP
    ANGLE_DOWN = 70    # 手肘彎曲拉上 → DOWN

    # ── 肩膀對稱閾值（正規化 y 差值） ──
    SHOULDER_WARN  = 0.03   # 輕微傾斜
    SHOULDER_ERROR = 0.07   # 嚴重高低肩

    # ── 手肘張開閾值（超出肩寬的比例） ──
    ELBOW_WIDE_RATIO = 0.35

    # ── 身體搖晃閾值（肩部中心 vs 髖部中心水平偏移） ──
    SWING_THRESHOLD = 0.12

    # ── HUD 設定 ──
    def get_name(self) -> str:
        return "PULL-UPS"

    def get_hud_color(self) -> tuple[int, int, int]:
        return (255, 130, 180)   # 紫色調

    # ── 主要偵測邏輯 ──
    def process(self, landmarks, w: int, h: int) -> ExerciseResult:
        lms = landmarks

        l_shoulder = lm_px(lms, PL.LEFT_SHOULDER.value,  w, h)
        l_elbow    = lm_px(lms, PL.LEFT_ELBOW.value,     w, h)
        l_wrist    = lm_px(lms, PL.LEFT_WRIST.value,     w, h)
        r_shoulder = lm_px(lms, PL.RIGHT_SHOULDER.value, w, h)
        r_elbow    = lm_px(lms, PL.RIGHT_ELBOW.value,    w, h)
        r_wrist    = lm_px(lms, PL.RIGHT_WRIST.value,    w, h)

        l_angle = calc_angle(l_shoulder, l_elbow, l_wrist)
        r_angle = calc_angle(r_shoulder, r_elbow, r_wrist)
        avg     = (l_angle + r_angle) / 2

        # 下巴過槓（鼻子 y < 肩膀中點 y）
        nose        = lms[PL.NOSE.value]
        sh_mid_y    = (lms[PL.LEFT_SHOULDER.value].y
                     + lms[PL.RIGHT_SHOULDER.value].y) / 2
        chin_over   = nose.y < sh_mid_y

        warnings = self._check_form(lms)

        # 狀態機
        if avg > self.ANGLE_UP:
            self.stage = "up"

        if avg < self.ANGLE_DOWN and self.stage == "up":
            self.stage = "down"
            if chin_over:
                self.counter += 1
                warnings.insert(0, FormWarning("good", "Good rep!", severity="good"))
            else:
                warnings.insert(0, FormWarning(
                    "chin_bar", "Chin must clear the bar!", severity="error"
                ))

        return ExerciseResult(
            counter     = self.counter,
            stage       = self.stage,
            angle_left  = l_angle,
            angle_right = r_angle,
            warnings    = warnings,
            extra       = {"chin_over_bar": chin_over},
        )

    # ── 姿勢檢查 ──
    def _check_form(self, lms) -> list[FormWarning]:
        warnings: list[FormWarning] = []

        l_sy = lms[PL.LEFT_SHOULDER.value].y
        r_sy = lms[PL.RIGHT_SHOULDER.value].y
        l_sx = lms[PL.LEFT_SHOULDER.value].x
        r_sx = lms[PL.RIGHT_SHOULDER.value].x
        l_ex = lms[PL.LEFT_ELBOW.value].x
        r_ex = lms[PL.RIGHT_ELBOW.value].x

        # 1. 肩膀對稱性（高低肩）
        tilt = abs(l_sy - r_sy)
        if tilt >= self.SHOULDER_ERROR:
            side = "左" if l_sy > r_sy else "右"
            warnings.append(FormWarning(
                "shoulder_tilt",
                f"Shoulder uneven! {side} side dropping",
                severity="error",
            ))
        elif tilt >= self.SHOULDER_WARN:
            warnings.append(FormWarning(
                "shoulder_tilt", "Keep shoulders level", severity="warn"
            ))

        # 2. 手肘過寬（Chicken Wing）
        shoulder_w = abs(l_sx - r_sx) + 1e-6
        l_flare    = (l_sx - l_ex) / shoulder_w
        r_flare    = (r_ex - r_sx) / shoulder_w
        if l_flare > self.ELBOW_WIDE_RATIO or r_flare > self.ELBOW_WIDE_RATIO:
            warnings.append(FormWarning(
                "elbow_wide", "Elbows too wide", severity="warn"
            ))

        # 3. 身體搖晃
        l_hx     = lms[PL.LEFT_HIP.value].x
        r_hx     = lms[PL.RIGHT_HIP.value].x
        sh_cx    = (l_sx + r_sx) / 2
        hip_cx   = (l_hx + r_hx) / 2
        if abs(sh_cx - hip_cx) > self.SWING_THRESHOLD:
            warnings.append(FormWarning(
                "body_swing", "Reduce body swing", severity="warn"
            ))

        return warnings

    # ── 視覺輔助：肩膀對稱線（顏色反映傾斜程度） ──
    def draw_overlay(self, frame, landmarks, w: int, h: int) -> None:
        lms = landmarks
        ls  = lms[PL.LEFT_SHOULDER.value]
        rs  = lms[PL.RIGHT_SHOULDER.value]
        lx, ly = int(ls.x * w), int(ls.y * h)
        rx, ry = int(rs.x * w), int(rs.y * h)

        tilt = abs(ls.y - rs.y)
        if tilt >= self.SHOULDER_ERROR:
            color = (0, 50, 255)    # 紅：嚴重
        elif tilt >= self.SHOULDER_WARN:
            color = (0, 165, 255)   # 橘：輕微
        else:
            color = (0, 220, 100)   # 綠：正常

        cv2.line(frame, (lx, ly), (rx, ry), color, 3, cv2.LINE_AA)
        cv2.circle(frame, (lx, ly), 7, color, -1)
        cv2.circle(frame, (rx, ry), 7, color, -1)
