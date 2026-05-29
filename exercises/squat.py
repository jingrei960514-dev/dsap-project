from __future__ import annotations
import cv2
import math
import mediapipe as mp

from .base_exercise import (
    BaseExercise, ExerciseResult, FormWarning,
    calc_angle, lm_px, landmarks_visible,
)

mp_pose = mp.solutions.pose
PL      = mp_pose.PoseLandmark


class Squat(BaseExercise):

    # ── 角度閾值（髖-膝-踝） ──
    ANGLE_DOWN = 105   # 微調：稍微放寬蹲下的角度（原 100）
    ANGLE_UP   = 160   # 高於此值視為站直（up）

    # ── 膝蓋超過腳尖容忍值（正規化 x 差值） ──
    KNEE_TOE_TOLERANCE = 0.03   

    # ── 膝蓋內扣門檻（膝寬 / 踝寬 比值） ──
    KNEE_CAVE_RATIO = 0.85   

    # ── 背部前傾門檻（肩-髖與垂直軸夾角，0°=直立） ──
    # 💡 側身拍攝時，深蹲正常的背部前傾角度較大，調高至 50 度避免誤判
    BACK_LEAN_MIN = 50   

    # ── 需要可見的關節 ──
    REQUIRED_LANDMARKS = [
        PL.LEFT_HIP.value,    PL.RIGHT_HIP.value,
        PL.LEFT_KNEE.value,   PL.RIGHT_KNEE.value,
        PL.LEFT_ANKLE.value,  PL.RIGHT_ANKLE.value,
    ]

    def get_name(self) -> str:
        return "SQUATS"

    def get_hud_color(self) -> tuple[int, int, int]:
        return (100, 255, 180)   

    # ── 主要偵測邏輯 ──
    def process(self, landmarks, w: int, h: int) -> ExerciseResult:
        lms = landmarks

        if not landmarks_visible(lms, self.REQUIRED_LANDMARKS, threshold=0.5):
            return ExerciseResult(counter=self.counter, stage=self.stage)

        l_hip   = lm_px(lms, PL.LEFT_HIP.value,   w, h)
        l_knee  = lm_px(lms, PL.LEFT_KNEE.value,  w, h)
        l_ankle = lm_px(lms, PL.LEFT_ANKLE.value, w, h)

        r_hip   = lm_px(lms, PL.RIGHT_HIP.value,   w, h)
        r_knee  = lm_px(lms, PL.RIGHT_KNEE.value,  w, h)
        r_ankle = lm_px(lms, PL.RIGHT_ANKLE.value, w, h)

        l_angle = calc_angle(l_hip, l_knee, l_ankle)
        r_angle = calc_angle(r_hip, r_knee, r_ankle)
        avg     = (l_angle + r_angle) / 2

        # 狀態機變更：先更動狀態，再送入 _check_form 檢查才有意義
        if avg > self.ANGLE_UP:
            self.stage = "up"

        # 進行姿勢檢查
        warnings = self._check_form(lms, w, h)

        if avg < self.ANGLE_DOWN and self.stage == "up":
            self.stage = "down"
            # 判斷本次是否姿勢正確（無 error 級警告）
            has_error = any(w.severity == "error" for w in warnings)
            if not has_error:
                self.counter += 1
                warnings.insert(0, FormWarning("good", "Good rep!", severity="good"))

        return ExerciseResult(
            counter     = self.counter,
            stage       = self.stage,
            angle_left  = l_angle,
            angle_right = r_angle,
            warnings    = warnings,
        )

    # ── 姿勢檢查 ──
    def _check_form(self, lms, w: int, h: int) -> list[FormWarning]:
        warnings: list[FormWarning] = []

        # 1. 膝蓋內扣（Knee Cave）— 修正：只有在「蹲下 (down)」時才檢查內扣
        # 💡 注意：側身拍攝時，左右膝與左右腳踝的 x 軸會重疊，此時不建議啟用內扣偵測（容易誤判）
        if self.stage == "down":
            knee_w  = abs(lms[PL.LEFT_KNEE.value].x  - lms[PL.RIGHT_KNEE.value].x)
            ankle_w = abs(lms[PL.LEFT_ANKLE.value].x - lms[PL.RIGHT_ANKLE.value].x) + 1e-6
            if knee_w / ankle_w < self.KNEE_CAVE_RATIO:
                warnings.append(FormWarning(
                    "knee_cave",
                    "Knees caving in — push knees out",
                    severity="error",
                ))

        # 2. 背部過度前傾
        l_sh  = lms[PL.LEFT_SHOULDER.value]
        l_hip = lms[PL.LEFT_HIP.value]
        dx = abs(l_sh.x - l_hip.x)
        dy = abs(l_sh.y - l_hip.y) + 1e-6
        back_lean_angle = math.degrees(math.atan2(dx, dy))

        if back_lean_angle > self.BACK_LEAN_MIN:
            warnings.append(FormWarning(
                "back_lean",
                "Keep chest up — back too forward",
                severity="warn", # 如果不希望背前傾卡住計數，維持 warn 即可；若要卡住計數請改 error
            ))

        return warnings

    # ── 視覺輔助 ──
    def draw_overlay(self, frame, landmarks, w: int, h: int) -> None:
        lms = landmarks

        if not landmarks_visible(lms, self.REQUIRED_LANDMARKS, threshold=0.5):
            return

        for hip_idx, knee_idx, ankle_idx in [
            (PL.LEFT_HIP.value,  PL.LEFT_KNEE.value,  PL.LEFT_ANKLE.value),
            (PL.RIGHT_HIP.value, PL.RIGHT_KNEE.value, PL.RIGHT_ANKLE.value),
        ]:
            hip   = (int(lms[hip_idx].x   * w), int(lms[hip_idx].y   * h))
            knee  = (int(lms[knee_idx].x  * w), int(lms[knee_idx].y  * h))
            ankle = (int(lms[ankle_idx].x * w), int(lms[ankle_idx].y * h))

            # 同步修正繪圖的內扣判斷邏輯
            knee_w  = abs(lms[PL.LEFT_KNEE.value].x  - lms[PL.RIGHT_KNEE.value].x)
            ankle_w = abs(lms[PL.LEFT_ANKLE.value].x - lms[PL.RIGHT_ANKLE.value].x) + 1e-6
            caving  = (self.stage == "down") and (knee_w / ankle_w < self.KNEE_CAVE_RATIO)
            color   = (0, 60, 255) if caving else (0, 220, 100)

            cv2.line(frame, hip,   knee,  color, 3, cv2.LINE_AA)
            cv2.line(frame, knee,  ankle, color, 3, cv2.LINE_AA)
            cv2.circle(frame, knee, 8, color, -1)

        l_hip_y  = int(lms[PL.LEFT_HIP.value].y   * h)
        l_knee_y = int(lms[PL.LEFT_KNEE.value].y  * h)

        cv2.line(frame, (0, l_hip_y), (w, l_hip_y), (180, 180, 180), 1, cv2.LINE_AA)

        depth_ok = l_knee_y > l_hip_y  
        depth_color = (0, 220, 100) if depth_ok else (0, 165, 255)
        label = "Depth OK" if depth_ok else "Go deeper"
        cv2.putText(frame, label, (w - 130, l_hip_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, depth_color, 2, cv2.LINE_AA)