"""
pullup_exercise.py
──────────────────
引體向上偵測模組，包含：
  - 肩-肘-腕角度狀態機（up / down 計次）
  - 肩膀對稱性即時檢查（高低肩警報）
  - 手肘張開角度警告（過寬傷膝，過窄效率差）
  - 下巴過槓檢查（用鼻子 y 座標近似）
  - 統一回傳 ExerciseResult，供主程式 HUD 使用
"""

from __future__ import annotations
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field

mp_pose = mp.solutions.pose
PL      = mp_pose.PoseLandmark   # 簡寫


# ──────────────────────────────────────────────
# 資料結構
# ──────────────────────────────────────────────

@dataclass
class FormWarning:
    """單條姿勢警告"""
    key:      str          # 唯一識別鍵（用於冷卻計時）
    message:  str          # 顯示給使用者的文字
    severity: str = "warn" # "warn" | "error"


@dataclass
class ExerciseResult:
    """每幀回傳給主程式的結果"""
    counter:       int              = 0
    stage:         str | None       = None   # "up" | "down" | None
    angle_left:    float            = 0.0
    angle_right:   float            = 0.0
    warnings:      list[FormWarning] = field(default_factory=list)
    chin_over_bar: bool             = False  # 是否過槓


# ──────────────────────────────────────────────
# 角度 / 幾何工具
# ──────────────────────────────────────────────

def _angle(a, b, c) -> float:
    """計算 a-b-c 三點夾角，b 為頂點，回傳 0~180°"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    r = (np.arctan2(c[1]-b[1], c[0]-b[0])
       - np.arctan2(a[1]-b[1], a[0]-b[0]))
    deg = abs(np.degrees(r))
    return 360 - deg if deg > 180 else deg


def _px(lm, idx: int, w: int, h: int) -> list[float]:
    """landmark → 像素座標"""
    p = lm[idx]
    return [p.x * w, p.y * h]


def _norm(lm, idx: int) -> tuple[float, float]:
    """landmark → 正規化座標 (x, y)，不需要 w/h"""
    p = lm[idx]
    return p.x, p.y


# ──────────────────────────────────────────────
# 引體向上偵測器
# ──────────────────────────────────────────────

class PullUpExercise:
    """
    使用方式：
        detector = PullUpExercise()
        result   = detector.process(pose_landmarks, frame_w, frame_h)
    """

    # ── 角度閾值 ──
    ANGLE_UP   = 150   # 手臂伸直（懸掛）→ UP 狀態
    ANGLE_DOWN = 70    # 手肘彎曲（拉上去）→ DOWN 狀態

    # ── 肩膀對稱閾值 ──
    SHOULDER_TILT_WARN  = 0.03   # 正規化 y 差值（約 3% 畫面高度）
    SHOULDER_TILT_ERROR = 0.07   # 超過此值為嚴重高低肩

    # ── 手肘張開角度（俯視估算：用肩-肘水平偏移量） ──
    ELBOW_WIDE_WARN = 0.35   # 肘比肩寬出超過 35% 肩寬 → 過寬
    ELBOW_NARROW_WARN = 0.0  # 肘比肩還窄 → 過窄（= 0 即任何內縮都警告）

    def __init__(self):
        self.counter: int       = 0
        self.stage:   str|None  = None
        self._rep_warnings: list[FormWarning] = []   # 本次 rep 累積的警告

    # ── 主要處理函式 ──
    def process(self, landmarks, w: int, h: int) -> ExerciseResult:
        lm = landmarks

        # ── 取得關鍵座標 ──
        l_shoulder = _px(lm, PL.LEFT_SHOULDER.value,  w, h)
        l_elbow    = _px(lm, PL.LEFT_ELBOW.value,     w, h)
        l_wrist    = _px(lm, PL.LEFT_WRIST.value,     w, h)

        r_shoulder = _px(lm, PL.RIGHT_SHOULDER.value, w, h)
        r_elbow    = _px(lm, PL.RIGHT_ELBOW.value,    w, h)
        r_wrist    = _px(lm, PL.RIGHT_WRIST.value,    w, h)

        nose       = lm[PL.NOSE.value]

        # ── 計算左右手肘角度 ──
        l_angle = _angle(l_shoulder, l_elbow, l_wrist)
        r_angle = _angle(r_shoulder, r_elbow, r_wrist)
        avg     = (l_angle + r_angle) / 2

        # ── 即時姿勢檢查 ──
        warnings = self._check_form(lm, w, h)

        # ── 下巴過槓判斷（鼻子 y < 肩膀 y 中點） ──
        shoulder_mid_y = (lm[PL.LEFT_SHOULDER.value].y
                        + lm[PL.RIGHT_SHOULDER.value].y) / 2
        chin_over = nose.y < shoulder_mid_y

        # ── 狀態機 ──
        if avg > self.ANGLE_UP:
            self.stage = "up"
            self._rep_warnings.clear()   # 新的懸掛開始，清空本次警告

        if avg < self.ANGLE_DOWN and self.stage == "up":
            self.stage = "down"
            # 只有下巴過槓才計入有效次數
            if chin_over:
                self.counter += 1
            else:
                warnings.append(FormWarning(
                    key="chin_bar",
                    message="Chin must clear the bar!",
                    severity="error"
                ))

        return ExerciseResult(
            counter       = self.counter,
            stage         = self.stage,
            angle_left    = l_angle,
            angle_right   = r_angle,
            warnings      = warnings,
            chin_over_bar = chin_over,
        )

    # ── 姿勢檢查子函式 ──
    def _check_form(self, lm, w: int, h: int) -> list[FormWarning]:
        warnings: list[FormWarning] = []

        # 1. 肩膀對稱性（高低肩）
        l_sy = lm[PL.LEFT_SHOULDER.value].y
        r_sy = lm[PL.RIGHT_SHOULDER.value].y
        tilt = abs(l_sy - r_sy)

        if tilt >= self.SHOULDER_TILT_ERROR:
            side = "左" if l_sy > r_sy else "右"
            warnings.append(FormWarning(
                key      = "shoulder_tilt",
                message  = f"Shoulder uneven! {side} side dropping",
                severity = "error"
            ))
        elif tilt >= self.SHOULDER_TILT_WARN:
            warnings.append(FormWarning(
                key      = "shoulder_tilt",
                message  = "Keep shoulders level",
                severity = "warn"
            ))

        # 2. 手肘張開寬度（避免 Chicken Wing 或過窄）
        l_sx = lm[PL.LEFT_SHOULDER.value].x
        r_sx = lm[PL.RIGHT_SHOULDER.value].x
        l_ex = lm[PL.LEFT_ELBOW.value].x
        r_ex = lm[PL.RIGHT_ELBOW.value].x

        shoulder_w = abs(l_sx - r_sx) + 1e-6
        # 左肘外展量（負值 = 內縮）
        l_flare = (l_sx - l_ex) / shoulder_w   # 左肘往左超出左肩的比例
        r_flare = (r_ex - r_sx) / shoulder_w   # 右肘往右超出右肩的比例

        if l_flare > self.ELBOW_WIDE_WARN or r_flare > self.ELBOW_WIDE_WARN:
            warnings.append(FormWarning(
                key      = "elbow_wide",
                message  = "Elbows too wide — risk of injury",
                severity = "warn"
            ))

        # 3. 身體搖晃（髖部水平位移偏離肩部中心過多）
        l_hx = lm[PL.LEFT_HIP.value].x
        r_hx = lm[PL.RIGHT_HIP.value].x
        shoulder_cx = (l_sx + r_sx) / 2
        hip_cx      = (l_hx + r_hx) / 2
        swing       = abs(shoulder_cx - hip_cx)

        if swing > 0.12:
            warnings.append(FormWarning(
                key      = "body_swing",
                message  = "Reduce body swing",
                severity = "warn"
            ))

        return warnings
