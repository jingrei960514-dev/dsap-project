"""
exercises/base_exercise.py
──────────────────────────
所有動作偵測器的抽象父類別。
新增動作只需繼承 BaseExercise，實作三個抽象方法即可。
"""

from __future__ import annotations # Python 3.11+ for self-referential type hints
from abc        import ABC, abstractmethod # 抽象基類與方法
from dataclasses import dataclass, field # 簡化資料結構定義


# ──────────────────────────────────────────────
# 共用資料結構（所有動作共用）
# ──────────────────────────────────────────────

@dataclass
class FormWarning:
    """單條姿勢警告"""
    key:      str           # 唯一識別鍵（用於冷卻計時）
    message:  str           # 顯示給使用者的文字
    severity: str = "warn"  # "warn" | "error" | "good"


@dataclass
class ExerciseResult:
    """每幀回傳給主程式的統一結果格式"""
    counter:     int               = 0
    stage:       str | None        = None    # "up" | "down" | None
    angle_left:  float             = 0.0
    angle_right: float             = 0.0
    warnings:    list[FormWarning] = field(default_factory=list)
    extra:       dict              = field(default_factory=dict)
    # extra 供各動作傳遞額外資訊，例如 pullup 的 chin_over_bar


# ──────────────────────────────────────────────
# 共用幾何工具（所有子類別都能直接用）
# ──────────────────────────────────────────────

import numpy as np


def calc_angle(a, b, c) -> float:
    """計算 a-b-c 三點夾角，b 為頂點，回傳 0~180°"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = (np.arctan2(c[1] - b[1], c[0] - b[0])
         - np.arctan2(a[1] - b[1], a[0] - b[0]))
    deg = abs(np.degrees(rad))
    return 360 - deg if deg > 180 else deg


def lm_px(landmarks, index: int, w: int, h: int) -> list[float]:
    """Mediapipe landmark → 像素座標 [x, y]"""
    p = landmarks[index]
    return [p.x * w, p.y * h]


def landmarks_visible(landmarks, indices: list, threshold: float = 0.5) -> bool:
    """確認指定的關節點 visibility 都高於門檻，低於門檻視為未偵測到"""
    return all(landmarks[i].visibility > threshold for i in indices)


# ──────────────────────────────────────────────
# 抽象父類別
# ──────────────────────────────────────────────

class BaseExercise(ABC):
    """
    所有運動偵測器的父類別。

    子類別必須實作：
        process()        — 每幀偵測邏輯，回傳 ExerciseResult
        get_name()       — HUD 顯示名稱，例如 "PUSH-UPS"
        get_hud_color()  — HUD 標題顏色 (B, G, R)

    子類別可選覆寫：
        draw_overlay()   — 在 frame 上畫額外的視覺輔助（預設不畫）
        reset()          — 重置狀態（預設只清 counter 和 stage）
    """

    def __init__(self):
        self.counter: int      = 0
        self.stage:   str|None = None   # "up" | "down"

    # ── 必須實作 ──────────────────────────────

    @abstractmethod
    def process(self, landmarks, w: int, h: int) -> ExerciseResult:
        """
        接收當幀 pose landmarks 與畫面尺寸，
        回傳 ExerciseResult 供主程式 HUD 使用。
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """HUD 標題文字，例如 'PUSH-UPS'"""
        ...

    @abstractmethod
    def get_hud_color(self) -> tuple[int, int, int]:
        """HUD 標題 BGR 顏色"""
        ...

    # ── 可選覆寫 ──────────────────────────────

    def draw_overlay(self, frame, landmarks, w: int, h: int) -> None:
        """
        在 frame 上繪製動作專屬的視覺輔助線／標記。
        預設不畫任何東西，子類別視需要覆寫。
        """
        pass

    def reset(self) -> None:
        """重置計數與狀態機，子類別如有額外狀態可覆寫後 super().reset()"""
        self.counter = 0
        self.stage   = None