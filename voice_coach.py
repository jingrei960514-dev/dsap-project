"""
voice_coach.py
──────────────
語音回饋模組，特性：
  - 優先級排隊：error(0) > warn(1) > good(2)，同優先級照時間順序
  - 冷卻計時器：每個 key 有獨立冷卻，避免同一條訊息連續播報
  - 非阻塞播放：語音在背景 thread 執行，不影響主迴圈 FPS
  - 中文支援：優先使用 gTTS（線上，音質佳），離線自動降級到 pyttsx3
  - 靜音模式：按 M 鍵可即時開關

使用方式：
    coach = VoiceCoach(lang="zh-tw")   # 或 "en"
    coach.say("膝蓋內扣", key="knee_cave", priority=0, cooldown=4.0)
    coach.say("很好！", key="good_rep",  priority=2, cooldown=1.5)
    coach.set_mute(True)   # 靜音
"""

from __future__ import annotations

import heapq
import threading
import time
import tempfile
import os
import sys
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# TTS 後端自動選擇
# ──────────────────────────────────────────────

def _detect_tts_backend() -> str:
    """
    偵測可用的 TTS 後端，回傳 'gtts' 或 'pyttsx3'。
    優先選 gTTS（線上，中文音質好），無網路或未安裝則降級。
    """
    try:
        from gtts import gTTS
        import pygame
        return "gtts"
    except ImportError:
        pass

    try:
        import pyttsx3
        return "pyttsx3"
    except ImportError:
        pass

    return "none"


# ──────────────────────────────────────────────
# 語音訊息資料結構
# ──────────────────────────────────────────────

@dataclass(order=True)
class _VoiceMessage:
    """
    heapq 使用的語音訊息，以 (priority, timestamp) 排序。
    priority 越小越優先（0=error, 1=warn, 2=good）。
    """
    priority:  int
    timestamp: float
    key:       str   = field(compare=False)
    text:      str   = field(compare=False)


# ──────────────────────────────────────────────
# 主類別
# ──────────────────────────────────────────────

class VoiceCoach:
    """
    語音回饋管理器。

    Parameters
    ----------
    lang     : 語言代碼，'zh-tw'（繁中）、'zh-cn'（簡中）、'en'
    backend  : 'auto'（自動偵測）、'gtts'、'pyttsx3'、'none'（靜音測試）
    """

    # 預設冷卻時間（秒），各優先級不同
    DEFAULT_COOLDOWN = {0: 3.0, 1: 4.0, 2: 1.5}

    def __init__(self, lang: str = "zh-tw", backend: str = "auto"):
        self.lang    = lang
        self.muted   = False
        self._lock   = threading.Lock()

        # ── 優先級 Min-Heap ──
        self._queue: list[_VoiceMessage] = []

        # ── 每個 key 的上次播放時間 ──
        self._last_spoke: dict[str, float] = {}

        # ── 是否正在播放（避免同時播兩條） ──
        self._speaking = False

        # ── TTS 後端初始化 ──
        self._backend = _detect_tts_backend() if backend == "auto" else backend
        self._engine  = None   # pyttsx3 engine（lazy init）

        if self._backend == "gtts":
            try:
                import pygame
                pygame.mixer.init()
                print(f"[VoiceCoach] 使用 gTTS 後端（語言：{lang}）")
            except Exception as e:
                print(f"[VoiceCoach] pygame 初始化失敗，降級到 pyttsx3：{e}")
                self._backend = "pyttsx3"

        if self._backend == "pyttsx3":
            self._init_pyttsx3()
            print(f"[VoiceCoach] 使用 pyttsx3 後端")

        if self._backend == "none":
            print("[VoiceCoach] 未找到 TTS 後端，語音功能關閉")

    def _init_pyttsx3(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            # 嘗試設定中文語音（Windows 有內建，macOS/Linux 效果不一）
            voices = self._engine.getProperty("voices")
            for v in voices:
                if any(kw in v.name.lower() for kw in ["zh", "chinese", "mandarin", "taiwan"]):
                    self._engine.setProperty("voice", v.id)
                    break
            self._engine.setProperty("rate", 160)   # 語速（wpm）
        except Exception as e:
            print(f"[VoiceCoach] pyttsx3 初始化失敗：{e}")
            self._backend = "none"

    # ──────────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────────

    def say(
        self,
        text:     str,
        key:      str,
        priority: int  = 1,
        cooldown: Optional[float] = None,
    ) -> None:
        """
        把一條語音訊息加入佇列。

        Parameters
        ----------
        text     : 要說的文字
        key      : 唯一識別鍵（用於冷卻計時，相同 key 共享冷卻）
        priority : 0=error（最高）, 1=warn, 2=good（最低）
        cooldown : 此訊息的冷卻秒數，None 則用預設值
        """
        if self._backend == "none" or self.muted:
            return

        cd  = cooldown if cooldown is not None else self.DEFAULT_COOLDOWN.get(priority, 3.0)
        now = time.time()

        with self._lock:
            last = self._last_spoke.get(key, 0)
            if now - last < cd:
                return   # 冷卻中，直接丟棄

            msg = _VoiceMessage(
                priority  = priority,
                timestamp = now,
                key       = key,
                text      = text,
            )
            heapq.heappush(self._queue, msg)

        # 若目前沒有在播放，啟動播放迴圈
        if not self._speaking:
            threading.Thread(target=self._play_loop, daemon=True).start()

    def set_mute(self, muted: bool) -> None:
        """切換靜音狀態"""
        self.muted = muted
        print(f"[VoiceCoach] {'🔇 靜音' if muted else '🔊 開啟'}")

    def toggle_mute(self) -> bool:
        """切換靜音並回傳新狀態"""
        self.set_mute(not self.muted)
        return self.muted

    def clear(self) -> None:
        """清空佇列（切換動作時呼叫）"""
        with self._lock:
            self._queue.clear()

    # ──────────────────────────────────────────
    # 內部播放迴圈
    # ──────────────────────────────────────────

    def _play_loop(self) -> None:
        """在背景 thread 持續取出佇列並播放，直到佇列清空"""
        self._speaking = True
        try:
            while True:
                with self._lock:
                    if not self._queue:
                        break
                    msg = heapq.heappop(self._queue)

                # 更新冷卻時間戳（pop 之後才更新，避免佇列裡同 key 重複）
                self._last_spoke[msg.key] = time.time()
                self._speak(msg.text)
        finally:
            self._speaking = False

    def _speak(self, text: str) -> None:
        """實際呼叫 TTS 播放（阻塞直到播完）"""
        if self._backend == "gtts":
            self._speak_gtts(text)
        elif self._backend == "pyttsx3":
            self._speak_pyttsx3(text)

    def _speak_gtts(self, text: str) -> None:
        try:
            from gtts import gTTS
            import pygame

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name

            gTTS(text=text, lang=self.lang, slow=False).save(tmp_path)

            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

            pygame.mixer.music.unload()
            os.unlink(tmp_path)

        except Exception as e:
            print(f"[VoiceCoach] gTTS 播放失敗：{e}")

    def _speak_pyttsx3(self, text: str) -> None:
        try:
            if self._engine is None:
                return
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            print(f"[VoiceCoach] pyttsx3 播放失敗：{e}")


# ──────────────────────────────────────────────
# 各動作的語音訊息對照表
# ──────────────────────────────────────────────

# 格式：{ warning_key: (text_zh, text_en, priority, cooldown) }
VOICE_MESSAGES: dict[str, tuple[str, str, int, float]] = {
    # ── 通用 ──
    "good":          ("很好，繼續！",         "Good rep!",              2, 1.5),

    # ── 伏地挺身 ──
    "body_align":    ("保持身體直線",          "Keep body straight",     0, 4.0),
    "head_drop":     ("頭部保持中立位置",       "Keep head neutral",      1, 5.0),

    # ── 引體向上 ──
    "shoulder_tilt": ("肩膀不平衡，注意高低肩", "Keep shoulders level",   0, 3.0),
    "chin_bar":      ("下巴要過槓",            "Chin over the bar",      1, 3.0),
    "elbow_wide":    ("手肘不要太開",           "Elbows too wide",        1, 5.0),
    "body_swing":    ("減少身體搖晃",           "Reduce body swing",      1, 4.0),

    # ── 深蹲 ──
    "knee_cave":     ("膝蓋往外推",            "Push knees out",         0, 3.0),
    "back_lean":     ("挺胸，背部不要過度前傾", "Keep chest up",          1, 5.0),
    "knee_over_toe": ("重心往後，膝蓋不要超過腳尖", "Push hips back",     1, 5.0),
}


def warnings_to_voice(
    coach:    VoiceCoach,
    warnings: list,
    lang:     str = "zh-tw",
) -> None:
    """
    把 ExerciseResult.warnings 轉換成語音播報。
    供 virtual_coach.py 主迴圈呼叫。
    """
    severity_to_priority = {"error": 0, "warn": 1, "good": 2}

    for w in warnings:
        if w.key not in VOICE_MESSAGES:
            continue

        zh_text, en_text, default_priority, cooldown = VOICE_MESSAGES[w.key]
        text     = zh_text if lang.startswith("zh") else en_text
        priority = severity_to_priority.get(w.severity, default_priority)

        coach.say(text, key=w.key, priority=priority, cooldown=cooldown)