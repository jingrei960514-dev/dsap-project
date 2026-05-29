"""
camera_selector.py
──────────────────
啟動時自動偵測可用攝影機，讓使用者選擇「電腦鏡頭」或「手機 IP 串流」。
支援：
  - 自動掃描本機 USB/內建攝影機（index 0~4）
  - 手機 IP Webcam / DroidCam / Camo 串流（HTTP MJPEG）
  - 互動式終端選單，選好後回傳 cv2.VideoCapture 物件
"""

import cv2
import socket
import urllib.request
import time
import sys


# ──────────────────────────────────────────────
# 常數設定
# ──────────────────────────────────────────────

SCAN_INDICES   = list(range(5))          # 掃描 index 0~4
SCAN_TIMEOUT   = 1.5                     # 每個 index 等待秒數
HTTP_TIMEOUT   = 3                       # IP 串流連線逾時秒數
DEFAULT_WIDTH  = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS    = 30

# 常見手機 IP 串流 App 的路徑格式
IP_STREAM_TEMPLATES = {
    "IP Webcam (Android)": "http://{ip}:{port}/video",
    "DroidCam":            "http://{ip}:{port}/video",
    "Camo (iOS/macOS)":    "http://{ip}:{port}/video",
    "自訂 MJPEG URL":      "{ip}",        # 使用者直接貼完整 URL
}

DEFAULT_PORT = 8080


# ──────────────────────────────────────────────
# 顯示工具
# ──────────────────────────────────────────────

def _hr(char="─", width=52):
    print(char * width)

def _title(text):
    _hr()
    print(f"  {text}")
    _hr()

def _ok(text):
    print(f"  ✓  {text}")

def _warn(text):
    print(f"  ⚠  {text}")

def _info(text):
    print(f"     {text}")


# ──────────────────────────────────────────────
# 攝影機偵測
# ──────────────────────────────────────────────

def _probe_local_camera(index: int) -> dict | None:
    """
    嘗試開啟本機攝影機，成功則回傳裝置資訊 dict，失敗回傳 None。
    使用 CAP_DSHOW（Windows）或預設後端，避免等太久。
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        # 讀一幀確認真的有畫面（避免虛假偵測）
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or (frame.shape[1] if frame is not None else 0))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or (frame.shape[0] if frame is not None else 0))
            return {
                "type":    "local",
                "index":   index,
                "backend": backend,
                "label":   f"本機攝影機 #{index}",
                "res":     f"{w}×{h}" if w and h else "未知解析度",
            }
    return None


def scan_local_cameras(indices: list = SCAN_INDICES) -> list[dict]:
    """掃描本機所有可用攝影機，回傳裝置清單。"""
    print()
    _title("🔍  掃描本機攝影機中…")
    found = []
    for idx in indices:
        _info(f"測試 index {idx}…", )
        result = _probe_local_camera(idx)
        if result:
            _ok(f"找到 {result['label']}  ({result['res']})")
            found.append(result)
    if not found:
        _warn("未找到任何本機攝影機")
    return found


def _check_ip_reachable(ip: str, port: int) -> bool:
    """快速 TCP 連線測試，確認手機 IP 是否可達。"""
    try:
        with socket.create_connection((ip, port), timeout=HTTP_TIMEOUT):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def _check_http_stream(url: str) -> bool:
    """確認 HTTP 串流 URL 回傳正確內容類型。"""
    try:
        req = urllib.request.urlopen(url, timeout=HTTP_TIMEOUT)
        ct  = req.headers.get("Content-Type", "")
        return "multipart" in ct or "video" in ct or "jpeg" in ct
    except Exception:
        return False


def probe_ip_camera(ip_or_url: str, port: int = DEFAULT_PORT, app_name: str = "IP Webcam") -> dict | None:
    """
    測試手機 IP 串流，成功回傳裝置資訊 dict，失敗回傳 None。
    ip_or_url 可以是：
      - "192.168.1.5"          → 自動加 port 和路徑
      - "http://192.168.1.5:8080/video" → 直接使用完整 URL
    """
    if ip_or_url.startswith("http"):
        url = ip_or_url
        label = f"手機串流 ({app_name})"
    else:
        url   = f"http://{ip_or_url}:{port}/video"
        label = f"手機串流 {ip_or_url}:{port} ({app_name})"

    _info(f"測試 {url} …")

    # 先測 TCP 連通性（比 HTTP 快）
    try:
        host = url.split("//")[1].split("/")[0].split(":")[0]
        p    = int(url.split("//")[1].split("/")[0].split(":")[1]) if ":" in url.split("//")[1].split("/")[0] else 80
    except Exception:
        host, p = ip_or_url, port

    if not _check_ip_reachable(host, p):
        return None

    # 再測 OpenCV 能否開啟
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        cap.release()
        return None

    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        h, w = frame.shape[:2]
        _ok(f"連線成功  ({w}×{h})")
        return {
            "type":  "ip",
            "url":   url,
            "label": label,
            "res":   f"{w}×{h}",
        }
    return None


# ──────────────────────────────────────────────
# 互動選單
# ──────────────────────────────────────────────

def _input_ip_info() -> tuple[str, int, str]:
    """引導使用者輸入手機 IP / port / App 類型。"""
    print()
    print("  手機串流 App 類型：")
    apps = list(IP_STREAM_TEMPLATES.keys())
    for i, name in enumerate(apps, 1):
        print(f"    {i}. {name}")

    while True:
        raw = input("  選擇 App（預設 1）: ").strip() or "1"
        if raw.isdigit() and 1 <= int(raw) <= len(apps):
            app_name = apps[int(raw) - 1]
            break
        _warn("請輸入有效數字")

    if app_name == "自訂 MJPEG URL":
        url = input("  輸入完整 URL（例：http://192.168.1.5:8080/video）: ").strip()
        return url, DEFAULT_PORT, app_name

    ip   = input(f"  手機 IP 位址（例：192.168.1.5）: ").strip()
    port_raw = input(f"  Port（預設 {DEFAULT_PORT}）: ").strip() or str(DEFAULT_PORT)
    port = int(port_raw) if port_raw.isdigit() else DEFAULT_PORT
    return ip, port, app_name


def _build_capture(device: dict) -> cv2.VideoCapture:
    """根據選擇的裝置建立並設定 VideoCapture。"""
    if device["type"] == "local":
        backend = device.get("backend", cv2.CAP_ANY)
        cap = cv2.VideoCapture(device["index"], backend)
    else:
        cap = cv2.VideoCapture(device["url"])

    # 統一設定畫質與緩衝
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DEFAULT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          DEFAULT_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # 減少串流延遲

    return cap


def select_camera() -> cv2.VideoCapture:
    """
    主要公開函式：
    顯示互動選單讓使用者選擇攝影機來源，
    回傳已開啟且設定好的 cv2.VideoCapture 物件。
    """
    print()
    _title("📷  Virtual Coach — 攝影機設定")

    # ── 先自動掃描本機攝影機 ──
    local_cams = scan_local_cameras()

    # ── 建立選項清單 ──
    options: list[dict] = []

    for cam in local_cams:
        options.append(cam)

    options.append({
        "type":  "ip_new",
        "label": "手機鏡頭（IP 串流）— 輸入 IP 位址",
        "res":   "",
    })

    # ── 顯示選單 ──
    print()
    _title("請選擇攝影機來源")
    for i, opt in enumerate(options, 1):
        res_str = f"  [{opt['res']}]" if opt.get("res") else ""
        print(f"  {i}.  {opt['label']}{res_str}")
    print()

    while True:
        raw = input(f"  輸入數字（1–{len(options)}）: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            chosen = options[int(raw) - 1]
            break
        _warn("請輸入有效數字")

    # ── 手機 IP 串流：引導輸入並測試 ──
    if chosen["type"] == "ip_new":
        while True:
            ip, port, app_name = _input_ip_info()
            print()
            _title("🔗  測試手機串流連線…")
            result = probe_ip_camera(ip, port, app_name)
            if result:
                chosen = result
                break
            print()
            _warn("無法連線，請確認：")
            _info("1. 手機與電腦在同一 Wi-Fi 網路")
            _info("2. 手機 App 已啟動並顯示串流畫面")
            _info("3. IP 位址與 Port 正確")
            print()
            retry = input("  重新輸入？(y/n，預設 y): ").strip().lower() or "y"
            if retry != "y":
                _warn("取消手機串流，改用預設電腦鏡頭（index 0）")
                chosen = {"type": "local", "index": 0,
                          "backend": cv2.CAP_ANY, "label": "本機攝影機 #0", "res": ""}
                break

    # ── 建立 VideoCapture ──
    print()
    _title(f"▶  啟動  {chosen['label']}")
    cap = _build_capture(chosen)

    if not cap.isOpened():
        _warn("攝影機開啟失敗，嘗試預設裝置（index 0）…")
        cap = cv2.VideoCapture(0)

    # ── 顯示最終解析度 ──
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _ok(f"攝影機就緒  {w}×{h}")
    _hr()
    print()

    return cap


# ──────────────────────────────────────────────
# 整合範例：直接取代原本的 cv2.VideoCapture(0)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # 測試模組是否正常運作
    cap = select_camera()

    print("按 Q 退出預覽…")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera Preview", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
