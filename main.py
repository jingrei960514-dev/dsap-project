import cv2
import mediapipe as mp

# 初始化 MediaPipe Pose 模組
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,        # 處理影片流而非單張圖片
    model_complexity=1,             # 複雜度 (0, 1, 2)，越高越準但越吃效能
    enable_segmentation=False,       # 是否需要人體去背
    min_detection_confidence=0.5,    # 偵測門檻
    min_tracking_confidence=0.5      # 追蹤門檻
)

# 啟動攝影機 (通常內建為 0)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 轉換顏色空間：OpenCV 預設是 BGR，但 MediaPipe 需要 RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. 進行姿勢偵測
    results = pose.process(image_rgb)

    # 3. 繪製關節點在原始畫面（BGR）上
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    # 顯示視窗
    cv2.imshow('Virtual Coach - Pose Tracking', frame)

    # 按下 'q' 退出
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()