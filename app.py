import cv2
import mediapipe as mp
import numpy as np
import time
import math
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(4, 1080)
cap.set(4, 1080)

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28)
]

MAX_TRAIL_FRAMES = 5
trail_buffer = []

start_time = time.time()
last_time = start_time

# ---- DATA LOGGING ARRAYS ----
fps_history = []
confidence_history = []
motion_history = []
frame_times = []

def robot_palette(t):
    base = int(150 + 80 * math.sin(t * 1.5))
    glow = int(220 + 35 * math.sin(t * 2.2))
    return (glow, base + 20, base - 10)


def draw_robot_line(img, p1, p2, color):
    cv2.line(img, p1, p2, tuple(min(c + 80, 255) for c in color), 12, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, 5, cv2.LINE_AA)


def draw_robot_joint(img, center, color):
    cv2.circle(img, center, 10, color, -1, cv2.LINE_AA)
    cv2.circle(img, center, 4, (255, 255, 255), -1, cv2.LINE_AA)

prev_pts = None  # For motion intensity

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    fps = 1 / dt if dt > 0 else 0
    fps_history.append(fps)
    frame_times.append(current_time - start_time)

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    h, w, _ = frame.shape
    t = current_time - start_time
    robot_color = robot_palette(t)

    glow = np.zeros_like(frame)

    if results.pose_landmarks:

        # Save confidence score
        conf = results.pose_landmarks.landmark[0].visibility
        confidence_history.append(conf)

        pts = [(int((1 - lm.x) * w), int(lm.y * h))
               for lm in results.pose_landmarks.landmark]

        # Motion intensity (difference between frames)
        if prev_pts is not None:
            movement = np.mean([np.linalg.norm(np.array(pts[i]) - np.array(prev_pts[i]))
                                for i in range(len(pts))])
        else:
            movement = 0
        motion_history.append(movement)
        prev_pts = pts.copy()

        # Skeleton
        for i, j in POSE_CONNECTIONS:
            draw_robot_line(glow, pts[i], pts[j], robot_color)

        # Joints
        frame_joints = []
        for (x, y) in pts:
            if 0 <= x < w and 0 <= y < h:
                frame_joints.append((x, y))
                draw_robot_joint(glow, (x, y), robot_color)

        trail_buffer.append(frame_joints)
        if len(trail_buffer) > MAX_TRAIL_FRAMES:
            trail_buffer.pop(0)

        for idx, joints in enumerate(trail_buffer[:-1]):
            alpha = idx / MAX_TRAIL_FRAMES
            faded = tuple(int(c * alpha * 0.3) for c in robot_color)

            for (x, y) in joints:
                cv2.circle(glow, (x, y), 3, faded, -1, cv2.LINE_AA)

        frame = cv2.addWeighted(frame, 0.55, glow, 0.85, 0)

    else:
        confidence_history.append(0)
        motion_history.append(0)

        text = "ROBOT SCANNING..."
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.putText(frame, text,
                    ((w - size[0]) // 2, (h + size[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, robot_color, 2, cv2.LINE_AA)
    
    
    cv2.imshow("ROBOT CLONE TRACKER ", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

plt.figure(figsize=(12, 8))

# FPS Graph
plt.subplot(3, 1, 1)
plt.plot(frame_times, fps_history, label="FPS", linewidth=2)
plt.title("System FPS Over Time")
plt.xlabel("Time (s)")
plt.ylabel("FPS")
plt.grid(True)

# Pose Confidence Graph
plt.subplot(3, 1, 2)
plt.plot(frame_times, confidence_history, color="green", linewidth=2)
plt.title("Pose Detection Confidence")
plt.xlabel("Time (s)")
plt.ylabel("Confidence")
plt.grid(True)

# Motion Graph
plt.subplot(3, 1, 3)
plt.plot(frame_times, motion_history, color="red", linewidth=2)
plt.title("Movement Intensity (Robot Tracking)")
plt.xlabel("Time (s)")
plt.ylabel("Motion Value")
plt.grid(True)

plt.tight_layout()
plt.show()