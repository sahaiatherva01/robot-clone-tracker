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
cap.set(3, 1280)
cap.set(4, 720)

PRIMARY_BLUE = (30, 30, 180)  
ACCENT_CYAN = (255, 180, 80)  
WHITE_CORE = (255, 255, 220)   
SHADOW = (10, 10, 80)           

# ---- Data Recording ----
fps_history = []
confidence_history = []
motion_history = []
frame_times = []

MAX_TRAIL_FRAMES = 5
trail_buffer = []
prev_pts = None

start_time = time.time()
last_time = start_time

def draw_robot_head(img, center, scale=1.0):
    x, y = center
    r = int(45 * scale)
    cv2.ellipse(img, (x, y), (r, int(r * 0.85)), 0, 0, 360, PRIMARY_BLUE, -1, cv2.LINE_AA)
    visor_h = int(r * 0.45)
    visor_w = int(r * 1.4)
    cv2.ellipse(img, (x, y), (visor_w // 2, visor_h // 2), 0, 0, 360, ACCENT_CYAN, -1, cv2.LINE_AA)
    cv2.ellipse(img, (x, y), (visor_w // 2 - 6, visor_h // 2 - 4), 0, 0, 360, (10, 40, 120), 2, cv2.LINE_AA)

def draw_robot_torso(img, sL, sR, hL, hR):
    pts = np.array([
        [sL[0], sL[1] + 6],
        [sR[0], sR[1] + 6],
        [hR[0], hR[1] - 10],
        [hL[0], hL[1] - 10]
    ], np.int32)

    cv2.fillPoly(img, [pts], PRIMARY_BLUE)
    cv2.polylines(img, [pts], True, ACCENT_CYAN, 3, cv2.LINE_AA)

    c1 = ((sL[0] + sR[0]) // 2, (sL[1] + sR[1]) // 2 + 20)
    for t in [0.25, 0.5, 0.75]:
        Lp = (int(c1[0] + (hL[0] - c1[0]) * t), int(c1[1] + (hL[1] - c1[1]) * t))
        Rp = (int(c1[0] + (hR[0] - c1[0]) * t), int(c1[1] + (hR[1] - c1[1]) * t))
        cv2.line(img, Lp, Rp, (80,130,230), 2, cv2.LINE_AA)

    core_x = (sL[0] + sR[0] + hL[0] + hR[0]) // 4
    core_y = (sL[1] + sR[1] + hL[1] + hR[1]) // 4
    cv2.circle(img, (core_x, core_y), 18, ACCENT_CYAN, -1)
    cv2.circle(img, (core_x, core_y), 10, PRIMARY_BLUE, -1)

def draw_spine(img, sL, sR, hL, hR):
    top = ((sL[0]+sR[0])//2, (sL[1]+sR[1])//2)
    bot = ((hL[0]+hR[0])//2, (hL[1]+hR[1])//2)

    cv2.line(img, top, bot, SHADOW, 16, cv2.LINE_AA)
    cv2.line(img, top, bot, PRIMARY_BLUE, 10, cv2.LINE_AA)

    for t in np.linspace(0.2, 0.8, 4):
        px = int(top[0] + (bot[0] - top[0]) * t)
        py = int(top[1] + (bot[1] - top[1]) * t)
        cv2.circle(img, (px,py), 8, ACCENT_CYAN, -1)

def draw_cylinder_limb(img, p1, p2):
    length = int(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))
    thickness = max(14, length // 6)

    cv2.line(img, p1, p2, SHADOW, thickness + 6, cv2.LINE_AA)
    cv2.line(img, p1, p2, PRIMARY_BLUE, thickness, cv2.LINE_AA)

    cuff = thickness // 2 + 4
    for pt in [p1, p2]:
        cv2.circle(img, pt, cuff, ACCENT_CYAN, -1)
        cv2.circle(img, pt, cuff - 5, PRIMARY_BLUE, -1)

def draw_joint_ball(img, center):
    cv2.circle(img, center, 18, SHADOW, -1)
    cv2.circle(img, center, 12, ACCENT_CYAN, -1)
    cv2.circle(img, center, 6, WHITE_CORE, -1)


while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = now - last_time
    last_time = now
    fps = 1/dt if dt > 0 else 0

    fps_history.append(fps)
    frame_times.append(now - start_time)

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    glow = np.zeros_like(frame)

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark
        
        confidence_history.append(float(landmarks[0].visibility) if len(landmarks) > 0 else 0.0)

        pts = [(int((1 - l.x) * w), int(l.y * h)) for l in landmarks]

        if prev_pts is not None and len(prev_pts) == len(pts):
            movement = np.mean([np.linalg.norm(np.array(pts[i]) - np.array(prev_pts[i])) for i in range(len(pts))])
        else:
            movement = 0
        motion_history.append(movement)
        prev_pts = pts.copy()

        def _pt(idx):
            return pts[idx] if 0 <= idx < len(pts) else (w // 2, h // 2)

        nose = _pt(0)
        Ls, Rs = _pt(11), _pt(12)
        Le, Re = _pt(13), _pt(14)
        Lw, Rw = _pt(15), _pt(16)
        Lh, Rh = _pt(23), _pt(24)
        Lk, Rk = _pt(25), _pt(26)
        La, Ra = _pt(27), _pt(28)

        draw_robot_head(glow, nose)
        draw_robot_torso(glow, Ls, Rs, Lh, Rh)
        draw_spine(glow, Ls, Rs, Lh, Rh)

        draw_cylinder_limb(glow, Ls, Le)
        draw_cylinder_limb(glow, Le, Lw)
        draw_cylinder_limb(glow, Rs, Re)
        draw_cylinder_limb(glow, Re, Rw)

        draw_cylinder_limb(glow, Lh, Lk)
        draw_cylinder_limb(glow, Lk, La)
        draw_cylinder_limb(glow, Rh, Rk)
        draw_cylinder_limb(glow, Rk, Ra)

        for j in [Ls, Rs, Le, Re, Lw, Rw, Lh, Rh, Lk, Rk, La, Ra]:
            draw_joint_ball(glow, j)

        frame = cv2.addWeighted(frame, 0.55, glow, 0.9, 0)

    else:
        confidence_history.append(0)
        motion_history.append(0)

    cv2.imshow("HUMANOID ROBOT CLONE", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(frame_times, fps_history, linewidth=2)
plt.title("FPS Over Time")
plt.xlabel("Time (s)")
plt.ylabel("FPS")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(frame_times, confidence_history, color="green", linewidth=2)
plt.title("Pose Confidence Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Confidence")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(frame_times, motion_history, color="red", linewidth=2)
plt.title("Motion Intensity Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Movement")
plt.grid(True)

plt.tight_layout()
plt.show()
