from flask import Flask, Response, jsonify, send_file
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os

from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


# Check if camera is available
camera_available = True
try:
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        camera_available = False
        print("WARNING: No camera detected!")
    test_cap.release()
except:
    camera_available = False
    print("WARNING: Camera initialization failed!")

# Try old API first
try:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    use_old_api = True
    print("Using Computer Vision API")
except:
    # Use new API
    mp_pose = mp.tasks.vision.PoseLandmarker
    pose_options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    pose = mp_pose.create_from_options(pose_options)
    use_old_api = False
    print("Using Advanced Computer Vision API")

PRIMARY_BLUE = (30, 30, 180)
ACCENT_CYAN = (255, 180, 80)
WHITE_CORE = (255, 255, 220)
SHADOW = (10, 10, 80)

# Global variables for stats
start_time = None
stats = {
    'fps': 0,
    'confidence': 0,
    'motion': 0,
    'keypoints': 0,
    'fps_history': [],
    'confidence_history': [],
    'motion_history': [],
    'camera_status': 'active' if camera_available else 'inactive',
    'start_time': None
}

prev_pts = None
last_time = time.time()
cap = None
recording_started = False

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

def generate_frames():
    global prev_pts, last_time, stats, cap, start_time, recording_started
    
    # Initialize start_time at the beginning of recording
    if start_time is None:
        start_time = time.time()
        recording_started = True
        stats['start_time'] = start_time
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        # Return error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "CAMERA NOT AVAILABLE", (100, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    
    print("Camera opened successfully!")
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
                
            now = time.time()
            dt = now - last_time
            last_time = now
            fps = 1/dt if dt > 0 else 0
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if use_old_api:
                results = pose.process(rgb)
                pose_detected = results.pose_landmarks is not None
                if pose_detected:
                    landmarks = results.pose_landmarks.landmark
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = pose.detect(mp_image)
                pose_detected = len(results.pose_landmarks) > 0
                if pose_detected:
                    landmarks = results.pose_landmarks[0].landmarks
            
            glow = np.zeros_like(frame)
            
            if pose_detected:
                confidence = float(landmarks[0].visibility) if len(landmarks) > 0 else 0.0
                
                pts = [(int((1 - l.x) * w), int(l.y * h)) for l in landmarks]
                
                if prev_pts is not None and len(prev_pts) == len(pts):
                    movement = np.mean([np.linalg.norm(np.array(pts[i]) - np.array(prev_pts[i])) for i in range(len(pts))])
                else:
                    movement = 0
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
                
                elapsed = time.time() - start_time
                stats['fps'] = fps
                stats['confidence'] = confidence
                stats['motion'] = movement
                stats['keypoints'] = len(landmarks)
                stats['fps_history'].append({'time': elapsed, 'value': fps})
                stats['confidence_history'].append({'time': elapsed, 'value': confidence})
                stats['motion_history'].append({'time': elapsed, 'value': movement})
                stats['camera_status'] = 'active'
                
                # Keep all history for full timeline
            else:
                elapsed = time.time() - start_time
                stats['confidence'] = 0
                stats['motion'] = 0
                stats['keypoints'] = 0
                stats['fps_history'].append({'time': elapsed, 'value': fps})
                stats['confidence_history'].append({'time': elapsed, 'value': 0})
                stats['motion_history'].append({'time': elapsed, 'value': 0})
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            break
    
    if cap:
        cap.release()

@app.route('/')
def index():
    # Serve the HTML file
    html_path = os.path.join(os.path.dirname(__file__), 'index.html')
    if os.path.exists(html_path):
        return send_file(html_path)
    else:
        return """
        <html>
        <body>
            <h1>Advanced Pose Tracker Backend</h1>
            <p>Backend is running!</p>
            <p>Video feed: <a href="/video_feed">/video_feed</a></p>
            <p>Stats: <a href="/stats">/stats</a></p>
            <p><strong>Note:</strong> Place index.html in the same directory as server.py</p>
        </body>
        </html>
        """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    return jsonify(stats)

@app.route('/test')
def test():
    return jsonify({
        'status': 'Backend is running',
        'camera_available': camera_available,
        'endpoints': {
            'video': '/video_feed',
            'stats': '/stats'
        }
    })

@app.route('/reset')
def reset():
    global start_time, stats
    start_time = None
    stats['fps_history'] = []
    stats['confidence_history'] = []
    stats['motion_history'] = []
    return jsonify({'status': 'Stats reset'})

if __name__ == '__main__':
    print("="*50)
    print("Starting Advanced Pose Tracker Backend")
    print("="*50)
    print(f"Camera Available: {camera_available}")
    print("Server running on: http://localhost:5000")
    print("Video feed: http://localhost:5000/video_feed")
    print("Stats API: http://localhost:5000/stats")
    print("\n⚠️  Press CTRL+C to stop the server\n")
    print("="*50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n\n" + "="*50)
        print("Server stopped by user")
        print("="*50)
        if cap:
            cap.release()
        pose.close()