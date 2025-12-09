# **Robot Clone Tracker**

### *A Real-Time Human Motion Mirroring.*

## **Overview**

**Robot Clone Tracker** started as a small personal experiment while I was exploring what’s possible with lightweight Computer Vision systems — 
especially ones that feel responsive, visual, and fun to interact with.

I didn’t want to build “just another pose-estimation demo.”
My goal was to create something that actually feels alive: a system that reads your movements, reacts instantly, and paints a glowing robotic version of you on the screen.

This project blends MediaPipe Pose, OpenCV, and a bit of creative math to build an experience where your real-world actions turn into a clean, animated robot-style visual. 
Along the way, I added frame-level analytics like FPS, pose confidence, and motion intensity to understand how the system behaves under different conditions.

## **Features**

* Real-Time Human Pose Tracking
* A glowing “Robot-Style” Skeleton overlay
* Movement intensity tracking (frame-to-frame changes)
* FPS and model confidence monitoring
* Simple analytics graph after the session ends

The final output feels responsive and visually pleasing without being too complex.


## **Tech Stack**

* **Python**
* **OpenCV** – video capture & drawing
* **MediaPipe Pose** – landmark detection
* **NumPy** – calculations for motion
* **Matplotlib** – basic analytics plots


## **How It Works**

The flow behind the tracker is pretty simple:

1. The webcam sends a continuous stream of frames.
2. MediaPipe extracts pose landmarks for each frame.
3. The coordinates are converted into pixel values.
4. A robot-like skeleton is drawn on top of the video feed.
5. A few previous frames are stored to create motion trails.
6. FPS, confidence, and movement values are logged quietly in the background.
7. When you close the window, the graphs are displayed automatically.

Nothing complicated — just a clean loop that runs smoothly and reacts to movement in real time.


## **Why I Built It**

I wanted a hands-on way to explore:

* Real-time computer vision
* Human-centered interaction design
* Lightweight motion tracking
* How performance metrics shape system behavior

And honestly, I just wanted something that looks cool on screen — a simple project that feels fun but still teaches a lot about how modern CV pipelines work.


## **Future Add-Ons**

* Better motion smoothing
* Multi-person tracking
* Gesture triggers
* Activity classification (jumping, stretching, posture detection)
* Heatmaps for movement
* Saving real-time recordings with overlays
* Exportable analytics reports
