🚆 RailGuard AI – Multi-Camera Railway Track Monitoring System

RailGuard AI is a real-time, AI-driven railway track monitoring system built using Python, OpenCV, and CustomTkinter.
It supports 1–4 cameras simultaneously and detects track intrusion, obstacles, and tampering using motion analysis and computer vision.

🎯 Focus: Railway safety, intrusion detection & tampering alerts

---

🔥 Features

🎥 Multi-camera support (1–4 camera(s))

📷 Supports:
- Laptop/USB webcams (`webcam-0`, `webcam-1`, …)
- IP cameras (e.g. Android phones via IP Webcam)

🧵 Each camera runs in its own background thread

🛤️ Manual track region selection per camera  
- For each camera, you draw a rectangle around the railway track (ROI) once at startup.

🚨 Intelligent status detection:
- 🟢 **GREEN** – Track clear  
- 🟡 **YELLOW** – Short-lived obstacle (movement/crossing)  
- 🔴 **RED** – Continuous obstacle / tampering (motion persists ≥ 2 seconds)

🧠 Smart alert workflow for RED events:
- When RED is detected:
  - The system announces an alert using text-to-speech, including the affected camera IDs.
  - It listens for the operator’s voice reply:
    - “All is well” → Treat as safe, do not save captures.
    - “Taking action” or any other reply → Treat as incident, save captures.
    - No reply within timeout → Play a danger alarm sound (`danger.mp3`) and save captures.
  - All RED events that occur while this workflow is running are merged into a **single alert session**.

📸 Capture handling:
- RED frames are buffered during an alert session.
- If the operator says **“All is well”** → all buffered captures for that session are **discarded**.
- If the operator says **“taking action”** or there is **no reply** → all buffered captures are **saved** under `captures/`.

⏸️ Temporary monitoring disable per camera:
- Each camera row has a checkbox:  
  **“Temporarily disable track monitoring for this camera”**
- When enabled:
  - The camera video stays ON as a normal camera.
  - Track monitoring, status updates (GREEN/YELLOW/RED), and alerts for that camera are paused.
  - On the camera window:
    - Status shows: **“Status: MONITORING OFF”**
    - A message appears: **“Track monitoring is temporarily disabled.”**
  - At the bottom of the main window:
    - Single camera off:  
      `Track monitoring is currently disabled for camera CAM01.`
    - Multiple cameras off:  
      `Track monitoring is currently disabled for multiple cameras (IDs: CAM01, CAM02).`

📊 Live plots:
- Motion intensity over time
- Track occupation duration
- Status timeline (GREEN/YELLOW/RED)

🖥️ Clean CustomTkinter GUI

🪟 Individual OpenCV video window per camera

📜 Per-camera live log (updated every second).

---

📁 Project Structure

```text
railguard_ai/
│
├── railguard.py          # Main application
├── requirements.txt      # Python dependencies
├── app_icon.ico          # App icon (Windows)
├── captures/             # Saved RED alert snapshots (auto-created)
├── logs/                 # Per-camera CSV logs (auto-created)
├── danger.mp3            # Danger alarm sound (expected in root folder)
├── .gitignore
└── README.md             # Documentation
```
⚙️ System Requirements

    OS: Windows / Linux / macOS
    Python: 3.9 – 3.12 recommended
    Hardware:

    At least one webcam or IP camera
    Minimum 4 GB RAM (8 GB recommended)
    Microphone and speakers (or headphones) for:
        Text-to-speech alerts
        Voice reply recognition
        Danger alarm playback

🐍 Installation Guide

1️⃣ Clone the Repository
``` Bash
git clone https://github.com/dyrpsf/railguard_ai.git
cd railguard_ai
```
2️⃣ Create a Virtual Environment (Recommended)

Windows:

``` Bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

``` Bash
python3 -m venv venv
source venv/bin/activate
```

3️⃣ Install Dependencies

``` Bash
pip install -r requirements.txt
```

⚠️ If opencv-python fails to install, try:

```Bash
pip install opencv-python-headless
```

📷 IP Camera Setup (Optional)

You can use a mobile phone as an IP camera.

Android

    Install IP Webcam or similar app from Play Store.
    Start the server.
    Use a URL like:
        http://192.168.0.101:8080/video

iOS

    Use apps like:
        DroidCam
        IP Camera Lite

▶️ Running the Application

```Bash
python railguard.py
```

🖥️ How to Use the GUI

1. Select number of cameras (1–4) at the top.

2. For each camera:

    Choose a source:
        webcam-0, webcam-1, etc.
        or ip-url and enter your IP camera URL.
    (Optional) Check or uncheck:
        “Temporarily disable track monitoring for this camera”
            ON → Camera acts as a normal video feed; no detection/alerts.
            OFF → Full monitoring and alerting enabled.

3. Click Start Monitoring.

4. For each camera, a window will appear to select the track region (ROI):

    Draw a rectangle around the railway track.
    Press ENTER/SPACE to confirm, or ESC to cancel.
    If no ROI is selected, that camera will show “NO ROI (Select track)” and will not monitor.
    
5. Watch:
    Live camera windows.
    Status updates per camera.
    Motion & status plots at the bottom.
    Per-camera logs and event history in the GUI.

6. To stop:

    Click Stop in the main window.
    Or press `q` in any camera window to close that specific window.

🚨 Alert Logic Explained

Condition	Status	Meaning
No motion on track	🟢 GREEN	Track clear
Brief motion	🟡 YELLOW	Crossing / transient object
Continuous motion ≥ 2 sec	🔴 RED	Obstacle / possible tampering

When a camera first enters RED:

1. A global alert session starts (or joins an existing active one).

2. The system:

    Announces: a spoken alert mentioning all cameras currently in RED.
    Listens for up to 15 seconds for a microphone reply:
        If you say “All is well”:
            Acknowledgement: “It is good that everything is fine.”
            All buffered captures for this alert session are discarded.
        If you say “taking action” or something else:
            Acknowledgement: “Thank you for taking action. I have also saved the captures.”
            All buffered captures for this alert session are saved.
        If there is no reply (timeout / silence / mic unavailable):
            The system plays danger.mp3.
            All buffered captures for this alert session are saved.
3. Any additional RED events that occur while the workflow is still running:

    Are merged into the same alert session.
    Their frames and camera IDs are added to the same capture set.
    No overlapping or repeated alert speech is triggered.
📸 Saved captures are stored as:

``` Text
captures/
└── CAM01_20260103_142530_RED.jpg
```

(if the session outcome requires saving).

📊 Graph Explanation

Top Graph

    Average motion per second (per camera)
    Smoothed motion trend
    Motion threshold line
Bottom Graph

    Status timeline:
        -1 = NO_ROI
        0 = GREEN
        1 = YELLOW
        2 = RED
    Track occupied duration (scaled)

🛠️ Configuration Parameters

You can tweak these in railguard.py:

```Python

MIN_MOTION_AREA = 200.0      # Motion threshold to consider the track "occupied"
OBSTACLE_MIN_AREA = 300.0    # Minimum contour area to consider an obstacle
TAMPERING_MIN_TIME = 2.0     # Seconds of continuous motion before status becomes RED
HISTORY_SECONDS = 60         # Seconds shown in plots
MAX_CAMERAS = 4              # Maximum supported cameras
```

🧯 Troubleshooting

Camera not opening

    Check camera index (webcam-0, webcam-1, …).
    Ensure no other app is using the camera.
Black screen / no detection

    Improve lighting near the track.
    Make sure the track is clearly visible in the ROI.
    Adjust MIN_MOTION_AREA if needed.
IP camera lag

    Ensure the PC and phone are on the same Wi-Fi network.
    Reduce resolution / frame rate in the IP camera app.
No sound / TTS errors

    Check system audio output (speakers/headphones).
    Ensure pyttsx3 and playsound3 installed correctly.
    On some systems, you may need extra audio backends (e.g. pyaudio).
    Microphone / speech recognition issues

    Ensure a working microphone is selected as default input.
    Check that speech_recognition and its dependencies are installed.
    If the mic is unavailable or recognition fails, the system treats it as no reply and plays the danger alarm.

🚀 Future Improvements

    YOLO-based object detection
    SMS / Email / WhatsApp alerts
    Cloud dashboard
    Railway-specific dataset training
    GPU acceleration
    Centralized multi-station monitoring

📜 License

    This project is open-source and intended for educational & research purposes.

👨‍💻 Author

Deepak Yadav <br>
B.Tech. CSE Core <br>
VIT Bhopal University