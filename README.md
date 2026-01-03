🚆 RailGuard AI – Multi-Camera Railway Track Monitoring System

RailGuard AI is a real-time, AI-driven railway track monitoring system built using Python, OpenCV, and CustomTkinter.
It supports 1–4 cameras simultaneously and detects track intrusion, obstacles, and tampering using motion analysis and computer vision.

🏆 Built for Hack4Delhi
🎯 Focus: Railway safety, intrusion detection & tampering alerts

🔥 Features

🎥 Multi-camera support (1–4 cameras)

📷 Supports:

Laptop/USB webcams (webcam-0, webcam-1, …)

IP cameras (e.g. Android phone via IP Webcam)

🧵 Each camera runs in its own background thread

🛤️ Automatic railway track region detection

🚨 Intelligent status detection:

🟢 GREEN – Track clear

🟡 YELLOW – Short-lived obstacle (movement/crossing)

🔴 RED – Continuous obstacle / tampering

📸 Automatic snapshot capture on RED alert

📊 Live plots:

Motion intensity

Track occupation duration

Status timeline

🖥️ Clean CustomTkinter GUI

🪟 Individual OpenCV video window per camera

📜 Per-camera live log (updated every second)

📁 Project Structure
railguard_ai/
│
├── railguard.py          # Main application
├── requirements.txt      # Python dependencies
├── app_icon.ico          # App icon (Windows)
├── captures/             # Saved RED alert snapshots (auto-created)
├── .gitignore
└── README.md             # Documentation

⚙️ System Requirements

OS: Windows / Linux / macOS

Python: 3.9 – 3.12 recommended

Hardware:

Webcam or IP camera

Minimum 4 GB RAM (8 GB recommended)

🐍 Installation Guide
1️⃣ Clone the Repository
git clone https://github.com/dyrpsf/railguard_ai.git
cd railguard_ai

2️⃣ Create a Virtual Environment (Recommended)
Windows
python -m venv venv
venv\Scripts\activate

Linux / macOS
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt


⚠️ If opencv-python fails, try:

pip install opencv-python-headless

📷 IP Camera Setup (Optional)

You can use a mobile phone as an IP camera.

Android

Install IP Webcam from Play Store

Start server

Use URL like:

http://192.168.0.101:8080/video

iOS

Use apps like:

DroidCam

IP Camera Lite

▶️ Running the Application
python railguard.py

🖥️ How to Use the GUI

Select number of cameras (1–4)

For each camera:

Choose:

webcam-0, webcam-1, etc

OR ip-url and enter IP camera URL

Click Start Monitoring

Watch:

Live camera windows

Status updates

Motion & status plots

Press Stop to end monitoring

Press q in any camera window to close it manually.

🚨 Alert Logic Explained
Condition	Status	Meaning
No motion on track	🟢 GREEN	Track clear
Brief motion	🟡 YELLOW	Crossing / transient object
Continuous motion ≥ 2 sec	🔴 RED	Obstacle / tampering

📸 On RED, a snapshot is saved automatically:

captures/
└── CAM01_20260103_142530_RED.jpg

📊 Graph Explanation
Top Graph

Average motion per second

Smoothed motion trend

Motion threshold line

Bottom Graph

Status timeline (0=GREEN, 1=YELLOW, 2=RED)

Track occupied duration (scaled)

🛠️ Configuration Parameters

You can tweak these in railguard.py:

MIN_MOTION_AREA = 500.0
OBSTACLE_MIN_AREA = 800.0
TAMPERING_MIN_TIME = 2.0
HISTORY_SECONDS = 60
MAX_CAMERAS = 4

🧯 Troubleshooting
Camera not opening

Check camera index (webcam-0, webcam-1)

Ensure no other app is using the camera

Black screen / no detection

Improve lighting

Adjust MIN_MOTION_AREA

Ensure track is visible in frame

IP camera lag

Use same Wi-Fi network

Reduce camera resolution

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

Deepak Yadav
B.Tech. CSE Core at VIT Bhopal University
AI • Computer Vision • Railway Safety
Hack4Delhi Participant
