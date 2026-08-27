import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Set
from pathlib import Path

import cv2
import numpy as np
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import queue
import pyttsx3
import speech_recognition as sr
from playsound3 import playsound


MIN_MOTION_AREA = 200.0
OBSTACLE_MIN_AREA = 300.0
TAMPERING_MIN_TIME = 2.0
HISTORY_SECONDS = 60
MAX_CAMERAS = 4
TRACK_MARGIN = 25


@dataclass
class CameraConfig:
    camera_id: str           # "CAM01"
    source: Union[int, str]  # webcam index or IP URL


@dataclass
class SecondReport:
    camera_id: str
    timestamp: float
    avg_motion: float
    occupied_duration: float
    status: str  # "GREEN", "YELLOW", "RED", "NO_ROI", or "ERROR_*"


@dataclass
class AlertSession:
    """
    Global alert session that merges all RED events occurring
    while the alert workflow (speech / mic / alarm) is running.
    """
    id: int
    start_time: float
    captures: List[Dict[str, Any]]
    workflow_started: bool = False
    resolved: bool = False
    camera_ids: Set[str] = None  # cameras involved in this session


class RailGuardMultiCamApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("RailGuard AI Multi-Camera - Track Monitor")
        self.iconbitmap("app_icon.ico")
        self.geometry("1200x800")

        # Global state
        self.running = False
        self.monitor_threads: List[threading.Thread] = []
        self.data_queue: "queue.Queue[SecondReport]" = queue.Queue()
        self.history: Dict[str, List[SecondReport]] = {}   # camera_id -> list[SecondReport]
        self.current_status: Dict[str, str] = {}           # camera_id -> status
        self.camera_rows: Dict[str, Dict[str, ctk.CTkBaseClass]] = {}  # widgets per camera

        # New: per-camera monitoring disable flags
        self.monitoring_disabled: Dict[str, bool] = {}

        # Extra state for metrics / risk / logs
        self.red_counts: Dict[str, int] = {}
        self.risk_scores: Dict[str, int] = {}
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.csv_paths: Dict[str, Path] = {}
        self.session_start: float = 0.0
        self.session_ts: str = ""

        # Directory to save RED-event snapshots
        self.capture_dir = Path("captures")
        self.capture_dir.mkdir(exist_ok=True)

        self.alert_lock = threading.Lock()
        self.current_alert_session: Optional[AlertSession] = None
        self.next_alert_session_id: int = 1

        # TTS engine (if available)
        self.tts_engine = None
        if pyttsx3 is not None:
            try:
                self.tts_engine = pyttsx3.init()
            except Exception as e:
                print("[WARN] Failed to initialize TTS engine:", e)
                self.tts_engine = None

        # Build UI
        self._build_ui()

        # Schedule periodic UI update
        self.after(500, self.update_ui)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _build_ui(self):
        # Top control frame
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.pack(side="top", fill="x", padx=10, pady=10)

        # Number of cameras
        self.num_cams_var = ctk.IntVar(value=1)
        ctk.CTkLabel(
            self.top_frame,
            text="Number of cameras:",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.num_cams_menu = ctk.CTkOptionMenu(
            self.top_frame,
            values=[str(i) for i in range(1, MAX_CAMERAS + 1)],
            variable=self.num_cams_var,
            command=self._on_num_cams_changed,
        )
        self.num_cams_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Start / Stop buttons
        self.start_button = ctk.CTkButton(
            self.top_frame, text="Start Monitoring", command=self.start_monitoring
        )
        self.start_button.grid(row=0, column=2, padx=10, pady=5)

        self.stop_button = ctk.CTkButton(
            self.top_frame, text="Stop", fg_color="red", command=self.stop_monitoring
        )
        self.stop_button.grid(row=0, column=3, padx=10, pady=5)

        # Progress bar
        self.progress = ctk.CTkProgressBar(self.top_frame, width=200)
        self.progress.grid(row=1, column=0, columnspan=4, padx=5, pady=(5, 0), sticky="w")
        self.progress.set(0.0)
        self.progress.stop()

        # Error label
        self.error_label = ctk.CTkLabel(
            self.top_frame,
            text="",
            text_color="red",
            font=ctk.CTkFont(size=12),
        )
        self.error_label.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky="w")

        # Scrollable frame that will contain per-camera sections
        self.cams_frame = ctk.CTkScrollableFrame(self, height=260)
        self.cams_frame.pack(side="top", fill="x", padx=10, pady=10)

        # Bottom frame for plots
        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.ax_motion = self.fig.add_subplot(211)
        self.ax_status = self.fig.add_subplot(212, sharex=self.ax_motion)

        self.ax_motion.set_ylabel("Avg Motion")
        self.ax_status.set_ylabel("Status\n-1=NoROI,0=G,1=Y,2=R")
        self.ax_status.set_xlabel("Seconds (most recent at 0)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.bottom_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        # Global bottom status bar for monitoring-off info
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 10))

        self.monitoring_off_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="orange",
        )
        self.monitoring_off_label.pack(side="left", padx=5, pady=5)

        self._rebuild_camera_rows()

    def _on_num_cams_changed(self, _value: str):
        if self.running:
            self.error_label.configure(text="Stop monitoring before changing camera count.")
            return
        self.error_label.configure(text="")
        self._rebuild_camera_rows()

    def _rebuild_camera_rows(self):
        for child in self.cams_frame.winfo_children():
            child.destroy()
        self.camera_rows.clear()
        self.history.clear()
        self.current_status.clear()
        self.red_counts.clear()
        self.risk_scores.clear()
        self.monitoring_disabled.clear()

        n = self.num_cams_var.get()
        for i in range(n):
            cam_id = f"CAM{i+1:02d}"

            cam_frame = ctk.CTkFrame(self.cams_frame)
            cam_frame.pack(side="top", fill="x", padx=5, pady=5)

            # Row 0: name + status
            label = ctk.CTkLabel(cam_frame, text=f"{cam_id}", font=ctk.CTkFont(size=14, weight="bold"))
            label.grid(row=0, column=0, padx=5, pady=2, sticky="w")

            status_label = ctk.CTkLabel(
                cam_frame,
                text="Status: NO ROI (Select track)",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="grey",
            )
            status_label.grid(row=0, column=1, padx=5, pady=2, sticky="w")

            # Row 1: source config
            mode_label = ctk.CTkLabel(cam_frame, text="Source:", font=ctk.CTkFont(size=12))
            mode_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")

            default_mode = f"webcam-{i}"
            if i == 0:
                default_mode = "webcam-0"

            mode_var = ctk.StringVar(value=default_mode)
            mode_combo = ctk.CTkOptionMenu(
                cam_frame,
                values=[f"webcam-{j}" for j in range(4)] + ["ip-url"],
                variable=mode_var,
            )
            mode_combo.grid(row=1, column=1, padx=5, pady=2, sticky="w")

            ip_entry = ctk.CTkEntry(
                cam_frame,
                width=350,
                placeholder_text="IP camera URL if 'ip-url' selected (e.g. http://192.168.0.101:8080/video)",
            )
            ip_entry.grid(row=1, column=2, padx=5, pady=2, sticky="w")

            # Row 2: metrics label
            metrics_label = ctk.CTkLabel(
                cam_frame,
                text="Motion: 0.0 | Occupied: 0.0s | RED events: 0",
                font=ctk.CTkFont(size=11),
            )
            metrics_label.grid(row=2, column=0, columnspan=3, padx=5, pady=2, sticky="w")

            # Row 3: risk bar + label
            risk_bar = ctk.CTkProgressBar(cam_frame, width=150)
            risk_bar.grid(row=3, column=0, padx=5, pady=2, sticky="w")
            risk_bar.set(0.0)

            risk_label = ctk.CTkLabel(
                cam_frame,
                text="Risk: 0 / 100",
                font=ctk.CTkFont(size=11),
            )
            risk_label.grid(row=3, column=1, columnspan=2, padx=5, pady=2, sticky="w")

            # Row 4: per-second log
            log_box = ctk.CTkTextbox(cam_frame, width=600, height=80)
            log_box.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

            # Row 5: event log
            event_box = ctk.CTkTextbox(cam_frame, width=600, height=60)
            event_box.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

            # Row 6: monitoring toggle
            monitoring_var = ctk.BooleanVar(value=False)
            monitoring_checkbox = ctk.CTkCheckBox(
                cam_frame,
                text="Temporarily disable track monitoring for this camera",
                variable=monitoring_var,
                command=lambda cid=cam_id, var=monitoring_var: self.on_monitoring_toggle(cid, var),
            )
            monitoring_checkbox.grid(row=6, column=0, columnspan=3, padx=5, pady=(2, 5), sticky="w")

            cam_frame.grid_columnconfigure(2, weight=1)

            self.camera_rows[cam_id] = {
                "mode_var": mode_var,
                "mode_combo": mode_combo,
                "ip_entry": ip_entry,
                "status_label": status_label,
                "metrics_label": metrics_label,
                "risk_bar": risk_bar,
                "risk_label": risk_label,
                "log_box": log_box,
                "event_box": event_box,
                "monitoring_var": monitoring_var,
                "monitoring_checkbox": monitoring_checkbox,
            }

            self.history[cam_id] = []
            self.current_status[cam_id] = "NO_ROI"
            self.red_counts[cam_id] = 0
            self.risk_scores[cam_id] = 0
            self.monitoring_disabled[cam_id] = False

        self.update_monitoring_status_label()

    def on_monitoring_toggle(self, cam_id: str, var: ctk.BooleanVar):
        """
        Called when the 'disable monitoring' checkbox for a camera is toggled.
        """
        self.monitoring_disabled[cam_id] = bool(var.get())
        self.update_monitoring_status_label()

    def update_monitoring_status_label(self):
        """
        Updates the global bottom label summarizing which cameras have monitoring disabled.
        """
        if not hasattr(self, "monitoring_off_label"):
            return

        disabled_cams = sorted([cid for cid, disabled in self.monitoring_disabled.items() if disabled])
        if not disabled_cams:
            self.monitoring_off_label.configure(text="")
        elif len(disabled_cams) == 1:
            cid = disabled_cams[0]
            self.monitoring_off_label.configure(
                text=f"Track monitoring is currently disabled for camera {cid}."
            )
        else:
            ids_str = ", ".join(disabled_cams)
            self.monitoring_off_label.configure(
                text=f"Track monitoring is currently disabled for multiple cameras (IDs: {ids_str})."
            )

    def start_monitoring(self):
        if self.running:
            return

        self.error_label.configure(text="")
        self.running = True
        self.history = {cid: [] for cid in self.camera_rows.keys()}
        self.current_status = {cid: "NO_ROI" for cid in self.camera_rows.keys()}
        self.red_counts = {cid: 0 for cid in self.camera_rows.keys()}
        self.risk_scores = {cid: 0 for cid in self.camera_rows.keys()}
        self.csv_paths.clear()
        self.session_start = time.time()
        self.session_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.session_start))

        self.progress.configure(mode="indeterminate")
        self.progress.start()

        for cam_id, row in self.camera_rows.items():
            row["log_box"].delete("1.0", "end")
            row["event_box"].delete("1.0", "end")
            row["status_label"].configure(text="Status: NO ROI (Select track)", text_color="grey")
            row["metrics_label"].configure(
                text="Motion: 0.0 | Occupied: 0.0s | RED events: 0"
            )
            row["risk_bar"].set(0.0)
            row["risk_label"].configure(text="Risk: 0 / 100")

        # Prepare CSV files
        for cam_id in self.camera_rows.keys():
            path = self.logs_dir / f"{cam_id}_{self.session_ts}.csv"
            self.csv_paths[cam_id] = path
            with open(path, "w", newline="") as f:
                f.write("timestamp,elapsed_s,avg_motion,occupied_duration,status,risk_score\n")

        # Launch monitoring threads
        self.monitor_threads = []
        for cam_id, row in self.camera_rows.items():
            mode = row["mode_var"].get()
            source: Union[int, str]
            if mode.startswith("webcam-"):
                try:
                    idx = int(mode.split("-")[1])
                except ValueError:
                    idx = 0
                source = idx
            else:
                url = row["ip_entry"].get().strip()
                if not url:
                    self.error_label.configure(
                        text=f"{cam_id}: Please enter IP URL or select a webcam mode."
                    )
                    self.running = False
                    self.progress.stop()
                    self.progress.set(0.0)
                    return
                source = url

            print(f"[DEBUG] Starting {cam_id} with mode={mode}, source={source!r}")

            cam_cfg = CameraConfig(camera_id=cam_id, source=source)
            t = threading.Thread(
                target=self.monitor_loop, args=(cam_cfg,), daemon=True
            )
            self.monitor_threads.append(t)
            t.start()

    def stop_monitoring(self):
        self.running = False
        self.progress.stop()
        self.progress.set(0.0)
        time.sleep(0.5)
        cv2.destroyAllWindows()

    def save_capture(self, cam_id: str, frame, timestamp: float):
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        filename = self.capture_dir / f"{cam_id}_{ts_str}_RED.jpg"
        try:
            cv2.imwrite(str(filename), frame)
            print(f"[INFO] Saved RED capture for {cam_id} at {filename}")
        except Exception as e:
            print(f"[WARN] Failed to save capture for {cam_id}: {e}")

    def handle_red_frame(self, cam_id: str, frame, timestamp: float):
        """
        Called instead of saving immediately whenever a new RED event starts.
        It buffers captures into a global AlertSession and triggers the
        speech / listening / alarm workflow once per session.
        """
        with self.alert_lock:
            session = self.current_alert_session

            # If there is no active session (or it was resolved), start a new one
            if session is None or session.resolved:
                session_id = self.next_alert_session_id
                self.next_alert_session_id += 1
                session = AlertSession(
                    id=session_id,
                    start_time=timestamp,
                    captures=[],
                    workflow_started=False,
                    resolved=False,
                    camera_ids=set(),
                )
                self.current_alert_session = session
                print(f"[INFO] Created new alert session {session.id} at {timestamp}")

            # Track which cameras are involved in this session
            session.camera_ids.add(cam_id)

            # Merge all RED captures while the workflow is running
            session.captures.append(
                {
                    "camera_id": cam_id,
                    "frame": frame.copy(),
                    "timestamp": timestamp,
                }
            )
            print(f"[DEBUG] Alert session {session.id}: added capture from {cam_id}")

            # Start the workflow only once for this session
            if not session.workflow_started:
                session.workflow_started = True
                t = threading.Thread(
                    target=self.alert_workflow,
                    args=(session.id,),
                    daemon=True,
                )
                t.start()

    def alert_workflow(self, session_id: int):

        print(f"[INFO] Starting alert workflow for session {session_id}")

        # Get camera IDs for announcement at the start of the workflow
        with self.alert_lock:
            session = self.current_alert_session
            if session is None or session.id != session_id:
                return
            cam_ids_for_alert = sorted(session.camera_ids) if session.camera_ids else []

        # 1) Speak alert (with camera id(s) included in the message)
        try:
            self.speak_alert_message(cam_ids_for_alert)
        except Exception as e:
            print(f"[WARN] Failed to speak alert message: {e}")

        # 2) Listen for reply (up to 15s)
        reply_text: Optional[str] = None
        try:
            reply_text = self.listen_for_reply(timeout=15)
        except Exception as e:
            print(f"[WARN] Error while listening for reply: {e}")
            reply_text = None

        # 3) Decide based on reply
        # None => no voice activity / timeout / mic issue
        normalized = reply_text.strip().lower() if reply_text is not None else None

        if normalized is None:
            decision = "NO_REPLY"
        elif "all is well" in normalized:
            decision = "ALL_IS_WELL"
        elif "taking action" in normalized:
            decision = "TAKING_ACTION"
        elif normalized == "":
            # Some speech but not understood -> treat as "other reply"
            decision = "OTHER"
        else:
            decision = "OTHER"

        print(f"[INFO] Session {session_id} decision={decision}, recognized={reply_text!r}")

        # 3a) Reaction speech / alarm according to your requirements:
        #   - ALL_IS_WELL  -> speak polite short 'everything is fine' message
        #   - TAKING_ACTION or OTHER -> speak short 'thank you for taking action' message
        #   - NO_REPLY     -> directly play danger.mp3 (no spoken reply)
        try:
            if decision == "ALL_IS_WELL":
                self.speak_all_is_well_ack()
            elif decision in ("TAKING_ACTION", "OTHER"):
                self.speak_taking_action_ack()
            elif decision == "NO_REPLY":
                self.play_danger_alarm()
        except Exception as e:
            print(f"[WARN] Error during acknowledgement / alarm: {e}")

        # 4) Finalize: snapshot captures & mark session resolved.
        with self.alert_lock:
            session = self.current_alert_session
            if session is None or session.id != session_id:
                # Session replaced or already cleared
                return

            # Mark resolved so future REDs create a new session.
            session.resolved = True
            captures = list(session.captures)
            # End this session; new reds after this moment will create a new one.
            self.current_alert_session = None

        # Capture-saving logic remains the same:
        #   - ALL_IS_WELL: discard captures
        #   - others: save captures
        if decision == "ALL_IS_WELL":
            print(f"[INFO] Session {session_id}: 'All is well' -> discarding {len(captures)} capture(s).")
            return

        print(f"[INFO] Session {session_id}: saving {len(captures)} capture(s).")
        for cap in captures:
            try:
                self.save_capture(cap["camera_id"], cap["frame"], cap["timestamp"])
            except Exception as e:
                print(f"[WARN] Failed to save merged capture for {cap['camera_id']}: {e}")

    def speak_alert_message(self, camera_ids: Optional[List[str]] = None):
        """
        Speaks the alert message, including camera id(s) in the text.
        If TTS is unavailable, just logs a warning.
        """
        if self.tts_engine is None:
            print("[WARN] TTS engine not available; cannot speak alert message.")
            return

        # Base alert text
        if camera_ids:
            if len(camera_ids) == 1:
                location = f" in {camera_ids[0]}"
            else:
                # e.g., "in cameras CAM01 and CAM02"
                if len(camera_ids) == 2:
                    location = f" in cameras {camera_ids[0]} and {camera_ids[1]}"
                else:
                    location = " in cameras " + ", ".join(camera_ids[:-1]) + f", and {camera_ids[-1]}"
        else:
            location = ""

        text = (
            f"Alert! miscellaneous thing found on the track{location}. "
            "Please reply 'All is well' if all things are fine, "
            "or reply 'taking action'."
        )

        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"[WARN] TTS error: {e}")

    def speak_all_is_well_ack(self):
        """
        Speaks a short polite message when the operator replies 'All is well'.
        """
        if self.tts_engine is None:
            print("[INFO] ALL_IS_WELL acknowledgement (TTS not available).")
            return

        text = "It is good that everything is fine."
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"[WARN] TTS error in ALL_IS_WELL acknowledgement: {e}")

    def speak_taking_action_ack(self):
        """
        Speaks a short message when the operator says 'taking action'
        or any other reply.
        """
        if self.tts_engine is None:
            print("[INFO] TAKING_ACTION / OTHER acknowledgement (TTS not available).")
            return

        text = "Thank you for taking action. I have saved the captures and log of the movement"
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"[WARN] TTS error in TAKING_ACTION acknowledgement: {e}")

    def listen_for_reply(self, timeout: int = 15) -> Optional[str]:
        """
        Listens to microphone for up to `timeout` seconds.
        Returns:
          - recognized text (string) if there is a reply,
          - "" if speech is heard but not understood,
          - None if there is no reply / timeout / mic unavailable.
        """
        if sr is None:
            print("[WARN] speech_recognition not available; treating as no reply.")
            return None

        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                print(f"[INFO] Listening for operator reply (up to {timeout}s)...")
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                except Exception:
                    # Non-fatal; continue without adjustment
                    pass

                try:
                    audio = recognizer.listen(source, timeout=timeout)
                except sr.WaitTimeoutError:
                    print("[INFO] No voice input detected within timeout.")
                    return None
        except Exception as e:
            print(f"[WARN] Could not access microphone: {e}")
            return None

        try:
            text = recognizer.recognize_google(audio)
            print(f"[INFO] Recognized reply: {text}")
            return text
        except sr.UnknownValueError:
            print("[INFO] Speech heard but not understood.")
            return ""
        except sr.RequestError as e:
            print(f"[WARN] Speech recognition service error: {e}")
            return ""

    def play_danger_alarm(self):
        """
        Plays the danger alarm (danger.mp3) when there is no reply.
        """
        if playsound is None:
            print("[WARN] playsound not available; cannot play danger alarm.")
            return

        sound_path = Path("danger.mp3")
        if not sound_path.exists():
            print("[WARN] danger.mp3 not found in current folder; cannot play alarm.")
            return

        try:
            print("[INFO] Playing danger alarm...")
            playsound(str(sound_path))
        except Exception as e:
            print(f"[WARN] Error playing danger alarm: {e}")

    @staticmethod
    def rectangles_intersect(box, roi) -> bool:
        """
        Check if a bounding box intersects the track ROI, with a margin so
        'near the track' is also counted.
        """
        x, y, w, h = box
        rx1, ry1, rx2, ry2 = roi

        # Expand ROI by TRACK_MARGIN in all directions
        rx1 -= TRACK_MARGIN
        ry1 -= TRACK_MARGIN
        rx2 += TRACK_MARGIN
        ry2 += TRACK_MARGIN

        x2 = x + w
        y2 = y + h
        if x > rx2 or x2 < rx1 or y > ry2 or y2 < ry1:
            return False
        return True

    def select_track_roi(self, cap, cam_id: str) -> Optional[tuple]:
        """
        Grab one frame from the camera and let the user draw the track ROI.
        Returns (x1, y1, x2, y2) or None if user cancels / invalid.
        """
        print(f"[INFO] Selecting track ROI for {cam_id}...")

        # Read a few frames to let camera auto-exposure stabilise
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                break
            time.sleep(0.05)

        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Could not read frame for ROI selection on {cam_id}")
            return None

        clone = frame.copy()
        cv2.putText(
            clone,
            f"{cam_id}: Draw rectangle around TRACK, then press ENTER/SPACE (ESC to cancel)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        roi = cv2.selectROI(f"Select TRACK region - {cam_id}", clone, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(f"Select TRACK region - {cam_id}")

        x, y, w_box, h_box = roi
        if w_box <= 0 or h_box <= 0:
            print(f"[WARN] No ROI selected for {cam_id}")
            return None

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w_box), int(y + h_box)
        print(f"[INFO] {cam_id} ROI selected: {(x1, y1, x2, y2)}")
        return x1, y1, x2, y2

    def monitor_loop(self, cam_cfg: CameraConfig):
        cam_id = cam_cfg.camera_id
        source = cam_cfg.source
        window_name = f"RailGuard - {cam_id}"

        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.data_queue.put(
                    SecondReport(
                        camera_id=cam_id,
                        timestamp=time.time(),
                        avg_motion=0.0,
                        occupied_duration=0.0,
                        status="ERROR_OPEN",
                    )
                )
                return

            # One-time manual ROI selection
            track_roi = self.select_track_roi(cap, cam_id)
            if track_roi is None:
                # No ROI selected -> NO_ROI status
                self.data_queue.put(
                    SecondReport(
                        camera_id=cam_id,
                        timestamp=time.time(),
                        avg_motion=0.0,
                        occupied_duration=0.0,
                        status="NO_ROI",
                    )
                )
                cap.release()
                return

            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=False
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            last_second = int(time.time())
            motion_acc = 0.0
            frame_count = 0
            occupied_since: Optional[float] = None
            last_status_for_capture: Optional[str] = None

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                h_frame, w_frame = frame.shape[:2]
                now = time.time()
                current_sec = int(now)

                monitoring_off = self.monitoring_disabled.get(cam_id, False)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fgmask_full = bg_subtractor.apply(gray)
                _, fgmask = cv2.threshold(fgmask_full, 250, 255, cv2.THRESH_BINARY)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)

                rx1, ry1, rx2, ry2 = track_roi
                motion_area = 0.0
                obstacle_boxes = []

                contours, _ = cv2.findContours(
                    fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for c in contours:
                    area = cv2.contourArea(c)
                    if area < OBSTACLE_MIN_AREA:
                        continue
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    if not self.rectangles_intersect((x, y, w_box, h_box), track_roi):
                        continue
                    motion_area += area
                    obstacle_boxes.append((x, y, w_box, h_box))

                motion_acc += motion_area
                frame_count += 1

                if current_sec != last_second:
                    if frame_count > 0:
                        avg_motion = motion_acc / frame_count
                    else:
                        avg_motion = 0.0

                    if not monitoring_off:
                        if avg_motion > MIN_MOTION_AREA:
                            if occupied_since is None:
                                occupied_since = now
                            occupied_duration = now - occupied_since

                            if occupied_duration >= TAMPERING_MIN_TIME:
                                status = "RED"
                            else:
                                status = "YELLOW"
                        else:
                            occupied_since = None
                            occupied_duration = 0.0
                            status = "GREEN"

                        report = SecondReport(
                            camera_id=cam_id,
                            timestamp=now,
                            avg_motion=avg_motion,
                            occupied_duration=occupied_duration,
                            status=status,
                        )
                        self.data_queue.put(report)

                        # Buffer RED captures instead of saving immediately
                        if status == "RED" and last_status_for_capture != "RED":
                            try:
                                self.handle_red_frame(cam_id, frame, now)
                            except Exception as e:
                                print(f"[WARN] Error handling RED frame for {cam_id}: {e}")

                        last_status_for_capture = status
                    else:
                        # When monitoring is off, reset occupancy-related state
                        occupied_since = None
                        last_status_for_capture = None

                    motion_acc = 0.0
                    frame_count = 0
                    last_second = current_sec

                # Draw overlays
                display_frame = frame.copy()
                cv2.rectangle(
                    display_frame,
                    (rx1, ry1),
                    (rx2, ry2),
                    (200, 200, 0),
                    2,
                )

                cv2.putText(
                    display_frame,
                    f"{cam_id}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                ts_str = time.strftime("%H:%M:%S", time.localtime(now))
                cv2.putText(
                    display_frame,
                    ts_str,
                    (w_frame - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # status from last report in UI
                current_status = self.current_status.get(cam_id, "GREEN")
                status_text = current_status
                color = (
                    (0, 255, 0)
                    if current_status == "GREEN"
                    else (0, 255, 255)
                    if current_status == "YELLOW"
                    else (0, 0, 255)
                )

                if monitoring_off:
                    status_text = "MONITORING OFF"
                    color = (180, 180, 180)

                cv2.putText(
                    display_frame,
                    f"Status: {status_text}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    display_frame,
                    "TRACK",
                    (int(w_frame * 0.40), h_frame - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )

                if monitoring_off:
                    cv2.putText(
                        display_frame,
                        "Track monitoring is temporarily disabled.",
                        (20, h_frame - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if not monitoring_off:
                    for (x, y, w_box, h_box) in obstacle_boxes:
                        cv2.rectangle(
                            display_frame,
                            (x, y),
                            (x + w_box, y + h_box),
                            (255, 0, 255),
                            2,
                        )
                        cv2.putText(
                            display_frame,
                            "Obstacle",
                            (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                cv2.imshow(window_name, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyWindow(window_name)

        except Exception as e:
            print(f"Error in monitor_loop for {cam_id}:", e)
            self.data_queue.put(
                SecondReport(
                    camera_id=cam_id,
                    timestamp=time.time(),
                    avg_motion=0.0,
                    occupied_duration=0.0,
                    status="ERROR",
                )
            )

    def update_ui(self):
        updated_any = False
        while not self.data_queue.empty():
            report = self.data_queue.get()
            cam_id = report.camera_id
            prev_status = self.current_status.get(cam_id, "NO_ROI")

            if report.status.startswith("ERROR"):
                self.error_label.configure(text=f"{cam_id}: Camera error: {report.status}")
                continue

            if cam_id not in self.history:
                self.history[cam_id] = []

            self.history[cam_id].append(report)
            if len(self.history[cam_id]) > HISTORY_SECONDS:
                self.history[cam_id] = self.history[cam_id][-HISTORY_SECONDS:]

            self.current_status[cam_id] = report.status
            updated_any = True

            row = self.camera_rows.get(cam_id)
            if row:
                status_label = row["status_label"]
                metrics_label = row["metrics_label"]
                risk_bar = row["risk_bar"]
                risk_label = row["risk_label"]
                log_box = row["log_box"]
                event_box = row["event_box"]

                # Status label
                if report.status == "NO_ROI":
                    text = "Status: NO ROI (Select track)"
                    color = "grey"
                elif report.status == "GREEN":
                    text = "Status: GREEN"
                    color = "green"
                elif report.status == "YELLOW":
                    text = "Status: YELLOW"
                    color = "yellow"
                else:
                    text = "Status: RED"
                    color = "red"

                status_label.configure(text=text, text_color=color)

                # Risk score (0-100) from motion + occupied duration
                motion = report.avg_motion
                occ = report.occupied_duration
                motion_factor = min(1.0, motion / (MIN_MOTION_AREA * 3.0)) if MIN_MOTION_AREA > 0 else 0.0
                occ_factor = min(1.0, occ / (TAMPERING_MIN_TIME * 2.0)) if TAMPERING_MIN_TIME > 0 else 0.0
                risk = int(100 * (0.5 * motion_factor + 0.5 * occ_factor))
                self.risk_scores[cam_id] = risk
                risk_bar.set(risk / 100.0)
                risk_label.configure(text=f"Risk: {risk} / 100")

                # Detect events on status transitions
                ts_str = time.strftime("%H:%M:%S", time.localtime(report.timestamp))
                if prev_status != report.status:
                    if prev_status != "RED" and report.status == "RED":
                        # new RED event
                        self.red_counts[cam_id] += 1
                        event_box.insert(
                            "end",
                            f"[{ts_str}] RED event START (motion={motion:.1f}, occupied={occ:.1f}s)\n",
                        )
                    elif prev_status != "YELLOW" and report.status == "YELLOW":
                        event_box.insert(
                            "end",
                            f"[{ts_str}] Crossing detected (YELLOW)\n",
                        )
                    elif prev_status == "YELLOW" and report.status == "GREEN":
                        event_box.insert(
                            "end",
                            f"[{ts_str}] Crossing cleared\n",
                        )
                    elif prev_status == "RED" and report.status == "GREEN":
                        event_box.insert(
                            "end",
                            f"[{ts_str}] Tampering cleared\n",
                        )
                    event_box.see("end")

                # Metrics label
                metrics_label.configure(
                    text=f"Motion: {motion:.1f} | Occupied: {occ:.1f}s | RED events: {self.red_counts[cam_id]}"
                )

                # Per-second log
                log_ts_str = ts_str
                log_box.insert(
                    "end",
                    f"[{log_ts_str}] status={report.status:<8} "
                    f"motion={report.avg_motion:>7.1f} "
                    f"occupied={report.occupied_duration:>4.1f}s\n",
                )
                log_box.see("end")

                # CSV logging (unchanged)
                if cam_id in self.csv_paths:
                    elapsed = report.timestamp - self.session_start
                    csv_line = (
                        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))},"
                        f"{elapsed:.1f},{report.avg_motion:.1f},"
                        f"{report.occupied_duration:.1f},{report.status},{risk}\n"
                    )
                    try:
                        with open(self.csv_paths[cam_id], "a", newline="") as f:
                            f.write(csv_line)
                    except Exception as e:
                        print(f"[WARN] Failed to write CSV for {cam_id}: {e}")

        if updated_any:
            self.progress.stop()
            self.progress.set(0.0)
            self.error_label.configure(text="")
            self.update_plot()

        self.after(500, self.update_ui)

    def update_plot(self):
        if not self.history:
            return

        self.ax_motion.clear()
        self.ax_status.clear()

        colors = ["cyan", "magenta", "yellow", "lime"]
        status_map = {"NO_ROI": -1, "GREEN": 0, "YELLOW": 1, "RED": 2}

        for idx, (cam_id, reports) in enumerate(self.history.items()):
            if not reports:
                continue

            xs = list(range(-len(reports) + 1, 1))
            motions = np.array([r.avg_motion for r in reports], dtype=float)
            levels = [status_map.get(r.status, 0) for r in reports]
            durations = [r.occupied_duration for r in reports]

            color = colors[idx % len(colors)]

            # Top: motion + smoothed motion
            self.ax_motion.plot(xs, motions, label=f"{cam_id} motion", color=color, linewidth=1.0)

            if len(motions) >= 3:
                kernel = np.ones(3, dtype=float) / 3.0
                smooth = np.convolve(motions, kernel, mode="same")
            else:
                smooth = motions.copy()

            self.ax_motion.plot(
                xs, smooth, label=f"_{cam_id}_smooth",
                color=color, linestyle="--", linewidth=1.0, alpha=0.7
            )

            # Bottom: status + occupied duration scaled
            self.ax_status.step(
                xs, levels, where="mid",
                label=f"{cam_id} status",
                color=color, linewidth=1.5,
            )

            if durations:
                scaled = [
                    min(2.0, (d / max(TAMPERING_MIN_TIME, 0.1)) * 2.0)
                    for d in durations
                ]
                self.ax_status.plot(
                    xs, scaled, label=f"_{cam_id}_occupied",
                    color=color, linestyle=":", linewidth=1.0, alpha=0.7,
                )

        self.ax_motion.axhline(
            MIN_MOTION_AREA, color="red",
            linestyle=":", linewidth=1.0,
            label="motion threshold",
        )

        self.ax_motion.set_ylabel("Average Motion")
        self.ax_status.set_ylabel("Status / Occupied\n-1=NoROI,0=G,1=Y,2=R")
        self.ax_status.set_xlabel("Seconds (most recent at 0)")

        self.ax_motion.legend(loc="upper left", fontsize=8)
        self.ax_status.legend(loc="upper left", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    def on_closing(self):
        self.running = False
        time.sleep(0.5)
        cv2.destroyAllWindows()
        self.destroy()



if __name__ == "__main__":
    app = RailGuardMultiCamApp()
    app.mainloop()