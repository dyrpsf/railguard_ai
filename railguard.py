"""
RailGuard Multi-Camera GUI – Track Monitoring with CustomTkinter

Logic:
- For each camera, we try to detect the railway track (two rails) automatically
  on EVERY frame using vertical-line detection in the CENTRAL part of the image.
- ONLY when tracks are visible do we run tampering logic:
    GREEN  – track clear (no obstacle in track region)
    YELLOW – short-lived obstacle crossing the track
    RED    – persistent obstacle staying on/near the track
- If tracks are not visible / not detected in the current frame, we show:
    "TRACK NOT VISIBLE" and do NOT classify tampering.
- When a camera first enters RED, we save a snapshot in 'captures/' with timestamp.
"""

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path

import cv2
import numpy as np
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import queue

# ---------------- Parameters (tune for your toy setup) ---------------- #

# More sensitive for a small toy track and hand-sized obstacles
MIN_MOTION_AREA = 200.0         # sum of obstacle areas per second -> "something present"
OBSTACLE_MIN_AREA = 300.0       # minimum contour area to draw "Obstacle" box
TAMPERING_MIN_TIME = 2.0        # seconds: continuous presence -> RED
HISTORY_SECONDS = 60            # show last N seconds in plots/logs
MAX_CAMERAS = 4                 # hard limit for GUI
TRACK_MARGIN = 25               # extra pixels around track ROI for "near track"


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
    status: str  # "GREEN", "YELLOW", "RED", "NO_TRACK", or "ERROR_*"


class RailGuardMultiCamApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("RailGuard Multi-Camera – Track Monitor")
        self.iconbitmap("app_icon.ico")
        self.geometry("1200x800")

        # Global state
        self.running = False
        self.monitor_threads: List[threading.Thread] = []
        self.data_queue: "queue.Queue[SecondReport]" = queue.Queue()
        self.history: Dict[str, List[SecondReport]] = {}   # camera_id -> list[SecondReport]
        self.current_status: Dict[str, str] = {}           # camera_id -> status
        self.camera_rows: Dict[str, Dict[str, ctk.CTkBaseClass]] = {}  # widgets per camera

        # Directory to save RED-event snapshots
        self.capture_dir = Path("captures")
        self.capture_dir.mkdir(exist_ok=True)

        # Build UI
        self._build_ui()

        # Schedule periodic UI update
        self.after(500, self.update_ui)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # -------------- UI construction -------------- #

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
        self.ax_status.set_ylabel("Status\n-1=NoTrack,0=G,1=Y,2=R")
        self.ax_status.set_xlabel("Seconds (most recent at 0)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.bottom_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

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

        n = self.num_cams_var.get()
        for i in range(n):
            cam_id = f"CAM{i+1:02d}"

            cam_frame = ctk.CTkFrame(self.cams_frame)
            cam_frame.pack(side="top", fill="x", padx=5, pady=5)

            label = ctk.CTkLabel(cam_frame, text=f"{cam_id}", font=ctk.CTkFont(size=14, weight="bold"))
            label.grid(row=0, column=0, padx=5, pady=2, sticky="w")

            status_label = ctk.CTkLabel(
                cam_frame,
                text="Status: NO TRACK",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="grey",
            )
            status_label.grid(row=0, column=1, padx=5, pady=2, sticky="w")

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

            log_box = ctk.CTkTextbox(cam_frame, width=600, height=80)
            log_box.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
            cam_frame.grid_columnconfigure(2, weight=1)

            self.camera_rows[cam_id] = {
                "mode_var": mode_var,
                "mode_combo": mode_combo,
                "ip_entry": ip_entry,
                "status_label": status_label,
                "log_box": log_box,
            }

            self.history[cam_id] = []
            self.current_status[cam_id] = "NO_TRACK"

    # -------------- Monitoring logic -------------- #

    def start_monitoring(self):
        if self.running:
            return

        self.error_label.configure(text="")
        self.running = True
        self.history = {cid: [] for cid in self.camera_rows.keys()}
        self.current_status = {cid: "NO_TRACK" for cid in self.camera_rows.keys()}

        self.progress.configure(mode="indeterminate")
        self.progress.start()

        for cam_id, row in self.camera_rows.items():
            row["log_box"].delete("1.0", "end")
            row["status_label"].configure(text="Status: NO TRACK", text_color="grey")

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

    def detect_track_roi_from_frame(self, frame) -> Optional[tuple]:
        """
        Detect approximate track region from a frame by finding two long vertical rails
        in the CENTRAL band of the image. Tuned for your toy track drawn as two
        vertical dark lines on white paper.

        Returns (x1, y1, x2, y2) or None if detection fails.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Crop central band horizontally to ignore left/right borders
        x_crop1 = int(w * 0.20)
        x_crop2 = int(w * 0.80)
        cropped = gray[:, x_crop1:x_crop2]

        # Threshold: rails are dark on light → invert binary
        _, bw = cv2.threshold(
            cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Emphasize vertical structures
        k_h = max(15, h // 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_h))
        vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            x_full = x + x_crop1
            # rails should be tall and thin
            if h_box < h * 0.6:
                continue
            if w_box > w * 0.2:
                continue
            aspect = h_box / max(w_box, 1)
            if aspect < 4.0:
                continue
            x_center = x_full + w_box / 2.0
            # stay away from extreme borders even inside crop
            if not (w * 0.25 <= x_center <= w * 0.75):
                continue
            candidates.append((h_box, x_full, y, x_full + w_box, y + h_box))

        if len(candidates) < 2:
            return None

        # Sort by height descending
        candidates.sort(reverse=True, key=lambda t: t[0])
        best1 = candidates[0]
        x_center1 = (best1[1] + best1[3]) / 2.0

        best2 = None
        for cand in candidates[1:]:
            x_center2 = (cand[1] + cand[3]) / 2.0
            sep = abs(x_center2 - x_center1)
            if w * 0.05 <= sep <= w * 0.5:
                best2 = cand
                break

        if best2 is None:
            return None

        xs = [best1[1], best1[3], best2[1], best2[3]]
        ys = [best1[2], best1[4], best2[2], best2[4]]

        x1 = max(0, min(xs) - 10)
        x2 = min(w - 1, max(xs) + 10)
        y1 = max(0, min(ys) - 10)
        y2 = min(h - 1, max(ys) + 10)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None

        return int(x1), int(y1), int(x2), int(y2)

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

                # --- Detect track on THIS frame ---
                track_roi = self.detect_track_roi_from_frame(frame)

                if track_roi is None:
                    # TRACK NOT VISIBLE: report NO_TRACK once per second
                    if current_sec != last_second:
                        report = SecondReport(
                            camera_id=cam_id,
                            timestamp=now,
                            avg_motion=0.0,
                            occupied_duration=0.0,
                            status="NO_TRACK",
                        )
                        self.data_queue.put(report)
                        last_second = current_sec
                        motion_acc = 0.0
                        frame_count = 0

                    display_frame = frame.copy()
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
                    cv2.putText(
                        display_frame,
                        "TRACK NOT VISIBLE",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (180, 180, 180),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.imshow(window_name, display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue  # skip tampering logic until track visible

                # --- Track is visible: run motion/tampering logic --- #
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

                    if status == "RED" and last_status_for_capture != "RED":
                        try:
                            self.save_capture(cam_id, frame, now)
                        except Exception as e:
                            print(f"[WARN] Error saving capture for {cam_id}: {e}")

                    last_status_for_capture = status
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

                current_status = self.current_status.get(cam_id, "GREEN")
                if current_status == "NO_TRACK":
                    status_text = "TRACK NOT VISIBLE"
                    color = (180, 180, 180)
                else:
                    status_text = f"Status: {current_status}"
                    color = (
                        (0, 255, 0)
                        if current_status == "GREEN"
                        else (0, 255, 255)
                        if current_status == "YELLOW"
                        else (0, 0, 255)
                    )

                cv2.putText(
                    display_frame,
                    status_text,
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )

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

    # -------------- UI update logic -------------- #

    def update_ui(self):
        updated_any = False
        while not self.data_queue.empty():
            report = self.data_queue.get()
            cam_id = report.camera_id
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
                if report.status == "NO_TRACK":
                    text = "Status: TRACK NOT VISIBLE"
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

                row["status_label"].configure(text=text, text_color=color)

                ts_str = time.strftime("%H:%M:%S", time.localtime(report.timestamp))
                row["log_box"].insert(
                    "end",
                    f"[{ts_str}] status={report.status:<8} "
                    f"motion={report.avg_motion:>7.1f} "
                    f"occupied={report.occupied_duration:>4.1f}s\n",
                )
                row["log_box"].see("end")

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
        status_map = {"NO_TRACK": -1, "GREEN": 0, "YELLOW": 1, "RED": 2}

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

        self.ax_motion.set_ylabel("Avg Motion")
        self.ax_status.set_ylabel("Status / Occupied\n-1=NoTrack,0=G,1=Y,2=R")
        self.ax_status.set_xlabel("Seconds (most recent at 0)")

        self.ax_motion.legend(loc="upper left", fontsize=8)
        self.ax_status.legend(loc="upper left", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    # -------------- Misc -------------- #

    def on_closing(self):
        self.running = False
        time.sleep(0.5)
        cv2.destroyAllWindows()
        self.destroy()


if __name__ == "__main__":
    app = RailGuardMultiCamApp()
    app.mainloop()