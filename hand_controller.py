"""
Hand controller - camera-based hand tracking for JARVIS.
Runs in a background thread and sends gesture events via a thread-safe queue.
"""

import os
import queue
import threading
import time
import logging
import urllib.request
from typing import Optional

import cv2
import numpy as np

from core.event_bus import bus
from clap_detector import ClapDetector
from gesture_detector import GestureDetector, GestureType

logger = logging.getLogger(__name__)

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
_MODEL_NAME = "hand_landmarker.task"
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODEL_NAME)

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17), (0, 18), (17, 18), (2, 17), (5, 17),
]


def _ensure_model() -> Optional[str]:
    if os.path.exists(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) > 0:
        return _MODEL_PATH
    parent = os.path.dirname(_MODEL_PATH)
    if not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    try:
        logger.info("[HAND] Downloading hand landmarker model...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        if os.path.getsize(_MODEL_PATH) > 100_000:
            logger.info("[HAND] Model downloaded: %s", _MODEL_PATH)
            return _MODEL_PATH
    except Exception as e:
        logger.error("[HAND] Failed to download model: %s", e)
    return None


class HandController:
    def __init__(self, camera_index=0, width=1280, height=720,
                 detection_conf=0.65, tracking_conf=0.65,
                 event_queue=None, debug=False, jarvis_state_callback=None):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf
        self.event_queue = event_queue or queue.Queue()
        self.debug = debug
        self.jarvis_state_callback = jarvis_state_callback or (lambda: "OFFLINE")

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap = None
        self._fps = 0.0
        self._frame_count = 0
        self._fps_time = time.time()
        self._detector = None
        self._clap_detector = None
        self._gesture_detector = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="hand-controller")
        self._thread.start()
        logger.info("[HAND] Controller started")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release()
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
        logger.info("[HAND] Controller stopped")

    def _release(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _init_camera(self) -> bool:
        try:
            backend = cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0
            self._cap = cv2.VideoCapture(self.camera_index, backend)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                logger.error("[HAND] Cannot open camera %d", self.camera_index)
                return False
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            for _ in range(5):
                self._cap.read()
            return True
        except Exception as e:
            logger.error("[HAND] Camera init failed: %s", e)
            return False

    def _init_detector(self) -> bool:
        if self._detector is not None:
            return True
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _ensure_model()
            if model_path is None:
                logger.error("[HAND] No model available for hand detection")
                return False

            base_options = mp_python.BaseOptions(
                model_asset_path=model_path,
                delegate=mp_python.BaseOptions.Delegate.CPU,
            )
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=self.detection_conf,
                min_tracking_confidence=self.tracking_conf,
                min_hand_presence_confidence=0.5,
            )
            self._detector = mp_vision.HandLandmarker.create_from_options(options)
            logger.info("[HAND] HandLandmarker initialized")
            return True
        except Exception as e:
            logger.error("[HAND] Detector init failed: %s", e)
            return False

    def _loop(self):
        if not self._init_camera():
            self._emit("camera_error", {"error": "Camera unavailable"})
            return
        if not self._init_detector():
            self._emit("camera_error", {"error": "Hand detector unavailable"})
            self._release()
            return

        self._clap_detector = ClapDetector()
        self._gesture_detector = GestureDetector()

        last_gesture_time = 0.0
        gesture_cooldown = 0.5

        while self._running:
            try:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = None
                try:
                    import mediapipe as mp
                    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
                except Exception as _e:
                    logger.error("[HAND] Image conversion failed: %s", _e)
                    time.sleep(0.1)
                    continue

                now = time.time()
                self._frame_count += 1
                if now - self._fps_time >= 1.0:
                    self._fps = self._frame_count / (now - self._fps_time)
                    self._frame_count = 0
                    self._fps_time = now

                result = self._detector.detect(mp_image)

                hands = []
                if result.hand_landmarks and result.handedness:
                    for hand_lms, hand_hd in zip(result.hand_landmarks, result.handedness):
                        if not hand_lms:
                            continue
                        label = "Left"
                        if hand_hd and len(hand_hd) > 0:
                            cat = hand_hd[0]
                            label = getattr(cat, 'category_name', None) or "Left"
                            label = "Left" if label.lower() == "left" else "Right"
                        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms])
                        hands.append((label, pts))
                        if self.debug:
                            self._draw_landmarks(frame, hand_lms, label)

                left_palm = right_palm = None
                left_wrist = right_wrist = None
                gesture = None
                gesture_hand = None

                for label, pts in hands:
                    wrist = pts[0]
                    palm = pts[9]
                    if label == "Left":
                        left_palm = palm
                        left_wrist = wrist
                    else:
                        right_palm = palm
                        right_wrist = wrist

                if left_palm is not None and right_palm is not None:
                    clap = self._clap_detector.update(left_palm, right_palm, left_wrist, right_wrist, now)
                    if clap:
                        self._emit("gesture_clap", {
                            "confidence": clap.confidence,
                            "timestamp": clap.timestamp,
                            "left_palm": clap.left_palm.tolist(),
                            "right_palm": clap.right_palm.tolist(),
                        })

                if now - last_gesture_time > gesture_cooldown:
                    for label, pts in hands:
                        g = self._gesture_detector.detect(pts, label, now)
                        if g != GestureType.NONE:
                            gesture = g
                            gesture_hand = label
                            last_gesture_time = now
                            break

                if gesture and gesture != GestureType.CLAP:
                    self._emit(f"gesture_{gesture.value.lower()}", {
                        "gesture": gesture.value,
                        "hand": gesture_hand,
                        "timestamp": now,
                    })

                if self.debug:
                    self._draw_debug(frame, hands, left_palm, right_palm, gesture)

                elapsed = time.time() - now
                if elapsed < 1.0 / 30.0:
                    time.sleep(max(0, 1.0 / 30.0 - elapsed))

            except Exception as e:
                logger.error("[HAND] Loop error: %s", e)
                time.sleep(0.5)

        if self.debug:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _emit(self, event_type: str, data: dict):
        self._emit_metric(event_type)
        try:
            self.event_queue.put_nowait({
                "type": event_type,
                "data": data,
                "timestamp": time.time(),
                "jarvis_state": self.jarvis_state_callback(),
            })
        except queue.Full:
            pass

    def _emit_metric(self, event_type: str):
        try:
            from core.observability import metrics
            if event_type == "gesture_clap":
                metrics.inc("hand_clap")
            elif event_type == "camera_error":
                metrics.inc("hand_camera_error")
            elif event_type.startswith("gesture_"):
                metrics.inc("hand_gesture")
            metrics.set("hand_fps", round(self._fps, 1))
        except Exception:
            pass

    def _draw_landmarks(self, frame, landmarks, label):
        h, w = frame.shape[:2]
        pts = []
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            pts.append((x, y))
        color = (0, 255, 0) if label == "Right" else (255, 0, 0)
        for i, j in _HAND_CONNECTIONS:
            if i < len(pts) and j < len(pts):
                cv2.line(frame, pts[i], pts[j], color, 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, color, -1)

    def _draw_debug(self, frame, hands, left_palm, right_palm, gesture):
        h, w = frame.shape[:2]
        state = self.jarvis_state_callback()
        lines = [
            f"JARVIS: {state}",
            f"GESTURE: {gesture.value if gesture else 'NONE'}",
            f"HANDS: {len(hands)}",
            f"FPS: {self._fps:.1f}",
        ]
        if left_palm is not None and right_palm is not None:
            dist = float(np.linalg.norm(left_palm - right_palm))
            lines.append(f"PALM DIST: {dist:.3f}")
        y = 30
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 25
        cv2.imshow("JARVIS Hand Control", frame)
        cv2.waitKey(1)
