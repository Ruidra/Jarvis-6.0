"""
JARVIS configuration extensions for hand control and clap activation.
"""

# ── Camera ─────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ── Hand detection ─────────────────────────────────────────────────────────────
HAND_DETECTION_CONFIDENCE = 0.65
HAND_TRACKING_CONFIDENCE = 0.65

# ── Clap detection ─────────────────────────────────────────────────────────────
CLAP_SENSITIVITY = 1.0
CLAP_COOLDOWN = 2.0

# ── Wake word timeout after clap ───────────────────────────────────────────────
WAKE_TIMEOUT = 7.0

# ── Debug mode ─────────────────────────────────────────────────────────────────
HAND_DEBUG = True

# ── Wake words ─────────────────────────────────────────────────────────────────
WAKE_WORDS = ["jarvis", "hey jarvis"]

# ── Performance ────────────────────────────────────────────────────────────────
HAND_FPS_TARGET = 30
