#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
chess_piece_vision_node.py
ROS Melodic / Python 2.7
Niryo Ned 1 — overhead camera chess vision

Approach: homography warp + L2-norm square comparison (repo approach)
- 4 corner clicks → cv2.getPerspectiveTransform → 400x400 flat board
- find_moves(): L2 norm per 50x50 square, dynamic threshold
- Turn-based before_w: captured ONCE per turn (not per motion event)
  so putting a piece back to its spot = zero delta

Topics published
----------------
/chess_vision/human_move    (std_msgs/String) UCI e.g. "e2e4"
/chess_vision/cleanup_event (std_msgs/String) "cleanup:e5" after capture removal
"""

from __future__ import print_function
import sys
import os
import cv2
import json
import time
import numpy as np
import threading
import argparse

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

try:
    import chess as _chess_mod
    _CHESS_AVAILABLE = True
except ImportError:
    _CHESS_AVAILABLE = False
    rospy.logwarn_once("python-chess not found — board tracking disabled")

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
HOMOGRAPHY_PATH = os.path.expanduser("~/.chess_vision/homography.json")

# ──────────────────────────────────────────────────────────────
# Tunables
# ──────────────────────────────────────────────────────────────
MOTION_THRESH      = 6       # frame-to-frame mean diff to call "motion"
STABLE_FRAMES_NEED = 8       # consecutive quiet frames before comparing
BOARD_PIX          = 400     # warped board size
SQ_PIX             = 50      # pixels per square (400/8)
GRACE_WINDOW_SEC   = 4.0     # seconds after move accepted for cleanup
CAPTURE_CLEANUP_MAX = 2      # max squares changed during grace

# ──────────────────────────────────────────────────────────────
# Homography helpers
# ──────────────────────────────────────────────────────────────

def save_homography(path, H):
    dirp = os.path.dirname(path)
    if dirp and not os.path.isdir(dirp):
        os.makedirs(dirp)
    with open(path, 'w') as f:
        json.dump(H.tolist(), f)
    rospy.loginfo("Homography saved to %s", path)


def load_homography(path):
    with open(path, 'r') as f:
        return np.array(json.load(f), dtype=np.float64)


def warp_board(img, H):
    """Warp camera image to flat 400x400 board view."""
    return cv2.warpPerspective(img, H, (BOARD_PIX, BOARD_PIX))


# ──────────────────────────────────────────────────────────────
# Move detection (repo approach)
# ──────────────────────────────────────────────────────────────
# Warped image orientation (standard overhead, camera behind white):
#   a8 = top-left,  h8 = top-right
#   a1 = bottom-left, h1 = bottom-right
# Row 0-7 (top to bottom) = rank 8 down to rank 1
# Col 0-7 (left to right) = file a to file h
# Square name: files[col] + str(8 - row)

FILES = 'abcdefgh'


def _sq_name(row, col):
    return FILES[col] + str(8 - row)


def find_moves(warped_before, warped_after):
    """
    Compare two 400x400 warped board images.
    Returns list of 2-4 most-changed square names (UCI notation).
    Uses L2-norm per 50x50 square + dynamic threshold (repo approach).
    """
    size = SQ_PIX
    largest = [0.0, 0.0, 0.0, 0.0]
    coordinates = ['', '', '', '']

    for row in range(8):
        for col in range(8):
            r0, r1 = row * size, (row + 1) * size
            c0, c1 = col * size, (col + 1) * size
            sq1 = warped_before[r0:r1, c0:c1]
            sq2 = warped_after[r0:r1, c0:c1]
            dist = cv2.norm(sq2, sq1, cv2.NORM_L2)
            name = _sq_name(row, col)
            for z in range(4):
                if dist >= largest[z]:
                    largest.insert(z, dist)
                    coordinates.insert(z, name)
                    largest.pop()
                    coordinates.pop()
                    break

    # dynamic threshold: anything below half the average of top-2 is noise
    thresh = (largest[0] + largest[1]) / 2.0 * 0.5
    # prune from the bottom up
    result_large = list(largest)
    result_coords = list(coordinates)
    for t in range(3, 1, -1):
        if result_large[t] < thresh:
            result_coords.pop(t)
            result_large.pop(t)

    return result_coords, result_large


# ──────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────

def calibrate_homography(cap, rotate_deg, mirror, hom_path):
    """
    Interactive 4-corner calibration.
    Click order: a8 (top-left) → h8 (top-right) → h1 (bottom-right) → a1 (bottom-left)
    Press S to save, R to retry, Q to quit.
    """
    CORNER_NAMES = ['a8', 'h8', 'h1', 'a1']
    # Destination corners in 400x400 warped image
    # a8→TL, h8→TR, h1→BR, a1→BL
    DST_PTS = np.array([
        [0,         0        ],
        [BOARD_PIX, 0        ],
        [BOARD_PIX, BOARD_PIX],
        [0,         BOARD_PIX],
    ], dtype=np.float32)

    rospy.loginfo("=== CALIBRATION MODE ===")
    rospy.loginfo("Click corners in order: %s(TL) %s(TR) %s(BR) %s(BL)",
                  *CORNER_NAMES)
    rospy.loginfo("Press S to save, R to retry, Q to quit")

    WIN = "Calibration: click a8 h8 h1 a1 then S"
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            rospy.loginfo("Click %d: (%d,%d) => %s",
                          len(clicks), x, y, CORNER_NAMES[len(clicks) - 1])

    frame = None
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.1)
    if frame is None:
        rospy.logerr("Cannot read from camera during calibration")
        return None

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 600)
    cv2.setMouseCallback(WIN, on_mouse)

    H = None
    while True:
        ret, raw = cap.read()
        if ret and raw is not None:
            frame = apply_transform(raw, rotate_deg, mirror)

        disp = frame.copy()
        h, w = disp.shape[:2]

        # corner labels
        lbl_pos = [(8, 28), (w - 80, 28), (w - 80, h - 8), (8, h - 8)]
        for lbl, pos in zip(CORNER_NAMES, lbl_pos):
            cv2.putText(disp, lbl, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 255), 2)

        for i, pt in enumerate(clicks):
            cv2.circle(disp, pt, 8, (0, 255, 0), -1)
            cv2.putText(disp, CORNER_NAMES[i], (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(clicks) == 4:
            pts = np.array(clicks, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(disp, [pts], True, (0, 255, 0), 2)
            cv2.putText(disp, "Press S to save, R to retry",
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 0), 2)
            # show preview warp
            src = np.array(clicks, dtype=np.float32)
            H_preview = cv2.getPerspectiveTransform(src, DST_PTS)
            warped = warp_board(frame, H_preview)
            cv2.imshow("Warped preview", warped)

        status = "Clicks: %d/4" % len(clicks)
        if len(clicks) < 4:
            status += "  Next: click %s" % CORNER_NAMES[len(clicks)]
        cv2.putText(disp, status, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 0), 2)

        cv2.imshow(WIN, disp)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('r'), ord('R')):
            clicks[:] = []
            H = None
            rospy.loginfo("Retry — click a8 h8 h1 a1")
        elif key in (ord('s'), ord('S')):
            if len(clicks) == 4:
                src = np.array(clicks, dtype=np.float32)
                H = cv2.getPerspectiveTransform(src, DST_PTS)
                save_homography(hom_path, H)
                cv2.destroyAllWindows()
                return H
            else:
                rospy.logwarn("Need 4 clicks first")
        elif key in (ord('q'), ord('Q')):
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return None


# ──────────────────────────────────────────────────────────────
# Image transform helpers
# ──────────────────────────────────────────────────────────────

def apply_transform(img, rotate_deg, mirror):
    if rotate_deg == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_deg == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotate_deg == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mirror:
        img = cv2.flip(img, 1)
    return img


# ──────────────────────────────────────────────────────────────
# Motion gate
# ──────────────────────────────────────────────────────────────

class MotionGate(object):
    def __init__(self, thresh=MOTION_THRESH):
        self.thresh = thresh
        self._prev_gray = None

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return False, 0.0
        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        mean_d = float(np.mean(diff))
        return mean_d > self.thresh, mean_d


# ──────────────────────────────────────────────────────────────
# State machine
# ──────────────────────────────────────────────────────────────

STATE_IDLE     = 'idle'
STATE_MOTION   = 'motion'
STATE_SETTLING = 'settling'
STATE_GRACE    = 'grace'


class MoveDetector(object):
    def __init__(self, H, move_pub, cleanup_pub):
        self.H           = H
        self.move_pub    = move_pub
        self.cleanup_pub = cleanup_pub

        self.motion_gate  = MotionGate()
        self.state        = STATE_IDLE
        self.stable_count = 0
        self.last_frame   = None
        self.grace_start  = 0.0

        # Turn-based reference: set once per turn (not per motion event).
        # This means pick-up-and-put-back = zero delta → no phantom move.
        self.before_w     = None
        self._before_w_ready = False

        # Engine move handling
        self._expecting_engine   = False
        self._pending_engine_uci = None

        # Board tracking
        if _CHESS_AVAILABLE:
            self._board          = _chess_mod.Board()
            self._board_tracking = True
        else:
            self._board          = None
            self._board_tracking = False

        self._lock = threading.Lock()

    # ── public ──────────────────────────────────────────────

    def feed(self, frame):
        with self._lock:
            self._process(frame)

    def on_engine_move(self, uci):
        """Called when GUI publishes engine move. Robot arm will now move pieces."""
        with self._lock:
            self._expecting_engine   = True
            self._pending_engine_uci = uci
            rospy.loginfo("[vision] Expecting engine move %s — waiting for arm", uci)

    def on_move_rejected(self):
        with self._lock:
            self._try_pop()
            if self.last_frame is not None:
                self.before_w = warp_board(self.last_frame, self.H)
            self.state        = STATE_IDLE
            self.stable_count = 0
        rospy.logwarn("[vision] Move rejected — before_w reset")

    def on_game_reset(self):
        with self._lock:
            if _CHESS_AVAILABLE:
                self._board          = _chess_mod.Board()
                self._board_tracking = True
            if self.last_frame is not None:
                self.before_w = warp_board(self.last_frame, self.H)
                self._before_w_ready = True
            self.state        = STATE_IDLE
            self.stable_count = 0
            self._expecting_engine   = False
            self._pending_engine_uci = None
        rospy.loginfo("[vision] Game reset — board and before_w reset")

    def set_initial_baseline(self, frame):
        """Call once on startup with a stable frame."""
        with self._lock:
            self.before_w        = warp_board(frame, self.H)
            self._before_w_ready = True
            self.last_frame      = frame
        rospy.loginfo("[vision] Initial before_w captured")

    # ── board tracking ───────────────────────────────────────

    def _try_push(self, uci):
        if not self._board_tracking:
            return
        try:
            self._board.push(_chess_mod.Move.from_uci(uci))
        except Exception as e:
            rospy.logwarn("Board tracking desync push(%s): %s", uci, e)
            self._board_tracking = False

    def _try_pop(self):
        if not self._board_tracking:
            return
        try:
            self._board.pop()
        except Exception:
            pass

    def _find_legal_move(self, candidates):
        """
        Given a list of changed square names (sorted by change magnitude),
        search pairs among them for a legal move on the tracking board.
        Returns UCI string or None.
        """
        if not self._board_tracking or self._board is None:
            # No board tracking: just return top-2 as-is
            if len(candidates) >= 2:
                return candidates[0] + candidates[1]
            return None

        # Build all pairs from candidates (up to 16 squares checked)
        checked = candidates[:16] if len(candidates) > 16 else candidates
        legal_ucis = set(m.uci() for m in self._board.legal_moves)

        for i in range(len(checked)):
            for j in range(len(checked)):
                if i == j:
                    continue
                uci = checked[i] + checked[j]
                if uci in legal_ucis:
                    return uci
                # promotion
                if uci + 'q' in legal_ucis:
                    return uci + 'q'
        return None

    # ── state machine ────────────────────────────────────────

    def _process(self, frame):
        self.last_frame = frame
        motion, mean_d = self.motion_gate.update(frame)

        # Initialize before_w if not yet done
        if not self._before_w_ready:
            if not motion:
                self.before_w        = warp_board(frame, self.H)
                self._before_w_ready = True
                rospy.loginfo("[vision] before_w initialized")
            return

        if self.state == STATE_IDLE:
            if motion:
                self.state        = STATE_MOTION
                self.stable_count = 0
                rospy.logdebug("[vision] IDLE→MOTION mean_d=%.1f", mean_d)

        elif self.state == STATE_MOTION:
            if not motion:
                self.state        = STATE_SETTLING
                self.stable_count = 1
                rospy.logdebug("[vision] MOTION→SETTLING")

        elif self.state == STATE_SETTLING:
            if motion:
                # Hand moved again — reset settling
                self.state        = STATE_MOTION
                self.stable_count = 0
                return

            self.stable_count += 1
            if self.stable_count >= STABLE_FRAMES_NEED:
                self.state = STATE_IDLE
                self._on_settled(frame)

        elif self.state == STATE_GRACE:
            if not motion:
                self.stable_count += 1
                if self.stable_count >= STABLE_FRAMES_NEED:
                    self._check_cleanup(frame)
            else:
                self.stable_count = 0

            # Grace window timeout
            if time.time() - self.grace_start > GRACE_WINDOW_SEC:
                self.state = STATE_IDLE
                self.stable_count = 0

    def _on_settled(self, frame):
        """Board has settled after motion. Compare before_w vs current."""
        if self.before_w is None:
            rospy.logwarn("[vision] No before_w — skipping")
            return

        after_w = warp_board(frame, self.H)
        candidates, magnitudes = find_moves(self.before_w, after_w)

        rospy.loginfo("[vision] Settled. Changed squares: %s magnitudes: %s",
                      candidates,
                      [round(m, 1) for m in magnitudes])

        if self._expecting_engine:
            # Robot arm just finished — update before_w for player's turn
            if self._pending_engine_uci:
                self._try_push(self._pending_engine_uci)
            self.before_w            = after_w
            self._expecting_engine   = False
            self._pending_engine_uci = None
            rospy.loginfo("[vision] Engine move settled — before_w updated for player's turn")
            return

        # No candidates or top change too small → treat as noise
        if not candidates or (magnitudes and magnitudes[0] < 50.0):
            rospy.logdebug("[vision] Change too small (%.1f) — ignoring",
                           magnitudes[0] if magnitudes else 0.0)
            return

        uci = self._find_legal_move(candidates)
        if uci:
            rospy.loginfo("[vision] Human move detected: %s", uci)
            self._try_push(uci)
            self.move_pub.publish(uci)
            # Before_w is NOT updated here — updated in grace/_after_grace
            # so that cleanup (captured piece removal) can be detected
            self.grace_start  = time.time()
            self.state        = STATE_GRACE
            self.stable_count = 0
        else:
            rospy.logwarn("[vision] No legal move found in: %s", candidates)
            # before_w stays the same so player can retry

    def _check_cleanup(self, frame):
        """
        During grace window, check if removed captured piece is the only change.
        If so, publish cleanup event and update before_w.
        """
        after_w = warp_board(frame, self.H)
        candidates, magnitudes = find_moves(self.before_w, after_w)

        if not candidates:
            # Nothing changed since last before_w — just end grace
            self.before_w     = after_w
            self.state        = STATE_IDLE
            self.stable_count = 0
            return

        # If 1-2 squares changed it's probably captured piece removal
        if len(candidates) <= CAPTURE_CLEANUP_MAX:
            sq = candidates[0]
            rospy.loginfo("[vision] Cleanup event: %s", sq)
            self.cleanup_pub.publish("cleanup:" + sq)

        # Update before_w to current state for next turn
        self.before_w     = after_w
        self.state        = STATE_IDLE
        self.stable_count = 0


# ──────────────────────────────────────────────────────────────
# Display thread
# ──────────────────────────────────────────────────────────────

class DisplayThread(threading.Thread):
    def __init__(self, detector):
        super(DisplayThread, self).__init__()
        self.detector = detector
        self.daemon   = True
        self._frame   = None
        self._lock    = threading.Lock()

    def set_frame(self, frame):
        with self._lock:
            self._frame = frame

    def run(self):
        cv2.namedWindow("Chess Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Chess Vision", 900, 450)
        while not rospy.is_shutdown():
            with self._lock:
                frame = self._frame
            if frame is None:
                time.sleep(0.03)
                continue

            with self.detector._lock:
                H       = self.detector.H
                before_w = self.detector.before_w
                state   = self.detector.state

            warped = warp_board(frame, H)

            # Draw grid + state on warped
            disp_warp = warped.copy()
            for i in range(1, 8):
                cv2.line(disp_warp, (i * SQ_PIX, 0), (i * SQ_PIX, BOARD_PIX),
                         (80, 80, 80), 1)
                cv2.line(disp_warp, (0, i * SQ_PIX), (BOARD_PIX, i * SQ_PIX),
                         (80, 80, 80), 1)
            # File/rank labels
            for col in range(8):
                cv2.putText(disp_warp, FILES[col],
                            (col * SQ_PIX + 2, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            for row in range(8):
                cv2.putText(disp_warp, str(8 - row),
                            (2, row * SQ_PIX + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(disp_warp, "State:" + state, (5, BOARD_PIX - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            # Show diff vs before_w
            if before_w is not None:
                diff = cv2.absdiff(before_w, warped)
                diff_bright = cv2.convertScaleAbs(diff, alpha=3.0)
                combined = np.hstack([disp_warp, diff_bright])
            else:
                combined = disp_warp

            # Resize raw for display
            h, w = frame.shape[:2]
            scale = 450.0 / h
            raw_small = cv2.resize(frame, (int(w * scale), 450))
            disp_h = max(combined.shape[0], raw_small.shape[0])
            # pad to same height
            if combined.shape[0] < disp_h:
                pad = np.zeros((disp_h - combined.shape[0],
                                combined.shape[1], 3), np.uint8)
                combined = np.vstack([combined, pad])
            if raw_small.shape[0] < disp_h:
                pad = np.zeros((disp_h - raw_small.shape[0],
                                raw_small.shape[1], 3), np.uint8)
                raw_small = np.vstack([raw_small, pad])
            display = np.hstack([combined, raw_small])
            cv2.imshow("Chess Vision", display)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('b'), ord('B')):
                with self.detector._lock:
                    self.detector.before_w = warp_board(frame, H)
                    self.detector.state    = STATE_IDLE
                    self.detector.stable_count = 0
                rospy.loginfo("[vision] Manual before_w reset (B key)")
            elif key in (ord('q'), ord('Q')):
                rospy.signal_shutdown("User quit")

        cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────
# ROS node
# ──────────────────────────────────────────────────────────────

class ChessVisionNode(object):
    def __init__(self, args):
        self.args      = args
        self.rotate    = args.rotate
        self.mirror    = args.mirror
        self.cam_index = args.camera

        rospy.init_node('chess_piece_vision_node', anonymous=False)

        self.move_pub    = rospy.Publisher('/chess_vision/human_move',
                                           String, queue_size=5)
        self.cleanup_pub = rospy.Publisher('/chess_vision/cleanup_event',
                                           String, queue_size=5)

        # Open camera
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            rospy.logfatal("Cannot open camera %d", self.cam_index)
            sys.exit(1)

        # Load or calibrate homography
        H = None
        if os.path.isfile(HOMOGRAPHY_PATH) and not args.calibrate:
            try:
                H = load_homography(HOMOGRAPHY_PATH)
                rospy.loginfo("Loaded homography from %s", HOMOGRAPHY_PATH)
            except Exception as e:
                rospy.logwarn("Failed to load homography: %s", e)

        if H is None:
            rospy.loginfo("Starting calibration…")
            H = calibrate_homography(self.cap, self.rotate, self.mirror,
                                     HOMOGRAPHY_PATH)
            if H is None:
                rospy.logfatal("Calibration cancelled")
                sys.exit(1)

        self.detector = MoveDetector(H, self.move_pub, self.cleanup_pub)

        # Capture initial stable frame for before_w
        rospy.loginfo("Capturing initial baseline — keep board still…")
        for _ in range(60):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = apply_transform(frame, self.rotate, self.mirror)
                time.sleep(0.05)
        ret, frame = self.cap.read()
        if ret and frame is not None:
            frame = apply_transform(frame, self.rotate, self.mirror)
            self.detector.set_initial_baseline(frame)

        self.display = DisplayThread(self.detector)
        self.display.start()

        # GUI feedback subscriptions
        rospy.Subscriber('/chess_vision/move_rejected', String,
                         self._on_rejected)
        rospy.Subscriber('/chess_vision/game_reset',    String,
                         self._on_reset)
        rospy.Subscriber('/chess_vision/engine_move',   String,
                         self._on_engine_move)

        # Camera topic (optional — for ROS image transport)
        if args.ros_camera:
            rospy.Subscriber(args.ros_camera, CompressedImage,
                             self._on_ros_image)
            self._use_ros_camera = True
        else:
            self._use_ros_camera = False

        rospy.loginfo("Chess vision node ready. Board: %dx%d, SQ: %dpx",
                      BOARD_PIX, BOARD_PIX, SQ_PIX)

    # ── callbacks ────────────────────────────────────────────

    def _on_rejected(self, msg):
        self.detector.on_move_rejected()

    def _on_reset(self, msg):
        self.detector.on_game_reset()

    def _on_engine_move(self, msg):
        self.detector.on_engine_move(msg.data.strip())

    def _on_ros_image(self, msg):
        try:
            arr  = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = apply_transform(frame, self.rotate, self.mirror)
                self.detector.feed(frame)
                self.display.set_frame(frame)
        except Exception as e:
            rospy.logwarn_throttle(5, "ROS image decode error: %s", e)

    # ── main loop ────────────────────────────────────────────

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if not self._use_ros_camera:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    frame = apply_transform(frame, self.rotate, self.mirror)
                    self.detector.feed(frame)
                    self.display.set_frame(frame)
            rate.sleep()
        self.cap.release()


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Chess piece vision node")
    p.add_argument('--camera',     type=int,   default=0,
                   help='OpenCV camera index (default 0)')
    p.add_argument('--rotate',     type=int,   default=0,
                   choices=[0, 90, 180, 270],
                   help='Rotate camera image before processing')
    p.add_argument('--mirror',     action='store_true',
                   help='Mirror camera image left-right')
    p.add_argument('--calibrate',  action='store_true',
                   help='Force recalibration even if homography exists')
    p.add_argument('--ros-camera', type=str,   default='',
                   dest='ros_camera',
                   help='ROS CompressedImage topic (e.g. /niryo_robot_vision/compressed)')
    # consume ROS remapping args
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()
    node = ChessVisionNode(args)
    node.run()


if __name__ == '__main__':
    main()
