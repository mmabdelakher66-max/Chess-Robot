#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
chess_piece_vision_node.py
ROS Melodic / Python 2.7
Niryo Ned 1 — overhead camera chess vision

Calibration labels on the board:
  a8 = top-left    h8 = top-right
  a1 = bottom-left h1 = bottom-right
  (when rotate=0 and human is at top of image)

Orientation logic
-----------------
--rotate 0   : human at top, a8 TL, h1 BR  (default)
--rotate 180 : physical board flipped 180, same result after transform
--mirror     : flip left-right after rotation

Topics published
----------------
/chess_vision/board_state   (std_msgs/String) JSON
/chess_vision/human_move    (std_msgs/String) UCI e.g. "e2e4"
/chess_vision/cleanup_event (std_msgs/String) "cleanup:e5" after capture removal
"""

from __future__ import print_function
import sys
import os
import cv2
import json
import math
import time
import copy
import numpy as np
import threading
import argparse

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

# ──────────────────────────────────────────────────────────────
# Constants / tunables
# ──────────────────────────────────────────────────────────────
SQDICT_PATH   = os.path.expanduser("~/.chess_vision/sqdict.json")
SVM_PATH      = os.path.expanduser("~/.chess_vision/piece_svm.yml")
CLASSES_PATH  = os.path.expanduser("~/.chess_vision/piece_classes.json")

# ══════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS — edit these to tune detection behaviour
# ══════════════════════════════════════════════════════════════
#
# MOTION_THRESH (default 6)
#   Frame-to-frame absdiff mean that triggers "motion detected".
#   Your camera noise at idle = ~2.1.  Hand moving a piece ≈ 8-20.
#   Too low  → phantom motion from vibration/shadows.
#   Too high → misses slow piece placements.
#   Recommended range: 4 – 8
#
MOTION_THRESH       = 6
#
# STABLE_FRAMES_NEED (default 5)
#   How many consecutive frames below MOTION_THRESH before we vote.
#   Higher = waits longer after piece is placed before measuring.
#   Recommended: 4 – 8
#
STABLE_FRAMES_NEED  = 5
#
# VOTE_FRAMES (default 3)
#   Number of frames averaged together during the vote.
#   More frames = more robust but slightly slower response.
#
VOTE_FRAMES         = 3
#
# OCC_THRESHOLD (default 15)
#   Per-square mean-delta (vs baseline) to call a square "changed".
#   From your data: real piece moves give 14–70; noise gives 1–7.
#   Too low  → false squares flagged (shadows, reflections).
#   Too high → destination square not detected (your h4 was 14.3).
#   Recommended range: 12 – 20.  Start at 12 if moves are missed.
#
OCC_THRESHOLD       = 12.0
#
# LIGHTING_DRIFT_MAX (default 6)
#   If the board-wide mean delta exceeds this, subtract it as
#   global lighting drift. Prevents all 64 squares triggering on
#   a light flicker.  Should be > camera noise but < real move.
#
LIGHTING_DRIFT_MAX  = 6.0
#
# BASELINE_IDLE_LOCK (default 300)
#   Frames of pure idle before auto-refreshing the baseline.
#   At 15 fps this is 20 seconds.  Prevents slow drift but also
#   prevents re-baselining during a paused game.
#   Set higher (600+) if you keep getting phantom moves.
#
BASELINE_IDLE_LOCK  = 300
#
# GRACE_WINDOW_SEC (default 4)
#   Seconds after a legal move is accepted where a single-square
#   change is treated as "captured piece cleanup" not a new move.
#   Reduced from 12 → 4 so the next move can be detected quickly.
#   If you remove captured pieces slowly, raise this to 6-8.
#
GRACE_WINDOW_SEC    = 4.0
#
# CAPTURE_CLEANUP_MAX (default 2)
#   Max squares that may change during grace window (cleanup).
#
CAPTURE_CLEANUP_MAX = 2
# ══════════════════════════════════════════════════════════════

HOG_WIN  = (32, 32)
HOG_CELL = (8, 8)
HOG_BLK  = (2, 2)

# ──────────────────────────────────────────────────────────────
# Square name helpers
# ──────────────────────────────────────────────────────────────
FILES = 'abcdefgh'
RANKS = '12345678'

def sq_name(col, row):
    """col 0-7 = a-h, row 0-7 = rank 1-8."""
    return FILES[col] + RANKS[row]

def sq_indices(name):
    """Return (col 0-7, row 0-7) from 'e2'."""
    return FILES.index(name[0]), RANKS.index(name[1])

# ──────────────────────────────────────────────────────────────
# sqdict helpers
# ──────────────────────────────────────────────────────────────

def load_sqdict(path):
    with open(path, 'r') as f:
        raw = json.load(f)
    # values might be list-of-lists; convert to numpy arrays
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


def save_sqdict(path, sqdict):
    dirp = os.path.dirname(path)
    if dirp and not os.path.isdir(dirp):
        os.makedirs(dirp)
    out = {k: v.tolist() for k, v in sqdict.items()}
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    rospy.loginfo("sqdict saved to %s", path)


def poly_center(pts):
    return np.mean(pts, axis=0)


def extract_patch(img, pts, size=48):
    """Warp the polygon defined by pts into a square patch."""
    pts = pts.astype(np.float32)
    dst = np.array([[0, 0],[size-1, 0],[size-1, size-1],[0, size-1]], dtype=np.float32)
    if len(pts) == 4:
        M = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(img, M, (size, size))
    else:
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        roi = img[max(0,y):y+h, max(0,x):x+w]
        return cv2.resize(roi, (size, size)) if roi.size > 0 else np.zeros((size,size,3),np.uint8)


def patch_mean(img, pts):
    patch = extract_patch(img, pts)
    return np.mean(patch.astype(np.float32))


def patch_delta(img1, img2, pts):
    p1 = extract_patch(img1, pts).astype(np.float32)
    p2 = extract_patch(img2, pts).astype(np.float32)
    return np.mean(np.abs(p1 - p2))


def is_occupied(img, pts, baseline_mean, threshold=35.0):
    """
    Compare current patch mean against baseline.
    threshold: delta above which we call the square occupied/changed.
    """
    cur = patch_mean(img, pts)
    return abs(cur - baseline_mean) > threshold

# ──────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# Corner → chess-square mapping for every rotate+mirror combo
#
# Physical camera view (raw, rotate=0, no mirror):
#   TL = a8  (black queenside rook corner)
#   TR = h8  (black kingside rook corner)
#   BR = h1  (white kingside rook corner)
#   BL = a1  (white queenside rook corner)
#
# After apply_transform(rotate, mirror) those corners shift.
# Table gives (TL_sq, TR_sq, BR_sq, BL_sq) in the DISPLAYED image.
# ──────────────────────────────────────────────────────────────
_CORNER_MAP = {
    # (rotate_deg, mirror): (TL_sq, TR_sq, BR_sq, BL_sq)
    (0,   False): ('a8', 'h8', 'h1', 'a1'),
    (90,  False): ('a1', 'a8', 'h8', 'h1'),
    (180, False): ('h1', 'a1', 'a8', 'h8'),
    (270, False): ('h8', 'h1', 'a1', 'a8'),
    (0,   True):  ('h8', 'a8', 'a1', 'h1'),
    (90,  True):  ('a8', 'a1', 'h1', 'h8'),
    (180, True):  ('a1', 'h1', 'h8', 'a8'),
    (270, True):  ('h1', 'h8', 'a8', 'a1'),
}

def get_corner_names(rotate_deg, mirror):
    """Return [TL, TR, BR, BL] chess square names for this transform."""
    key = (rotate_deg % 360, bool(mirror))
    return list(_CORNER_MAP.get(key, ('a8', 'h8', 'h1', 'a1')))


def calibrate_interactive(cap, rotate_deg, mirror, sqdict_path):
    """
    Interactive calibration.
    The window shows the board AFTER rotation+mirror is applied.
    Labels in each corner tell you exactly which chess square to click.
    Click order: TOP-LEFT → TOP-RIGHT → BOTTOM-RIGHT → BOTTOM-LEFT
    Press S to save, R to retry, Q to quit.
    """
    corner_names = get_corner_names(rotate_deg, mirror)  # [TL, TR, BR, BL]

    rospy.loginfo("=== CALIBRATION MODE ===  rotate=%d  mirror=%s", rotate_deg, mirror)
    rospy.loginfo("Click board corners in order: %s(TL)  %s(TR)  %s(BR)  %s(BL)",
                  *corner_names)
    rospy.loginfo("Press S to save, R to retry, Q to quit")

    WIN = "Calibration: click %s %s %s %s then press S" % tuple(corner_names)
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            rospy.loginfo("Click %d: (%d,%d) => %s",
                          len(clicks), x, y, corner_names[len(clicks)-1])

    frame = None
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.1)
    if frame is None:
        rospy.logerr("Cannot read from camera during calibration")
        return None

    frame = apply_transform(frame, rotate_deg, mirror)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 800)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        ret, raw = cap.read()
        if ret and raw is not None:
            frame = apply_transform(raw, rotate_deg, mirror)

        disp = frame.copy()
        h, w = disp.shape[:2]

        # Draw chess-square labels at the 4 image corners so user knows where to click
        # Positions: TL, TR, BR, BL
        lbl_positions = [(8, 28), (w-70, 28), (w-70, h-8), (8, h-8)]
        for lbl, pos in zip(corner_names, lbl_positions):
            cv2.putText(disp, lbl, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 255), 2)

        # Draw already-clicked points
        for i, pt in enumerate(clicks):
            cv2.circle(disp, pt, 8, (0, 255, 0), -1)
            cv2.putText(disp, corner_names[i], (pt[0]+10, pt[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(clicks) == 4:
            pts = np.array(clicks, dtype=np.float32)
            cv2.polylines(disp, [pts.astype(np.int32).reshape(-1, 1, 2)],
                          True, (0, 255, 0), 2)
            cv2.putText(disp, "Press S to save, R to retry",
                        (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        status = "Clicks: %d/4" % len(clicks)
        if len(clicks) < 4:
            status += "  Next: click %s" % corner_names[len(clicks)]
        cv2.putText(disp, status, (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 0), 2)

        cv2.imshow(WIN, disp)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('r'), ord('R')):
            clicks = []
            rospy.loginfo("Retry — click %s %s %s %s", *corner_names)
        elif key in (ord('s'), ord('S')):
            if len(clicks) == 4:
                sqdict = build_sqdict_from_corners(clicks, frame.shape,
                                                   rotate_deg, mirror)
                save_sqdict(sqdict_path, sqdict)
                cv2.destroyWindow(WIN)
                return sqdict
            else:
                rospy.logwarn("Need 4 clicks first")
        elif key in (ord('q'), ord('Q')):
            cv2.destroyWindow(WIN)
            return None

    cv2.destroyWindow(WIN)
    return None


def build_sqdict_from_corners(corners, img_shape, rotate_deg=0, mirror=False):
    """
    corners = [TL_click, TR_click, BR_click, BL_click] in the displayed image.

    Uses the rotate+mirror-aware corner map to determine which chess square
    (a1..h8) corresponds to each image corner, then bilinear-interpolates
    all 64 square polygons using the 4 named chess-corner anchors.

    The interpolation is always in chess-canonical space:
      a8 = top-left of chess board
      h8 = top-right
      h1 = bottom-right
      a1 = bottom-left
    """
    corner_names = get_corner_names(rotate_deg, mirror)  # [TL_sq, TR_sq, BR_sq, BL_sq]

    # Map chess corner name → image point
    img_pt = {}
    for name, click in zip(corner_names, corners):
        img_pt[name] = np.array(click, dtype=np.float64)

    # Chess-canonical anchor points in IMAGE space
    pt_a8 = img_pt['a8']   # chess top-left
    pt_h8 = img_pt['h8']   # chess top-right
    pt_h1 = img_pt['h1']   # chess bottom-right
    pt_a1 = img_pt['a1']   # chess bottom-left

    def interp(u, v):
        """
        u = file fraction (0=file-a, 1=file-h)
        v = rank fraction (0=rank-8/top, 1=rank-1/bottom) in chess space
        """
        return (1-v)*((1-u)*pt_a8 + u*pt_h8) + v*((1-u)*pt_a1 + u*pt_h1)

    sqdict = {}
    for rank_idx in range(8):    # 0 = rank1, 7 = rank8
        for file_idx in range(8):  # 0 = file-a, 7 = file-h
            u0 = file_idx / 8.0
            u1 = (file_idx + 1) / 8.0
            # rank8 → v=0 (chess top), rank1 → v=1 (chess bottom)
            v0 = (7 - rank_idx) / 8.0
            v1 = (7 - rank_idx + 1) / 8.0

            ptTL = interp(u0, v0)
            ptTR = interp(u1, v0)
            ptBR = interp(u1, v1)
            ptBL = interp(u0, v1)

            name = sq_name(file_idx, rank_idx)
            sqdict[name] = np.array([ptTL, ptTR, ptBR, ptBL], dtype=np.float32)

    return sqdict

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
# HOG feature + SVM piece labeler (optional)
# ──────────────────────────────────────────────────────────────

class PieceLabeler(object):
    def __init__(self):
        self.svm = None
        self.classes = []
        self._last_labels = {}
        self._load()

    def _load(self):
        if os.path.isfile(SVM_PATH) and os.path.isfile(CLASSES_PATH):
            try:
                self.svm = cv2.ml.SVM_load(SVM_PATH)
                with open(CLASSES_PATH, 'r') as f:
                    self.classes = json.load(f)
                rospy.loginfo("PieceLabeler loaded: %d classes", len(self.classes))
            except Exception as e:
                rospy.logwarn("PieceLabeler load failed: %s", e)
                self.svm = None
        else:
            rospy.loginfo("No SVM model found — piece labeling disabled")

    def _hog(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, HOG_WIN)
        hog = cv2.HOGDescriptor(
            HOG_WIN, HOG_BLK,
            (HOG_CELL[0]//2, HOG_CELL[1]//2),
            HOG_CELL, 9)
        feat = hog.compute(gray)
        return feat.flatten()

    def label(self, img, sqdict):
        """Return dict sq_name -> label_str for all 64 squares."""
        if self.svm is None:
            return {}
        out = {}
        for name, pts in sqdict.items():
            patch = extract_patch(img, pts, size=HOG_WIN[0])
            feat = self._hog(patch).reshape(1, -1).astype(np.float32)
            _, res = self.svm.predict(feat)
            idx = int(res[0][0])
            if 0 <= idx < len(self.classes):
                lbl = self.classes[idx]
            else:
                lbl = '??'
            # keep last stable label — never go to unknown
            if lbl == '??' or lbl == 'empty':
                lbl = self._last_labels.get(name, lbl)
            else:
                self._last_labels[name] = lbl
            out[name] = lbl
        return out

# ──────────────────────────────────────────────────────────────
# Motion detector
# ──────────────────────────────────────────────────────────────

class MotionGate(object):
    def __init__(self, thresh=MOTION_THRESH):
        self.thresh = thresh
        self._prev_gray = None

    def update(self, frame):
        """Return (motion_detected:bool, mean_delta:float)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return False, 0.0
        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        mean_d = float(np.mean(diff))
        return mean_d > self.thresh, mean_d

# ──────────────────────────────────────────────────────────────
# Baseline occupancy map
# ──────────────────────────────────────────────────────────────

class BaselineMap(object):
    def __init__(self, sqdict):
        self.sqdict = sqdict
        self.means = {}   # sq -> float mean of baseline patch

    def update(self, img):
        for name, pts in self.sqdict.items():
            self.means[name] = patch_mean(img, pts)

    def delta_map(self, img):
        """Return dict sq -> float delta vs baseline."""
        out = {}
        for name, pts in self.sqdict.items():
            cur = patch_mean(img, pts)
            out[name] = abs(cur - self.means.get(name, cur))
        return out

# ──────────────────────────────────────────────────────────────
# Move detector state machine
# ──────────────────────────────────────────────────────────────

STATE_IDLE      = 'idle'
STATE_MOTION    = 'motion'
STATE_WAIT_STABLE = 'wait_stable'
STATE_VOTING    = 'voting'
STATE_GRACE     = 'grace'   # after legal move, waiting for capture cleanup


class MoveDetector(object):
    def __init__(self, sqdict, labeler, move_pub, state_pub, cleanup_pub):
        self.sqdict      = sqdict
        self.labeler     = labeler
        self.move_pub    = move_pub
        self.state_pub   = state_pub
        self.cleanup_pub = cleanup_pub

        self.baseline    = BaselineMap(sqdict)
        self.motion_gate = MotionGate()

        self.state           = STATE_IDLE
        self.stable_count    = 0
        self.idle_count      = 0
        self.vote_frames     = []
        self.last_frame      = None
        self.grace_start     = 0.0
        self.grace_baseline  = None   # snapshot at start of grace

        # Pre-move snapshot: saved when motion first detected.
        # Used as the "before" reference in _analyse_vote so that
        # piece shadows and lighting are identical in before/after frames.
        self._last_stable_frame = None
        self.pre_move_frame     = None

        # Set True when engine_move published; vision waits for
        # physical board change then re-baselines WITHOUT publishing.
        self._expecting_engine  = False

        self._baseline_ready = False
        self._lock           = threading.Lock()

    # ── public ──────────────────────────────────────────────

    def feed(self, frame):
        """Process one frame. Thread-safe."""
        with self._lock:
            self._process(frame)

    def reset_after_rejection(self):
        """
        Called when GUI rejected our published move as illegal.
        The physical board has already changed, but we published
        wrong square names.  Re-capture baseline from the CURRENT
        frame so we stop seeing that change as 'new'.
        """
        with self._lock:
            if self.last_frame is not None:
                self.baseline.update(self.last_frame)
            self.state        = STATE_IDLE
            self.stable_count = 0
            self.idle_count   = 0
            self.vote_frames  = []
        rospy.logwarn("Baseline reset after rejected move — board re-learned")

    def force_baseline(self, frame):
        """Manually set baseline (called after calibration or 'B' key)."""
        with self._lock:
            self.baseline.update(frame)
            self._baseline_ready = True
            self.state = STATE_IDLE
            rospy.loginfo("Baseline forced from calibration frame")

    # ── internal ────────────────────────────────────────────

    def _process(self, frame):
        motion, delta = self.motion_gate.update(frame)
        self.last_frame = frame

        # Always publish current state JSON
        self._pub_state(frame, motion, delta)

        if self.state == STATE_IDLE:
            self._idle(frame, motion, delta)
        elif self.state == STATE_MOTION:
            self._in_motion(frame, motion, delta)
        elif self.state == STATE_WAIT_STABLE:
            self._wait_stable(frame, motion, delta)
        elif self.state == STATE_VOTING:
            self._voting(frame, motion)
        elif self.state == STATE_GRACE:
            self._grace(frame, motion, delta)

    def _idle(self, frame, motion, delta):
        if not self._baseline_ready:
            self.baseline.update(frame)
            self._baseline_ready = True
            self._last_stable_frame = frame
            return

        if motion:
            # Save the last quiet frame as fresh "before" reference for this vote.
            # It has the same lighting/shadows as the post-move frame, so piece
            # shadows from other pieces cancel out completely.
            if self._last_stable_frame is not None:
                self.pre_move_frame = self._last_stable_frame
            else:
                self.pre_move_frame = frame
            self.state = STATE_MOTION
            self.stable_count = 0
            rospy.logdebug("IDLE -> MOTION  delta=%.2f", delta)
        else:
            self._last_stable_frame = frame
            self.idle_count += 1
            if self.idle_count >= BASELINE_IDLE_LOCK:
                # Very slow baseline drift correction
                self.baseline.update(frame)
                self.idle_count = 0

    def _in_motion(self, frame, motion, delta):
        if not motion:
            self.stable_count += 1
            if self.stable_count >= STABLE_FRAMES_NEED:
                self.state = STATE_VOTING
                self.vote_frames = [frame]
                rospy.logdebug("MOTION -> VOTING after %d stable frames", self.stable_count)
        else:
            self.stable_count = 0

    def _wait_stable(self, frame, motion, delta):
        # same as in_motion but we arrive here from VOTING fallback
        self._in_motion(frame, motion, delta)

    def _voting(self, frame, motion):
        if motion:
            # motion restarted — go back to MOTION
            self.state = STATE_MOTION
            self.stable_count = 0
            self.vote_frames = []
            return

        self.vote_frames.append(frame)
        if len(self.vote_frames) < VOTE_FRAMES:
            return

        # We have enough frames — analyze
        move = self._analyse_vote()
        self.vote_frames = []

        if move:
            if self._expecting_engine:
                # This is the engine's physical move being placed on the board.
                # Do NOT publish it as a human move — just re-baseline.
                rospy.loginfo("Engine move physically executed (%s) — re-baselining", move)
                self._expecting_engine = False
                self.baseline.update(frame)
                self.state = STATE_IDLE
                self.idle_count = 0
            else:
                rospy.loginfo("Move detected: %s", move)
                self.move_pub.publish(move)
                # Start grace window for capture cleanup
                self.state = STATE_GRACE
                self.grace_start = time.time()
                self.grace_baseline = copy.deepcopy(self.baseline)
                # Update baseline to post-move state
                self.baseline.update(frame)
                self.idle_count = 0
        else:
            # False trigger — update baseline and return to idle
            self.baseline.update(frame)
            self.state = STATE_IDLE
            self.idle_count = 0

    def _grace(self, frame, motion, delta):
        elapsed = time.time() - self.grace_start
        if elapsed > GRACE_WINDOW_SEC:
            # Grace expired naturally — baseline the now-settled board
            self.baseline.update(frame)
            self._last_stable_frame = frame
            self.state = STATE_IDLE
            self.idle_count = 0
            rospy.logdebug("Grace window expired — re-baselined")
            return

        if motion:
            return  # ignore motion during grace (hand removing captured piece)

        # Check for single-square cleanup (captured piece removal)
        dm = self.baseline.delta_map(frame)
        changed = [sq for sq, d in dm.items() if d > OCC_THRESHOLD]

        if 0 < len(changed) <= CAPTURE_CLEANUP_MAX:
            # Publish cleanup event but do NOT update baseline here.
            # The user's hand may still be in frame.  Grace expires
            # naturally and will baseline the clean board.
            for sq in changed:
                rospy.loginfo("Cleanup event: %s", sq)
                self.cleanup_pub.publish("cleanup:" + sq)

    def _analyse_vote(self):
        """
        Find the two squares with the highest average delta.

        KEY: uses pre_move_frame (saved just before motion started) as the
        "before" reference instead of the aging baseline.  Both frames are
        taken within seconds of each other, so all piece shadows and
        reflections are IDENTICAL and cancel out in the diff.  Only squares
        where a piece actually moved will show large delta.

        Returns UCI string (src+dst) or None.
        """
        # Build per-square reference means from the pre-move snapshot.
        # Fall back to stored baseline if no snapshot is available.
        if self.pre_move_frame is not None:
            ref_means = {}
            for name, pts in self.sqdict.items():
                ref_means[name] = patch_mean(self.pre_move_frame, pts)
            rospy.logdebug("Using pre-move snapshot as vote reference")
        else:
            ref_means = self.baseline.means
            rospy.logdebug("No pre-move snapshot — using baseline")

        # Accumulate per-square delta across all vote frames
        delta_acc = {}
        for frame in self.vote_frames:
            for name, pts in self.sqdict.items():
                after  = patch_mean(frame, pts)
                before = ref_means.get(name, after)
                d      = abs(after - before)
                delta_acc[name] = delta_acc.get(name, 0.0) + d

        n = max(len(self.vote_frames), 1)
        ranked = sorted(delta_acc.items(), key=lambda x: -x[1])

        # Always log top-6 so user can verify coordinate mapping
        rospy.loginfo("=== VOTE RESULT — top changed squares ===")
        for sq, acc in ranked[:6]:
            avg = acc / n
            marker = " <<< CHANGED" if avg > OCC_THRESHOLD else ""
            rospy.loginfo("  %s : avg_delta=%.1f%s", sq, avg, marker)

        above = [(sq, acc / n) for sq, acc in ranked if acc / n > OCC_THRESHOLD]

        if len(above) < 2:
            rospy.loginfo("Only %d square(s) above threshold %.1f — ignoring (false trigger)",
                          len(above), OCC_THRESHOLD)
            return None

        sq1, d1 = above[0]
        sq2, d2 = above[1]

        # Determine src vs dst using signed brightness change:
        #   Source square: piece was removed → square now looks more like bare board
        #   Destination  : piece was added   → square brightness changed toward piece color
        #
        # Heuristic: the src is the square whose current mean is CLOSER to neighboring
        # empty squares.  Simpler proxy: the square that got BRIGHTER is more likely to
        # be the source of a dark piece, or the square that got DARKER is source of a
        # light piece.  Because we don't know piece color, we use the square where
        # |signed change| is larger — that's the square that changed MORE = source.
        last = self.vote_frames[-1]
        cur1 = patch_mean(last, self.sqdict[sq1])
        cur2 = patch_mean(last, self.sqdict[sq2])
        b1   = ref_means.get(sq1, cur1)
        b2   = ref_means.get(sq2, cur2)

        chg1 = abs(cur1 - b1)
        chg2 = abs(cur2 - b2)

        if chg1 >= chg2:
            src, dst = sq1, sq2
        else:
            src, dst = sq2, sq1

        rospy.loginfo("  sq1=%s cur=%.1f ref=%.1f |chg|=%.1f", sq1, cur1, b1, chg1)
        rospy.loginfo("  sq2=%s cur=%.1f ref=%.1f |chg|=%.1f", sq2, cur2, b2, chg2)

        rospy.loginfo("Best move candidate: %s -> %s  (delta %.1f -> %.1f)",
                      src, dst, d1, d2)
        return src + dst

    def _pub_state(self, frame, motion, delta):
        dm = self.baseline.delta_map(frame) if self._baseline_ready else {}
        occ = {sq: bool(d > OCC_THRESHOLD) for sq, d in dm.items()}
        data = {
            'state':   self.state,
            'motion':  bool(motion),
            'delta':   round(delta, 3),
            'stable':  self.stable_count,
            'occupied': occ,
        }
        self.state_pub.publish(json.dumps(data))

# ──────────────────────────────────────────────────────────────
# ROS Image subscriber
# ──────────────────────────────────────────────────────────────

class ImageSubscriber(object):
    def __init__(self, topic, rotate_deg, mirror, callback):
        self.rotate_deg = rotate_deg
        self.mirror     = mirror
        self.callback   = callback
        self._sub = rospy.Subscriber(topic, CompressedImage,
                                     self._cb, queue_size=1,
                                     buff_size=2**24)
        rospy.loginfo("Subscribed to %s", topic)

    def _cb(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return
            frame = apply_transform(frame, self.rotate_deg, self.mirror)
            self.callback(frame)
        except Exception as e:
            rospy.logwarn_throttle(5, "Image decode error: %s", e)

# ──────────────────────────────────────────────────────────────
# Camera fallback (USB / local)
# ──────────────────────────────────────────────────────────────

def open_local_camera(index=0):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        rospy.loginfo("Opened local camera index %d", index)
    return cap

# ──────────────────────────────────────────────────────────────
# Display thread
# ──────────────────────────────────────────────────────────────

def _sq_under_point(sqdict, x, y):
    """Return the square name whose polygon contains pixel (x, y), or ''."""
    for name, pts in sqdict.items():
        poly = pts.astype(np.float32).reshape((-1, 1, 2))
        if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
            return name
    return ''


class DisplayThread(threading.Thread):
    def __init__(self, detector):
        super(DisplayThread, self).__init__()
        self.detector  = detector
        self.daemon    = True
        self._running  = True
        self._mouse_x  = 0
        self._mouse_y  = 0

    def run(self):
        WIN = "Chess Vision"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        def on_mouse(event, x, y, flags, param):
            self._mouse_x = x
            self._mouse_y = y

        cv2.setMouseCallback(WIN, on_mouse)

        while self._running and not rospy.is_shutdown():
            with self.detector._lock:
                frame  = self.detector.last_frame
                sqdict = self.detector.sqdict
                dm     = self.detector.baseline.delta_map(frame) if (
                    frame is not None and self.detector._baseline_ready) else {}
                state  = self.detector.state

            if frame is None:
                time.sleep(0.05)
                continue

            disp = frame.copy()
            h, w = disp.shape[:2]

            for name, pts in sqdict.items():
                ctr     = poly_center(pts).astype(int)
                d       = dm.get(name, 0.0)
                changed = d > OCC_THRESHOLD
                color   = (0, 0, 220) if changed else (0, 200, 0)
                cv2.polylines(disp,
                              [pts.astype(np.int32).reshape(-1, 1, 2)],
                              True, color, 1)
                cv2.putText(disp, name,
                            tuple(ctr - np.array([14, 5])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

            # Mouse-hover: show which square the cursor is over
            hovered = _sq_under_point(sqdict, self._mouse_x, self._mouse_y)
            if hovered:
                hover_pts = sqdict[hovered].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(disp, [hover_pts], True, (0, 255, 255), 2)
                cv2.putText(disp, ">> " + hovered,
                            (self._mouse_x + 10, self._mouse_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            status = "State:%-12s Hover:%-4s  [B]=reset baseline  [Q]=quit" % (
                state, hovered)
            cv2.putText(disp, status, (6, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1)

            cv2.imshow(WIN, disp)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                rospy.signal_shutdown("User quit display")
                break
            elif key in (ord('b'), ord('B')):
                # Force re-baseline from current frame
                with self.detector._lock:
                    if self.detector.last_frame is not None:
                        self.detector.baseline.update(self.detector.last_frame)
                        self.detector.state        = STATE_IDLE
                        self.detector.stable_count = 0
                        self.detector.idle_count   = 0
                        self.detector.vote_frames  = []
                        self.detector._baseline_ready = True
                rospy.logwarn("=== BASELINE RESET by user (B key) ===")

        cv2.destroyAllWindows()

    def stop(self):
        self._running = False

# ──────────────────────────────────────────────────────────────
# Main node
# ──────────────────────────────────────────────────────────────

def main():
    rospy.init_node('chess_vision_node', anonymous=False)

    # ── Read config: ROS params first, then CLI args as override ──
    # ROS params (set by launch file <param> tags)
    rotate_deg     = rospy.get_param('~rotate',    0)
    mirror         = rospy.get_param('~mirror',    False)
    need_calibrate = rospy.get_param('~calibrate', False)
    show           = rospy.get_param('~show',      False)
    cam_index      = rospy.get_param('~cam',       -1)
    topic          = rospy.get_param('~topic',
                        '/niryo_robot_vision/compressed_video_stream')

    # Allow CLI overrides (for direct python invocation without launch)
    argv = rospy.myargv(argv=sys.argv)
    parser = argparse.ArgumentParser(description='Chess Vision Node', add_help=False)
    parser.add_argument('--rotate',    type=int,   default=None)
    parser.add_argument('--mirror',    action='store_true', default=None)
    parser.add_argument('--calibrate', action='store_true', default=None)
    parser.add_argument('--cam',       type=int,   default=None)
    parser.add_argument('--topic',     type=str,   default=None)
    parser.add_argument('--show',      action='store_true', default=None)
    cli, _ = parser.parse_known_args(argv[1:])
    if cli.rotate    is not None: rotate_deg     = cli.rotate
    if cli.mirror:                mirror         = True
    if cli.calibrate:             need_calibrate = True
    if cli.cam       is not None: cam_index      = cli.cam
    if cli.topic     is not None: topic          = cli.topic
    if cli.show:                  show           = True

    # Publishers
    move_pub    = rospy.Publisher('/chess_vision/human_move',    String, queue_size=5)
    state_pub   = rospy.Publisher('/chess_vision/board_state',   String, queue_size=5)
    cleanup_pub = rospy.Publisher('/chess_vision/cleanup_event', String, queue_size=5)

    # Open camera
    cap = None
    using_ros_topic = (cam_index == -1)

    if not using_ros_topic:
        cap = open_local_camera(cam_index)
        if not cap.isOpened():
            rospy.logerr("Cannot open camera index %d", cam_index)
            return

    # ── Calibration ─────────────────────────────────────────
    need_calibrate = need_calibrate or not os.path.isfile(SQDICT_PATH)

    if need_calibrate:
        rospy.loginfo("Starting calibration…")
        if using_ros_topic:
            # Temporarily use local cam for calibration if needed, else grab from topic
            tmp_cap = open_local_camera(0)
            if not tmp_cap.isOpened():
                rospy.logerr("Need a camera for calibration. Use --cam 0 or connect USB camera.")
                return
            sqdict = calibrate_interactive(tmp_cap, rotate_deg, mirror, SQDICT_PATH)
            tmp_cap.release()
        else:
            sqdict = calibrate_interactive(cap, rotate_deg, mirror, SQDICT_PATH)

        if sqdict is None:
            rospy.logwarn("Calibration cancelled — exiting")
            return
    else:
        sqdict = load_sqdict(SQDICT_PATH)
        rospy.loginfo("Loaded sqdict with %d squares", len(sqdict))

    # ── Build detector ───────────────────────────────────────
    labeler  = PieceLabeler()
    detector = MoveDetector(sqdict, labeler, move_pub, state_pub, cleanup_pub)

    # ── GUI feedback subscriptions ───────────────────────────
    def _on_move_rejected(msg):
        """GUI tells us the move was illegal — reset baseline from current frame."""
        rospy.logwarn("Move rejected by GUI: %s — resetting baseline", msg.data)
        detector.reset_after_rejection()

    def _on_game_reset(msg):
        """New Game pressed — reset state machine and re-baseline."""
        rospy.loginfo("Game reset received — re-baselining")
        with detector._lock:
            if detector.last_frame is not None:
                detector.baseline.update(detector.last_frame)
            detector.state        = STATE_IDLE
            detector.stable_count = 0
            detector.idle_count   = 0
            detector.vote_frames  = []
            detector._baseline_ready = True

    def _on_engine_move(msg):
        """Engine played — wait for user to physically make that move, then re-baseline."""
        rospy.loginfo("Engine move: %s — waiting for physical execution on board", msg.data)
        with detector._lock:
            detector._expecting_engine = True
            # Reset to IDLE so we can detect the physical motion
            detector.state        = STATE_IDLE
            detector.stable_count = 0
            detector.idle_count   = 0
            detector.vote_frames  = []

    rospy.Subscriber('/chess_vision/move_rejected', String, _on_move_rejected, queue_size=5)
    rospy.Subscriber('/chess_vision/game_reset',    String, _on_game_reset,    queue_size=5)
    rospy.Subscriber('/chess_vision/engine_move',   String, _on_engine_move,   queue_size=5)

    if show:
        disp_thread = DisplayThread(detector)
        disp_thread.start()

    # ── Feed frames ──────────────────────────────────────────
    if using_ros_topic:
        def ros_frame_cb(frame):
            detector.feed(frame)

        _ = ImageSubscriber(topic, rotate_deg, mirror, ros_frame_cb)
        rospy.loginfo("Chess Vision Node running (ROS topic mode)")
        rospy.spin()

    else:
        rospy.loginfo("Chess Vision Node running (local camera mode)")
        rate = rospy.Rate(15)
        # Grab initial baseline frame
        for _ in range(10):
            ret, raw = cap.read()
            if ret and raw is not None:
                frame = apply_transform(raw, rotate_deg, mirror)
                detector.force_baseline(frame)
                break
            time.sleep(0.05)

        while not rospy.is_shutdown():
            ret, raw = cap.read()
            if not ret or raw is None:
                rospy.logwarn_throttle(5, "Camera read failed")
                rate.sleep()
                continue
            frame = apply_transform(raw, rotate_deg, mirror)
            detector.feed(frame)
            rate.sleep()

        cap.release()

    if show:
        disp_thread.stop()

    rospy.loginfo("Chess Vision Node stopped")


if __name__ == '__main__':
    main()
