#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
chess_piece_vision_node.py  –  Chess Vision for ROS Melodic / Python 2.7
=========================================================================
* Subscribes to a compressed-image ROS topic (Niryo camera) **or** opens
  a local USB camera (--cam 0, 1, …).
* Interactive 4-corner calibration  (click a1 → h1 → a8 → h8, press S).
* Perspective-warps the board to 400×400, splits into 64 square patches.
* Motion-gated, stability-locked, 3-frame-voted move detector.
* Publishes  /chess_vision/human_move   (UCI string, e.g. "e2e4")
* Publishes  /chess_vision/board_state  (JSON per-square occupancy + color)
* Publishes  /chess_vision/cleanup_event (square name after capture removal)
* Piece-color classifier  (white / black / empty) via brightness + texture.
* Optional HOG/SVM piece-type labelling if model files exist.
* Capture-removal grace window (default 12 s) prevents freeze.
"""
from __future__ import print_function

import sys
import os
import json
import time
import copy
import math
import threading
import numpy as np
import cv2
from collections import Counter

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
CONFIG_DIR = os.path.expanduser("~/.chess_vision")
SQDICT_DEFAULT = os.path.join(CONFIG_DIR, "sqdict.json")
SVM_PATH = os.path.join(CONFIG_DIR, "piece_svm.yml")
CLASSES_PATH = os.path.join(CONFIG_DIR, "piece_classes.json")

# Also check package-local config dir
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_CONFIG = os.path.join(os.path.dirname(_THIS_DIR), "config")

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
BOARD_PX = 400                    # warped board is 400×400
SQ_PX   = BOARD_PX // 8          # 50 px per square

STABLE_FRAMES   = 8              # consecutive low-motion frames before "stable"
MOTION_FRAC_THR = 0.015          # fraction of changed pixels to count as motion
PIXEL_DIFF_THR  = 25             # per-pixel grey diff to flag as changed

VOTE_ROUNDS     = 3              # snapshot votes after stability
VOTE_AGREE      = 2              # minimum agreement

MOVE_SCORE_THR  = 12.0           # per-square mean-diff threshold
MIN_CHANGED_SQ  = 1
MAX_CHANGED_SQ  = 4

CLEANUP_GRACE_S = 12.0           # seconds to watch for capture-piece removal
LIGHTING_DRIFT  = 3.5            # max global brightness drift before slow adapt
BASELINE_ALPHA  = 0.015          # EMA alpha for slow lighting adaptation

DUP_MOVE_COOL_S = 4.0            # ignore duplicate UCI strings within this window


# ╔══════════════════════════════════════════════════════════════╗
# ║  Helpers                                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def _ensure_dir(d):
    if not os.path.isdir(d):
        os.makedirs(d)


def _sq_name(fi, ri):
    """file index 0-7, rank index 0-7  →  'a1'..'h8'."""
    return chr(ord('a') + fi) + str(ri + 1)


def _patch(img, sq, margin=4):
    """Extract the interior of a square (avoids grid-line artefacts)."""
    x = sq['x'] + margin
    y = sq['y'] + margin
    w = sq['w'] - 2 * margin
    h = sq['h'] - 2 * margin
    return img[y:y + h, x:x + w]


def _gray(img):
    if img is None:
        return None
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _piece_color(patch):
    """Classify a square patch → 'w', 'b', or 'empty'."""
    g = _gray(patch)
    if g is None or g.size == 0:
        return 'empty'
    h, w = g.shape[:2]
    centre = g[h // 4:3 * h // 4, w // 4:3 * w // 4]
    cmean = float(np.mean(centre))
    cstd  = float(np.std(centre))
    # Very low texture → probably empty square
    if cstd < 8:
        return 'empty'
    edges = cv2.Canny(g, 40, 130)
    edensity = np.sum(edges > 0) / float(edges.size)
    if edensity < 0.03 and cstd < 15:
        return 'empty'
    if cmean > 145:
        return 'w'
    if cmean < 95:
        return 'b'
    # Ambiguous mid-range – use edge density tie-break
    if cmean > 115:
        return 'w'
    return 'b'


def _build_sqdict():
    """Return a fresh sqdict (pixel rects in the 400×400 warped image)."""
    sqdict = {}
    for fi in range(8):
        for ri in range(8):
            name = _sq_name(fi, ri)
            sqdict[name] = {
                'x': fi * SQ_PX,
                'y': (7 - ri) * SQ_PX,   # rank 8 at y=0, rank 1 at y=350
                'w': SQ_PX,
                'h': SQ_PX,
            }
    return sqdict


def _H_from_corners(corners):
    """
    corners : dict  {'a1':(x,y), 'h1':(x,y), 'a8':(x,y), 'h8':(x,y)}
    in the (possibly rotated/mirrored) raw camera image.

    Returns 3×3 perspective matrix that maps raw→400×400 board where
        top-left  = a8   (rank 8, file a)
        top-right = h8
        bot-right = h1
        bot-left  = a1
    """
    src = np.float32([corners['a8'], corners['h8'],
                      corners['h1'], corners['a1']])
    dst = np.float32([[0, 0], [BOARD_PX, 0],
                      [BOARD_PX, BOARD_PX], [0, BOARD_PX]])
    return cv2.getPerspectiveTransform(src, dst)


# ╔══════════════════════════════════════════════════════════════╗
# ║  Main node                                                   ║
# ╚══════════════════════════════════════════════════════════════╝

class ChessVisionNode(object):

    # ── init ──────────────────────────────────────────────────
    def __init__(self):
        rospy.init_node('chess_vision_node', anonymous=False)

        # Parameters (set from launch file or command line)
        self.rotate      = int(rospy.get_param('~rotate', 0))
        self.mirror       = bool(rospy.get_param('~mirror', False))
        self.do_calibrate = bool(rospy.get_param('~calibrate', False))
        self.cam_id       = int(rospy.get_param('~cam', -1))
        self.topic        = rospy.get_param('~topic',
                                '/niryo_robot_vision/compressed_video_stream')
        self.human_color  = rospy.get_param('~human_color', 'w')
        self.sqdict_path  = rospy.get_param('~sqdict_path', SQDICT_DEFAULT)

        # ---- state ---------------------------------------------------
        self.frame       = None
        self.frame_lock  = threading.Lock()
        self.H           = None
        self.sqdict      = None
        self.baseline    = None
        self.prev_gray   = None

        self.stable_cnt  = 0
        self.in_motion   = False
        self.vote_buf    = []
        self.last_move_t = 0.0
        self.last_uci    = ""
        self.cleanup_until = 0.0

        self.svm          = None
        self.piece_classes = None
        self.label_cache   = {}       # sq_name → last stable label

        # ---- ROS publishers ------------------------------------------
        self.pub_board   = rospy.Publisher('/chess_vision/board_state',
                                          String, queue_size=1)
        self.pub_move    = rospy.Publisher('/chess_vision/human_move',
                                          String, queue_size=1)
        self.pub_cleanup = rospy.Publisher('/chess_vision/cleanup_event',
                                          String, queue_size=1)

        # ---- camera --------------------------------------------------
        self.cap = None
        self.use_topic = False
        if self.cam_id >= 0:
            self.cap = cv2.VideoCapture(self.cam_id)
            if not self.cap.isOpened():
                rospy.logfatal("Cannot open camera %d", self.cam_id)
                sys.exit(1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            rospy.loginfo("Opened local camera %d", self.cam_id)
        else:
            self.use_topic = True
            rospy.Subscriber(self.topic, CompressedImage,
                             self._img_cb, queue_size=1,
                             buff_size=2**24)
            rospy.loginfo("Waiting for images on %s …", self.topic)

        # ---- optional SVM --------------------------------------------
        self._load_svm()

        # ---- calibration / load --------------------------------------
        need_cal = self.do_calibrate
        if not need_cal:
            loaded = self._load_cal()
            if not loaded:
                rospy.logwarn("No sqdict found – entering calibration.")
                need_cal = True
        if need_cal:
            self._run_calibration()

        if self.H is None:
            rospy.logfatal("Calibration failed – exiting.")
            sys.exit(1)

        # ---- baseline ------------------------------------------------
        rospy.loginfo("Capturing baseline …")
        rospy.sleep(1.0)
        self._take_baseline()
        rospy.loginfo("Vision node READY.")

    # ── camera helpers ────────────────────────────────────────
    def _img_cb(self, msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            with self.frame_lock:
                self.frame = img

    def _grab(self):
        """Return one BGR frame (or None)."""
        if self.use_topic:
            with self.frame_lock:
                return self.frame.copy() if self.frame is not None else None
        else:
            for _ in range(4):
                self.cap.grab()
            ok, f = self.cap.read()
            return f if ok else None

    def _pre(self, img):
        """Rotate + mirror the raw frame (before perspective warp)."""
        if img is None:
            return None
        if self.rotate != 0:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0),
                                        self.rotate, 1.0)
            cos_a = abs(M[0, 0])
            sin_a = abs(M[0, 1])
            nw = int(h * sin_a + w * cos_a)
            nh = int(h * cos_a + w * sin_a)
            M[0, 2] += (nw - w) / 2.0
            M[1, 2] += (nh - h) / 2.0
            img = cv2.warpAffine(img, M, (nw, nh))
        if self.mirror:
            img = cv2.flip(img, 1)
        return img

    def _warp(self, img):
        """Full pipeline: pre-transform → perspective warp → 400×400."""
        img = self._pre(img)
        if img is None or self.H is None:
            return None
        return cv2.warpPerspective(img, self.H, (BOARD_PX, BOARD_PX))

    # ── SVM ───────────────────────────────────────────────────
    def _load_svm(self):
        if os.path.isfile(SVM_PATH) and os.path.isfile(CLASSES_PATH):
            try:
                self.svm = cv2.ml.SVM_load(SVM_PATH)
                with open(CLASSES_PATH, 'r') as f:
                    self.piece_classes = json.load(f)
                rospy.loginfo("Loaded piece SVM from %s", SVM_PATH)
            except Exception as exc:
                rospy.logwarn("SVM load failed: %s", exc)
                self.svm = None

    def _svm_label(self, patch):
        """Return e.g. 'wP', 'bN' or None."""
        if self.svm is None or self.piece_classes is None:
            return None
        try:
            g = _gray(patch)
            g = cv2.resize(g, (64, 64))
            hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
            feat = hog.compute(g).flatten()
            _, res = self.svm.predict(np.array([feat], dtype=np.float32))
            idx = int(res[0][0])
            return self.piece_classes.get(str(idx), None)
        except Exception:
            return None

    # ── calibration ───────────────────────────────────────────
    def _load_cal(self):
        for p in [self.sqdict_path,
                  os.path.join(_PKG_CONFIG, "sqdict.json"),
                  SQDICT_DEFAULT]:
            if os.path.isfile(p):
                try:
                    with open(p) as f:
                        data = json.load(f)
                    self.H = np.array(data['H'], dtype=np.float64)
                    self.sqdict = data['sqdict']
                    rospy.loginfo("Loaded calibration from %s", p)
                    return True
                except Exception as exc:
                    rospy.logwarn("Bad sqdict %s: %s", p, exc)
        return False

    def _save_cal(self, path):
        _ensure_dir(os.path.dirname(path))
        data = {'H': self.H.tolist(), 'sqdict': self.sqdict}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        # Also save to package config dir
        pkg_path = os.path.join(_PKG_CONFIG, "sqdict.json")
        try:
            _ensure_dir(_PKG_CONFIG)
            with open(pkg_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
        rospy.loginfo("Calibration saved → %s", path)

    def _run_calibration(self):
        """
        Interactive OpenCV window.  User clicks four corners in order:
            a1  →  h1  →  a8  →  h8
        then presses  S  to save,  R  to reset,  Q  to quit.
        """
        names  = ['a1', 'h1', 'a8', 'h8']
        labels = [
            'a1  (white QR – bottom-left)',
            'h1  (white KR – bottom-right)',
            'a8  (black QR – top-left)',
            'h8  (black KR – top-right)',
        ]
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        corners = {}
        pts = []
        idx = [0]

        def _mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and idx[0] < 4:
                pts.append((x, y))
                corners[names[idx[0]]] = (x, y)
                idx[0] += 1

        win = 'Calibration – click a1 h1 a8 h8, then S'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 800, 600)
        cv2.setMouseCallback(win, _mouse)
        rospy.loginfo("=== CALIBRATION: click a1, h1, a8, h8 then press S ===")

        while not rospy.is_shutdown():
            raw = self._grab()
            if raw is None:
                rospy.sleep(0.05)
                continue
            disp = self._pre(raw.copy())
            if disp is None:
                continue

            # Header
            if idx[0] < 4:
                txt = "Click: " + labels[idx[0]]
            else:
                txt = "All set!  S=save  R=reset  Q=quit"
            cv2.putText(disp, txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw points + labels
            for i, p in enumerate(pts):
                cv2.circle(disp, p, 8, colors[i], -1)
                cv2.putText(disp, names[i], (p[0] + 12, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

            # Outline
            if len(pts) == 4:
                poly = np.array([pts[2], pts[3], pts[1], pts[0]], np.int32)
                cv2.polylines(disp, [poly], True, (0, 255, 0), 2)
                # Also show warped preview in corner
                try:
                    H_tmp = _H_from_corners(corners)
                    preview = cv2.warpPerspective(
                        self._pre(raw), H_tmp, (BOARD_PX, BOARD_PX))
                    small = cv2.resize(preview, (160, 160))
                    disp[10:170, disp.shape[1] - 170:disp.shape[1] - 10] = small
                except Exception:
                    pass

            cv2.imshow(win, disp)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('s'), ord('S')):
                if idx[0] == 4:
                    self.H = _H_from_corners(corners)
                    self.sqdict = _build_sqdict()
                    self._save_cal(self.sqdict_path)
                    break
                else:
                    rospy.logwarn("Need 4 corners first")
            elif key in (ord('r'), ord('R')):
                idx[0] = 0
                pts[:] = []
                corners.clear()
            elif key in (ord('q'), ord('Q')):
                break

        cv2.destroyAllWindows()
        # Small delay so the window actually closes
        for _ in range(5):
            cv2.waitKey(1)

    # ── baseline ──────────────────────────────────────────────
    def _take_baseline(self):
        for _ in range(30):
            f = self._grab()
            if f is not None:
                w = self._warp(f)
                if w is not None:
                    self.baseline = w.copy()
                    self.prev_gray = _gray(w)
                    return True
            rospy.sleep(0.1)
        rospy.logwarn("Could not capture baseline!")
        return False

    # ── motion ────────────────────────────────────────────────
    def _motion_frac(self, cur_gray):
        if self.prev_gray is None:
            return 0.0
        diff = cv2.absdiff(cur_gray, self.prev_gray)
        _, thr = cv2.threshold(diff, PIXEL_DIFF_THR, 255, cv2.THRESH_BINARY)
        return np.sum(thr > 0) / float(thr.size)

    def _global_drift(self, cur_gray):
        if self.baseline is None:
            return 0.0
        bg = _gray(self.baseline)
        return abs(float(np.mean(cur_gray)) - float(np.mean(bg)))

    # ── square comparison ─────────────────────────────────────
    def _sq_scores(self, warped):
        """Per-square change score  baseline → warped."""
        if self.baseline is None or self.sqdict is None:
            return {}
        scores = {}
        for name, sq in self.sqdict.items():
            bp = _gray(_patch(self.baseline, sq))
            cp = _gray(_patch(warped, sq))
            diff_mean = float(np.mean(cv2.absdiff(bp, cp)))
            # Edge texture difference (helps with same-colour moves)
            be = cv2.Canny(bp, 40, 130)
            ce = cv2.Canny(cp, 40, 130)
            edge_d = float(np.mean(cv2.absdiff(be, ce)))
            scores[name] = diff_mean + edge_d * 0.25
        return scores

    def _occupancy_change(self, scores, warped):
        """
        Return (src, dst) square names  or  None.
        src = became empty,  dst = became occupied.
        """
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        changed = [(n, s) for n, s in ranked if s > MOVE_SCORE_THR]
        if len(changed) < MIN_CHANGED_SQ:
            return None
        changed = changed[:MAX_CHANGED_SQ]

        became_empty  = []
        became_filled = []

        for name, sc in changed:
            sq = self.sqdict[name]
            bp = _gray(_patch(self.baseline, sq))
            cp = _gray(_patch(warped, sq))
            bstd = float(np.std(bp))
            cstd = float(np.std(cp))
            # Higher texture = likely occupied
            if bstd > cstd + 2.5:
                became_empty.append((name, sc))
            elif cstd > bstd + 2.5:
                became_filled.append((name, sc))
            else:
                # Tie-break: assign to whichever list is shorter
                became_empty.append((name, sc * 0.5))
                became_filled.append((name, sc * 0.5))

        if became_empty and became_filled:
            src = max(became_empty, key=lambda x: x[1])[0]
            dst = max(became_filled, key=lambda x: x[1])[0]
            if src != dst:
                return (src, dst)

        # Fallback: top-2
        if len(changed) >= 2:
            n1, _ = changed[0]
            n2, _ = changed[1]
            sq1 = self.sqdict[n1]
            sq2 = self.sqdict[n2]
            cs1 = float(np.std(_gray(_patch(warped, sq1))))
            cs2 = float(np.std(_gray(_patch(warped, sq2))))
            if cs1 > cs2:
                return (n2, n1)
            else:
                return (n1, n2)

        return None

    def _cleanup_check(self, scores):
        """In grace window, detect single-square removal."""
        changed = [(n, s) for n, s in scores.items() if s > MOVE_SCORE_THR]
        if len(changed) == 1:
            return changed[0][0]
        # Also accept if one square dominates heavily
        if len(changed) >= 2:
            changed.sort(key=lambda x: x[1], reverse=True)
            if changed[0][1] > changed[1][1] * 2.0:
                return changed[0][0]
        return None

    # ── board state ───────────────────────────────────────────
    def _board_state_json(self, warped, motion_frac):
        """Build JSON dict published on /chess_vision/board_state."""
        out = {
            'motion': self.in_motion,
            'motion_frac': round(motion_frac, 4),
            'stable': self.stable_cnt >= STABLE_FRAMES,
            'in_cleanup': time.time() < self.cleanup_until,
        }
        if self.sqdict is None or warped is None:
            return json.dumps(out)

        sq_map = {}
        for name, sq in self.sqdict.items():
            p = _patch(warped, sq)
            col = _piece_color(p)
            lbl = col
            # SVM label (only if occupied)
            if col != 'empty':
                svm_lbl = self._svm_label(p)
                if svm_lbl is not None:
                    lbl = svm_lbl
                    self.label_cache[name] = lbl
                elif name in self.label_cache:
                    lbl = self.label_cache[name]
            else:
                self.label_cache.pop(name, None)
            sq_map[name] = {'color': col, 'label': lbl}

        out['squares'] = sq_map
        return json.dumps(out)

    # ── main loop ─────────────────────────────────────────────
    def run(self):
        rate = rospy.Rate(15)
        board_pub_interval = 1.0   # seconds between full board-state publishes
        last_board_pub = 0.0

        while not rospy.is_shutdown():
            raw = self._grab()
            if raw is None:
                rate.sleep()
                continue

            warped = self._warp(raw)
            if warped is None:
                rate.sleep()
                continue

            cur_gray = _gray(warped)
            now = time.time()

            # ── motion gate ───────────────────────────────────
            mfrac = self._motion_frac(cur_gray)
            self.prev_gray = cur_gray.copy()

            if mfrac > MOTION_FRAC_THR:
                if not self.in_motion:
                    rospy.logdebug("Motion started (%.4f)", mfrac)
                self.in_motion = True
                self.stable_cnt = 0
                self.vote_buf = []
            else:
                if self.in_motion:
                    self.stable_cnt += 1
                    if self.stable_cnt >= STABLE_FRAMES:
                        self.in_motion = False
                        self.vote_buf = []
                        rospy.logdebug("Board stable after motion")

            # ── lighting drift slow-adapt ─────────────────────
            if self.baseline is not None:
                drift = self._global_drift(cur_gray)
                if drift > LIGHTING_DRIFT and drift < 25:
                    self.baseline = cv2.addWeighted(
                        self.baseline, 1.0 - BASELINE_ALPHA,
                        warped, BASELINE_ALPHA, 0)

            # ── move / cleanup detection ──────────────────────
            in_grace = now < self.cleanup_until

            if (not self.in_motion
                    and self.stable_cnt >= STABLE_FRAMES
                    and self.baseline is not None):

                scores = self._sq_scores(warped)

                if in_grace:
                    # Cleanup mode
                    cln = self._cleanup_check(scores)
                    if cln is not None:
                        rospy.loginfo("Cleanup: %s removed", cln)
                        self.pub_cleanup.publish(String(data=cln))
                        self.baseline = warped.copy()
                        self.cleanup_until = 0.0
                        self.stable_cnt = 0
                        self.vote_buf = []
                else:
                    # Normal vote-based detection
                    if len(self.vote_buf) < VOTE_ROUNDS:
                        cand = self._occupancy_change(scores, warped)
                        if cand is not None:
                            self.vote_buf.append(cand)
                        else:
                            # No move seen – reset votes if buffer non-empty
                            # (avoids stale partial votes)
                            if self.vote_buf:
                                self.vote_buf = []
                                self.stable_cnt = 0

                    if len(self.vote_buf) >= VOTE_ROUNDS:
                        counts = Counter(self.vote_buf)
                        best, cnt = counts.most_common(1)[0]
                        if cnt >= VOTE_AGREE:
                            src, dst = best
                            uci = src + dst
                            # Duplicate guard
                            if (uci != self.last_uci
                                    or (now - self.last_move_t) > DUP_MOVE_COOL_S):
                                rospy.loginfo(">>> MOVE: %s", uci)
                                self.pub_move.publish(String(data=uci))
                                self.last_uci = uci
                                self.last_move_t = now
                                self.cleanup_until = now + CLEANUP_GRACE_S
                                self.baseline = warped.copy()
                        self.vote_buf = []
                        self.stable_cnt = 0

            # ── publish board state (throttled) ───────────────
            if now - last_board_pub > board_pub_interval:
                msg = self._board_state_json(warped, mfrac)
                self.pub_board.publish(String(data=msg))
                last_board_pub = now

            rate.sleep()

        # Cleanup
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


# ╔══════════════════════════════════════════════════════════════╗
# ║  Entry point                                                 ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    # Strip ROS remapping args before any argparse
    argv = rospy.myargv(argv=sys.argv)
    try:
        node = ChessVisionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
