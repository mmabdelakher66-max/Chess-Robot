#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
chess_gui_stockfish_node.py
ROS Melodic / Python 2.7
Tkinter GUI + Stockfish engine for chess-robot

Roles
-----
- Engine = BLACK  (plays automatically after each validated human move)
- Human  = WHITE

Subscribes
----------
/chess_vision/human_move    (String UCI)
/chess_vision/board_state   (String JSON)  -- for status display
/chess_vision/cleanup_event (String)       -- "cleanup:e5" — ignored for game logic

Publishes
---------
/chess_vision/engine_move   (String UCI)
"""

from __future__ import print_function
import sys
import os
import json
import time
import threading
import argparse
import subprocess

import rospy
from std_msgs.msg import String

# ── Tkinter (Python 2.7) ────────────────────────────────────
import Tkinter as tk
import ttk

# ── python-chess 0.23.x compatible (no parse_square) ────────
import chess
import chess.uci

# ──────────────────────────────────────────────────────────────
# Compatibility shim — chess.parse_square was added in 0.26
# ──────────────────────────────────────────────────────────────

def parse_square(name):
    """Convert square name like 'e4' -> chess square index (0-63)."""
    file_idx = 'abcdefgh'.index(name[0])
    rank_idx = int(name[1]) - 1
    return chess.square(file_idx, rank_idx)

# ──────────────────────────────────────────────────────────────
# Stockfish paths to try
# ──────────────────────────────────────────────────────────────
STOCKFISH_PATHS = [
    '/usr/games/stockfish',
    '/usr/bin/stockfish',
    '/usr/local/bin/stockfish',
    os.path.expanduser('~/stockfish'),
]

def find_stockfish():
    for p in STOCKFISH_PATHS:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    # Try which
    try:
        result = subprocess.check_output(['which', 'stockfish']).strip()
        if result:
            return result
    except Exception:
        pass
    return None

# ──────────────────────────────────────────────────────────────
# Board image helpers
# ──────────────────────────────────────────────────────────────
SQ_SIZE = 64  # pixels per square in GUI

LIGHT_COLOR = '#F0D9B5'
DARK_COLOR  = '#B58863'
HIGHLIGHT_SRC = '#AAD742'
HIGHLIGHT_DST = '#CDF048'
LAST_MOVE_COLOR = '#BACA44'

PIECE_UNICODE = {
    'P': u'\u2659', 'N': u'\u2658', 'B': u'\u2657',
    'R': u'\u2656', 'Q': u'\u2655', 'K': u'\u2654',
    'p': u'\u265F', 'n': u'\u265E', 'b': u'\u265D',
    'r': u'\u265C', 'q': u'\u265B', 'k': u'\u265A',
}

# ──────────────────────────────────────────────────────────────
# GUI Application
# ──────────────────────────────────────────────────────────────

class ChessGUI(object):
    def __init__(self, root, stockfish_path, engine_skill, engine_depth):
        self.root = root
        self.root.title("Chess Robot — Engine=BLACK  Human=WHITE")
        self.root.resizable(False, False)

        self.board = chess.Board()
        self.engine = None
        self.engine_path  = stockfish_path
        self.engine_skill = engine_skill
        self.engine_depth = engine_depth

        self.last_move       = None
        self.status_text     = tk.StringVar()
        self.move_history    = []
        self._engine_lock    = threading.Lock()
        self._game_over      = False
        self._processing     = False   # prevent re-entrant engine calls

        self._build_ui()
        self._start_engine()
        self.draw_board()

    # ── UI construction ──────────────────────────────────────

    def _build_ui(self):
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(padx=8, pady=8)

        # Board canvas
        board_size = SQ_SIZE * 8
        self.canvas = tk.Canvas(main_frame, width=board_size, height=board_size,
                                borderwidth=0, highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=(0, 8))

        # Right panel
        right = tk.Frame(main_frame, bg='#2b2b2b', width=260)
        right.grid(row=0, column=1, sticky='ns')

        tk.Label(right, text="Status", bg='#2b2b2b', fg='white',
                 font=('Helvetica', 11, 'bold')).pack(anchor='w')

        self.status_lbl = tk.Label(right, textvariable=self.status_text,
                                   bg='#333', fg='#00FF99',
                                   font=('Courier', 10),
                                   wraplength=240, justify='left',
                                   relief='sunken', padx=6, pady=6)
        self.status_lbl.pack(fill='x', pady=(0, 8))

        tk.Label(right, text="Move History", bg='#2b2b2b', fg='white',
                 font=('Helvetica', 11, 'bold')).pack(anchor='w')

        self.history_text = tk.Text(right, height=24, width=28,
                                    bg='#1a1a1a', fg='#dddddd',
                                    font=('Courier', 9),
                                    state='disabled')
        self.history_text.pack(fill='both')

        btn_frame = tk.Frame(right, bg='#2b2b2b')
        btn_frame.pack(fill='x', pady=(8, 0))

        tk.Button(btn_frame, text="New Game",
                  command=self._new_game,
                  bg='#4CAF50', fg='white',
                  font=('Helvetica', 10, 'bold')).pack(side='left', expand=True, fill='x')

        tk.Button(btn_frame, text="Undo",
                  command=self._undo,
                  bg='#FF9800', fg='white',
                  font=('Helvetica', 10, 'bold')).pack(side='left', expand=True, fill='x', padx=(4,0))

        self._set_status("Waiting for human move (WHITE)…")

    # ── Engine ───────────────────────────────────────────────

    def _start_engine(self):
        if not self.engine_path:
            self._set_status("ERROR: Stockfish not found!\nInstall: sudo apt install stockfish")
            return
        try:
            self.engine = chess.uci.popen_engine(self.engine_path)
            self.engine.uci()
            if self.engine_skill is not None:
                self.engine.setoption({'Skill Level': self.engine_skill})
            rospy.loginfo("Stockfish started: %s", self.engine_path)
        except Exception as e:
            rospy.logerr("Failed to start engine: %s", e)
            self.engine = None
            self._set_status("ERROR starting Stockfish:\n" + str(e))

    # ── Move handling (called from ROS thread) ───────────────

    def on_human_move(self, uci_str):
        """Called from ROS subscriber thread."""
        self.root.after(0, lambda: self._handle_human_move(uci_str))

    def on_board_state(self, json_str):
        """Called from ROS subscriber thread."""
        self.root.after(0, lambda: self._handle_board_state(json_str))

    def on_cleanup_event(self, event_str):
        """Called from ROS subscriber thread. No game logic change needed."""
        rospy.loginfo("Cleanup event received: %s", event_str)
        self.root.after(0, lambda: self._set_status(
            "Cleanup: " + event_str + "\n(capture piece removed)"))

    # ── Internal move handlers (run on Tk main thread) ───────

    def _handle_human_move(self, uci_str):
        uci_str = uci_str.strip()
        if not uci_str or self._game_over or self._processing:
            return
        if self.board.turn != chess.WHITE:
            self._set_status("Not WHITE's turn — ignoring: " + uci_str)
            return

        # Validate
        try:
            move = chess.Move.from_uci(uci_str)
        except Exception:
            self._set_status("Bad UCI: " + uci_str)
            return

        if move not in self.board.legal_moves:
            # Try promotion (auto-queen)
            promo = chess.Move.from_uci(uci_str + 'q')
            if promo in self.board.legal_moves:
                move = promo
            else:
                self._set_status("Illegal move: " + uci_str)
                rospy.logwarn("Illegal move: %s  FEN: %s", uci_str, self.board.fen())
                return

        self.board.push(move)
        self.last_move = move
        self.move_history.append(move.uci())
        self._update_history()
        self.draw_board()
        self._check_game_over()
        if not self._game_over:
            self._trigger_engine_move()

    def _handle_board_state(self, json_str):
        try:
            data = json.loads(json_str)
            state  = data.get('state', '')
            motion = data.get('motion', False)
            delta  = data.get('delta', 0.0)
            stable = data.get('stable', 0)
            turn   = 'WHITE' if self.board.turn == chess.WHITE else 'BLACK'
            msg = "Vision: %s\nMotion: %s  delta=%.2f  stable=%d\nTurn: %s" % (
                state, motion, delta, stable, turn)
            self._set_status(msg)
        except Exception:
            pass

    # ── Engine move ──────────────────────────────────────────

    def _trigger_engine_move(self):
        if self._game_over or self._processing:
            return
        if self.board.turn != chess.BLACK:
            return
        self._processing = True
        self._set_status("Engine thinking…")
        t = threading.Thread(target=self._engine_move_thread)
        t.daemon = True
        t.start()

    def _engine_move_thread(self):
        try:
            with self._engine_lock:
                if self.engine is None:
                    self.root.after(0, lambda: self._set_status("No engine available"))
                    return
                self.engine.position(self.board)
                cmd = chess.uci.InfoHandler()
                self.engine.info_handlers.append(cmd)
                move, ponder = self.engine.go(depth=self.engine_depth)
                self.engine.info_handlers.remove(cmd)

            if move:
                self.root.after(0, lambda: self._apply_engine_move(move))
            else:
                self.root.after(0, lambda: self._set_status("Engine returned no move"))
        except Exception as e:
            rospy.logerr("Engine error: %s", e)
            self.root.after(0, lambda: self._set_status("Engine error: " + str(e)))
        finally:
            self._processing = False

    def _apply_engine_move(self, move):
        if move not in self.board.legal_moves:
            self._set_status("Engine illegal move: " + str(move))
            return
        self.board.push(move)
        self.last_move = move
        self.move_history.append("(engine) " + move.uci())
        self._update_history()
        self.draw_board()
        self._check_game_over()
        # Publish engine move
        engine_move_pub.publish(move.uci())
        rospy.loginfo("Engine move: %s", move.uci())
        if not self._game_over:
            self._set_status("Waiting for human move (WHITE)…")

    # ── Game state ───────────────────────────────────────────

    def _check_game_over(self):
        if self.board.is_checkmate():
            winner = 'BLACK' if self.board.turn == chess.WHITE else 'WHITE'
            self._set_status("CHECKMATE — %s wins!" % winner)
            self._game_over = True
        elif self.board.is_stalemate():
            self._set_status("STALEMATE — Draw!")
            self._game_over = True
        elif self.board.is_insufficient_material():
            self._set_status("DRAW — Insufficient material")
            self._game_over = True
        elif self.board.is_check():
            self._set_status("CHECK!")

    def _new_game(self):
        self.board = chess.Board()
        self.last_move    = None
        self.move_history = []
        self._game_over   = False
        self._processing  = False
        self._update_history()
        self.draw_board()
        self._set_status("New game started. Human=WHITE, Engine=BLACK")

    def _undo(self):
        if len(self.board.move_stack) >= 2:
            self.board.pop()
            self.board.pop()
            if self.move_history:
                self.move_history.pop()
            if self.move_history:
                self.move_history.pop()
            self.last_move = None
            self._game_over   = False
            self._processing  = False
            self._update_history()
            self.draw_board()
            self._set_status("Undo — waiting for human move (WHITE)…")

    # ── Drawing ──────────────────────────────────────────────

    def draw_board(self):
        self.canvas.delete('all')
        last_from = self.last_move.from_square if self.last_move else None
        last_to   = self.last_move.to_square   if self.last_move else None

        for sq in chess.SQUARES:
            file_ = chess.square_file(sq)
            rank_ = chess.square_rank(sq)

            # Draw with white at bottom (engine=black at top)
            x = file_ * SQ_SIZE
            y = (7 - rank_) * SQ_SIZE

            # Color
            is_light = (file_ + rank_) % 2 == 1
            if sq == last_from:
                color = HIGHLIGHT_SRC
            elif sq == last_to:
                color = HIGHLIGHT_DST
            else:
                color = LIGHT_COLOR if is_light else DARK_COLOR

            self.canvas.create_rectangle(x, y, x + SQ_SIZE, y + SQ_SIZE,
                                          fill=color, outline='')

            # Rank/file labels
            if file_ == 0:
                self.canvas.create_text(x + 3, y + 3,
                                        text=str(rank_ + 1),
                                        anchor='nw', font=('Helvetica', 8),
                                        fill=DARK_COLOR if is_light else LIGHT_COLOR)
            if rank_ == 0:
                self.canvas.create_text(x + SQ_SIZE - 3, y + SQ_SIZE - 3,
                                        text='abcdefgh'[file_],
                                        anchor='se', font=('Helvetica', 8),
                                        fill=DARK_COLOR if is_light else LIGHT_COLOR)

            # Piece
            piece = self.board.piece_at(sq)
            if piece:
                sym = piece.symbol()
                ucode = PIECE_UNICODE.get(sym, sym)
                fill = '#FFFFFF' if piece.color == chess.WHITE else '#000000'
                # outline for visibility
                out = '#000000' if piece.color == chess.WHITE else '#FFFFFF'
                self.canvas.create_text(x + SQ_SIZE // 2, y + SQ_SIZE // 2,
                                        text=ucode,
                                        font=('Helvetica', int(SQ_SIZE * 0.68)),
                                        fill=fill)

    # ── History / status ─────────────────────────────────────

    def _update_history(self):
        self.history_text.config(state='normal')
        self.history_text.delete('1.0', 'end')
        for i in range(0, len(self.move_history), 2):
            move_num = i // 2 + 1
            white_m = self.move_history[i]
            black_m = self.move_history[i+1] if i+1 < len(self.move_history) else ''
            line = "%3d. %-10s %s\n" % (move_num, white_m, black_m)
            self.history_text.insert('end', line)
        self.history_text.config(state='disabled')
        self.history_text.see('end')

    def _set_status(self, msg):
        try:
            self.status_text.set(msg)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────
# Module-level publisher (set after rospy.init_node)
# ──────────────────────────────────────────────────────────────
engine_move_pub = None

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    global engine_move_pub

    rospy.init_node('chess_gui_stockfish_node', anonymous=False)

    # ROS params (set by launch file)
    skill_level    = rospy.get_param('~skill',     5)
    engine_depth   = rospy.get_param('~depth',     12)
    stockfish_path = rospy.get_param('~stockfish', '')

    # CLI overrides (for direct python invocation)
    argv = rospy.myargv(argv=sys.argv)
    parser = argparse.ArgumentParser(description='Chess GUI + Stockfish Node',
                                     add_help=False)
    parser.add_argument('--skill',     type=int, default=None)
    parser.add_argument('--depth',     type=int, default=None)
    parser.add_argument('--stockfish', type=str, default=None)
    cli, _ = parser.parse_known_args(argv[1:])
    if cli.skill     is not None: skill_level    = cli.skill
    if cli.depth     is not None: engine_depth   = cli.depth
    if cli.stockfish is not None: stockfish_path = cli.stockfish

    stockfish_path = stockfish_path or find_stockfish()

    engine_move_pub = rospy.Publisher('/chess_vision/engine_move', String, queue_size=5)
    if not stockfish_path:
        rospy.logwarn("Stockfish not found — GUI will run without engine")

    # ── Create Tk root FIRST, then any StringVar ─────────────
    root = tk.Tk()

    app = ChessGUI(root,
                   stockfish_path=stockfish_path,
                   engine_skill=skill_level,
                   engine_depth=engine_depth)

    # ── ROS Subscribers ──────────────────────────────────────
    def human_move_cb(msg):
        app.on_human_move(msg.data)

    def board_state_cb(msg):
        app.on_board_state(msg.data)

    def cleanup_cb(msg):
        app.on_cleanup_event(msg.data)

    rospy.Subscriber('/chess_vision/human_move',    String, human_move_cb,    queue_size=10)
    rospy.Subscriber('/chess_vision/board_state',   String, board_state_cb,   queue_size=5)
    rospy.Subscriber('/chess_vision/cleanup_event', String, cleanup_cb,       queue_size=10)

    # ── ROS spin in background thread ────────────────────────
    spin_thread = threading.Thread(target=rospy.spin)
    spin_thread.daemon = True
    spin_thread.start()

    rospy.loginfo("Chess GUI Node started — engine=%s", stockfish_path)

    # ── Tkinter mainloop (must run on main thread) ────────────
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        rospy.signal_shutdown("GUI closed")

    rospy.loginfo("Chess GUI Node stopped")


if __name__ == '__main__':
    main()
