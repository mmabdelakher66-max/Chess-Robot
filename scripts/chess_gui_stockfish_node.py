#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
chess_gui_stockfish_node.py  –  Tkinter Chess GUI + Stockfish for ROS Melodic
=============================================================================
* Human = WHITE,  Engine = BLACK  (configurable via param).
* Subscribes to  /chess_vision/human_move,  /chess_vision/cleanup_event,
                 /chess_vision/board_state.
* Publishes      /chess_vision/engine_move   (UCI string).
* Uses python-chess 0.23.x (no chess.parse_square – we supply our own).
* Uses chess.uci (not chess.engine) for Python-2.7 / old python-chess compat.
* Tkinter root is created BEFORE any StringVar.
* rospy.myargv() strips __name:= / __log:= before argparse.

Python 2.7 / ROS Melodic compatible.
"""
from __future__ import print_function

import sys
import os
import json
import time
import threading

import rospy
from std_msgs.msg import String

# ── Tkinter (Py2 / Py3) ──────────────────────────────────────
try:
    import Tkinter as tk
    import tkFont as tkfont
    import tkMessageBox as messagebox
except ImportError:
    import tkinter as tk
    import tkinter.font as tkfont
    import tkinter.messagebox as messagebox

import chess
import chess.uci


# ╔══════════════════════════════════════════════════════════════╗
# ║  Helpers (python-chess 0.23 compat)                          ║
# ╚══════════════════════════════════════════════════════════════╝

def parse_square(name):
    """'e2' → chess square index.  Works with every python-chess version."""
    f = ord(name[0]) - ord('a')
    r = int(name[1]) - 1
    return chess.square(f, r)


# Unicode piece symbols (works on most modern terminals / Tkinter)
_PIECE_UNI = {
    'R': u'\u2656', 'N': u'\u2658', 'B': u'\u2657',
    'Q': u'\u2655', 'K': u'\u2654', 'P': u'\u2659',
    'r': u'\u265C', 'n': u'\u265E', 'b': u'\u265D',
    'q': u'\u265B', 'k': u'\u265A', 'p': u'\u265F',
}

# Fallback ASCII piece map (safe everywhere)
_PIECE_ASCII = {
    'R': 'R', 'N': 'N', 'B': 'B', 'Q': 'Q', 'K': 'K', 'P': 'P',
    'r': 'r', 'n': 'n', 'b': 'b', 'q': 'q', 'k': 'k', 'p': 'p',
}


# ╔══════════════════════════════════════════════════════════════╗
# ║  GUI node                                                    ║
# ╚══════════════════════════════════════════════════════════════╝

class ChessGUINode(object):

    LIGHT = '#F0D9B5'
    DARK  = '#B58863'
    HL    = '#AAD751'   # highlight colour for last move

    def __init__(self):
        # ROS init  (BEFORE Tk – Tk is created next)
        argv = rospy.myargv(argv=sys.argv)
        rospy.init_node('chess_gui_stockfish_node', anonymous=False)

        self.skill    = int(rospy.get_param('~skill', 5))
        self.depth    = int(rospy.get_param('~depth', 12))
        self.sf_path  = rospy.get_param('~stockfish', '')
        self.human_is_white = True          # human = WHITE, engine = BLACK

        # ── find stockfish ────────────────────────────────────
        if not self.sf_path:
            for p in ['/usr/games/stockfish',
                      '/usr/bin/stockfish',
                      '/usr/local/bin/stockfish',
                      os.path.expanduser('~/stockfish')]:
                if os.path.isfile(p):
                    self.sf_path = p
                    break
        if not self.sf_path:
            rospy.logfatal("Stockfish not found!  Set ~stockfish param.")
            sys.exit(1)
        rospy.loginfo("Stockfish path: %s", self.sf_path)

        # ── chess state ───────────────────────────────────────
        self.board = chess.Board()
        self.engine = None
        self.game_on = True
        self.human_turn = True
        self.move_history = []
        self.last_move_squares = []   # for highlight
        self.pending_cleanup = False

        # ── start engine ──────────────────────────────────────
        self._start_engine()

        # ── Tk root (MUST exist before any StringVar) ─────────
        self.root = tk.Tk()
        self.root.title("Chess  –  Human (White) vs Engine (Black)")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # StringVar created AFTER Tk()
        self.status_var = tk.StringVar(value="Your turn (White).")
        self.use_unicode = tk.BooleanVar(value=True)

        self._build_ui()
        self._draw_board()

        # ── ROS topics ────────────────────────────────────────
        rospy.Subscriber('/chess_vision/human_move', String,
                         self._cb_human_move, queue_size=1)
        rospy.Subscriber('/chess_vision/cleanup_event', String,
                         self._cb_cleanup, queue_size=1)
        rospy.Subscriber('/chess_vision/board_state', String,
                         self._cb_board_state, queue_size=1)

        self.pub_engine = rospy.Publisher('/chess_vision/engine_move',
                                         String, queue_size=1)

        # Periodic ROS spin inside Tk mainloop
        self.root.after(100, self._tk_spin)

    # ── engine ────────────────────────────────────────────────
    def _start_engine(self):
        try:
            self.engine = chess.uci.popen_engine(self.sf_path)
            self.engine.uci()
            self.engine.setoption({"Skill Level": self.skill})
            info_handler = chess.uci.InfoHandler()
            self.engine.info_handlers.append(info_handler)
            self.engine.isready()
            rospy.loginfo("Stockfish ready  skill=%d  depth=%d",
                          self.skill, self.depth)
        except Exception as exc:
            rospy.logerr("Stockfish failed: %s", exc)
            self.engine = None

    # ── UI build ──────────────────────────────────────────────
    def _build_ui(self):
        self.sq_px = 64
        board_px = self.sq_px * 8

        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Label(top, textvariable=self.status_var,
                 font=("Helvetica", 14, "bold"),
                 wraplength=500).pack()

        mid = tk.Frame(self.root)
        mid.pack(side=tk.TOP)

        # ── board canvas with file/rank labels ────────────────
        lbl_size = 20
        self.canvas = tk.Canvas(mid,
                                width=board_px + 2 * lbl_size,
                                height=board_px + 2 * lbl_size,
                                bg='white')
        self.canvas.pack(side=tk.LEFT, padx=5, pady=5)
        self.board_x0 = lbl_size
        self.board_y0 = lbl_size

        # ── right panel ───────────────────────────────────────
        right = tk.Frame(mid)
        right.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

        tk.Label(right, text="Move history",
                 font=("Helvetica", 12, "bold")).pack(pady=(0, 5))

        self.move_text = tk.Text(right, width=22, height=26,
                                 font=("Courier", 11), state=tk.DISABLED)
        self.move_text.pack()

        btn_fr = tk.Frame(right)
        btn_fr.pack(pady=8)
        tk.Button(btn_fr, text="New Game", width=10,
                  command=self._new_game).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_fr, text="Quit", width=10,
                  command=self._on_close).pack(side=tk.LEFT, padx=3)

    # ── drawing ───────────────────────────────────────────────
    def _draw_board(self):
        c = self.canvas
        c.delete("all")
        sq = self.sq_px
        x0 = self.board_x0
        y0 = self.board_y0
        files = 'abcdefgh'

        hl_set = set(self.last_move_squares)

        for rank in range(8):
            for fil in range(8):
                px = x0 + fil * sq
                # White at bottom → rank 0 at bottom row
                py = y0 + (7 - rank) * sq

                sqidx = chess.square(fil, rank)
                is_light = (fil + rank) % 2 == 0
                if sqidx in hl_set:
                    colour = self.HL
                else:
                    colour = self.LIGHT if is_light else self.DARK
                c.create_rectangle(px, py, px + sq, py + sq,
                                   fill=colour, outline='')

                piece = self.board.piece_at(sqidx)
                if piece is not None:
                    sym = piece.symbol()
                    if self.use_unicode.get():
                        display = _PIECE_UNI.get(sym, sym)
                    else:
                        display = _PIECE_ASCII.get(sym, sym)
                    fill = 'black'
                    c.create_text(px + sq // 2, py + sq // 2,
                                  text=display,
                                  font=("Arial", int(sq * 0.55)),
                                  fill=fill)

        # File labels (a-h)
        for fi in range(8):
            cx = x0 + fi * sq + sq // 2
            c.create_text(cx, y0 + 8 * sq + 10, text=files[fi],
                          font=("Helvetica", 10))
            c.create_text(cx, y0 - 10, text=files[fi],
                          font=("Helvetica", 10))
        # Rank labels (1-8)
        for ri in range(8):
            cy = y0 + (7 - ri) * sq + sq // 2
            c.create_text(x0 - 10, cy, text=str(ri + 1),
                          font=("Helvetica", 10))
            c.create_text(x0 + 8 * sq + 10, cy, text=str(ri + 1),
                          font=("Helvetica", 10))

    def _append_move(self, san, is_white):
        self.move_text.config(state=tk.NORMAL)
        if is_white:
            num = (len(self.move_history) + 1) // 2 + 1
            self.move_text.insert(tk.END, "%d. %s " % (num, san))
        else:
            self.move_text.insert(tk.END, "%s\n" % san)
        self.move_text.see(tk.END)
        self.move_text.config(state=tk.DISABLED)

    # ── ROS callbacks ─────────────────────────────────────────
    def _cb_human_move(self, msg):
        """Called on /chess_vision/human_move."""
        if not self.game_on or not self.human_turn:
            return
        uci_str = msg.data.strip()
        rospy.loginfo("Vision says: %s", uci_str)
        # Schedule on Tk thread
        self.root.after(0, lambda u=uci_str: self._apply_human(u))

    def _cb_cleanup(self, msg):
        """Capture-piece removal – just log, don't touch game state."""
        rospy.loginfo("Cleanup event: square %s", msg.data)
        self.pending_cleanup = False
        self.root.after(0, lambda: self.status_var.set("Cleanup done. Your turn."))

    def _cb_board_state(self, msg):
        """Board state from vision – currently used for debug only."""
        pass

    # ── move logic ────────────────────────────────────────────
    def _apply_human(self, uci_str):
        if not self.game_on or not self.human_turn:
            return

        # Parse
        try:
            move = chess.Move.from_uci(uci_str)
        except (ValueError, IndexError):
            self.status_var.set("Bad format: " + uci_str)
            rospy.logwarn("Bad UCI: %s", uci_str)
            return

        # Legality check (try auto-promote to queen)
        if move not in self.board.legal_moves:
            promo = chess.Move.from_uci(uci_str + 'q')
            if promo in self.board.legal_moves:
                move = promo
            else:
                self.status_var.set("Illegal: " + uci_str)
                rospy.logwarn("Illegal move: %s  (legal: %s)",
                              uci_str,
                              ' '.join(m.uci() for m in self.board.legal_moves))
                return

        san = self.board.san(move)
        is_capture = self.board.is_capture(move)
        self.board.push(move)
        self.move_history.append(move)
        self.last_move_squares = [move.from_square, move.to_square]
        self.human_turn = False

        self._draw_board()
        self._append_move(san, True)

        if is_capture:
            self.pending_cleanup = True

        if self.board.is_game_over():
            self._game_over()
            return

        info = "You played: %s" % san
        if self.board.is_check():
            info += "  CHECK!"
        self.status_var.set(info + "   Engine thinking…")

        # Engine turn after short delay
        self.root.after(300, self._engine_turn)

    def _engine_turn(self):
        if self.engine is None or not self.game_on:
            self.status_var.set("Engine not available!")
            self.human_turn = True
            return

        def think():
            try:
                self.engine.position(self.board)
                result = self.engine.go(depth=self.depth, movetime=2000)
                best = result.bestmove
                if best is None:
                    self.root.after(0, lambda: self.status_var.set(
                        "Engine returned no move."))
                    self.human_turn = True
                    return

                san = self.board.san(best)
                is_capture = self.board.is_capture(best)
                self.board.push(best)
                self.move_history.append(best)
                self.last_move_squares = [best.from_square, best.to_square]

                uci_out = best.uci()
                self.pub_engine.publish(String(data=uci_out))
                rospy.loginfo("Engine: %s (%s)", san, uci_out)

                def update():
                    self._draw_board()
                    self._append_move(san, False)

                    if self.board.is_game_over():
                        self._game_over()
                        return

                    msg = "Engine played: %s" % san
                    if self.board.is_check():
                        msg += "  CHECK!"
                    msg += "   Your turn."
                    self.status_var.set(msg)
                    self.human_turn = True

                self.root.after(0, update)

            except Exception as exc:
                rospy.logerr("Engine error: %s", exc)
                def err():
                    self.status_var.set("Engine error – your turn.")
                    self.human_turn = True
                self.root.after(0, err)

        t = threading.Thread(target=think)
        t.daemon = True
        t.start()

    def _game_over(self):
        self.game_on = False
        res = self.board.result()
        if self.board.is_checkmate():
            msg = "CHECKMATE!  " + res
        elif self.board.is_stalemate():
            msg = "Stalemate!  " + res
        elif self.board.is_insufficient_material():
            msg = "Draw (insufficient material)  " + res
        else:
            msg = "Game over: " + res
        self.status_var.set(msg)
        rospy.loginfo("GAME OVER: %s", msg)

    # ── buttons ───────────────────────────────────────────────
    def _new_game(self):
        self.board = chess.Board()
        self.game_on = True
        self.human_turn = True
        self.move_history = []
        self.last_move_squares = []
        self.pending_cleanup = False
        self.move_text.config(state=tk.NORMAL)
        self.move_text.delete("1.0", tk.END)
        self.move_text.config(state=tk.DISABLED)
        self._draw_board()
        self.status_var.set("New game!  Your turn (White).")
        if self.engine:
            self.engine.ucinewgame()
            self.engine.isready()
        rospy.loginfo("New game started.")

    def _on_close(self):
        self.game_on = False
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass
        self.root.destroy()
        rospy.signal_shutdown("GUI closed")

    # ── Tk ↔ ROS spin ────────────────────────────────────────
    def _tk_spin(self):
        if rospy.is_shutdown():
            self._on_close()
            return
        self.root.after(100, self._tk_spin)

    # ── run ───────────────────────────────────────────────────
    def run(self):
        self.root.mainloop()


# ╔══════════════════════════════════════════════════════════════╗
# ║  Entry point                                                 ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    argv = rospy.myargv(argv=sys.argv)
    try:
        gui = ChessGUINode()
        gui.run()
    except rospy.ROSInterruptException:
        pass
