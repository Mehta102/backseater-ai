# Backseater AI - Tic-Tac-Toe coaching tool
# Uses computer vision to read the board and ML to decide what to say

import cv2
import numpy as np
import json
import mss
import sys
import subprocess
try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# --- Game Logic ---

# maps cell index to a readable name for commentary
POS_NAMES = {
    0: "top left", 1: "top middle", 2: "top right",
    3: "middle left", 4: "center", 5: "middle right",
    6: "bottom left", 7: "bottom middle", 8: "bottom right"
}

# all 8 ways to win (rows, cols, diagonals)
WIN_LINES = [
    [0,1,2], [3,4,5], [6,7,8],
    [0,3,6], [1,4,7], [2,5,8],
    [0,4,8], [2,4,6]
]


def check_winner(board):
    for line in WIN_LINES:
        a, b, c = line
        if board[a] != ' ' and board[a] == board[b] == board[c]:
            return board[a]
    return None


def find_threats(board, player):
    # returns cells where player can win next move
    threats = []
    for line in WIN_LINES:
        cells = [board[i] for i in line]
        if cells.count(player) == 2 and cells.count(' ') == 1:
            empty_idx = line[cells.index(' ')]
            threats.append(empty_idx)
    return threats


def find_move(old_board, new_board):
    # find which cell changed between two board states
    for i in range(9):
        if old_board[i] == ' ' and new_board[i] != ' ':
            return i, new_board[i]
    return None, None


# --- Computer Vision ---

def classify_cell(cell):
    # figure out if a cell contains X, O, or nothing
    h, w = cell.shape[:2]
    if h < 20 or w < 20:
        return ' '

    # convert to grayscale so we only deal with brightness, not color
    if len(cell.shape) == 3:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell

    # pick threshold method based on how bright the background is
    mean_val = np.mean(gray)
    if mean_val > 200:
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    elif mean_val < 80:
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    total_pixels = h * w
    white_pixels = np.sum(binary > 0)
    fill_ratio = white_pixels / total_pixels

    if fill_ratio < 0.05:
        return ' '

    # trim edges to avoid grid lines messing up detection
    margin = int(min(h, w) * 0.15)
    core = binary[margin:h-margin, margin:w-margin]
    if core.size == 0:
        return ' '

    ch, cw = core.shape

    # X detection - check how much of each diagonal is filled
    diag1_fill = 0
    for i in range(min(ch, cw)):
        if core[i, i] > 0:
            diag1_fill += 1
    diag1_fill = diag1_fill / min(ch, cw) if min(ch, cw) > 0 else 0

    diag2_fill = 0
    for i in range(min(ch, cw)):
        if core[i, cw-1-i] > 0:
            diag2_fill += 1
    diag2_fill = diag2_fill / min(ch, cw) if min(ch, cw) > 0 else 0

    # O detection - circle has filled edges but hollow center
    center_h, center_w = ch//3, cw//3
    center_region = core[center_h:2*center_h, center_w:2*center_w]
    center_fill = np.sum(center_region > 0) / center_region.size if center_region.size > 0 else 0

    edge_top = core[0:ch//4, :]
    edge_bot = core[3*ch//4:ch, :]
    edge_left = core[:, 0:cw//4]
    edge_right = core[:, 3*cw//4:cw]

    edge_fill = 0
    for edge in [edge_top, edge_bot, edge_left, edge_right]:
        if edge.size > 0:
            edge_fill += np.sum(edge > 0) / edge.size
    edge_fill = edge_fill / 4

    if (diag1_fill > 0.4 and diag2_fill > 0.4) or (diag1_fill + diag2_fill) / 2 > 0.45:
        return 'X'

    if edge_fill > 0.3 and fill_ratio > 0.15:
        if center_fill < edge_fill * 0.7:
            return 'O'

    # fallback checks
    if fill_ratio > 0.2 and (diag1_fill + diag2_fill) / 2 < 0.3:
        return 'O'

    if (diag1_fill + diag2_fill) / 2 > 0.35:
        return 'X'

    return ' '


def read_board(frame):
    # split the frame into 9 cells and classify each one
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    h, w = gray.shape
    cell_w = w // 3
    cell_h = h // 3

    cells = []
    for row in range(3):
        for col in range(3):
            # small padding so we don't read the grid lines as part of the symbol
            pad_x = int(cell_w * 0.12)
            pad_y = int(cell_h * 0.12)

            x1 = col * cell_w + pad_x
            y1 = row * cell_h + pad_y
            x2 = (col + 1) * cell_w - pad_x
            y2 = (row + 1) * cell_h - pad_y

            if x2 <= x1 or y2 <= y1:
                cells.append(' ')
                continue

            cell = gray[y1:y2, x1:x2]
            cells.append(classify_cell(cell))

    return cells


# --- AI Commentary (Decision Tree) ---

class AICommentary:

    def __init__(self):
        self.model = None
        self.model_file = "commentary_model.pkl"
        # load existing model if we have one, otherwise train from scratch
        if os.path.exists(self.model_file):
            self.load_model()
        else:
            self.train_model()

    def extract_features(self, board, player, pos):
        # turn the board state into a list of numbers the model can use
        opp = 'O' if player == 'X' else 'X'
        corners = [0, 2, 6, 8]

        return [
            board.count(player),
            board.count(opp),
            board.count(' '),
            1 if board[4] == player else 0,       # center control
            sum(1 for i in corners if board[i] == player),
            self.count_threats(board, player),
            self.count_threats(board, opp),
            1 if pos == 4 else 0,
            1 if pos in corners else 0
        ]

    def count_threats(self, board, player):
        # count lines where player has 2 pieces and 1 empty (one move from winning)
        return sum(1 for line in WIN_LINES
                   if [board[i] for i in line].count(player) == 2
                   and [board[i] for i in line].count(' ') == 1)

    def generate_training_data(self):
        # simulate random games and label each board state
        X, y = [], []
        print("Generating training data...")

        for _ in range(300):
            board = [' '] * 9

            for i in range(random.randint(1, 7)):
                empty = [j for j, cell in enumerate(board) if cell == ' ']
                if empty:
                    board[random.choice(empty)] = 'X' if i % 2 == 0 else 'O'

            empty = [j for j, cell in enumerate(board) if cell == ' ']
            if not empty:
                continue

            pos = random.choice(empty)
            features = self.extract_features(board, 'X', pos)

            my_threats = self.count_threats(board, 'X')
            opp_threats = self.count_threats(board, 'O')

            if my_threats >= 2:
                label = 'fork'
            elif my_threats == 1:
                label = 'win'
            elif opp_threats >= 2:
                label = 'danger'
            elif opp_threats == 1:
                label = 'block'
            elif pos == 4:
                label = 'center'
            elif pos in [0, 2, 6, 8]:
                label = 'corner'
            else:
                label = 'neutral'

            X.append(features)
            y.append(label)

        return X, y

    def train_model(self):
        print("Training AI model...")
        X, y = self.generate_training_data()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = DecisionTreeClassifier(max_depth=8, random_state=42)
        self.model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        print(f"Model trained! Accuracy: {accuracy:.1%}")
        self.save_model()

    def save_model(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {self.model_file}")

    def load_model(self):
        with open(self.model_file, 'rb') as f:
            self.model = pickle.load(f)
        print("AI model loaded")

    def predict(self, board, player, pos):
        features = self.extract_features(board, player, pos)
        return self.model.predict([features])[0]


# --- Screen Capture ---

def grab_screen():
    with mss.mss() as cap:
        monitor = cap.monitors[1]
        img = cap.grab(monitor)
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)


def grab_region(x, y, w, h):
    with mss.mss() as cap:
        region = {'left': x, 'top': y, 'width': w, 'height': h}
        img = cap.grab(region)
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)


def save_roi(roi):
    with open("roi.json", 'w') as f:
        json.dump({'x': roi[0], 'y': roi[1], 'w': roi[2], 'h': roi[3]}, f)


def load_roi():
    try:
        with open("roi.json", 'r') as f:
            d = json.load(f)
        return (d['x'], d['y'], d['w'], d['h'])
    except:
        return None


# --- Text to Speech ---

class TTS:
    # runs speech in a background thread so it doesn't block the UI
    # uses macOS 'say' on Mac, pyttsx3 on Windows

    def __init__(self):
        self.q = queue.Queue()
        self.running = True
        self.volume = 1.0
        self._is_mac = sys.platform == "darwin"
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while self.running:
            try:
                text = self.q.get(timeout=0.5)
                if text:
                    try:
                        if self._is_mac:
                            subprocess.run(["say", "-r", "160", text], check=False, timeout=15)
                        elif _PYTTSX3_AVAILABLE:
                            engine = pyttsx3.init()
                            engine.setProperty('rate', 160)
                            engine.setProperty('volume', self.volume)
                            engine.say(text)
                            engine.runAndWait()
                            engine.stop()
                            del engine
                        else:
                            print(f"[TTS] {text}")
                    except Exception as e:
                        print(f"TTS error: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS worker error: {e}")
                continue

    def say(self, text):
        if text:
            self.q.put(text)

    def set_volume(self, v):
        self.volume = max(0.0, min(1.0, v))

    def stop(self):
        self.running = False


# --- Commentary Engine ---

class Commentary:

    COMMENTS = {
        'win':     ["You can WIN at {pos}!", "Victory at {pos}!", "Finish them at {pos}!"],
        'block':   ["Block {pos} or lose!", "URGENT! Block {pos}!", "Defend {pos} now!"],
        'fork':    ["Fork at {pos}!", "Double threat at {pos}!", "Set up fork at {pos}!"],
        'danger':  ["They forked you!", "Multiple threats!", "Tough spot!"],
        'center':  ["Take center at {pos}!", "Center is key!", "Control center!"],
        'corner':  ["Corner at {pos}!", "Good corner play!", "Take the corner!"],
        'neutral': ["Played {pos}.", "Move to {pos}.", "{pos} taken."]
    }

    def __init__(self):
        self.ai = AICommentary()
        self.tts = TTS()

    def generate(self, old_board, new_board, player):
        # always comment from the player's perspective
        pos, piece = find_move(old_board, new_board)
        if pos is None:
            return None

        winner = check_winner(new_board)
        if winner:
            return "You won! Great game!" if winner == player else "They won this time."

        if new_board.count(' ') == 0:
            return "It's a draw!"

        opp = 'O' if player == 'X' else 'X'
        my_threats = find_threats(new_board, player)
        opp_threats = find_threats(new_board, opp)
        is_player_move = (piece == player)

        if is_player_move:
            category = self.ai.predict(new_board, player, pos)
            templates = self.COMMENTS.get(category, self.COMMENTS['neutral'])
            comment = random.choice(templates)

            if '{pos}' in comment:
                if my_threats:
                    fill_pos = my_threats[0]
                elif opp_threats:
                    fill_pos = opp_threats[0]
                else:
                    fill_pos = pos
                comment = comment.format(pos=POS_NAMES[fill_pos])
        else:
            # opponent just moved, tell the player what to do
            if my_threats:
                comment = f"You can WIN at {POS_NAMES[my_threats[0]]}!"
            elif len(opp_threats) >= 2:
                comment = "They forked you! Tough spot."
            elif len(opp_threats) == 1:
                comment = f"Block {POS_NAMES[opp_threats[0]]} or you lose!"
            else:
                comment = f"They played {POS_NAMES[pos]}."

        self.tts.say(comment)
        return comment

    def set_volume(self, v):
        self.tts.set_volume(v)

    def stop(self):
        self.tts.stop()


# --- GUI ---

BG     = "#0d1117"
PANEL  = "#1c2128"
ACCENT = "#58a6ff"
ACCENT2 = "#f85149"
GREEN  = "#3fb950"
TEXT   = "#c9d1d9"
TEXT2  = "#8b949e"
BORDER = "#30363d"


class BackseaterGUI:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Backseater AI - Game Coach")
        self.root.configure(bg=BG)
        self.root.geometry("1200x800")

        self.roi = load_roi()
        self.watching = False
        self.player_side = "X"
        self.commentary = Commentary()
        self.prev_board = [' '] * 9

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=canvas.yview)

        main_frame = tk.Frame(canvas, bg=BG)
        main_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=main_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)

        self.build_left_panel(main_frame)
        self.build_right_panel(main_frame)

    def build_left_panel(self, parent):
        left = tk.Frame(parent, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        capture_frame = tk.Frame(left, bg=PANEL, highlightbackground=BORDER, highlightthickness=2)
        capture_frame.pack(fill="both", expand=True, pady=(0,10))

        header = tk.Frame(capture_frame, bg=PANEL)
        header.pack(fill="x", padx=16, pady=14)

        tk.Label(header, text="🎮 GAME CAPTURE", font=("Segoe UI", 13, "bold"),
                 fg=TEXT, bg=PANEL).pack(side="left")

        self.roi_label = tk.Label(header, text=self.get_roi_text(), font=("Segoe UI", 9),
                                  fg=TEXT2, bg=PANEL)
        self.roi_label.pack(side="right", padx=(0,10))

        tk.Button(header, text="📐 Select Region", font=("Segoe UI", 10, "bold"),
                  bg=ACCENT, fg="white", relief="flat", padx=14, pady=6,
                  command=self.select_roi).pack(side="right")

        self.canvas = tk.Canvas(capture_frame, bg="#161b22", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=16, pady=(0,14))

        analysis_frame = tk.Frame(left, bg=PANEL, highlightbackground=BORDER, highlightthickness=2)
        analysis_frame.pack(fill="x")

        tk.Label(analysis_frame, text="📊 GAME ANALYSIS", font=("Segoe UI", 13, "bold"),
                 fg=TEXT, bg=PANEL).pack(anchor="w", padx=16, pady=14)

    def build_right_panel(self, parent):
        right = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=2)
        right.grid(row=0, column=1, sticky="nsew", padx=(0,10), pady=10)

        tk.Label(right, text="⚙️ CONTROLS", font=("Segoe UI", 13, "bold"),
                 fg=TEXT, bg=PANEL).pack(anchor="w", padx=16, pady=14)

        self.start_btn = tk.Button(right, text="▶ START COACHING", font=("Segoe UI", 10, "bold"),
                                   bg=GREEN, fg="white", relief="flat", padx=14, pady=8,
                                   command=self.toggle_watch)
        self.start_btn.pack(fill="x", padx=16, pady=(0,12))

        tk.Label(right, text="🎮 Playing As", font=("Segoe UI", 11, "bold"),
                 fg=TEXT, bg=PANEL).pack(anchor="w", padx=16, pady=(8,4))

        side_frame = tk.Frame(right, bg=PANEL)
        side_frame.pack(fill="x", padx=16, pady=(0,8))

        self.x_btn = tk.Button(side_frame, text="❌ X", font=("Segoe UI", 10),
                               bg=ACCENT, fg="white", relief="flat", pady=6,
                               command=lambda: self.set_side("X"))
        self.x_btn.pack(side="left", fill="x", expand=True, padx=(0,4))

        self.o_btn = tk.Button(side_frame, text="⭕ O", font=("Segoe UI", 10),
                               bg="#161b22", fg=TEXT2, relief="flat", pady=6,
                               command=lambda: self.set_side("O"))
        self.o_btn.pack(side="left", fill="x", expand=True, padx=(4,0))

        tk.Label(right, text="🔊 Volume", font=("Segoe UI", 11, "bold"),
                 fg=TEXT, bg=PANEL).pack(anchor="w", padx=16, pady=(8,4))

        vol_frame = tk.Frame(right, bg=PANEL)
        vol_frame.pack(fill="x", padx=16, pady=(0,8))

        self.vol_scale = tk.Scale(vol_frame, from_=0, to=100, orient="horizontal",
                                  bg=PANEL, fg=TEXT, troughcolor="#161b22",
                                  highlightthickness=0, showvalue=0,
                                  command=lambda v: self.commentary.set_volume(int(v)/100))
        self.vol_scale.set(100)
        self.vol_scale.pack(side="left", fill="x", expand=True)

        self.vol_label = tk.Label(vol_frame, text="100%", font=("Segoe UI", 9, "bold"),
                                  fg=TEXT, bg=PANEL, width=4)
        self.vol_label.pack(side="right", padx=(8,0))

        tk.Frame(right, bg=BORDER, height=2).pack(fill="x", padx=16, pady=12)

        tk.Label(right, text="💬 LATEST COMMENT", font=("Segoe UI", 11, "bold"),
                 fg=TEXT, bg=PANEL).pack(anchor="w", padx=16, pady=(0,6))

        comment_box = tk.Frame(right, bg="#161b22", highlightbackground=BORDER, highlightthickness=1)
        comment_box.pack(fill="x", padx=16, pady=(0,8))

        self.comment_label = tk.Label(comment_box, text="Ready to coach!", font=("Segoe UI", 11),
                                      fg=ACCENT, bg="#161b22", wraplength=280, justify="left",
                                      padx=10, pady=10)
        self.comment_label.pack(fill="both")

        tk.Frame(right, bg=BORDER, height=2).pack(fill="x", padx=16, pady=12)

        tk.Label(right, text="📜 HISTORY", font=("Segoe UI", 11, "bold"),
                 fg=TEXT, bg=PANEL).pack(anchor="w", padx=16, pady=(0,6))

        self.history_text = tk.Text(right, bg="#161b22", fg=TEXT2, font=("Segoe UI", 9),
                                    relief="flat", wrap="word", state="disabled",
                                    highlightthickness=0, padx=10, pady=8, height=10)
        self.history_text.pack(fill="both", expand=True, padx=16, pady=(0,14))

    def get_roi_text(self):
        if self.roi:
            return f"ROI: {self.roi[2]}x{self.roi[3]} at ({self.roi[0]},{self.roi[1]})"
        return "No ROI set"


    def select_roi(self):
        self.root.iconify()
        time.sleep(0.8 if sys.platform == "darwin" else 0.3)

        try:
            screen = grab_screen()
        except:
            self.root.deiconify()
            return

        self.root.deiconify()

        sel_win = tk.Toplevel(self.root)
        sel_win.title("Select Region - Drag rectangle, press Enter")
        sel_win.attributes("-topmost", True)

        h, w = screen.shape[:2]
        scale = min(1200/w, 800/h, 1.0)
        dw, dh = int(w*scale), int(h*scale)

        rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (dw, dh))
        img = ImageTk.PhotoImage(Image.fromarray(resized))

        canvas = tk.Canvas(sel_win, width=dw, height=dh, cursor="crosshair")
        canvas.pack()
        canvas.create_image(0, 0, anchor="nw", image=img)
        canvas._img = img

        rect_data = {"start": None, "rect_id": None, "roi": None, "scale": scale}

        def on_press(e):
            rect_data["start"] = (e.x, e.y)
            if rect_data["rect_id"]:
                canvas.delete(rect_data["rect_id"])
            rect_data["rect_id"] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#00ff00", width=2)

        def on_drag(e):
            if rect_data["start"] and rect_data["rect_id"]:
                canvas.coords(rect_data["rect_id"], rect_data["start"][0], rect_data["start"][1], e.x, e.y)

        def on_release(e):
            if rect_data["start"]:
                sx, sy = rect_data["start"]
                sc = rect_data["scale"]
                x1 = int(min(sx, e.x) / sc)
                y1 = int(min(sy, e.y) / sc)
                rw = int(abs(e.x - sx) / sc)
                rh = int(abs(e.y - sy) / sc)
                if rw > 10 and rh > 10:
                    rect_data["roi"] = (x1, y1, rw, rh)

        def on_enter(e=None):
            if rect_data["roi"]:
                self.roi = rect_data["roi"]
                save_roi(self.roi)
                self.roi_label.config(text=self.get_roi_text())
            sel_win.destroy()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        sel_win.bind("<Return>", on_enter)
        sel_win.bind("<Escape>", lambda e: sel_win.destroy())

    def set_side(self, side):
        self.player_side = side
        if side == "X":
            self.x_btn.config(bg=ACCENT, fg="white")
            self.o_btn.config(bg="#161b22", fg=TEXT2)
        else:
            self.x_btn.config(bg="#161b22", fg=TEXT2)
            self.o_btn.config(bg=ACCENT, fg="white")

    def toggle_watch(self):
        if self.watching:
            self.watching = False
            self.start_btn.config(text="▶ START COACHING", bg=GREEN)
        else:
            if not self.roi:
                self.show_comment("⚠️ Select a region first!")
                return
            self.watching = True
            self.prev_board = [' '] * 9
            self.start_btn.config(text="⏹ STOP", bg=ACCENT2)
            threading.Thread(target=self.watch_loop, daemon=True).start()

    def watch_loop(self):
        board_buffer = []

        while self.watching:
            try:
                frame = grab_region(self.roi[0], self.roi[1], self.roi[2], self.roi[3])
                self.update_preview(frame)

                board = read_board(frame)

                # only act on a board state if we see it twice in a row (avoids false reads)
                board_buffer.append(board)
                if len(board_buffer) > 3:
                    board_buffer.pop(0)

                if len(board_buffer) >= 2 and board_buffer[-1] == board_buffer[-2]:
                    stable_board = board
                else:
                    time.sleep(0.3)
                    continue

                if stable_board != self.prev_board:
                    print(f"Board: {stable_board}")
                    print(f"Winner: {check_winner(stable_board)}")

                    comment = self.commentary.generate(self.prev_board, stable_board, self.player_side)
                    if comment:
                        self.show_comment(comment)

                    winner = check_winner(stable_board)
                    if winner or stable_board.count(' ') == 0:
                        time.sleep(3)
                        self.prev_board = [' '] * 9
                        board_buffer = []
                    else:
                        self.prev_board = stable_board

                time.sleep(0.5)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

    def update_preview(self, frame):
        try:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw < 10 or ch < 10:
                return

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fh, fw = rgb.shape[:2]

            cell_w = fw // 3
            cell_h = fh // 3

            # draw grid lines over the preview
            for i in range(1, 3):
                cv2.line(rgb, (i * cell_w, 0), (i * cell_w, fh), (0, 255, 0), 2)
                cv2.line(rgb, (0, i * cell_h), (fw, i * cell_h), (0, 255, 0), 2)

            # label detected symbols on the preview
            board = read_board(frame)
            for idx, symbol in enumerate(board):
                if symbol != ' ':
                    row = idx // 3
                    col = idx % 3
                    x = col * cell_w + cell_w // 2
                    y = row * cell_h + cell_h // 2
                    cv2.putText(rgb, symbol, (x-15, y+15), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (255, 0, 0), 3)

            scale = min(cw/fw, ch/fh)
            nw, nh = int(fw*scale), int(fh*scale)
            resized = cv2.resize(rgb, (nw, nh))

            img = ImageTk.PhotoImage(Image.fromarray(resized))
            self.canvas.delete("all")
            self.canvas.create_image(cw//2, ch//2, image=img)
            self.canvas._img = img
        except Exception as e:
            print(f"Preview error: {e}")

    def show_comment(self, text):
        self.comment_label.config(text=text)
        self.history_text.config(state="normal")
        self.history_text.insert("end", f"> {text}\n")
        self.history_text.see("end")
        self.history_text.config(state="disabled")

    def on_close(self):
        self.watching = False
        self.commentary.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = BackseaterGUI()
    app.run()
