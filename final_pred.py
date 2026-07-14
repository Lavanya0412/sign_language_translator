import cv2
import numpy as np
import time
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import pyttsx3
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import threading
from collections import deque
import traceback

# ------------------- CONFIG -------------------
MODEL_PATH = r"E:\test_data_2.0\sign2text_mobilenetv2_skeleton_model.h5"
IMG_SIZE = 128
SMOOTHING_WINDOW = 5
DISPLAY_DELAY = 3.0  # seconds before appending predicted symbol
CAMERA_W, CAMERA_H = 400, 300

# ------------------- LOAD MODEL -------------------
print(f" Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ------------------- GLOBALS -------------------
current_symbol = ''
sentence = ''
current_word = ''  # buffer for current letters
suggestions = ["", "", "", "", ""]  # now 5 suggestions
classes = [chr(i) for i in range(65, 91)]  # A-Z
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
stable_char = ''
stable_time = 0

# ------------------- FUNCTIONS -------------------
def add_space():
    """Add a visible space in the sentence after current word."""
    global sentence, current_word
    if current_word:
        sentence += current_word + ' '
        current_word = ''
    else:
        if not sentence.endswith(' '):
            sentence += ' '
    update_labels()

def speak_fun():
    global sentence
    if not sentence.strip():
        return

    def worker():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
        engine.say(sentence)
        engine.runAndWait()
        engine.stop()

    threading.Thread(target=worker, daemon=True).start()

def clear_fun():
    """Remove last letter from current_word or sentence."""
    global current_word, sentence
    if current_word:
        current_word = current_word[:-1]
    elif sentence:
        sentence = sentence.rstrip()
        sentence = sentence[:-1]
    update_labels()

def update_labels():
    """Update sentence, current word, and suggestions."""
    display_text = sentence + current_word
    label_sentence.config(text=f"Sentence: {display_text}")
    label_char.config(text=f"Character: {current_symbol}")
    for i in range(5):
        btns[i].config(text=suggestions[i])

def choose_suggestion(idx):
    """Fix the selected suggestion in the sentence."""
    global sentence, current_word
    sugg = suggestions[idx]
    if sugg:
        sentence += sugg
        current_word = ''
        update_labels()

def update_suggestions(char):
    """Update the suggestion buttons dynamically."""
    global suggestions
    suggestions[0] = char
    suggestions[1] = char + 'E'
    suggestions[2] = char + 'S'
    suggestions[3] = char + 'ED'
    suggestions[4] = char + 'N'  # ✅ new suggestion with N

def get_skeleton_image(hand, w_canvas=CAMERA_W, h_canvas=CAMERA_H):
    skeleton_img = np.ones((h_canvas, w_canvas, 3), dtype=np.uint8) * 255
    lmList = hand['lmList']
    if len(lmList) < 21:
        return skeleton_img
    x, y, w, h = hand['bbox']
    os_x = ((w_canvas - w) // 2) - 15
    os_y = ((h_canvas - h) // 2) - 15
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13),
        (13, 17), (0, 5), (0, 17)
    ]
    lmList_adj = [(pt[0] + os_x - x, pt[1] + os_y - y) for pt in lmList]
    for a, b in connections:
        cv2.line(skeleton_img, lmList_adj[a], lmList_adj[b], (0, 255, 0), 3)
    for pt in lmList_adj:
        cv2.circle(skeleton_img, pt, 5, (0, 0, 255), -1)
    return skeleton_img

def video_loop():
    global current_symbol, current_word, suggestions, stable_char, stable_time
    try:
        ret, frame = cap.read()
        if not ret:
            root.after(10, video_loop)
            return

        frame = cv2.flip(frame, 1)
        hands, _ = hd.findHands(frame, draw=False)
        current_symbol = ''
        skeleton_img = np.ones((CAMERA_H, CAMERA_W, 3), dtype=np.uint8) * 255

        if hands:
            hand = hands[0]
            skeleton_img = get_skeleton_image(hand)
            img_resized = cv2.resize(skeleton_img, (IMG_SIZE, IMG_SIZE))
            img_norm = img_resized.astype('float32') / 255.0
            input_tensor = np.expand_dims(img_norm, axis=0)
            pred = model.predict(input_tensor, verbose=0)[0]
            ch1 = np.argmax(pred)
            prediction_buffer.append(ch1)
            smoothed = max(set(prediction_buffer), key=prediction_buffer.count)
            current_symbol = classes[smoothed]
            update_suggestions(current_symbol)

            # Delay logic for stable letter
            if stable_char != current_symbol:
                stable_char = current_symbol
                stable_time = time.time()
            elif time.time() - stable_time >= DISPLAY_DELAY:
                current_word += current_symbol
                stable_char = ''
                stable_time = 0
                prediction_buffer.clear()

        # Update camera panel
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        camera_panel.configure(image=img_pil)
        camera_panel.image = img_pil

        # Update skeleton panel
        skeleton_rgb = cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2RGB)
        skeleton_pil = ImageTk.PhotoImage(Image.fromarray(skeleton_rgb))
        hand_panel.configure(image=skeleton_pil)
        hand_panel.image = skeleton_pil

        update_labels()
        root.after(10, video_loop)
    except Exception:
        print(traceback.format_exc())
        root.after(10, video_loop)

# ------------------- TKINTER SETUP -------------------
root = tk.Tk()
root.geometry("950x650")
root.title("Sign Language To Text")
root.configure(bg="#e6f2ff")

# Webcam & Skeleton panels
camera_panel = Label(root, bg="black", bd=3, relief="ridge")
camera_panel.place(x=20, y=20, width=400, height=300)

hand_panel = Label(root, bg="white", bd=3, relief="ridge")
hand_panel.place(x=450, y=20, width=400, height=300)

# Character label
label_char = Label(root, text="Character: ", font=("Courier New", 20, "bold"), bg="#e6f2ff", anchor="w")
label_char.place(x=20, y=340, width=400, height=40)

# Sentence label
label_sentence = Label(root, text="Sentence: ", font=("Courier New", 20, "bold"), bg="#e6f2ff", anchor="w")
label_sentence.place(x=20, y=390, width=850, height=40)

# Suggestions label
label_suggestions = Label(root, text="Suggestions:", font=("Courier New", 20, "bold"), bg="#e6f2ff", anchor="w")
label_suggestions.place(x=20, y=440, width=180, height=40)

# Suggestion buttons (5 total now)
btns = []
for i in range(5):
    b = Button(root, text="", font=("Arial", 16), bg="#4da6ff", fg="white", activebackground="#3399ff",
               command=lambda i=i: choose_suggestion(i))
    b.place(x=200 + i * 140, y=440, width=130, height=40)
    btns.append(b)

# Space, Speak & Clear buttons
btn_space = Button(root, text="Space", font=("Arial", 16), bg="#3399ff", fg="white", activebackground="#2673cc", command=add_space)
btn_space.place(x=20, y=500, width=120, height=40)

btn_speak = Button(root, text="Speak", font=("Arial", 16), bg="#4da6ff", fg="white", activebackground="#3399ff", command=speak_fun)
btn_speak.place(x=160, y=500, width=120, height=40)

btn_clear = Button(root, text="Clear", font=("Arial", 16), bg="#2673cc", fg="white", activebackground="#1a4d99", command=clear_fun)
btn_clear.place(x=300, y=500, width=120, height=40)

# ------------------- HAND DETECTOR & CAMERA -------------------
hd = HandDetector(maxHands=1, detectionCon=0.7)
cap = cv2.VideoCapture(0)

# ------------------- CLEAN EXIT -------------------
def on_closing():
    try:
        cap.release()
    finally:
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
video_loop()
root.mainloop()
