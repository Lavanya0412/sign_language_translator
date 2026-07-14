import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
import traceback
from collections import deque

# ------------------- CONFIG -------------------
MODEL_PATH = r"E:\test_data_2.0\sign2text_mobilenetv2_skeleton_model.h5"
MODEL_INPUT_SIZE = 128
OFFSET = 25  # margin for cropping
SMOOTHING_WINDOW = 5  # number of recent predictions to average

# ------------------- LOAD MODEL -------------------
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ------------------- INITIAL SETUP -------------------
capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

# ------------------- MAIN LOOP -------------------
while True:
    try:
        ret, frame = capture.read()
        if not ret:
            print("❌ Could not read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        hands, img = detector.findHands(frame, draw=False, flipType=True)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # --- Create white background ---
            white = np.ones((400, 400, 3), np.uint8) * 255

            # --- Draw skeleton ---
            lmList = hand['lmList']

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (5, 6), (6, 7), (7, 8),
                (9, 10), (10, 11), (11, 12),
                (13, 14), (14, 15), (15, 16),
                (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17), (0, 5), (0, 17)
            ]

            # Offset hand to center it in the 400x400 white canvas
            os_x = ((400 - w) // 2) - 15
            os_y = ((400 - h) // 2) - 15

            for a, b in connections:
                cv2.line(white, (lmList[a][0] + os_x, lmList[a][1] + os_y),
                         (lmList[b][0] + os_x, lmList[b][1] + os_y), (0, 255, 0), 3)

            for i in range(21):
                cv2.circle(white, (lmList[i][0] + os_x, lmList[i][1] + os_y),
                           3, (0, 0, 255), -1)

            # Show the skeleton image
            cv2.imshow("Skeleton", white)

            # --- Preprocess for model ---
            white_resized = cv2.resize(white, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
            white_norm = white_resized.astype('float32') / 255.0
            input_tensor = np.expand_dims(white_norm, axis=0)

            # --- Predict ---
            prediction = model.predict(input_tensor, verbose=0)[0]
            ch1 = np.argmax(prediction)
            prediction_buffer.append(ch1)

            # Smooth predictions
            smoothed = max(set(prediction_buffer), key=prediction_buffer.count)

            # --- Display result ---
            cv2.putText(frame, f"Predicted: {chr(65 + smoothed)}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Live Feed", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == 27:  # ESC key
            break

    except Exception:
        print("⚠️ Error in loop:", traceback.format_exc())

capture.release()
cv2.destroyAllWindows()
