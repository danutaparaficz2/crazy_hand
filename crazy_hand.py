
# Real-time hand gesture recognition using MediaPipe, outputting scaled fingertip distances

import cv2
import mediapipe as mp
import time
import math
import serial  # Add this import

# Set up serial port (adjust 'COM3' or '/dev/ttyUSB0' to your port)
ser = serial.Serial('/dev/tty.usbserial-0001', 115200, timeout=1)  # For Linux/Mac
# ser = serial.Serial('COM3', 9600, timeout=1)        # For Windows

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
PALM_POINTS = [0, 1, 2, 5, 9, 13, 17]  # Common palm landmarks

def get_palm_center(hand_landmarks):
    # Average the coordinates of palm landmarks
    xs = [hand_landmarks.landmark[i].x for i in PALM_POINTS]
    ys = [hand_landmarks.landmark[i].y for i in PALM_POINTS]
    zs = [hand_landmarks.landmark[i].z for i in PALM_POINTS]
    return sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)

cap = cv2.VideoCapture(0)
last_print_time = time.time()

# Set your frame size (adjust if needed)
FRAME_SIZE = 480

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            palm_x, palm_y, palm_z = get_palm_center(hand_landmarks)
            distances = []
            for tip_id in FINGER_TIPS:
                tip = hand_landmarks.landmark[tip_id]
                # Convert normalized coordinates to pixel space
                tip_px = tip.x * FRAME_SIZE
                tip_py = tip.y * FRAME_SIZE
                palm_px = palm_x * FRAME_SIZE
                palm_py = palm_y * FRAME_SIZE
                # Euclidean distance in pixel space
                dist = math.sqrt(
                    (tip_px - palm_px) ** 2 +
                    (tip_py - palm_py) ** 2
                )
                # Scale to range 40-140 (simple linear scaling)
                scaled_dist = 40 + (dist / FRAME_SIZE) * 200
                distances.append(scaled_dist)
            avg_distance = sum(distances) / len(distances)
            avg_distance_int = int(round(avg_distance))

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Print average scaled distance every 1 second
            current_time = time.time()
            if current_time - last_print_time >= 0.02:
                print(f"Average scaled fingertip distance to palm center: {avg_distance:.2f}")
                # Send the value through USART
                ser.write(f"{avg_distance_int}\n".encode())
                last_print_time = current_time

    cv2.imshow('MediaPipe Hands - Finger Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()
