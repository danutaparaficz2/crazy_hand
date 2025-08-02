# Recognize spoken words from mouth/lip movement using MediaPipe Face Mesh

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Indices for outer and inner lips in MediaPipe Face Mesh
OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

# Example: Define simple rules for a few words (open/closed mouth, wide/narrow)
def recognize_word(lip_landmarks):
    # Calculate mouth opening (vertical distance between upper and lower lip)
    top_lip = lip_landmarks[13]  # upper inner lip
    bottom_lip = lip_landmarks[14]  # lower inner lip
    mouth_opening = abs(top_lip[1] - bottom_lip[1])
    # Calculate mouth width (horizontal distance between corners)
    left_corner = lip_landmarks[78]
    right_corner = lip_landmarks[308]
    mouth_width = abs(left_corner[0] - right_corner[0])
    # Simple rules (you need to calibrate these for your camera and face)
    if mouth_opening > 0.06 and mouth_width > 0.12:
        return "Hello"
    elif mouth_opening < 0.03 and mouth_width > 0.12:
        return "Smile"
    elif mouth_opening > 0.06 and mouth_width < 0.10:
        return "Oh"
    else:
        return "Unknown"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get lip landmarks
            lip_landmarks = {}
            for idx in INNER_LIPS:
                x = face_landmarks.landmark[idx].x
                y = face_landmarks.landmark[idx].y
                lip_landmarks[idx] = (x, y)
            word = recognize_word(lip_landmarks)
            print("Recognized word:", word)
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_LIPS)

    cv2.imshow('Lip Reading', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()