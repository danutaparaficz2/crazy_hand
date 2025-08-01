import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained classifier and label names
clf = joblib.load("lip_classifier.pkl")
label_names = ["ok", "one", "thumbs_up", "two"]  # Match your trained model classes

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

def calculate_lip_distance(lip_landmarks):
    """Calculate the average distance between upper and lower lip landmarks"""
    upper_lip = [13, 14, 15, 16, 17, 18]  # Upper lip landmarks
    lower_lip = [78, 95, 88, 178, 87, 14]  # Lower lip landmarks
    
    distances = []
    for i in range(min(len(upper_lip), len(lower_lip))):
        if upper_lip[i] in lip_landmarks and lower_lip[i] in lip_landmarks:
            upper_point = lip_landmarks[upper_lip[i]]
            lower_point = lip_landmarks[lower_lip[i]]
            dist = ((upper_point[0] - lower_point[0])**2 + (upper_point[1] - lower_point[1])**2)**0.5
            distances.append(dist)
    
    return sum(distances) / len(distances) if distances else 0

def is_speaking(lip_landmarks, threshold=0.015):
    """Detect if person is speaking based on lip movement"""
    lip_distance = calculate_lip_distance(lip_landmarks)
    return lip_distance > threshold

def predict_word(lip_landmarks):
    # First check if person is speaking
    if not is_speaking(lip_landmarks):
        return "silence"
    
    sample = []
    for idx in INNER_LIPS:
        sample.extend(lip_landmarks[idx])
    sample = np.array(sample).reshape(1, -1)
    pred = clf.predict(sample)[0]
    return label_names[pred]

cap = cv2.VideoCapture(0)
frame_count = 0
print_interval = 30  # Print prediction every 30 frames (about 1 second)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lip_landmarks = {}
            for idx in INNER_LIPS:
                x = face_landmarks.landmark[idx].x
                y = face_landmarks.landmark[idx].y
                lip_landmarks[idx] = (x, y)
            word = predict_word(lip_landmarks)
            
            # Only print prediction every print_interval frames
            if frame_count % print_interval == 0:
                print("Predicted word:", word)
            
            # Display prediction on the frame
            cv2.putText(frame, f"Word: {word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_LIPS)
    
    frame_count += 1

    cv2.imshow('Lip Reading', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()