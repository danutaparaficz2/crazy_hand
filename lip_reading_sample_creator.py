import cv2
import mediapipe as mp
import numpy as np
import csv

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

# Define classes and samples per class
classes = ["ok", "thumbs_up", "one", "two"]
samples_per_class = 10
current_class_index = 0
current_sample_count = 0

current_label = classes[current_class_index]

csv_file = open("lip_data.csv", "a", newline="")
csv_writer = csv.writer(csv_file)

cap = cv2.VideoCapture(0)
print(f"Starting data collection for class: {current_label}")
print(f"Say '{current_label}' and press 's' to save sample. Need {samples_per_class} samples for this class.")
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get lip landmarks
            sample = []
            for idx in INNER_LIPS:
                x = face_landmarks.landmark[idx].x
                y = face_landmarks.landmark[idx].y
                sample.extend([x, y])
            # Show current class and progress on frame
            cv2.putText(frame, f"Class: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Samples: {current_sample_count}/{samples_per_class}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Class {current_class_index + 1}/{len(classes)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_LIPS)

    cv2.imshow('Lip Data Collection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save sample with label
        csv_writer.writerow([current_label] + sample)
        current_sample_count += 1
        print(f"Saved sample {current_sample_count}/{samples_per_class} for class: {current_label}")
        
        # Check if we've collected enough samples for current class
        if current_sample_count >= samples_per_class:
            current_class_index += 1
            current_sample_count = 0
            
            # Check if we've finished all classes
            if current_class_index >= len(classes):
                print("Data collection completed for all classes!")
                break
            else:
                current_label = classes[current_class_index]
                print(f"\nMoving to next class: {current_label}")
                print(f"Say '{current_label}' and press 's' to save sample. Need {samples_per_class} samples for this class.")
                
    elif key == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()