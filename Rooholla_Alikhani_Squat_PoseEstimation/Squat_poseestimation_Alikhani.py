import cv2 # version 3
import mediapipe as mp
import numpy as np
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def get_landmarks(results):
    landmarks = results.pose_landmarks.landmark
    return {
        "L_hip": [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        "L_knee": [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        "L_ankle": [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
        "R_hip": [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
        "R_knee": [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
        "R_ankle": [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y],
        "R_shoulder": [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        "L_shoulder": [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
    }

def squat_counter():
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None
    knee_angles = deque(maxlen=5)

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌errro to read Frame")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                points = get_landmarks(results)

                L_knee_angle = calculate_angle(points["L_hip"], points["L_knee"], points["L_ankle"])
                R_knee_angle = calculate_angle(points["R_hip"], points["R_knee"], points["R_ankle"])
                knee_angle = (L_knee_angle + R_knee_angle) / 2

                back_angle_R = calculate_angle(points["R_shoulder"], points["R_hip"], points["R_knee"])
                back_angle_L = calculate_angle(points["L_shoulder"], points["L_hip"], points["L_knee"])
                back_angle = (back_angle_R + back_angle_L) / 2

                knee_angles.append(knee_angle)
                smooth_knee_angle = np.mean(knee_angles)

                if stage is None and smooth_knee_angle > 150:
                    stage = "up"

                error_flag = False
                if back_angle < 60: 
                    error_flag = True

                if not error_flag:  
                    if smooth_knee_angle < 110 and stage == "up":
                        stage = "down"
                    if smooth_knee_angle > 150 and stage == "down":
                        stage = "up"
                        counter += 1

                cv2.putText(image, f'Knee: {int(smooth_knee_angle)}',
                            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(image, f'Back: {int(back_angle)}',
                            (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(image, f'Squats: {counter}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, f'Stage: {stage}', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                if error_flag:
                    cv2.putText(image, "❌Wrong move!!! The back is bent too much.",
                                (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            except Exception as e:
                print("⚠️ error:", e)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Squat Counter', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    squat_counter()
