import cv2
import dlib
import time
import pyttsx3
from scipy.spatial import distance

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Open the webcam (use 0 for default camera)
cap = cv2.VideoCapture(0)

# Check if the camera is opened properly
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Load dlib's face detector
face_detector = dlib.get_frontal_face_detector()

# Load shape predictor model
model_path = r"C:/Users/hp\Downloadsshape/shape_predictor_68_face_landmarks.dat"

try:
    dlib_facelandmark = dlib.shape_predictor(model_path)
except RuntimeError as e:
    print(f"Error: Could not load shape predictor model: {e}")
    exit()

# Function to calculate Eye Aspect Ratio (EAR)
def detect_eye(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

# Drowsiness detection variables
EAR_THRESHOLD = 0.25  # Below this value means eyes are closed
CONSEC_FRAMES = 20  # Number of frames to count drowsiness
counter = 0
alert_active = False
alert_time = 0  # Timestamp when alert starts

# Main loop
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_scale)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)
        left_eye = [] 
        right_eye = [] 

        # Detect right eye (dlib points 42-47)
        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            right_eye.append((x, y))
            next_point = 42 if n == 47 else n + 1
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        # Detect left eye (dlib points 36-41)
        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            left_eye.append((x, y))
            next_point = 36 if n == 41 else n + 1
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

        # Compute EAR for both eyes
        left_EAR = detect_eye(left_eye)
        right_EAR = detect_eye(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Display EAR on frame
        #cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (50, 50),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check if eyes are closed
        if avg_EAR < EAR_THRESHOLD:
            counter += 1
            if counter >= CONSEC_FRAMES and not alert_active:
                alert_active = True
                alert_time = time.time()  # Record time when alert starts
                cv2.putText(frame, "DROWSINESS ALERT!", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                engine.say("Wake up!")
                engine.runAndWait()
        else:
            counter = 0  # Reset counter when eyes are open

        # Display alert for 5 seconds
        if alert_active and (time.time() - alert_time < 5):
            cv2.putText(frame, "DROWSINESS ALERT!", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            alert_active = False  # Reset alert after 5 seconds

    cv2.imshow("Drowsiness Detector", frame)

    # Press ESC (27) to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()