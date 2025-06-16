import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
model = load_model('proposed_model.h5')

# Load Haar cascade for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw green rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Region of interest (ROI) for face
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face
        eyes = eye_cascade.detectMultiScale(face_roi_gray)

        prediction_text = "Not Drowsy"  # Default
        drowsy = False

        for (ex, ey, ew, eh) in eyes:
            # Inside your for (ex, ey, ew, eh) in eyes:
            eye = face_roi_color[ey:ey + eh, ex:ex + ew]  # use color eye region
            eye = cv2.resize(eye, (227, 227))  # resize to model input size
            eye = eye.astype("float") / 255.0
            eye = np.expand_dims(eye, axis=0)

            prediction = model.predict(eye, verbose=0)


            if prediction[0][0] < 0.5:
                prediction_text = "Drowsy"
                drowsy = True
                break

        # Display prediction above the face
        cv2.putText(frame, prediction_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255) if drowsy else (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
