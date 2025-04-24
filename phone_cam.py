import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the pre-trained gender detection model (adjust path to your model)
model = load_model('path_to_your_model.h5')

# Initialize the face detector (Haar Cascade for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# URL of the phone's camera stream (replace with your actual IP and port)
phone_cam_url = "http://192.168.1.4:8080/video"  # Replace with your IP and port
cap = cv2.VideoCapture(phone_cam_url)

while True:
    ret, frame = cap.read()  # Read the frame
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        # Resize and preprocess the face for prediction
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0  # Normalize the face image
        
        # Predict the gender
        prediction = model.predict(face)
        gender = "Male" if prediction[0][0] > 0.5 else "Female"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Put the gender label and confidence on the frame
        text = f"{gender} ({confidence*100:.2f}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with face and gender information
    cv2.imshow("Gender Detection - Live Feed", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  # Close all OpenCV windows
