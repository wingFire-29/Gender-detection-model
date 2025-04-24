import cv2
import os
import argparse

def highlightFace(net, frame, conf_threshold=0.7):
    """Detects faces in the frame and highlights them with rectangles."""
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frameOpencvDnn.shape[:2]

    # Prepare input blob for face detection
    blob = cv2.dnn.blobFromImage(
        frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False
    )

    # Set input to the network
    net.setInput(blob)
    detections = net.forward()  # Forward pass to get detections

    faceBoxes = []  # List to store face bounding boxes

    # Iterate over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Confidence score

        if confidence > conf_threshold:
            # Compute bounding box coordinates
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            # Validate bounding box coordinates
            x1 = max(0, min(x1, frameWidth - 1))
            y1 = max(0, min(y1, frameHeight - 1))
            x2 = max(0, min(x2, frameWidth - 1))
            y2 = max(0, min(y2, frameHeight - 1))

            if x2 <= x1 or y2 <= y1:
                print(f"Invalid bounding box: [{x1}, {y1}, {x2}, {y2}]. Skipping...")
                continue

            faceBoxes.append([x1, y1, x2, y2])  # Append to list

            # Draw rectangle around the face
            cv2.rectangle(
                frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2, 8
            )

    return frameOpencvDnn, faceBoxes  # Return the annotated frame and bounding boxes

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help="Path to the dataset folder", required=True)
args = parser.parse_args()

# Paths to the pre-trained model files for face and gender detection
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Labels and constants for gender detection
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

# Load the pre-trained models using OpenCV's DNN module
try:
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Initialize counters for correct and total predictions
correct_predictions = 0
total_predictions = 0

# Path to your dataset folder with subfolders named 'Male' and 'Female'
dataset_path = args.dataset

# Iterate over each gender label
for gender_label in genderList:
    gender_folder = os.path.join(dataset_path, gender_label)  # e.g., 'dataset/Male'

    if not os.path.exists(gender_folder):
        print(f"Folder {gender_folder} does not exist. Skipping...")
        continue

    # Iterate over each image in the gender folder
    for filename in os.listdir(gender_folder):
        # Construct full image path
        img_path = os.path.join(gender_folder, filename)

        # Read the image
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Could not read {img_path}. Skipping...")
            continue  # Skip if image is not readable
        else:
            print(f"Successfully read {img_path}")

        # Detect faces in the image
        resultImg, faceBoxes = highlightFace(faceNet, frame)

        if not faceBoxes:
            print(f"No face detected in {filename}. Skipping...")
            continue  # Skip if no face is detected

        # For simplicity, assume the first detected face is the subject
        faceBox = faceBoxes[0]

        # Extract the face region with padding
        padding = 20
        start_x = max(0, faceBox[1] - padding)
        end_x = min(faceBox[3] + padding, frame.shape[0] - 1)
        start_y = max(0, faceBox[0] - padding)
        end_y = min(faceBox[2] + padding, frame.shape[1] - 1)

        face = frame[start_x:end_x, start_y:end_y]

        # Debugging: Check the size of the extracted face
        if face.size == 0:
            print(f"Empty face region extracted for {filename}. Skipping...")
            continue
        else:
            print(f"Extracted face region for {filename}: {face.shape}")

        # (Optional) Display the extracted face for visual verification
        # cv2.imshow("Face", face)
        # cv2.waitKey(1)  # Display for 1 ms

        # Prepare input blob for gender prediction
        try:
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
            )
        except cv2.error as e:
            print(f"Error creating blob for {filename}: {e}. Skipping...")
            continue

        genderNet.setInput(blob)  # Set the input to the gender detection model

        # Predict gender
        genderPreds = genderNet.forward()
        predicted_gender_index = genderPreds[0].argmax()
        predicted_gender = genderList[predicted_gender_index]
        gender_prob = genderPreds[0][predicted_gender_index] * 100  # Convert to percentage

        # Update counters
        total_predictions += 1
        if predicted_gender == gender_label:
            correct_predictions += 1

        # Print prediction result
        print(f"Image: {filename}, Actual: {gender_label}, Predicted: {predicted_gender} ({gender_prob:.2f}%)")

# Calculate and print accuracy
if total_predictions > 0:
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\nTotal Predictions: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("No predictions were made.")



# to check the predictions and accuracy of the model run this code in the terminal
# python testing1.py --dataset "D:\college\Projects\Tech-She\age and gender detection\dataset100"
# python testing1.py --dataset "D:\college\Projects\Tech-She\age and gender detection\dataset1000"
