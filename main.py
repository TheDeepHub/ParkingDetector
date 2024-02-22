import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Path to your .h5 file
mask_image_path = r"C:\Users\jorge\Parking detector\ParkingLotDetectorAndCounter\mask\mask_1920_1080.png" # Replace with the path to your mask image
video_path = r"C:\Users\jorge\Parking detector\ParkingLotDetectorAndCounter\data\parking_1920_1080.mp4" # Replace with the path to your original video
model_path = r"C:\Users\jorge\Parking detector\ParkingLotDetectorAndCounter\model\weights\path_to_my_model.h5"


# Load your trained model
model = tf.keras.models.load_model(model_path)  # Update the path to where your model is saved

def preprocess_for_prediction(roi, target_size=(68, 29)):
    # Resize the ROI to the target size expected by the model
    roi_resized = cv2.resize(roi, target_size)
    # Normalize pixel values if your model expects normalization
    roi_normalized = roi_resized / 255.0
    # Expand dimensions to add the batch size
    roi_expanded = np.expand_dims(roi_normalized, axis=0)
    return roi_expanded

def draw_bounding_boxes_and_predict(frame, mask, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    segmented = cv2.bitwise_and(gray, gray, mask=mask)
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Extract ROI using the bounding box coordinates
        roi = frame[y:y+h, x:x+w]
        # Preprocess the ROI for model prediction
        roi_preprocessed = preprocess_for_prediction(roi)
        # Predict the occupancy using your model
        prediction = model.predict(roi_preprocessed)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Assuming binary classification: 0 for empty, 1 for occupied
        
        # Draw bounding box in green if empty, red if occupied
        color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    return frame

# Function to draw bounding boxes on the frame based on the provided mask
def draw_bounding_boxes(frame, mask):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply the mask to get the segmented image
    segmented = cv2.bitwise_and(gray, gray, mask=mask)
    # Find contours in the segmented image
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw bounding boxes around each contour on the original frame
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

# Load the mask image (the same mask used earlier)
mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

# Capture video from a file or camera (replace 'your_video.mp4' with your video file or use 0 for webcam)
video_capture = cv2.VideoCapture(video_path)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab a frame")
        break

    # Use the updated function that includes model predictions
    frame_with_predictions = draw_bounding_boxes_and_predict(frame, mask, model)

    cv2.imshow('Video with Parking Slot Occupancy', frame_with_predictions)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the capture when everything is done
video_capture.release()
cv2.destroyAllWindows()
