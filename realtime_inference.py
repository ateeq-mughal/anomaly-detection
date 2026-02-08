import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Load your trained model
# Ensure the file name matches exactly what you saved in the notebook
MODEL_PATH = 'biscuit_anomaly_detector.keras'
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# 2. Define Class Labels
# These correspond to the folders/classes in your dataset.
# Keras usually sorts them alphabetically. Based on your notebook metadata:
CLASS_LABELS = ['Defect_Color', 'Defect_No', 'Defect_Object', 'Defect_Shape']

def preprocess_frame(frame):
    """
    Preprocesses the video frame to match the training conditions.
    """
    # Convert BGR (OpenCV standard) to RGB (Model standard)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to the model's expected input size (150x150)
    resized_frame = cv2.resize(rgb_frame, (150, 150))
    
    # Convert to array and normalize pixel values to [0, 1]
    img_array = resized_frame.astype('float32') / 255.0
    
    # Expand dimensions to create a batch of size 1: (1, 150, 150, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch

# 3. Start Video Capture (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Starting video feed. Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame for the model
    processed_input = preprocess_frame(frame)

    # Make prediction
    predictions = model.predict(processed_input, verbose=0)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions)
    
    predicted_label = CLASS_LABELS[predicted_class_idx]

    # Visualization logic
    # Green text for 'Defect_No', Red for actual defects
    color = (0, 255, 0) if predicted_label == 'Defect_No' else (0, 0, 255)
    
    # Prepare display text
    text = f"{predicted_label} ({confidence_score*100:.1f}%)"
    
    # Draw a rectangle and label on the ORIGINAL frame (not the resized one)
    # Here we just put text in the corner, but you could add object detection bounding boxes later
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Display the result
    cv2.imshow('Biscuit Anomaly Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()