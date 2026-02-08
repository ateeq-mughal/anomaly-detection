import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# 1. Configuration
MODEL_PATH = 'biscuit_anomaly_detector.keras'
# Replace this with the actual path to your test image
# IMAGE_PATH = '0211.jpg' 
# IMAGE_PATH = '0162.jpg' 
# IMAGE_PATH = '0150.jpg' 
IMAGE_PATH = '0001.jpg' 

# Define your class labels (must match the training order)
# Usually alphabetical: ['Defect_Color', 'Defect_No', 'Defect_Object', 'Defect_Shape']
CLASS_LABELS = ['Defect_Color', 'Defect_No', 'Defect_Object', 'Defect_Shape']

def predict_single_image(model_path, image_path):
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # 2. Preprocessing
    # Load the image ensuring it is resized to 150x150 automatically
    # load_img loads in RGB format by default (unlike OpenCV which is BGR)
    img = load_img(image_path, target_size=(150, 150))
    
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    
    # Normalize pixel values to [0, 1] just like in training
    img_array = img_array / 255.0
    
    # Add a batch dimension: (150, 150, 3) -> (1, 150, 150, 3)
    # The model expects a batch of images, even if it's just one.
    img_batch = np.expand_dims(img_array, axis=0)

    # 3. Prediction
    print("Running prediction...")
    predictions = model.predict(img_batch)
    
    # Get the index of the highest probability
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    result_label = CLASS_LABELS[predicted_class_idx]
    
    # 4. Display Results
    print(f"---------------------------------")
    print(f"Prediction: {result_label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"---------------------------------")

    # Optional: Show the image using Matplotlib
    plt.imshow(img)
    plt.title(f"Pred: {result_label} ({confidence*100:.1f}%)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Check if the file exists before running
    import os
    if os.path.exists(IMAGE_PATH):
        predict_single_image(MODEL_PATH, IMAGE_PATH)
    else:
        print(f"Error: The image file '{IMAGE_PATH}' was not found.")
        print("Please edit the IMAGE_PATH variable in the script.")