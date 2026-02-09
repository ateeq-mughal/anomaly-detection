import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import io

# --- Configuration ---
MODEL_PATH = 'biscuit_anomaly_detector.keras'
# Ensure these match your training folder names alphabetically
CLASS_LABELS = ['Defect_Color', 'Defect_No', 'Defect_Object', 'Defect_Shape']

app = FastAPI(title="Biscuit Anomaly Detection API")

# --- Load Model Once at Startup ---
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# --- Helper Functions ---

def preprocess_image(image_array):
    # Resize to 150x150
    resized = cv2.resize(image_array, (150, 150))
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize
    normalized = rgb.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=0)

def predict_roi(roi_image):
    processed = preprocess_image(roi_image)
    predictions = model.predict(processed, verbose=0)
    
    class_idx = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    label = CLASS_LABELS[class_idx]
    
    return label, confidence

# --- Video Streaming Generator ---

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # --- 1. Define Region of Interest (ROI) ---
            # We want a square box in the center of the screen
            height, width, _ = frame.shape
            box_size = 250  # Size of the scanning box in pixels
            
            x1 = int((width - box_size) / 2)
            y1 = int((height - box_size) / 2)
            x2 = x1 + box_size
            y2 = y1 + box_size
            
            # Crop the frame to just the box area
            roi_frame = frame[y1:y2, x1:x2]
            
            # --- 2. Run Inference ONLY on the ROI ---
            # (We check if roi_frame is valid to avoid crashes on edge cases)
            if roi_frame.size != 0:
                label, confidence = predict_roi(roi_frame)
                
                # --- 3. Visualization ---
                # Determine color (Green for good, Red for defect)
                color = (0, 255, 0) if label == 'Defect_No' else (0, 0, 255)
                
                # Draw the ROI box on the main frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Display Label above the box
                text = f"{label} ({confidence*100:.0f}%)"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, color, 2, cv2.LINE_AA)
            
            # Encode Frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def home():
    """Simple HTML Dashboard to view video and upload images."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Biscuit Anomaly Detector</title>
        <style>
            body { font-family: sans-serif; text-align: center; padding: 20px; background-color: #000; color: #fff; }
            .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
            .box { border: 1px solid #ddd; padding: 20px; border-radius: 8px; width: 45%; background-color: #222; }
            img { max-width: 100%; border-radius: 4px; }
            pre { background: #f4f4f4; padding: 10px; text-align: left; }
        </style>
    </head>
    <body>
        <h1 style="color: #fff;">Intelligent Consumer Technologies</h1>
        <h2 style="color: #fff;">🍪 Biscuit Anomaly Detection</h2>
        
        <div class="container">
            <div class="box">
                <h2>Real-Time Live Feed</h2>
                <img src="/video_feed" width="640" height="480" />
            </div>

            <div class="box">
                <h2>Single Image Check</h2>
                <input type="file" id="fileInput" accept="image/*">
                <button onclick="uploadImage()">Analyze Image</button>
                <br><br>
                <div id="result"></div>
            </div>
        </div>

        <script>
            async function uploadImage() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                if (!file) return alert("Please select an image first.");

                const formData = new FormData();
                formData.append("file", file);

                document.getElementById('result').innerHTML = "Analyzing...";

                const response = await fetch('/predict/image', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                document.getElementById('result').innerHTML = 
                    `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video_feed")
async def video_feed():
    """Stream video with detection boxes."""
    return StreamingResponse(generate_frames(), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/predict/image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    """API Endpoint to predict a single uploaded image."""
    # Read image file
    contents = await file.read()
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return JSONResponse(content={"error": "Invalid image file"}, status_code=400)

    # Predict
    label, confidence = predict_frame(image)
    
    return {
        "filename": file.filename,
        "prediction": label,
        "confidence": f"{confidence*100:.2f}%",
        "is_defective": label != "Defect_No"
    }

