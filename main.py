import cv2
import numpy as np
import base64
import json
from tensorflow.keras.models import load_model
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# --- Configuration ---
MODEL_PATH = 'biscuit_anomaly_detector.keras'
CLASS_LABELS = ['Defect_Color', 'Defect_No', 'Defect_Object', 'Defect_Shape']

app = FastAPI(title="Biscuit Anomaly Detection API")

# --- Load Model ---
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# --- Helper Functions ---
def preprocess_image(image_array):
    # Resize to 150x150
    resized = cv2.resize(image_array, (150, 150))
    # Convert BGR to RGB (OpenCV uses BGR, Keras usually trained on RGB)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize
    normalized = rgb.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=0)

def predict_frame(frame):
    processed = preprocess_image(frame)
    predictions = model.predict(processed, verbose=0)
    
    class_idx = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    label = CLASS_LABELS[class_idx]
    
    return label, confidence

# --- WebSocket Endpoint (The Engine) ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 1. Receive the base64 encoded image from the browser
            data = await websocket.receive_text()
            
            # 2. Decode the base64 string to an image
            # The string looks like "data:image/jpeg;base64,/9j/4AAQTk..."
            header, encoded = data.split(",", 1)
            image_data = base64.b64decode(encoded)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # 3. Predict (We don't need ROI cropping here necessarily, 
            #    because the user can zoom/position the camera themselves)
            label, confidence = predict_frame(frame)
            
            # 4. Send result back to browser
            result = {
                "label": label,
                "confidence": f"{confidence*100:.1f}%",
                "color": "green" if label == "Defect_No" else "red"
            }
            await websocket.send_text(json.dumps(result))
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.close()
        except:
            pass

# --- Frontend (The UI) ---
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Biscuit Anomaly Detector</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: sans-serif; text-align: center; background-color: #111; color: #fff; margin: 0; padding: 0; }
            h1 { font-size: 1.5rem; margin: 10px; }
            
            /* Video Container */
            .video-container { 
                position: relative; 
                width: 100%; 
                max-width: 640px; 
                margin: 0 auto; 
                border: 2px solid #444; 
            }
            video { width: 100%; display: block; }
            
            /* Overlay Box */
            .overlay {
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(0, 0, 0, 0.7);
                padding: 10px;
                border-radius: 5px;
                font-size: 1.2rem;
                font-weight: bold;
                pointer-events: none;
            }
            
            /* Canvas is hidden, used for capturing frames */
            canvas { display: none; }
            
            button {
                padding: 15px 30px;
                font-size: 1rem;
                margin-top: 20px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <h1>🍪 Biscuit Scanner</h1>
        
        <div class="video-container">
            <video id="video" autoplay playsinline muted></video>
            <div id="result" class="overlay" style="color: yellow;">Waiting...</div>
        </div>
        
        <canvas id="canvas"></canvas>
        
        <button onclick="startCamera()">Start Camera</button>
        <p id="status" style="color: #aaa; font-size: 0.9rem;">Click Start to begin</p>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const resultDiv = document.getElementById('result');
            const statusDiv = document.getElementById('status');
            const ctx = canvas.getContext('2d');
            let ws;

            async function startCamera() {
                try {
                    // Access the camera (prefer back camera on phones)
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { facingMode: "environment" } 
                    });
                    video.srcObject = stream;
                    
                    // Wait for video to be ready
                    video.onloadedmetadata = () => {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        startWebSocket();
                    };
                    
                    document.querySelector('button').style.display = 'none';
                    statusDiv.innerText = "Camera Active. Connecting to server...";
                    
                } catch (err) {
                    console.error("Error accessing camera:", err);
                    statusDiv.innerText = "Error: Cannot access camera. Make sure you allowed permissions.";
                }
            }

            function startWebSocket() {
                // Connect to the WebSocket endpoint
                // Use wss:// for https (Render) and ws:// for localhost
                const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
                const wsUrl = `${protocol}://${window.location.host}/ws`;
                
                ws = new WebSocket(wsUrl);

                ws.onopen = () => {
                    statusDiv.innerText = "Connected! Scanning...";
                    // Start sending frames
                    sendFrameLoop();
                };

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    resultDiv.innerText = `${data.label} (${data.confidence})`;
                    resultDiv.style.color = data.color;
                    resultDiv.style.borderColor = data.color;
                };

                ws.onerror = (error) => {
                    console.error("WebSocket Error:", error);
                    statusDiv.innerText = "Connection Error.";
                };
            }

            function sendFrameLoop() {
                if (ws.readyState === WebSocket.OPEN) {
                    // 1. Draw video frame to hidden canvas
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    // 2. Convert canvas to base64 JPEG
                    // (0.7 quality reduces bandwidth usage)
                    const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
                    
                    // 3. Send to server
                    ws.send(dataUrl);
                }
                
                // Send next frame in 100ms (10 FPS) to avoid overloading server
                setTimeout(sendFrameLoop, 100);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

