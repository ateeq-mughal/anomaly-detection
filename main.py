import cv2
import numpy as np
import base64
import json
import asyncio
from tensorflow.keras.models import load_model
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# --- Configuration ---
MODEL_PATH = 'biscuit_anomaly_detector_20e.keras'
CLASS_LABELS = ['Defect_Color', 'Defect_No', 'Defect_Object', 'Defect_Shape']

app = FastAPI(title="Biscuit Anomaly Detection API")

# --- Load Model ---
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# --- Helper Functions ---
def preprocess_image(image_array):
    # Resize to 150x150 (Model Requirement)
    # Since we are receiving a square crop, this resize maintains aspect ratio perfectly.
    resized = cv2.resize(image_array, (150, 150))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=0)

def predict_sync(frame):
    processed = preprocess_image(frame)
    predictions = model.predict(processed, verbose=0)[0]
    
    class_idx = np.argmax(predictions)
    confidence = float(predictions[class_idx])
    label = CLASS_LABELS[class_idx]
    
    scores = {CLASS_LABELS[i]: float(predictions[i]) for i in range(len(CLASS_LABELS))}
    return label, confidence, scores

# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 1. Receive data (This is now the CROPPED image from frontend)
            data = await websocket.receive_text()
            
            # 2. Decode
            header, encoded = data.split(",", 1)
            image_data = base64.b64decode(encoded)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # 3. Predict
            label, confidence, scores = await asyncio.to_thread(predict_sync, frame)
            
            # 4. Response
            color = "#28a745" if label == "Defect_No" else "#dc3545"
            result = {
                "label": label,
                "confidence": f"{confidence*100:.1f}%",
                "scores": scores,
                "color": color
            }
            await websocket.send_text(json.dumps(result))
            
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.close()
        except:
            pass

# --- Frontend ---
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Biscuit Inspector ROI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: 'Segoe UI', sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; text-align: center; }
            
            /* Video Container */
            .video-wrapper { 
                position: relative; 
                max-width: 640px; 
                margin: 0 auto; 
                border: 2px solid #333; 
                border-radius: 8px; 
                overflow: hidden; 
                background: #000; 
            }
            video { width: 100%; display: block; }

            /* THE BOUNDING BOX OVERLAY */
            .roi-box {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 250px;
                height: 250px;
                border: 3px solid #00FF00;
                box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
                z-index: 10;
                pointer-events: none; /* Let clicks pass through */
            }
            .roi-label {
                position: absolute;
                top: -25px;
                left: 50%;
                transform: translateX(-50%);
                background: #00FF00;
                color: #000;
                padding: 2px 8px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 4px;
                white-space: nowrap;
            }

            /* Timers and Badges */
            .timer-badge { position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.8); padding: 5px 10px; border-radius: 4px; font-weight: bold; z-index: 11; }

            /* Results Area */
            .analysis-container { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; margin-top: 20px; }
            .box-panel { flex: 1; min-width: 250px; max-width: 400px; background: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #333; }
            
            /* Captured Image (Now shows cropped) */
            .captured-img { width: 150px; height: 150px; object-fit: cover; border: 2px solid #555; border-radius: 4px; display: block; margin: 0 auto; }
            
            /* Graphs */
            .bar-row { display: flex; align-items: center; margin-bottom: 8px; font-size: 0.85rem; }
            .bar-text { width: 90px; text-align: right; padding-right: 10px; color: #bbb; }
            .bar-track { flex-grow: 1; background: #333; height: 8px; border-radius: 4px; overflow: hidden; }
            .bar-fill { height: 100%; width: 0%; transition: width 0.5s ease; }
            .bar-percent { width: 45px; text-align: right; padding-left: 10px; font-family: monospace; }
            
            #mainPrediction { font-size: 1.5rem; font-weight: bold; margin: 10px 0; display: block; }
            button { padding: 12px 25px; font-size: 1rem; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 4px; margin-top: 10px; }
        </style>
    </head>
    <body>

        <h2>🍪 Biscuit Inspector</h2>

        <div class="video-wrapper">
            <span id="timerDisplay" class="timer-badge">Next Scan: 5s</span>
            <video id="video" autoplay playsinline muted></video>
            
            <div class="roi-box">
                <span class="roi-label">SCAN AREA</span>
            </div>
        </div>

        <button id="startBtn" onclick="startSystem()">Start Camera</button>
        <p id="status" style="color: #aaa;">Place biscuit inside the green box</p>

        <div class="analysis-container" id="resultsArea" style="opacity: 0.3;">
            
            <div class="box-panel">
                <div style="margin-bottom:10px; color:#aaa;">Analyzed Region (Crop)</div>
                <img id="capturedFrame" class="captured-img" src="" alt="Waiting..." />
            </div>

            <div class="box-panel">
                <div style="margin-bottom:10px; color:#aaa;">Prediction</div>
                <span id="mainPrediction">--</span>
                <div id="graphContainer"></div>
            </div>
        </div>

        <canvas id="canvas" style="display:none;"></canvas>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const capturedFrame = document.getElementById('capturedFrame');
            const mainPrediction = document.getElementById('mainPrediction');
            const graphContainer = document.getElementById('graphContainer');
            const timerDisplay = document.getElementById('timerDisplay');
            const ctx = canvas.getContext('2d');
            
            let ws;
            let countdown = 5;
            // ROI Size in pixels (must match the CSS width/height of .roi-box)
            const ROI_SIZE = 250; 

            // Initialize Bars
            const classes = ['Defect_Color', 'Defect_No', 'Defect_Object', 'Defect_Shape'];
            classes.forEach(cls => {
                graphContainer.innerHTML += `
                    <div class="bar-row">
                        <div class="bar-text">${cls}</div>
                        <div class="bar-track"><div class="bar-fill" id="fill-${cls}" style="background: #555;"></div></div>
                        <div class="bar-percent" id="val-${cls}">0%</div>
                    </div>
                `;
            });

            async function startSystem() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { facingMode: "environment" } 
                    });
                    video.srcObject = stream;
                    document.getElementById('startBtn').style.display = 'none';
                    document.getElementById('status').innerText = "System Active";
                    
                    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
                    ws = new WebSocket(`${protocol}://${window.location.host}/ws`);
                    
                    ws.onopen = () => {
                        document.getElementById('resultsArea').style.opacity = "1";
                        startTimer();
                    };

                    ws.onmessage = (event) => updateUI(JSON.parse(event.data));

                } catch (err) { alert("Error: " + err); }
            }

            function startTimer() {
                setInterval(() => {
                    countdown--;
                    timerDisplay.innerText = `Next Scan: ${countdown}s`;
                    if (countdown <= 0) {
                        captureAndSend();
                        countdown = 5;
                    }
                }, 1000);
            }

            function captureAndSend() {
                if (ws.readyState !== WebSocket.OPEN) return;
                
                timerDisplay.innerText = "Scanning...";

                // 1. Calculate Crop Coordinates (Center of video)
                const videoW = video.videoWidth;
                const videoH = video.videoHeight;
                
                const startX = (videoW - ROI_SIZE) / 2;
                const startY = (videoH - ROI_SIZE) / 2;

                // 2. Set Canvas to ROI size (250x250)
                canvas.width = ROI_SIZE;
                canvas.height = ROI_SIZE;

                // 3. Draw ONLY the center region to the canvas
                // ctx.drawImage(source, sx, sy, sw, sh, dx, dy, dw, dh)
                ctx.drawImage(video, startX, startY, ROI_SIZE, ROI_SIZE, 0, 0, ROI_SIZE, ROI_SIZE);

                // 4. Convert to Data URL
                const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

                // 5. Update UI immediately
                capturedFrame.src = dataUrl;

                // 6. Send to Backend
                ws.send(dataUrl);
            }

            function updateUI(data) {
                mainPrediction.innerText = `${data.label} (${data.confidence})`;
                mainPrediction.style.color = data.color;

                for (const [cls, score] of Object.entries(data.scores)) {
                    const pct = (score * 100).toFixed(1);
                    document.getElementById(`fill-${cls}`).style.width = `${pct}%`;
                    document.getElementById(`val-${cls}`).innerText = `${pct}%`;
                    document.getElementById(`fill-${cls}`).style.background = (cls === data.label) ? data.color : '#555';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)