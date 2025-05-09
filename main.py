import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Response, Form, Query
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import tempfile
import time
from collections import deque
import os
import json
from typing import Dict, List, Tuple, Optional
import uuid
from pathlib import Path

# Create app and mount static files
app = FastAPI()
os.makedirs("static", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Dictionary to store tracking information
tracks = {}
# Dictionary to store metrics
metrics = {
    "fps": 0,
    "object_count": 0,
    "class_counts": {},
    "avg_confidence": 0,
    "processing_time": 0,
    "detection_time": 0,
    "tracking_time": 0,
    "total_frames": 0,
    "current_frame": 0,
    "distance_traveled": {},  # Track distance traveled by each object
    "speed_estimates": {},    # Estimated speed of each object
    "total_objects_tracked": 0
}

# Maximum trail length
MAX_TRAIL_LENGTH = 25

# Generate unique colors for each track
def get_color(idx):
    """Generate a unique color for a track ID"""
    np.random.seed(idx)
    return tuple(map(int, np.random.randint(0, 255, 3)))

@app.get("/")
def main():
    """Serve the main HTML page"""
    html_content = open("index.html", "r").read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Handle video upload"""
    # Clear previous tracking data
    global tracks, metrics
    tracks = {}
    
    # Reset metrics
    metrics = {
        "fps": 0,
        "object_count": 0,
        "class_counts": {},
        "avg_confidence": 0,
        "processing_time": 0,
        "detection_time": 0,
        "tracking_time": 0,
        "total_frames": 0,
        "current_frame": 0,
        "distance_traveled": {},
        "speed_estimates": {},
        "total_objects_tracked": 0
    }
    
    # Save uploaded file to temp location
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp.write(await file.read())
    temp.close()
    return {"video_path": temp.name}

@app.post("/process_local_video")
async def process_local_video(video_path: str = Form(...)):
    """Process a video file from the local filesystem"""
    # Clear previous tracking data
    global tracks, metrics
    tracks = {}
    
    # Reset metrics
    metrics = {
        "fps": 0,
        "object_count": 0,
        "class_counts": {},
        "avg_confidence": 0,
        "processing_time": 0,
        "detection_time": 0,
        "tracking_time": 0,
        "total_frames": 0,
        "current_frame": 0,
        "distance_traveled": {},
        "speed_estimates": {},
        "total_objects_tracked": 0
    }
    
    # Check if file exists
    if not os.path.exists(video_path):
        return {"error": "Video file not found"}
    
    return {"video_path": video_path}

@app.get("/metrics")
def get_metrics():
    """Return current metrics as JSON"""
    return metrics

@app.get("/save_video")
def save_processed_video(video_path: str):
    """Save the processed video to a file"""
    output_path = f"outputs/processed_{uuid.uuid4().hex}.mp4"
    
    # Create a video writer
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO inference
        results = model(frame)[0]
        
        # Process frame with tracking
        processed_frame = process_frame(frame, results, frame_count, save_mode=True)
        
        # Write frame to output video
        out.write(processed_frame)
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    return {"output_path": output_path}

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def process_frame(frame, results, frame_count, save_mode=False):
    """Process a frame with detection and tracking"""
    global tracks, metrics
    
    # Start timing for metrics
    start_time = time.time()
    
    # Update total frames
    if frame_count == 0:
        # Get total frames for progress tracking
        metrics["total_frames"] = 0  # Will be updated if we're processing a video file
    
    metrics["current_frame"] = frame_count
    
    # Detection metrics
    detection_start = time.time()
    detection_results = results
    detection_end = time.time()
    detection_time = detection_end - detection_start
    
    # Tracking metrics
    tracking_start = time.time()
    
    # Get current detections
    current_detections = []
    class_counts = {}
    total_confidence = 0
    
    # Process each detection
    for box in detection_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]
        
        # Update class counts
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
        
        # Calculate centroid
        centroid_x = (x1 + x2) // 2
        centroid_y = (y1 + y2) // 2
        
        # Add to current detections
        current_detections.append({
            "bbox": (x1, y1, x2, y2),
            "centroid": (centroid_x, centroid_y),
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": conf
        })
        
        total_confidence += conf
    
    # Simple tracking based on IoU (Intersection over Union)
    # In a real application, you might want to use a more sophisticated tracker like SORT or ByteTrack
    if frame_count == 0:
        # First frame, initialize tracks
        for i, det in enumerate(current_detections):
            track_id = i + 1  # Start IDs from 1
            tracks[track_id] = {
                "bbox": det["bbox"],
                "centroid": det["centroid"],
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "trail": deque([det["centroid"]], maxlen=MAX_TRAIL_LENGTH),
                "color": get_color(track_id),
                "last_seen": frame_count
            }
    else:
        # Process subsequent frames
        # Match detections to existing tracks based on IoU
        matched_tracks = set()
        matched_detections = set()
        
        for track_id, track in tracks.items():
            if frame_count - track["last_seen"] > 30:  # Remove tracks not seen for 30 frames
                continue
                
            track_bbox = track["bbox"]
            best_iou = 0.3  # IoU threshold
            best_detection = -1
            
            for i, det in enumerate(current_detections):
                if i in matched_detections:
                    continue
                    
                det_bbox = det["bbox"]
                iou = calculate_iou(track_bbox, det_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_detection = i
            
            if best_detection >= 0:
                # Update track with new detection
                det = current_detections[best_detection]
                tracks[track_id]["bbox"] = det["bbox"]
                tracks[track_id]["centroid"] = det["centroid"]
                tracks[track_id]["class_id"] = det["class_id"]
                tracks[track_id]["class_name"] = det["class_name"]
                tracks[track_id]["confidence"] = det["confidence"]
                tracks[track_id]["trail"].append(det["centroid"])
                tracks[track_id]["last_seen"] = frame_count
                
                matched_tracks.add(track_id)
                matched_detections.add(best_detection)
        
        # Add new tracks for unmatched detections
        next_track_id = max(tracks.keys()) + 1 if tracks else 1
        for i, det in enumerate(current_detections):
            if i not in matched_detections:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = {
                    "bbox": det["bbox"],
                    "centroid": det["centroid"],
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "trail": deque([det["centroid"]], maxlen=MAX_TRAIL_LENGTH),
                    "color": get_color(track_id),
                    "last_seen": frame_count
                }
    
    # Draw bounding boxes, trails, and IDs
    active_tracks = 0
    for track_id, track in tracks.items():
        if frame_count - track["last_seen"] > 30:  # Skip tracks not seen recently
            continue
            
        active_tracks += 1
        x1, y1, x2, y2 = track["bbox"]
        color = track["color"]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with track ID
        label = f"{track['class_name']} #{track_id} {track['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw centroid
        centroid = track["centroid"]
        cv2.circle(frame, centroid, 4, color, -1)
        
        # Draw trail with gradient color
        trail = list(track["trail"])
        if len(trail) > 1:
            for i in range(1, len(trail)):
                # Calculate alpha for gradient effect (more recent = more opaque)
                alpha = 0.3 + 0.7 * (i / len(trail))
                # Create a gradient color (fade from color to white)
                trail_color = (
                    int(color[0] * alpha + 255 * (1 - alpha)),
                    int(color[1] * alpha + 255 * (1 - alpha)),
                    int(color[2] * alpha + 255 * (1 - alpha))
                )
                # Draw line segment
                cv2.line(frame, trail[i-1], trail[i], trail_color, 2)
                
        # Calculate and display distance traveled if we have enough trail points
        if track_id in metrics["distance_traveled"] and not save_mode:
            distance = metrics["distance_traveled"][track_id]
            cv2.putText(frame, f"Dist: {distance:.1f}px", 
                       (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Display speed if available
            if track_id in metrics["speed_estimates"]:
                speed = metrics["speed_estimates"][track_id]
                cv2.putText(frame, f"Speed: {speed:.1f}px/f", 
                           (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    tracking_end = time.time()
    tracking_time = tracking_end - tracking_start
    
    # Update metrics
    avg_confidence = total_confidence / len(current_detections) if current_detections else 0
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate total objects tracked
    metrics["total_objects_tracked"] = max(tracks.keys()) if tracks else 0
    
    metrics.update({
        "object_count": active_tracks,
        "class_counts": class_counts,
        "avg_confidence": avg_confidence,
        "processing_time": processing_time,
        "detection_time": detection_time,
        "tracking_time": tracking_time
    })
    
    # Draw metrics on frame if not in save mode
    if not save_mode:
        cv2.putText(frame, f"FPS: {metrics['fps']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Objects: {active_tracks}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Conf: {avg_confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add total objects tracked
        cv2.putText(frame, f"Total tracked: {metrics['total_objects_tracked']}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        y_offset = 180
        for class_name, count in class_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25
    
    return frame

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0

@app.get("/video_feed")
def video_feed(video_path: str):
    """Stream processed video with object tracking"""
    def generate():
        global metrics
        
        cap = cv2.VideoCapture(video_path)
        prev_time = time.time()
        fps_deque = deque(maxlen=30)
        frame_count = 0
        
        # Get total frames for progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        metrics["total_frames"] = total_frames
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Start timing for FPS calculation
            start = time.time()
            
            # Run YOLO inference
            results = model(frame)[0]
            
            # Process frame with tracking
            processed_frame = process_frame(frame, results, frame_count)
            
            # FPS calculation
            end = time.time()
            fps = 1 / (end - start)
            fps_deque.append(fps)
            avg_fps = np.mean(fps_deque)
            metrics["fps"] = avg_fps
            
            # Encode frame to JPEG
            _, jpeg = cv2.imencode('.jpg', processed_frame)
            
            # Yield frame for streaming
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            frame_count += 1
            
        cap.release()
        
    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/available_videos")
def get_available_videos():
    """Get a list of available video files in the current directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    videos = []
    
    # Check current directory
    for file in os.listdir('.'):
        if os.path.isfile(file) and any(file.lower().endswith(ext) for ext in video_extensions):
            videos.append({
                "path": file,
                "name": os.path.basename(file),
                "size": os.path.getsize(file) / (1024 * 1024)  # Size in MB
            })
    
    return {"videos": videos}

@app.get("/outputs")
def get_outputs():
    """Get a list of processed video outputs"""
    outputs = []
    
    if os.path.exists("outputs"):
        for file in os.listdir("outputs"):
            if file.endswith(".mp4"):
                outputs.append({
                    "path": f"outputs/{file}",
                    "name": file,
                    "size": os.path.getsize(f"outputs/{file}") / (1024 * 1024)  # Size in MB
                })
    
    return {"outputs": outputs}

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
