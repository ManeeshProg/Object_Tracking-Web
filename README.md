<<<<<<< HEAD
# Object_Tracking-Web-
A real-time object detection and tracking web application using YOLOv8, OpenCV, and Flask. The app streams video from a webcam or file, performs object tracking using YOLOv8, and displays the output via a Bootstrap-styled frontend.
=======
# YOLOv8 Object Tracking Web Application

A web-based application for real-time object tracking using YOLOv8 and FastAPI. This project provides a user-friendly interface for uploading and processing videos with state-of-the-art object detection and tracking capabilities.

![Object Tracking Demo](https://github.com/ManeeshProg/Object_Tracking-Web-/raw/main/static/demo.png)

## Features

- **Real-time Object Detection & Tracking**: Utilizes YOLOv8 for accurate object detection and implements tracking algorithms to follow objects across video frames
- **Interactive Web Interface**: User-friendly dashboard for uploading, processing, and analyzing videos
- **Detailed Metrics**: Real-time display of FPS, object counts, confidence scores, and processing times
- **Visual Tracking**: Displays bounding boxes, object trails, and unique IDs for each tracked object
- **Video Processing**: Upload your own videos or use local video files for processing
- **Save Processed Videos**: Option to save the processed videos with tracking visualization

## Technologies Used

- **Backend**: FastAPI, Python
- **Object Detection**: YOLOv8 (Ultralytics)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Video Processing**: OpenCV
- **Data Visualization**: Custom JavaScript for real-time metrics display

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ManeeshProg/Object_Tracking-Web-.git
   cd Object_Tracking-Web-
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model (if not already included):
   ```bash
   # The model will be downloaded automatically when first running the application
   # Or you can manually place yolov8n.pt in the project root directory
   ```

## Usage

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8000
   ```

3. Use the web interface to:
   - Upload a video file for processing
   - Select a local video file from your system
   - View real-time object tracking with metrics
   - Save the processed video with tracking visualization

## Project Structure

```
Object_Tracking-Web-/
├── main.py                # FastAPI application and backend logic
├── index.html             # Main web interface
├── static/                # Static files (CSS, images)
│   └── styles.css         # Custom CSS styles
├── outputs/               # Directory for processed video outputs
├── yolov8n.pt             # YOLOv8 nano model weights
└── requirements.txt       # Python dependencies
```

## Customization

- **Model Selection**: You can replace `yolov8n.pt` with other YOLOv8 models (s, m, l, x) for different accuracy/speed trade-offs
- **Tracking Parameters**: Adjust tracking parameters in the code to optimize for your specific use case
- **UI Customization**: Modify the HTML and CSS files to customize the user interface

## Requirements

- Python 3.8+
- FastAPI
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Modern web browser with JavaScript enabled

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection model
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [OpenCV](https://opencv.org/) for video processing capabilities
- [Bootstrap](https://getbootstrap.com/) for frontend components
>>>>>>> 680be56 (Intern First_commit)
