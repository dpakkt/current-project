from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import tempfile
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import base64
from io import BytesIO
from datetime import datetime
import threading
import time
import atexit
import logging

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global dictionary to store processing status
processing_status = {}
temp_files = []  # Track temporary files for cleanup

def cleanup_temp_files():
    """Clean up temporary files on exit"""
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up {file_path}: {e}")

# Register cleanup function
atexit.register(cleanup_temp_files)

class FootTrafficAnalyzer:
    def __init__(self):
        self.confidence_threshold = 0.3
        self.scale_factor = 1.1
        self.min_neighbors = 3
        self.cascade = None
        
    def load_detector(self):
        """Load person detection model"""
        try:
            # Try to load full body cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            if os.path.exists(cascade_path):
                self.cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("Loaded full body cascade")
            else:
                # Fallback to upper body cascade
                cascade_path = cv2.data.haarcascades + 'haarcascade_upperbody.xml'
                self.cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("Loaded upper body cascade as fallback")
                
            if self.cascade.empty():
                raise Exception("Failed to load cascade classifier")
                
        except Exception as e:
            logger.error(f"Error loading detector: {e}")
            raise e
    
    def detect_people(self, frame):
        """Detect people in frame using Haar cascade"""
        if self.cascade is None:
            self.load_detector()
            
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect people
        detections = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30),
            maxSize=(300, 300)
        )
        
        people_positions = []
        for (x, y, w, h) in detections:
            # Use bottom center of detection as person position (foot location)
            person_x = x + w // 2
            person_y = y + h  # Bottom of the detection box
            people_positions.append((person_x, person_y))
        
        return people_positions
    
    def process_video(self, video_path, job_id):
        """Process video and generate heatmap data"""
        try:
            processing_status[job_id] = {"status": "processing", "progress": 0}
            logger.info(f"Starting video processing for job {job_id}")
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                processing_status[job_id] = {"status": "error", "message": "Could not open video file"}
                return
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
            
            # Create heatmap accumulator
            heatmap = np.zeros((height, width), dtype=np.float64)
            
            frame_count = 0
            processed_count = 0
            skip_frames = max(1, fps // 2)  # Process every 0.5 seconds
            
            # Load detector
            self.load_detector()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % skip_frames == 0:
                    try:
                        # Detect people in frame
                        people_positions = self.detect_people(frame)
                        processed_count += 1
                        
                        # Add to heatmap
                        for x, y in people_positions:
                            if 0 <= x < width and 0 <= y < height:
                                # Add gaussian blob around person position
                                self.add_gaussian_blob(heatmap, x, y, sigma=25, amplitude=1.0)
                        
                        logger.info(f"Frame {frame_count}: detected {len(people_positions)} people")
                        
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_count}: {e}")
                
                # Update progress
                progress = int((frame_count / total_frames) * 90)  # Reserve 10% for final processing
                processing_status[job_id]["progress"] = progress
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Processed {processed_count} frames out of {frame_count} total frames")
            
            # Final processing
            processing_status[job_id]["progress"] = 95
            
            # Normalize heatmap
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            # Apply gaussian smoothing
            heatmap_smooth = gaussian_filter(heatmap, sigma=3)
            
            # Generate heatmap image
            heatmap_image = self.generate_heatmap_image(heatmap_smooth, width, height)
            
            # Calculate statistics
            total_detections = int(np.sum(heatmap > 0))
            max_intensity = float(heatmap_smooth.max())
            
            # Save results
            result_data = {
                "heatmap_array": heatmap_smooth.tolist(),
                "video_dimensions": {"width": width, "height": height},
                "total_detections": total_detections,
                "processed_frames": processed_count,
                "total_frames": frame_count,
                "max_intensity": max_intensity,
                "heatmap_image": heatmap_image,
                "processing_time": datetime.now().isoformat()
            }
            
            processing_status[job_id] = {
                "status": "completed",
                "progress": 100,
                "result": result_data
            }
            
            logger.info(f"Completed processing for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error in process_video: {str(e)}")
            processing_status[job_id] = {
                "status": "error",
                "message": f"Processing failed: {str(e)}"
            }
        finally:
            # Clean up video file
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    if video_path in temp_files:
                        temp_files.remove(video_path)
            except Exception as e:
                logger.error(f"Error cleaning up video file: {e}")
    
    def add_gaussian_blob(self, heatmap, x, y, sigma=25, amplitude=1.0):
        """Add a gaussian blob to the heatmap at specified location"""
        height, width = heatmap.shape
        
        # Create coordinate grids (limit range for performance)
        radius = int(3 * sigma)
        x_start = max(0, x - radius)
        x_end = min(width, x + radius + 1)
        y_start = max(0, y - radius)
        y_end = min(height, y + radius + 1)
        
        for yi in range(y_start, y_end):
            for xi in range(x_start, x_end):
                distance_sq = (xi - x)**2 + (yi - y)**2
                value = amplitude * np.exp(-distance_sq / (2 * sigma**2))
                heatmap[yi, xi] += value
    
    def generate_heatmap_image(self, heatmap, width, height):
        """Generate heatmap visualization as base64 image"""
        try:
            plt.figure(figsize=(12, 8))
            plt.imshow(heatmap, cmap='hot', interpolation='bilinear', alpha=0.8, aspect='auto')
            plt.colorbar(label='Foot Traffic Intensity', shrink=0.8)
            plt.title('Store Foot Traffic Heatmap', fontsize=16, pad=20)
            plt.xlabel('X Position (pixels)')
            plt.ylabel('Y Position (pixels)')
            
            # Add some statistics as text
            max_val = heatmap.max()
            plt.text(0.02, 0.98, f'Max Intensity: {max_val:.3f}', 
                    transform=plt.gca().transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor='white')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()  # Important: close the figure to free memory
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error generating heatmap image: {e}")
            return None

# Initialize analyzer
analyzer = FootTrafficAnalyzer()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Store Foot Traffic Heatmap API",
        "version": "1.0",
        "description": "Upload a video of your store to generate a foot traffic heatmap",
        "endpoints": {
            "upload": "POST /upload - Upload video file (MP4, AVI, MOV, MKV)",
            "status": "GET /status/<job_id> - Check processing status",
            "result": "GET /result/<job_id> - Get complete heatmap results",
            "heatmap": "GET /heatmap/<job_id> - Get heatmap image only",
            "health": "GET /health - Health check"
        },
        "supported_formats": ["MP4", "AVI", "MOV", "MKV"],
        "max_file_size": "100MB",
        "processing_time": "Typically 2-5 minutes for a 10-30 minute video"
    })

@app.route('/upload', methods=['POST'])
def upload_video():
    """Upload video file for processing"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided. Use 'video' as the form field name."}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check file extension
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({
                "error": "Unsupported file format", 
                "supported_formats": allowed_extensions
            }), 400
        
        # Generate unique job ID
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, f"{job_id}_{filename}")
        file.save(video_path)
        temp_files.append(video_path)  # Track for cleanup
        
        logger.info(f"Uploaded video for job {job_id}: {filename}")
        
        # Start processing in background
        thread = threading.Thread(target=analyzer.process_video, args=(video_path, job_id))
        thread.daemon = True  # Make thread daemon so it doesn't prevent shutdown
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "message": "Video uploaded successfully. Processing started.",
            "filename": filename,
            "status_url": f"/status/{job_id}",
            "result_url": f"/result/{job_id}",
            "heatmap_url": f"/heatmap/{job_id}",
            "estimated_time": "2-5 minutes"
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get processing status for a job"""
    if job_id not in processing_status:
        return jsonify({"error": "Job not found", "job_id": job_id}), 404
    
    status = processing_status[job_id].copy()
    
    # Add helpful messages based on status
    if status["status"] == "processing":
        if status["progress"] < 50:
            status["message"] = "Analyzing video frames..."
        elif status["progress"] < 90:
            status["message"] = "Detecting people and building heatmap..."
        else:
            status["message"] = "Finalizing heatmap generation..."
    
    return jsonify(status)

@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    """Get complete heatmap results for completed job"""
    if job_id not in processing_status:
        return jsonify({"error": "Job not found", "job_id": job_id}), 404
    
    status = processing_status[job_id]
    
    if status["status"] == "processing":
        return jsonify({
            "error": "Job still processing", 
            "progress": status.get("progress", 0),
            "message": "Please wait for processing to complete"
        }), 202
    elif status["status"] == "error":
        return jsonify({"error": status.get("message", "Processing failed")}), 500
    
    return jsonify(status["result"])

@app.route('/heatmap/<job_id>', methods=['GET'])
def get_heatmap_image(job_id):
    """Get heatmap image for completed job"""
    if job_id not in processing_status:
        return jsonify({"error": "Job not found", "job_id": job_id}), 404
    
    status = processing_status[job_id]
    
    if status["status"] == "processing":
        return jsonify({
            "error": "Job still processing", 
            "progress": status.get("progress", 0)
        }), 202
    elif status["status"] == "error":
        return jsonify({"error": status.get("message", "Processing failed")}), 500
    
    # Return base64 image
    image_data = status["result"]["heatmap_image"]
    if image_data:
        return jsonify({
            "heatmap_image": f"data:image/png;base64,{image_data}",
            "job_id": job_id,
            "dimensions": status["result"]["video_dimensions"]
        })
    else:
        return jsonify({"error": "Heatmap image not available"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len([k for k, v in processing_status.items() if v["status"] == "processing"])
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 100MB."}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Create temp directory if it doesn't exist
    import tempfile
    os.makedirs(tempfile.gettempdir(), exist_ok=True)
    
    print("=" * 50)
    print("Store Foot Traffic Heatmap API")
    print("=" * 50)
    print("Available endpoints:")
    print("  POST /upload - Upload video file")
    print("  GET /status/<job_id> - Check processing status")
    print("  GET /result/<job_id> - Get complete results")
    print("  GET /heatmap/<job_id> - Get heatmap image")
    print("  GET /health - Health check")
    print("=" * 50)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
