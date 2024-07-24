from flask import Flask, Response, render_template
import threading
import time
import io
import cv2
from picamera2 import Picamera2
try:  # If called as an imported module
    from pithermalcam import pithermalcam
except:  # If run directly
    from pi_therm_cam import pithermalcam

app = Flask(__name__)

# Global variables for HD and Thermal camera frames
hd_output_frame = None
thermal_output_frame = None
hd_lock = threading.Lock()
thermal_lock = threading.Lock()

# Define the desired offset dimensions for thermal camera positioning
OFFSET_TOP = 198
OFFSET_BOTTOM = 152
OFFSET_LEFT = 176
OFFSET_RIGHT = 294

# HD Camera Thread and Functionality
def capture_hd_frames():
    global hd_output_frame, hd_lock
    picam2_hd = Picamera2()
    config_hd = picam2_hd.create_preview_configuration(main={"size": (1270, 950)})  # Set capture size to 1270x950
    picam2_hd.configure(config_hd)
    picam2_hd.start()

    while True:
        image_hd = picam2_hd.capture_array()
        
        # Convert to RGB for compatibility with OpenCV
        image_hd = cv2.cvtColor(image_hd, cv2.COLOR_BGR2RGB)
        
        with hd_lock:
            hd_output_frame = image_hd.copy()

# Thermal Camera Thread and Functionality
def pull_images():
    global thermal_output_frame, thermal_lock
    thermcam = pithermalcam(output_folder='/home/pi/pithermalcam/saved_snapshots/')
    time.sleep(0.1)

    while True:
        current_frame = thermcam.update_image_frame()
        if current_frame is not None:
            with thermal_lock:
                thermal_output_frame = current_frame.copy()

# Flask Routes
@app.route("/")
def index():
    return render_template("index.html")

def generate():
    global hd_output_frame, thermal_output_frame, hd_lock, thermal_lock
    alpha = 0.5  # Transparency factor

    while True:
        with hd_lock:
            hd_frame = hd_output_frame.copy() if hd_output_frame is not None else None
        with thermal_lock:
            thermal_frame = thermal_output_frame.copy() if thermal_output_frame is not None else None
        
        if hd_frame is None or thermal_frame is None:
            continue

        # Define position to overlay thermal frame on the HD frame
        x_offset = OFFSET_LEFT  # Use defined offset
        y_offset = OFFSET_TOP   # Use defined offset

        # Create an ROI on the HD frame where the thermal frame will be placed
        y1, y2 = y_offset, y_offset + thermal_frame.shape[0]
        x1, x2 = x_offset, x_offset + thermal_frame.shape[1]

        # Ensure the overlay does not go out of bounds
        y2 = min(y2, hd_frame.shape[0])
        x2 = min(x2, hd_frame.shape[1])
        resized_thermal_frame = thermal_frame[:(y2 - y1), :(x2 - x1)]

        # Blend the frames
        roi = hd_frame[y1:y2, x1:x2]
        blended_roi = cv2.addWeighted(roi, 1 - alpha, resized_thermal_frame, alpha, 0)
        hd_frame[y1:y2, x1:x2] = blended_roi

        # Encode combined frame
        (flag, encoded_image) = cv2.imencode(".jpg", hd_frame)
        if not flag:
            continue
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    # Start HD camera thread
    hd_thread = threading.Thread(target=capture_hd_frames)
    hd_thread.daemon = True
    hd_thread.start()

    # Start Thermal camera thread
    thermal_thread = threading.Thread(target=pull_images)
    thermal_thread.daemon = True
    thermal_thread.start()

    # Run Flask app
    app.run(host='0.0.0.0', port=8010, debug=True, threaded=True, use_reloader=False)
