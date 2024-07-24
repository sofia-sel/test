from flask import Flask, Response, render_template
import threading
import time
import cv2
from picamera2 import Picamera2
import numpy as np
from scipy import ndimage
import adafruit_mlx90640
import busio
import board

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

# Define the desired crop dimensions for HD camera
CROP_TOP = 198
CROP_BOTTOM = 152
CROP_LEFT = 176
CROP_RIGHT = 294

# HD Camera Thread and Functionality
def capture_hd_frames():
    global hd_output_frame, hd_lock
    picam2_hd = Picamera2()
    config_hd = picam2_hd.create_preview_configuration(main={"size": (1270, 950)})  # Set capture size to 1270x950
    picam2_hd.configure(config_hd)
    picam2_hd.start()

    while True:
        image_hd = picam2_hd.capture_array()
        
        # Crop the image
        cropped_hd_image = image_hd[CROP_TOP:-CROP_BOTTOM, CROP_LEFT:-CROP_RIGHT]
        
        # Convert to RGB for compatibility with OpenCV
        cropped_hd_image = cv2.cvtColor(cropped_hd_image, cv2.COLOR_BGR2RGB)
        
        with hd_lock:
            hd_output_frame = cropped_hd_image.copy()
        time.sleep(0.03)  # Reduce CPU usage

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
        time.sleep(0.03)  # Reduce CPU usage

# Flask Routes
@app.route("/")
def index():
    return render_template("index.html")

def generate():
    global hd_output_frame, thermal_output_frame, hd_lock, thermal_lock
    alpha = 0.5  # Transparency factor

    # Create an instance of the thermal camera class
    thermcam = pithermalcam(output_folder='/home/pi/pithermalcam/saved_snapshots/')
    
    while True:
        with hd_lock:
            hd_frame = hd_output_frame.copy() if hd_output_frame is not None else None
        with thermal_lock:
            thermal_frame = thermal_output_frame.copy() if thermal_output_frame is not None else None

        if hd_frame is None or thermal_frame is None:
            continue

        # Resize thermal frame to match the HD frame size
        hd_frame = cv2.resize(hd_frame, (640, 480))
        thermal_frame = cv2.resize(thermal_frame, (640, 480))

        # Get temperature data and resize to match the image dimensions
        thermcam.update_image_frame()
        temperature_data = thermcam._raw_image.copy() if thermcam._raw_image is not None else None
        if temperature_data is None:
            continue

        # Reshape and resize temperature data to match the image size
        temperature_data = np.resize(temperature_data, (24, 32))
        temperature_data = ndimage.zoom(temperature_data, (20, 20))  # Upscale to match image size
        
        # Create an empty mask with the same size as the HD frame
        mask = np.ones_like(temperature_data, dtype=np.uint8)
        mask[temperature_data < 40] = 0  # Set mask to 0 (black) where temperature is below 40°C
        
        # Resize the mask to match HD frame size
        mask = cv2.resize(mask, (640, 480))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Blend the frames using the mask
        blended_frame = cv2.addWeighted(hd_frame, 1 - alpha, thermal_frame, alpha, 0)
        result_frame = cv2.bitwise_and(blended_frame, blended_frame, mask=mask[:, :, 0])

        # Encode combined frame
        (flag, encoded_image) = cv2.imencode(".jpg", result_frame)
        if not flag:
            continue
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
        time.sleep(0.03)  # Reduce CPU usage


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
