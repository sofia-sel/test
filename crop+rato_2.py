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

# HD Camera Thread and Functionality
def capture_hd_frames():
    global hd_output_frame, hd_lock
    picam2_hd = Picamera2()
    config_hd = picam2_hd.create_preview_configuration(main={"size": (1270, 950)})
    picam2_hd.configure(config_hd)
    picam2_hd.start()

    while True:
        image_hd = picam2_hd.capture_array()
        
        # Define the crop dimensions
        crop_left = 294
        crop_right = image_hd.shape[1] - 176
        crop_top = 198
        crop_bottom = image_hd.shape[0] - 152

        # Crop the image
        cropped_hd_image = image_hd[crop_top:crop_bottom, crop_left:crop_right]

        with hd_lock:
            hd_output_frame = cropped_hd_image.copy()

# Thermal Camera Thread and Functionality
def pull_images():
    global thermal_output_frame, thermal_lock
    thermcam = pithermalcam(output_folder='/home/pi/pithermalcam/saved_snapshots/')
    time.sleep(0.1)

    while True:
        current_frame = thermcam.update_image_frame()
        if current_frame is not None:
            # Resize thermal image to 800x600
            resized_thermal_frame = cv2.resize(current_frame, (800, 600))
            with thermal_lock:
                thermal_output_frame = resized_thermal_frame.copy()

# Flask Routes
@app.route("/")
def index():
    return render_template("index.html")

def generate():
    global hd_output_frame, thermal_output_frame, hd_lock, thermal_lock
    while True:
        with hd_lock:
            hd_frame = hd_output_frame.copy() if hd_output_frame is not None else None
        with thermal_lock:
            thermal_frame = thermal_output_frame.copy() if thermal_output_frame is not None else None
        
        if hd_frame is None or thermal_frame is None:
            continue
        
        # Resize frames if needed to display side by side
        hd_frame = cv2.resize(hd_frame, (320, 240))
        thermal_frame = cv2.resize(thermal_frame, (320, 240))
        
        # Combine frames horizontally
        combined_frame = cv2.hconcat([hd_frame, thermal_frame])

        # Encode combined frame
        (flag, encoded_image) = cv2.imencode(".jpg", combined_frame)
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
