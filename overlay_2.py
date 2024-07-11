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
show_thermal = True

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
    global hd_output_frame, thermal_output_frame, hd_lock, thermal_lock, show_thermal
    while True:
        with hd_lock:
            hd_frame = hd_output_frame.copy() if hd_output_frame is not None else None
        with thermal_lock:
            thermal_frame = thermal_output_frame.copy() if thermal_output_frame is not None else None
        
        if hd_frame is None:
            continue

        if show_thermal and thermal_frame is not None:
            # Resize thermal frame to match HD frame size
            thermal_frame = cv2.resize(thermal_frame, (hd_frame.shape[1], hd_frame.shape[0]))

            # Convert thermal frame to RGB and add alpha channel
            thermal_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2BGRA)
            alpha = 0.5  # Set transparency level (0.0 to 1.0)
            thermal_frame[:, :, 3] = (thermal_frame[:, :, 3] * alpha).astype(thermal_frame.dtype)

            # Overlay thermal frame onto HD frame
            overlay_frame = cv2.addWeighted(hd_frame, 1.0, thermal_frame[:, :, :3], alpha, 0)
        else:
            overlay_frame = hd_frame

        # Encode combined frame
        (flag, encoded_image) = cv2.imencode(".jpg", overlay_frame)
        if not flag:
            continue
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

def keyboard_listener():
    global show_thermal
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'):
            show_thermal = not show_thermal

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

    # Start keyboard listener thread
    keyboard_thread = threading.Thread(target=keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()

    # Run Flask app
    app.run(host='0.0.0.0', port=8010, debug=True, threaded=True, use_reloader=False)
