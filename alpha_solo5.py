from flask import Flask, Response
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

def capture_hd_frames():
    global hd_output_frame, hd_lock
    # Initialize the HD camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)}))
    picam2.start()
    
    while True:
        frame = picam2.capture_array()
        
        # Crop the frame
        frame = frame[CROP_TOP:-CROP_BOTTOM, CROP_LEFT:-CROP_RIGHT]
        
        with hd_lock:
            hd_output_frame = frame.copy()
        
        time.sleep(0.033)  # Adjust based on your camera's frame rate

def pull_images():
    global thermal_output_frame, thermal_lock
    i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ

    while True:
        try:
            frame = [0] * 768
            mlx.getFrame(frame)
            frame_2d = np.reshape(frame, (24, 32))
            thermal_output_frame = np.uint8((frame_2d - frame_2d.min()) / (frame_2d.ptp() / 255.0))
            thermal_output_frame = ndimage.zoom(thermal_output_frame, (20, 20), order=1)  # Scale to match HD size

            # Convert to 3 channel BGR
            thermal_output_frame = cv2.cvtColor(thermal_output_frame, cv2.COLOR_GRAY2BGR)
            
            with thermal_lock:
                thermal_output_frame = thermal_output_frame.copy()
            
            time.sleep(0.125)  # 8 Hz frame rate
        except Exception as e:
            print(f"Thermal camera error: {e}")

def generate():
    global hd_output_frame, thermal_output_frame, hd_lock, thermal_lock
    alpha = 0.5  # Transparency factor for blending

    # Define temperature range for the thermal camera
    min_temp = 20.0  # Minimum temperature in °C
    max_temp = 100.0  # Maximum temperature in °C
    threshold_temp = 40.0  # Temperature threshold in °C

    # Calculate pixel value threshold
    threshold_pixel_value = int((threshold_temp - min_temp) * 255 / (max_temp - min_temp))

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

        print(f"HD Frame shape: {hd_frame.shape}")
        print(f"Thermal Frame shape: {thermal_frame.shape}")

        # Convert thermal frame to grayscale to process temperature values
        thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)

        # Create masks for pixels above and below the threshold
        above_threshold_mask = thermal_gray >= threshold_pixel_value
        below_threshold_mask = ~above_threshold_mask

        # Create 3-channel masks for blending
        above_threshold_mask_3c = np.stack([above_threshold_mask] * 3, axis=-1)
        below_threshold_mask_3c = np.stack([below_threshold_mask] * 3, axis=-1)

        # Apply alpha blending to create translucent effect for pixels above the threshold
        blended_above_threshold = cv2.addWeighted(thermal_frame, alpha, hd_frame, 1 - alpha, 0)

        # Combine the frames using the masks
        blended_frame = np.where(above_threshold_mask_3c, blended_above_threshold, hd_frame)

        # Encode the blended frame
        (flag, encoded_image) = cv2.imencode(".jpg", blended_frame)
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
