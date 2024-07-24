from flask import Flask, Response
import threading
import time
import cv2
import board
import busio
import adafruit_mlx90640

app = Flask(__name__)

# Global variables to hold the video frames and locks
hd_output_frame = None
thermal_output_frame = None
hd_lock = threading.Lock()
thermal_lock = threading.Lock()

# Function to capture frames from HD camera
def capture_hd_camera():
    global hd_output_frame, hd_lock
    cap = cv2.VideoCapture(0)  # Change the camera index if necessary
    if not cap.isOpened():
        print("Error: Could not open HD camera.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with hd_lock:
            hd_output_frame = frame.copy()
        time.sleep(0.03)  # Reduce CPU usage

# Function to capture frames from thermal camera
def capture_thermal_camera():
    global thermal_output_frame, thermal_lock
    i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ

    while True:
        try:
            frame = [0] * 768
            mlx.getFrame(frame)
            # Process the frame here
            # Assuming frame processing results in an image `thermal_image`
            thermal_image = process_frame(frame)  # Define this function as needed
            with thermal_lock:
                thermal_output_frame = thermal_image.copy()
        except Exception as e:
            print(f"Error capturing thermal frame: {e}")
        time.sleep(0.5)  # Adjust sleep time as necessary

# Placeholder function to process the frame from MLX90640
def process_frame(frame):
    # Convert the frame to an image, apply colormap, etc.
    # This is just an example, adjust as needed for your application
    import numpy as np
    import cmapy

    # Assuming the frame is a 1D array with 768 values (24x32)
    thermal_image = np.array(frame).reshape(24, 32)
    thermal_image = cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX)
    thermal_image = np.uint8(thermal_image)
    thermal_image = cv2.applyColorMap(thermal_image, cmapy.cmap('inferno'))

    # Resize to match the HD frame size
    thermal_image = cv2.resize(thermal_image, (640, 480))

    return thermal_image

# Function to generate combined video stream
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

        # Check if thermal frame has an alpha channel
        if thermal_frame.shape[2] == 4:
            # Split alpha channel from thermal frame
            b, g, r, a = cv2.split(thermal_frame)
            overlay = cv2.merge((b, g, r))
            mask = a / 255.0
            inverse_mask = 1.0 - mask

            # Blend the frames using the alpha mask
            for c in range(0, 3):
                hd_frame[:, :, c] = (inverse_mask * hd_frame[:, :, c] + mask * overlay[:, :, c])
        else:
            # Blend the thermal frame with the HD frame if no alpha channel
            hd_frame = cv2.addWeighted(hd_frame, 1 - alpha, thermal_frame, alpha, 0)

        # Encode combined frame
        (flag, encoded_image) = cv2.imencode(".jpg", hd_frame)
        if not flag:
            continue
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

# Route to handle the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main function to start the threads and Flask app
if __name__ == '__main__':
    # Start HD camera thread
    hd_thread = threading.Thread(target=capture_hd_camera)
    hd_thread.daemon = True
    hd_thread.start()

    # Start thermal camera thread
    thermal_thread = threading.Thread(target=capture_thermal_camera)
    thermal_thread.daemon = True
    thermal_thread.start()

    # Start Flask app
    app.run(host='0.0.0.0', port=8030, debug=True, threaded=True, use_reloader=False)
