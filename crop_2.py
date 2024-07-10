from flask import Flask, Response, render_template
import threading
import time
import io
import cv2
try:  # If called as an imported module
	from pithermalcam import pithermalcam
except:  # If run directly
	from pi_therm_cam import pithermalcam

app = Flask(__name__)

output_frame = None
lock = threading.Lock()

# Define the desired crop dimensions
CROP_WIDTH = 550  # Adjusted based on calculations or requirements
CROP_HEIGHT = 280  # Adjusted based on calculations or requirements

def capture_frames():
    global output_frame, lock
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    while True:
        image = picam2.capture_array()
        
        # Example: Crop the image to a specific size (550x280)
        height, width, channels = image.shape
        start_x = (width - CROP_WIDTH) // 2
        start_y = (height - CROP_HEIGHT) // 2
        cropped_image = image[start_y:start_y+CROP_HEIGHT, start_x:start_x+CROP_WIDTH]

        # Convert color space if needed (example conversion)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        with lock:
            output_frame = cropped_image.copy()

@app.route("/")
def index():
    return render_template("index.html")

def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    t = threading.Thread(target=capture_frames)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=8010, debug=True, threaded=True, use_reloader=False)
