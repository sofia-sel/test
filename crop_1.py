from flask import Flask, Response, render_template
import threading
import cv2
from picamera2 import Picamera2

app = Flask(__name__)

# Global variables
output_frame = None
lock = threading.Lock()

# Define the crop region (top-left and bottom-right corners)
crop_top_left = (100, 100)
crop_bottom_right = (540, 380)

def capture_frames():
    global output_frame, lock
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    while True:
        try:
            # Capture frame-by-frame
            image = picam2.capture_array()
            # Convert the image color from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Crop the image
            cropped_image = image[crop_top_left[1]:crop_bottom_right[1], crop_top_left[0]:crop_bottom_right[0]]
            with lock:
                output_frame = cropped_image.copy()
        except Exception as e:
            print(f"Error capturing frame: {e}")

@app.route("/")
def index():
    # Serve the index HTML page
    return render_template("index.html")

def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            # Encode the frame in JPEG format
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        # Yield the output frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # Return the response generated along with the specific media type
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    # Start a thread that will perform the frame capturing
    t = threading.Thread(target=capture_frames)
    t.daemon = True
    t.start()
    # Start the Flask app
    app.run(host='0.0.0.0', port=8010, debug=True, threaded=True, use_reloader=False)
