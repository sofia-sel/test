from flask import Flask, Response, render_template
import threading
import time
import io
import cv2
from picamera2 import Picamera2, Preview

app = Flask(__name__)

output_frame = None
lock = threading.Lock()

def capture_frames():
    global output_frame, lock
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    while True:
        image = picam2.capture_array()
        # change color 
        image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB)
        with lock:
            output_frame = image.copy()

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
