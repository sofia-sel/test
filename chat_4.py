from flask import Flask, Response, render_template
import threading
import time
import io
import picamera
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2

app = Flask(__name__)

output_frame = None
lock = threading.Lock()

def capture_frames():
    global output_frame, lock
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.framerate = 32
        raw_capture = PiRGBArray(camera, size=(640, 480))
        time.sleep(0.1)

        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            image = frame.array
            raw_capture.truncate(0)
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
