# -*- coding: utf-8 -*-
#!/usr/bin/python3

import cv2
import io
import threading
from flask import Flask, Response, render_template

# Initialize OpenCV capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera (usually built-in webcam)

app = Flask(__name__)

def generate_frames():
    while True:
        success, frame = cap.read()  # Read frame from camera
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the index.html page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)

cap.release()
cv2.destroyAllWindows()
