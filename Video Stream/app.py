from flask import Flask, Response, render_template
import cv2

app = Flask(__name__)

def gen_frames():
    camera = cv2.VideoCapture(0)  # Use camera 0
    
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concatenate frame one by one and show result

@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Video streaming home page
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)