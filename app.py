from flask import Flask, render_template, url_for, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    # Check if 'output.mp4' exists
    if os.path.exists('output.mp4'):
        return render_template('display_video.html')
    else:
        return render_template('display_no.html')

@app.route('/video')
def video():
    return send_from_directory('.', 'output.mp4')

if __name__ == '__main__':
    app.run(debug=True)
