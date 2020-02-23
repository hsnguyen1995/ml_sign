'''ASL MACHINE LEARNING FLASK APP
How to run: In terminal type "python webstreaming.py -i 0.0.0.0 -o 5000"
You may need to adjust this for your IP address and port #
'''

from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import request, render_template, url_for, redirect
import threading
import argparse
import datetime
import imutils
import time
from cv2 import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

@app.route("/forward/", methods=['POST'])
def move_forward():
   
	
	cv2.imwrite(filename='data/saved_img.jpg', img=outputFrame)
	model = load_model("cnn_sign.h5")

	test_data_dir = "data/"

	def load_testing_data():
		img_size = 64
		testing_data = []
		labels = []
		for image in os.listdir(test_data_dir):
			try:
				img_array = cv2.imread(test_data_dir + image)
				new_array = cv2.resize(img_array, (img_size,img_size))
				testing_data.append(new_array)
				labels.append(image)
			except Exception as e:
				pass
				
		testing_data_arr = np.array(testing_data)
		testing_data_norm = testing_data_arr.astype('float32')/255

		return testing_data_norm, labels

	testing_data_norm, labels_categ = load_testing_data()

	predictions = [model.predict_classes(image.reshape(1,64,64,3))[0] for image in testing_data_norm]
	list=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
	print(list[predictions[0]])
	forward_message = list[predictions[0]]
	return render_template('index.html', forward_message=forward_message)


@app.route("/about", methods=['GET', 'POST'])
def about():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template("about.html")

@app.route("/more", methods=['GET', 'POST'])
def more():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template("more.html")


    


def detect_motion(frameCount):
	
	global vs, outputFrame, lock

	while True:
		
		frame = vs.read()
		frame = imutils.resize(frame, width=400)

	
		with lock:
			outputFrame = frame.copy()
		
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	

	# loop over frames from the output stream
	while True:
		
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
		

		
@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	#construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()

