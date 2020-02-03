# USAGE
# python webstreaming.py --model liveness.model --le le.pickle --detector face_detector --shape_predictor shapepredictor --ip 0.0.0.0 --port 8000

# import the necessary packages
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import dlib
from scipy.spatial import distance as dist

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup

video_capture = cv2.VideoCapture(0)  


@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def eye_aspect_ratio(eye):
    
    A = dist.euclidean(eye[1], eye[5])  
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])  
    ear = (A + B) / (2.0 * C)  
    return ear 




class Liveness1:
    
    def __init__(self, accumWeight):
        self.accumWeight = accumWeight
        self.bg = None
    
    def update(self, image):
        
        
        if self.bg is None:
                    
            self.bg = image.copy().astype("float")
            return
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)    
        
#    @classmethod
    def detect(frame, rects):
        
        global outputFrame, lock, total
    
        liv = Liveness1(accumWeight=0.1)
        total = 0
        
        
        for rect  in rects:
            
            
            x = rect.left()  
            y = rect.top()  
            x1 = rect.right()  
            y1 = rect.bottom()
            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])  
            left_eye = landmarks[LEFT_EYE_POINTS]  
            right_eye = landmarks[RIGHT_EYE_POINTS]  
            left_eye_hull = cv2.convexHull(left_eye)  
            right_eye_hull = cv2.convexHull(right_eye)  
            ear_left = eye_aspect_ratio(left_eye)  
            ear_right = eye_aspect_ratio(right_eye)
                    
            if ear_left < EYE_AR_THRESH:

                COUNTER_LEFT += 1
                   
            else:
                
                if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
                
                    TOTAL_LEFT += 1  
                    print("Left eye winked") 
                    COUNTER_LEFT = 0
    
            if ear_right < EYE_AR_THRESH:
                
                COUNTER_RIGHT += 1  
                    
            else:
                        
                if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES: 
                        
                    TOTAL_RIGHT += 1  
                    print("Right eye winked")  
                    COUNTER_RIGHT = 0
                    
            x = TOTAL_LEFT + TOTAL_RIGHT
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
            
    for i in range(0, detections.shape[2]):
        
                
        confidence = detections[0, 0, i, 2]
                
        if confidence > args["confidence"] and x>10:
                    
                    
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
    
        	# ensure the detected bounding box does fall outside the
        	# dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
        
        	# extract the face ROI and then preproces it in the exact
        	# same manner as our training data
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
        
        	# pass the face ROI through the trained liveness detector
        	# model to determine if the face is "real" or "fake"
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]
        
        	# draw the label and bounding box on the frame
            label = "{}: {:.4f}".format(label, preds[j])
            cv2.putText(frame, label, (startX, startY - 10),
    				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
        				(0, 0, 255), 2)
            #no return
            liv.update(gray)
            with lock:
                outputFrame = frame.copy()            
        
		
def detect_liveness(frameCount):
    global outputFrame, lock
    
    liv = Liveness1(accumWeight=0.1)
    total = 0
    
    while True:
        ret, frame = video_capture.read()
        
        timestamp = datetime.datetime.now()
        
        cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            frame = imutils.resize(frame, width=600)
        if total > frameCount: 
            
            liv.detect(gray, rects)
        
        total=total+1
   
        
        
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
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")
        
    
  
# check to see if this is the main thread of execution

if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
    		help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
    		help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
    		help="# of frames used to construct the background model")
    ap.add_argument("-m", "--model", type=str, required=True,
    	    help="path to trained model")
    ap.add_argument("-l", "--le", type=str, required=True,
    	    help="path to label encoder")
    ap.add_argument("-d", "--detector", type=str, required=True,
    	    help="path to OpenCV's deep learning face detector")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
    	    help="minimum probability to filter weak detections")
    ap.add_argument("-p", "--shape_predictor", required=True,
    	    help="path to facial landmark predictor")
    args = vars(ap.parse_args())
    x=0
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
    	"res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    
    # load the liveness detector model and label encoder from disk
    print("[INFO] loading liveness detector...")
    model = load_model(args["model"])
    le = pickle.loads(open(args["le"], "rb").read())
    #detetrmining all the face points in the form of list
    FULL_POINTS = list(range(0, 68))  
    FACE_POINTS = list(range(17, 68))  
    JAWLINE_POINTS = list(range(0, 17))  
    RIGHT_EYEBROW_POINTS = list(range(17, 22))  
    LEFT_EYEBROW_POINTS = list(range(22, 27))  
    NOSE_POINTS = list(range(27, 36))  
    RIGHT_EYE_POINTS = list(range(36, 42))  
    LEFT_EYE_POINTS = list(range(42, 48))  
    MOUTH_OUTLINE_POINTS = list(range(48, 61))  
    MOUTH_INNER_POINTS = list(range(61, 68))  
       
    EYE_AR_THRESH = 0.30 
    EYE_AR_CONSEC_FRAMES = 2  
       
    COUNTER_LEFT = 0  
    TOTAL_LEFT = 0  
       
    COUNTER_RIGHT = 0  
    TOTAL_RIGHT = 0 
    
    detector = dlib.get_frontal_face_detector()  
    predictor = dlib.shape_predictor(args["shape_predictor"])


    # start a thread that will perform liveness detection
    t = threading.Thread(target=detect_liveness, args=(
		args["frame_count"],))
    t.daemon = True
    t.start()

	# start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
    
# release the video stream pointer
cv2.destroyAllWindows()

