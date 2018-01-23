import cv2
import glob
import random
import numpy as np

emotions = ["happy","neutral","fear"] #Emotion list
face = cv2.face.createFisherFaceRecognizer() #Initialize fisher face classifier

data = {}

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
    
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


#emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions

def detect_faces(frame):

	faceDet = cv2.CascadeClassifier("/home/realsys2/opencv/data/haarcascades/haarcascade_frontalface_default.xml")
	faceDet2 = cv2.CascadeClassifier("/home/realsys2/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml")
	faceDet3 = cv2.CascadeClassifier("/home/realsys2/opencv/data/haarcascades/haarcascade_frontalface_alt.xml")
	faceDet4 = cv2.CascadeClassifier("/home/realsys2/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml")

	#frame = cv2.imread("sachin.jpg") #Open image
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
	#print("detect_faces")
	#Detect face using 4 different classifiers
	face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
	face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
	face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
	face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

	#Go over detected faces, stop at first detected face, return empty if no face.
	if len(face) == 1:
	    features = face
	elif len(face2) == 1:
	    features = face2
	elif len(face3) == 1:
	    features = face3
	elif len(face4) == 1:
	    features = face4
	else:
	    features = ""
        
	#Cut and save face
	for (x, y, w, h) in features: #get coordinates and size of rectangle containing face
	    #print "face found in file: %s" %f
	    gray = gray[y:y+h, x:x+w] #Cut the frame to size
	    
	    try:
		out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
		return out
		#cv2.imwrite("/home/realsys2/opencv/Workspace/EmotionDetect/gray.jpg", out) #Write image
		#print("writing file")
	    except:
	       pass 

def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    
    print "training fisher face classifier"
    print "size of training set is:", len(training_labels), "images"
    face.train(training_data, np.asarray(training_labels))

    print "predicting classification set"
    cnt = 0
    correct = 0
    incorrect = 0
    
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output1.avi',fourcc, 20.0, (640,480))

    happyCount = 0
    neutralCount = 0
    fearCount = 0
    frameCount = 0
    frameCountOfFear = 0

    while(cap.isOpened()):
      try:
	frameCount += 1
    	ret, frame = cap.read()
	
    	image = detect_faces(frame)
    	if image is None:
  		pass
	else:
		pred, conf = face.predict(image)
    		#print("Prediction: %s" %pred)
    		#print("Prediction: %s" %conf)
    		#print("Prediction_labels: %s" %prediction_labels)
    		#print("fisher")
    		expression = emotions[pred]
		if expression == "happy":
			happyCount += 1
		if expression == "neutral":
			neutralCount += 1
		if expression == "fear":
			fearCount += 1
			frameCountOfFear = frameCount
			frameTime = frameCountOfFear/3.3
			print(frameTime)
    		cv2.putText(image,expression, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA) #Draw the text
		#out.write(image)
    		cv2.imshow("Emotion", image)
    		if cv2.waitKey(1) & 0xFF == ord('q'):
        		break
      except Exception:
	print(happyCount)
   	print(neutralCount)
    	print(fearCount)
	return
    
    
    cap.release()
    out.release()	

run_recognizer()

cv2.destroyAllWindows()

#print ("Emotion: " expression)
    #metascore.append(correct)

#print "\n\nend score:", np.mean(metascore), "percent correct!"
