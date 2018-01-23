# Emotion-Detection
Detect emotions on human faces

This code uses an available functionality in OpenCV called Fisher Face Recognizer. HAAR Cascade Classifiers in OpenCV are used to detect a face in the frame. A model is trained on different facial emotions and is used to predict the emotion on the detected face in real-time. This code was used to detect and record emotions of fear when a drone is being flown around a human being to determine the areas where the drone inflicts fear. The code was around 70%-80% accurate and it is designed to train everytime the code is run. The next step would be to save the model so that it need not be trained everytime.

The dataset for training can be downloaded from here after your request has been approved: http://www.consortium.ri.cmu.edu/ckagree/
or you can create your own dataset of different people with different emotions in grayscale.


Datasets are based on the work of the following:

– Kanade, T., Cohn, J. F., & Tian, Y. (2000). Comprehensive database for facial expression analysis. Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG’00), Grenoble, France, 46-53.
– Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression. Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.

The idea is taken from the tutorial:

van Gent, P. (2016). Emotion Recognition With Python, OpenCV and a Face Dataset. A tech blog about fun things with Python and embedded electronics. Retrieved from:
http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
