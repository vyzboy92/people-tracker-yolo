# people-tracker-yolo
This project intends to build a people detection and tracking pipeline to detect and track people across a camera feed.
People are tracked once the YOLO network detects it. 
Users can either use IP camera streams or input video to do the inference. Tracking is done with dlib tracker.
# Inference flow
Since IP camera streams are used, latency in frames can occur. This leads to frame drops and frozen script.
To avoid this, the WebCamVideostream class from imutils package is used.
The captured frame is send to the YOLO model for inference, where it detects people and corresponding centroids are calculated.
The centroids are passed to the dlib tracker for id generation and tracking.
# Requirements
1. OpenCV 3.4
2. Tensorflow 1.7.0
3. Keras
4. PIL
5. imutils
6. logging
7. Mongo DB
8. pymongo
9. dlib
# Running the tracker
This pipeline relies on a configuration maintained in a Mongodb collection to access the video input.
Data for building this collection is available in the config.txt file.
## Steps to run the file
1. Create a Mongodb database named 'Main_DB'
2. Create a colletion named 'cam' and write the data in 'config.txt' to a new document in 'cam' (modify the content according to your usage)
3. Optional: create an environment to install the required packages and activate it.
4. Run the script as ```python person_tracker.py```
# Issues
The yolo inference runs optimally on a CUDA capable NVIDIA GPU. CPU inferences are slow (which is expected behaviour)
Tracker can  wrongly ID a person in crowded situation or when the camera is not static. This may cause ID switching.
# Future developement
Entire workflow will be switche dto the OpenVINO toolkit provided by Intel which accelerates CPU inference significantly.
