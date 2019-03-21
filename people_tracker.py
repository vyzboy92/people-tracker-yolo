#! /usr/bin/env python
# -*- coding: utf-8 -*-

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import numpy as np
import imutils
import dlib
import cv2
import time
import logging
import pymongo
from math import ceil
from datetime import datetime
import warnings, os
from multiprocessing import Process
from PIL import Image
from yolo import YOLO
logger = logging.Logger('catch_all')
warnings.filterwarnings('ignore')

def chk_movement(cent, a1, b1, p_id, cam_id, count_type):
    Dir = ((cent[0] - a1[0]) * (b1[1] - a1[1])) - ((cent[1] - a1[1]) * (b1[0] - a1[0]))

    Dir = 1 if Dir > 0 else -1
    flag = False

    if cam_id in buff_dict:
        if p_id in buff_dict[cam_id]:
            out = buff_dict[cam_id][p_id]
            if int(time.time()) - out['timestamp'] > 900:
                buff_dict[cam_id][p_id] = {'timestamp': int(time.time()), 'd': Dir}
            else:
                if out['d'] != Dir:
                    buff_dict[cam_id][p_id] = {'timestamp': int(time.time()), 'd': Dir}
                    flag = True
                else:
                    buff_dict[cam_id][p_id] = {'timestamp': int(time.time()), 'd': out['d']}
        else:
            buff_dict[cam_id][p_id] = {'timestamp': int(time.time()), 'd': Dir}
    else:
        buff_dict[cam_id] = dict()


    if flag:
        if Dir == count_type:
            return 1
        else:
            return -1
    else:
        return 0


def person_tracker(yolo, video, cam_id, a, b, count_type):

    print("[INFO] opening video file...")
    fvs = WebcamVideoStream(video).start()
    time.sleep(0.5)
    W = None
    H = None
    ct = CentroidTracker(maxDisappeared=1, maxDistance=500)
    trackers = []
    trackableObjects = {}
    totalFrames = 0
    cnt = 0
    exit_cnt = 0
    scale_factor = 1
    fps = FPS().start()
    init_frame = fvs.read()
    if init_frame is None:
        print('No frame')
    # print(init_frame.type)
    if init_frame.shape[1] == 1920:
        scale_factor = 4
    elif init_frame.shape[1] == 3072:
        scale_factor = 8
    frm_width = ceil(init_frame.shape[1] / scale_factor)
    frm_height = ceil(init_frame.shape[0] / scale_factor)
    a1 = [ceil(a_ / scale_factor) for a_ in a]
    b1 = [ceil(b_ / scale_factor) for b_ in b]
    while True:
        fps.update()
        skip_frames = 60
        frame = fvs.read()
        if frame is None:
            break

        frame = imutils.resize(frame, frm_width, frm_height)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        rects = []
        if totalFrames % skip_frames == 0:
            trackers = []
            image = Image.fromarray(frame)
            boxs = yolo.detect_image(image)
            print(boxs)
            for box in boxs:

                startX = box[0]
                startY = box[1]
                endX = box[2] + startX
                endY = box[3] + startY
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                trackers.append(tracker)


        else:
            # loop over the trackers
            for tracker in trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

                # draw a horizontal line in the center of the frame -- once an
                # object crosses this line we will determine whether they were
                # moving 'up' or 'down'

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        for (objectID, data) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            centroid = data[0]
            objectRect = data[1]
            # print(objectRect)
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # # 'up' and positive for 'down')
                # y = [c[1] for c in to.centroids]
                # direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

            trackableObjects[objectID] = to
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            count_flag = chk_movement([centroid[0], centroid[1]], a1, b1, int(objectID), int(cam_id), count_type)
            if count_flag == 1:
                cnt += 1
                cnt_col.update({'cam_id': cam_id, 'video_file': video}, {
                    '$set': {'entry_count': cnt, 'processed_timestamp': datetime.utcnow()}},
                               upsert=True)
            elif count_flag == -1:
                exit_cnt += 1
                cnt_col.update({'cam_id': cam_id, 'video_file': video}, {
                    '$set': {'exit_count': exit_cnt,
                             'processed_timestamp': datetime.utcnow()}},
                               upsert=True)

        info = [
            ("Exit", cnt),
            ("Entry", exit_cnt)
        ]
        #
        # # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Frame", cv2.resize(frame, (800,600)))
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    print("completed....")
    fvs.stop()

if __name__ == '__main__':

    em_client = pymongo.MongoClient("mongodb://localhost:27017/")
    dblist = em_client.list_database_names()
    if "Emotix_DB" in dblist:
        print("========================")
        print("Main_db found in Mongo")
        print("========================")
    em_db = em_client["Emotix_DB"]
    em_col = em_db["visitor_info"]  # logs emotions, person id, age, gender, time etc
    vid_col = em_db["cam"]
    cnt_col = em_db["entry_count"]
    buff_dict = dict()
    yolo = YOLO()

    for doc in vid_col.find({}):
        rtsp = doc['feed']
        cam_id = doc['cam_id']
        a = doc['a']
        b = doc['b']
        # _type = doc['type']
        count_type = doc['count_type']
        emotion = False
        person_tracker(yolo, rtsp, cam_id, a, b, count_type)

