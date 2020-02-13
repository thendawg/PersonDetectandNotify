# Live Person detection with alerts based on Tensorflow Object Detection Framework

import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import os
import logging
import shutil
import config
import datetime
from pushbullet import Pushbullet
from threading import Thread
from queue import Queue, LifoQueue
from flask import Flask, render_template, Response

class DetectorAPI:
    def __init__(self, model_path):
        self.path_to_ckpt = model_path
        config = tf.ConfigProto(
                device_count = {'GPU': 1}
            )
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.30
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(config=config, graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
    
    def processFrame(self, q, q_img):
        while True:
            # Get a fresh frame each time processFrame executes then clear the queue after 20 iterations - this prevents too much memory from being consumed.
            self.image = q_img.get()
            # Resize the frame to 720p
            self.imgrs = cv2.resize(self.image,(1280, 720))
            # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(self.imgrs, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})


            im_height, im_width,_ = self.imgrs.shape
            boxes_list = [None for i in range(boxes.shape[1])]
            for i in range(boxes.shape[1]):
                boxes_list[i] = (int(boxes[0,i,0] * im_height),
                            int(boxes[0,i,1]*im_width),
                            int(boxes[0,i,2] * im_height),
                            int(boxes[0,i,3]*im_width))

            q.put((self.imgrs, boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])))

    def close(self):
        self.sess.close()
        self.default_graph.close()
        
# Class for the seperate thread that will grab frames from the camera (this is much faster than single threading)

class FrameGrab:

    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        
    # This is the part that actually grabs the frames in loop and puts them in queue
    def get(self, q_img):
        while True:
            self.cap.grab()         
            if q_img.qsize() == 0:
                (self.grabbed, self.frame) = self.cap.retrieve()
                q_img.put(self.frame)
            
# Class for seperate thread for PushBullet Notifications - Only spawns if pb is enabled in config.

class PBAsync:
    
    # Checks if PB queue is empty, if it is not, push alert message from queue and task_done
    def sendpbalert(q_pb, pbapikey, pbch):
        pb = Pushbullet(pbapikey)
        pbipcam_channel = pb.get_channel(pbch)
        while True:
            if q_pb.qsize() > 0:
                pbmsg = q_pb.get()
                q_pb.task_done()
                pb.push_link(pbmsg[0], pbmsg[1], channel=pbipcam_channel)
            else:
                time.sleep(.10)
            
def runflask(numcams, q_imgout_cam1, q_imgout_cam2, q_imgout_cam3, q_imgout_cam4):
    
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template('index.html')

    def stream1():
        while True:
            frame1 = q_imgout_cam1.get()
            q_imgout_cam1.task_done()
            frame1dec = cv2.imencode('.jpg', frame1)[1].tobytes()            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame1dec + b'\r\n')
    
    if numcams >= 2:
        def stream2():
            while True:
                frame2 = q_imgout_cam2.get()
                q_imgout_cam2.task_done()
                frame2dec = cv2.imencode('.jpg', frame2)[1].tobytes()            
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame2dec + b'\r\n')
       
    if numcams >= 3:
        def stream3():
            while True:
                frame3 = q_imgout_cam3.get()
                q_imgout_cam3.task_done()
                frame3dec = cv2.imencode('.jpg', frame3)[1].tobytes()            
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame3dec + b'\r\n')
    
    if numcams >= 4:
        def stream4():
            while True:
                frame4 = q_imgout_cam4.get()
                q_imgout_cam4.task_done()
                frame4dec = cv2.imencode('.jpg', frame4)[1].tobytes()            
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame4dec + b'\r\n')

    @app.route('/video_feed1')
    def video_feed1():
        return Response(stream1(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
                        
    if numcams >= 2:
        @app.route('/video_feed2')
        def video_feed2():
            return Response(stream2(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
    
    if numcams >= 3:
        @app.route('/video_feed3')
        def video_feed3():
            return Response(stream3(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        
    if numcams >= 4:
        @app.route('/video_feed4')
        def video_feed4():
            return Response(stream4(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
                        
    app.run(host='0.0.0.0', debug=True, threaded=True, use_reloader=False)
    
        
    
def analyzeframe(img, boxes, scores, classes, num, hfilter, vfilter, hfilter2, vfilter2, fsize):
    humandetected = 0
    alertpb = 0
    
    # For function to iterate through all detection boxes and draw those that are over the cutoff and correct class (1 - Human) 
    for i in range(len(boxes)):
        if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]            
            yeval = int(box[2])
            y2eval = int(box[0])
            xeval = int(box[3])
            x2eval = int(box[1])
            scoreint = str(round(scores[i], 3))
            # *This is now set via config* This is a filter to determine if any vertex of the bounding box for a detected person is within the part of the image I care about ie: not across the street (remember opencv uses rows, so top is 0) Basically if you set this value to say 100, then at least some part of the detected object must be outside fo the top 100 rows/pixels in the image. 
            if (yeval > hfilter) or (y2eval > hfilter):
                if (yeval < hfilter2) or (y2eval < hfilter2):
                    if (xeval > vfilter) or (x2eval > vfilter):
                        if (xeval < vfilter2) or (x2eval < vfilter2):
                                # Draws the box, puts a confidence score under it, and alerts that we have detected a human
                                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(0,0,255),2)
                                cv2.putText(img,scoreint,(box[1]+10,box[2]+40),cv2.FONT_HERSHEY_DUPLEX,fsize,(0,0,255),1,cv2.LINE_AA)
                                humandetected = 1
    
    # A seperate for function to evaluate if scores of bounding boxes detected meet the second (higher) threshold for a pb notification.
    for i in range(len(boxes)):
        if classes[i] == 1 and scores[i] > thresholdpb:
            if (yeval > hfilter) or (y2eval > hfilter):
                if (yeval < hfilter2) or (y2eval < hfilter2):
                    if (xeval > vfilter) or (x2eval > vfilter):
                        if (xeval < vfilter2) or (x2eval < vfilter2):
                                alertpb = 1
                            
    return img, humandetected, alertpb
    
def humanevent(img, timeelap, dirtime, timebetweenevents, pbenabled, url, imgoutdir, alertpb, nametext, timeelappb, lastpbtime, q_pb, logpath):
    # Check to see how long its been since the last person detected, this avoids a new entry/notification for people hanging around. Tweak as needed       
    if timeelap > timebetweenevents:
        # This process builds a directory structure based on time since epoch + driveway/frontyard and then writes the image into that directory as well as a php file to display the images. It also pushes the notification and writes to our log.
        dirtime = str(round(time.time()))
        now = datetime.datetime.now()
        timenow = now.strftime("%Y-%m-%d %H:%M:%S")
        os.mkdir(imgoutdir + nametext + dirtime)
        cv2.imwrite(imgoutdir + nametext + dirtime + '/' + dirtime + nametext + '.jpg', img)
        shutil.copy2('showimgs.php', imgoutdir + nametext + dirtime)
        logfile = open(logpath,'a+')
        logfile.write(timenow + ' - <a href="' + url + nametext + dirtime + '/showimgs.php" target="_blank">Person detected ' + nametext + '</a>\n') 
        logfile.close()
        lastlogtime = int(round(time.time()))
    else:
        # If were still detecting people, but its within the elapse interval, well write the frame to the same directory as before.
        linktime = str(round(time.time(), 2))
        scrublinktime = linktime.replace(".", "-")
        cv2.imwrite(imgoutdir + nametext + dirtime + '/' + scrublinktime + nametext + '.jpg', img)
        lastlogtime = int(round(time.time()))
    
    # Seperately evaluate the PushBullet function, this requires the higher score threshold to be met and tracks the time since last pb push seperate from the last log entry.
    if (timeelappb > timebetweenevents) and (pbenabled == 1) and (alertpb == 1):
        # Puts pb notification in queue
        pbp1 = str("Person Detected " + nametext)
        pbp2 = str(url + nametext + dirtime + "/showimgs.php")
        q_pb.put([pbp1, pbp2])
        lastpbtime = int(round(time.time()))
    elif (pbenabled == 1) and (alertpb == 1):
        lastpbtime = int(round(time.time()))
    return lastlogtime, dirtime, lastpbtime
    

if __name__ == "__main__":

    # These are the variables defined in config.py, set them there.
    numcams = config.numcams
    previewmode = config.previewmode
    model_path = str(config.model_path)
    threshold = config.threshold
    thresholdpb = config.thresholdpb
    timebetweenevents = config.timebetweenevents
    coloraftertime = config.coloraftertime
    yfilter1cam1 = config.yfilter1cam1
    xfilter1cam1 = config.xfilter1cam1
    yfilter2cam1 = config.yfilter2cam1
    xfilter2cam1 = config.xfilter2cam1
    yfilter1cam2 = config.yfilter1cam2
    xfilter1cam2 = config.xfilter1cam2
    yfilter2cam2 = config.yfilter2cam2
    xfilter2cam2 = config.xfilter2cam2
    yfilter1cam3 = config.yfilter1cam3
    xfilter1cam3 = config.xfilter1cam3
    yfilter2cam3 = config.yfilter2cam3
    xfilter2cam3 = config.xfilter2cam3
    yfilter1cam4 = config.yfilter1cam4
    xfilter1cam4 = config.xfilter1cam4
    yfilter2cam4 = config.yfilter2cam4
    xfilter2cam4 = config.xfilter2cam4
    imgoutdir = str(config.imgoutdir)
    fsize = config.fontsize
    fsize2 = config.fontsize2
    fsize3 = config.fontsize3
    previewtime = config.previewtime
    hsizeout = config.imgwidth
    vsizeout = config.imgheight
    url = str(config.url)
    cam1text = str(config.cam1text)
    cam2text = str(config.cam2text)
    cam3text = str(config.cam3text)
    cam4text = str(config.cam4text)
    frametime = config.frametime
    pbenabled = config.pbenabled
    logpath = config.logpath
    
    # Setup initial timings based on number of cams 
    lastlogtimecam1 = int(round(time.time()))
    drtimecam1 = "0"
    lastpbtimecam1 = int(round(time.time()))
    if numcams >= 2:
        lastlogtimecam2 = int(round(time.time()))
        drtimecam2 = "0"
        lastpbtimecam2 = int(round(time.time()))
    if numcams >= 3:
        lastlogtimecam3 = int(round(time.time()))
        drtimecam3 = "0"
        lastpbtimecam3 = int(round(time.time()))
    if numcams >= 4:
        lastlogtimecam4 = int(round(time.time()))
        drtimecam4 = "0"
        lastpbtimecam4 = int(round(time.time()))

    # Setup the capture threads and queues
    q_cam1_img = Queue(maxsize=1)
    if numcams >= 2:
        q_cam2_img = Queue(maxsize=1)
    if numcams >= 3:
        q_cam3_img = Queue(maxsize=1)
    if numcams >= 4:
        q_cam4_img = Queue(maxsize=1)
    capurl = config.capurl
    if numcams >= 2:
        capurl2 = config.capurl2
    if numcams >= 3:
        capurl3 = config.capurl3
    if numcams >= 4:
        capurl4 = config.capurl4
    capcam1t = Thread(name='capcam1t', target=FrameGrab(capurl).get, daemon = True, args=(q_cam1_img,))
    if numcams >= 2:
        capcam2t = Thread(name='capcam2t', target=FrameGrab(capurl2).get, daemon = True, args=(q_cam2_img,))
    if numcams >= 3:
        capcam3t = Thread(name='capcam3t', target=FrameGrab(capurl3).get, daemon = True, args=(q_cam3_img,))
    if numcams >= 4:
        capcam4t = Thread(name='capcam4t', target=FrameGrab(capurl4).get, daemon = True, args=(q_cam4_img,))
    capcam1t.start()
    if numcams >= 2:
        capcam2t.start()
    if numcams >= 3:
        capcam3t.start()
    if numcams >= 4:
        capcam4t.start()
    
    # Spawn a seperate process and queue for PushBullet Notifications (Makes them async) - If PB is enabled in config - see README for additional pushbullet info. 
    if pbenabled == 1:
        pbapikey = config.pbapikey
        pbch = config.pbchannelname   
        q_pb = Queue()
        pbt = Thread(name='pushbullett', target=PBAsync.sendpbalert, daemon = True, args=(q_pb, pbapikey, pbch))
        pbt.start()
    
    # Build a queue and start a seperate thread for each stream being analyzed - the maxsize of 1 for the queue insures that TF cant outrun the image processing/output. 
    q_cam1 = Queue(maxsize=1)
    cam1t = Thread(name='acam1t', target=DetectorAPI(model_path).processFrame, daemon = True, args=(q_cam1, q_cam1_img))
    cam1t.start()
    if numcams >= 2:
        q_cam2 = Queue(maxsize=1)
        cam2t = Thread(name='acam2t', target=DetectorAPI(model_path).processFrame, daemon = True, args=(q_cam2, q_cam2_img))
        cam2t.start()
    if numcams >= 3:
        q_cam3 = Queue(maxsize=1)
        cam3t = Thread(name='acam3t', target=DetectorAPI(model_path).processFrame, daemon = True, args=(q_cam3, q_cam3_img))
        cam3t.start()
    if numcams >= 4:
        q_cam4 = Queue(maxsize=1)
        cam4t = Thread(name='acam4t', target=DetectorAPI(model_path).processFrame, daemon = True, args=(q_cam4, q_cam4_img))
        cam4t.start()
    
    # Start a thread for Flask
    q_imgout_cam1 = Queue(maxsize=1)
    q_imgout_cam2 = Queue(maxsize=1)
    q_imgout_cam3 = Queue(maxsize=1)
    q_imgout_cam4 = Queue(maxsize=1)
    flaskt = Thread(name='flaskt', target=runflask, daemon = True, args=(numcams, q_imgout_cam1, q_imgout_cam2, q_imgout_cam3, q_imgout_cam4))      
    flaskt.start()
    
    while True:
        
        # Get timestamps, start of process time
        start_time = time.time()
        timestamp = time.ctime()
        
        # Retrieve TF results from queues
        img, boxes, scores, classes, num = q_cam1.get()
        q_cam1.task_done()
        if numcams >= 2:
            img2, boxes2, scores2, classes2, num2 = q_cam2.get()
            q_cam2.task_done()
        if numcams >= 3:
            img3, boxes3, scores3, classes3, num3 = q_cam3.get()
            q_cam3.task_done()
        if numcams >= 4:
            img4, boxes4, scores4, classes4, num4 = q_cam4.get()
            q_cam4.task_done()
        
        # Determine if a human was detected and draw a bounding box if so along with scores
        imgoutcam1, humandetectcam1, alertpbcam1 = analyzeframe(img, boxes, scores, classes, num, yfilter1cam1, xfilter1cam1, yfilter2cam1, xfilter2cam1, fsize)
        if numcams >= 2:
            imgoutcam2, humandetectcam2, alertpbcam2 = analyzeframe(img2, boxes2, scores2, classes2, num2, yfilter1cam2, xfilter1cam2, yfilter2cam2, xfilter2cam2, fsize)
        if numcams >= 3:
            imgoutcam3, humandetectcam3, alertpbcam3 = analyzeframe(img3, boxes3, scores3, classes3, num3, yfilter1cam3, xfilter1cam3, yfilter2cam3, xfilter2cam3, fsize)
        if numcams >= 4:
            imgoutcam4, humandetectcam4, alertpbcam4 = analyzeframe(img4, boxes4, scores4, classes4, num4, yfilter1cam4, xfilter1cam4, yfilter2cam4, xfilter2cam4, fsize)
        
        # Determine time since last PB and log entry events
        curtime = int(round(time.time()))
        timeelapcam1 = curtime-lastlogtimecam1
        timeelappbcam1 = curtime-lastpbtimecam1
        if numcams >= 2:
            timeelapcam2 = curtime-lastlogtimecam2
            timeelappbcam2 = curtime-lastpbtimecam2
        if numcams >= 3:
            timeelapcam3 = curtime-lastlogtimecam3
            timeelappbcam3 = curtime-lastpbtimecam3
        if numcams >= 4:
            timeelapcam4 = curtime-lastlogtimecam4
            timeelappbcam4 = curtime-lastpbtimecam4
        
        # Add timestamps
        cv2.putText(imgoutcam1,timestamp,(35, 40),cv2.FONT_HERSHEY_DUPLEX,fsize,(211,211,211),1,cv2.LINE_AA)
        if numcams >= 2:
            cv2.putText(imgoutcam2,timestamp,(35, 40),cv2.FONT_HERSHEY_DUPLEX,fsize,(211,211,211),1,cv2.LINE_AA)
        if numcams >= 3:
            cv2.putText(imgoutcam3,timestamp,(35, 40),cv2.FONT_HERSHEY_DUPLEX,fsize,(211,211,211),1,cv2.LINE_AA)
        if numcams >= 4:
            cv2.putText(imgoutcam3,timestamp,(35, 40),cv2.FONT_HERSHEY_DUPLEX,fsize,(211,211,211),1,cv2.LINE_AA)
        
        # If a person was detected in the frame, run a function that creates a display directory, saves the frame, sends us a push notification, and writes to the log.
        if humandetectcam1 == 1:
            lastlogtimecam1, drtimecam1, lastpbtimecam1 = humanevent(imgoutcam1, timeelapcam1, drtimecam1, timebetweenevents, pbenabled, url, imgoutdir, alertpbcam1, cam1text, timeelappbcam1, lastpbtimecam1, q_pb, logpath)
        if numcams >= 2:
            if humandetectcam2 == 1:
                lastlogtimecam2, drtimecam2, lastpbtimecam2 = humanevent(imgoutcam2, timeelapcam2, drtimecam2, timebetweenevents, pbenabled, url, imgoutdir, alertpbcam2, cam2text, timeelappbcam2, lastpbtimecam2, q_pb, logpath)
        if numcams >= 3:
            if humandetectcam3 == 1:
                lastlogtimecam3, drtimecam3, lastpbtimecam3 = humanevent(imgoutcam3, timeelapcam3, drtimecam3, timebetweenevents, pbenabled, url, imgoutdir, alertpbcam3, cam3text, timeelappbcam3, lastpbtimecam3, q_pb, logpath)
        if numcams >= 4:
            if humandetectcam4 == 1:
                lastlogtimecam4, drtimecam4, lastpbtimecam4 = humanevent(imgoutcam4, timeelapcam4, drtimecam4, timebetweenevents, pbenabled, url, imgoutdir, alertpbcam4, cam4text, timeelappbcam4, lastpbtimecam4, q_pb, logpath)
        
        # Resize the images for output, or display null image if no person detected in preview mode.
        if previewmode:     
            imgresizecam1 = cv2.resize(imgoutcam1,(hsizeout, vsizeout),interpolation = cv2.INTER_AREA)
            if numcams >= 2:
                imgresizecam2 = cv2.resize(imgoutcam2,(hsizeout, vsizeout),interpolation = cv2.INTER_AREA)
            if numcams >= 3:
                imgresizecam3 = cv2.resize(imgoutcam3,(hsizeout, vsizeout),interpolation = cv2.INTER_AREA)
            if numcams >= 4:
                imgresizecam4 = cv2.resize(imgoutcam4,(hsizeout, vsizeout),interpolation = cv2.INTER_AREA)
        else:
            if (timeelapcam1 < previewtime):
                imgresizecam1 = cv2.resize(imgoutcam1,(hsizeout, vsizeout),interpolation = cv2.INTER_AREA)
            else:
                blank_img1 = np.zeros((720,1280,3), np.uint8)
                cv2.putText(blank_img1,timestamp,(35, 40),cv2.FONT_HERSHEY_DUPLEX,fsize3,(211,211,211),1,cv2.LINE_AA)
                cv2.putText(blank_img1,'No person detected in ' + cam1text,(140, 380),cv2.FONT_HERSHEY_DUPLEX,fsize2,(211,211,211),1,cv2.LINE_AA)
                imgresizecam1 = cv2.resize(blank_img1,(hsizeout, vsizeout),interpolation = cv2.INTER_AREA)
            if numcams >=2:
                if (timeelapcam2 < previewtime):
                    imgresizecam2 = cv2.resize(imgoutcam2,(hsizeout, vsizeout),interpolation = cv2.INTER_AREA)
                else:
                    blank_img2 = np.zeros((720,1280,3), np.uint8)
                    cv2.putText(blank_img2,timestamp,(35, 40),cv2.FONT_HERSHEY_DUPLEX,fsize3,(211,211,211),1,cv2.LINE_AA)
                    cv2.putText(blank_img2,'No person detected in ' + cam2text,(140, 380),cv2.FONT_HERSHEY_DUPLEX,fsize2,(211,211,211),1,cv2.LINE_AA)
                    imgresizecam2 = cv2.resize(blank_img2,(hsizeout, vsizeout),interpolation = cv2.INTER_AREA)
            if numcams >=3:
                if (timeelapcam3 < previewtime):
                    imgresizecam3 = cv2.resize(imgoutcam3,(hsizeout, vsizeout),interpolation = cv2.INTER_AREA)
                else:
                    blank_img3 = np.zeros((720,1280,3), np.uint8)
                    cv2.putText(blank_img3,timestamp,(35, 40),cv2.FONT_HERSHEY_DUPLEX,fsize3,(211,211,211),1,cv2.LINE_AA)
                    cv2.putText(blank_img3,'No person detected in ' + cam3text,(140, 380),cv2.FONT_HERSHEY_DUPLEX,fsize2,(211,211,211),1,cv2.LINE_AA)
                    imgresizecam3 = cv2.resize(blank_img3,(hsizeout, vsizeout),interpolation = cv2.INTER_AREA)
            if numcams >=4:
                if (timeelapcam4 < previewtime):
                    imgresizecam4 = cv2.resize(imgoutcam4,(hsizeout, vsizeout),interpolation = cv2.INTER_AREA)
                else:
                    blank_img4 = np.zeros((720,1280,3), np.uint8)
                    cv2.putText(blank_img4,timestamp,(35, 40),cv2.FONT_HERSHEY_DUPLEX,fsize3,(211,211,211),1,cv2.LINE_AA)
                    cv2.putText(blank_img4,'No person detected in ' + cam4text,(140, 380),cv2.FONT_HERSHEY_DUPLEX,fsize2,(211,211,211),1,cv2.LINE_AA)
                    imgresizecam4 = cv2.resize(blank_img4,(hsizeout, vsizeout),interpolation = cv2.INTER_AREA)
        
        # Write the images to queue to be served by Flask
        
        if q_imgout_cam1.qsize() == 0:
            q_imgout_cam1.put(imgresizecam1)
        if numcams >=2:
            if q_imgout_cam2.qsize() == 0:
                q_imgout_cam2.put(imgresizecam2)
        if numcams >=3:
            if q_imgout_cam3.qsize() == 0:
                q_imgout_cam3.put(imgresizecam3)
        if numcams >=4:
            if q_imgout_cam4.qsize() == 0:
                q_imgout_cam4.put(imgresizecam4)
        
        # Color the log background based on the time since last event
        if numcams == 1:
            if (timeelapcam1 < coloraftertime):
                colorf = open(imgoutdir + "color.txt", "w")
                colorf.write("#720808;")
            else:
                colorf = open(imgoutdir + "color.txt", "w")
                colorf.write("#1f1f1f;")
        if numcams == 2:
            if (timeelapcam1 < coloraftertime) or (timeelapcam2 < coloraftertime):
                colorf = open(imgoutdir + "color.txt", "w")
                colorf.write("#720808;")
            else:
                colorf = open(imgoutdir + "color.txt", "w")
                colorf.write("#1f1f1f;")
        if numcams == 3:
            if (timeelapcam1 < coloraftertime) or (timeelapcam2 < coloraftertime) or (timeelapcam3 < coloraftertime):
                colorf = open(imgoutdir + "color.txt", "w")
                colorf.write("#720808;")
            else:
                colorf = open(imgoutdir + "color.txt", "w")
                colorf.write("#1f1f1f;")
        if numcams == 4:
            if (timeelapcam1 < coloraftertime) or (timeelapcam2 < coloraftertime) or (timeelapcam3 < coloraftertime) or (timeelapcam4 < coloraftertime):
                colorf = open(imgoutdir + "color.txt", "w")
                colorf.write("#720808;")
            else:
                colorf = open(imgoutdir + "color.txt", "w")
                colorf.write("#1f1f1f;")
                
        #For debugging - print("Time since last cam1 log:", timeelapcam1)
        #For debugging - print("Time since last cam2 log:", timeelapcam2)
        #For debugging - print("Time since last cam3 log:", timeelapcam3)
        #For debugging - print("Time since last cam4 log:", timeelapcam4)
        
        #Flush stdout and set end time
        sys.stdout.flush()
        end_time = time.time()
        
        
        # Close task so analyzation queue can recieve next image
        q_cam1_img.task_done()
        if numcams >= 2:
            q_cam2_img.task_done()
        if numcams >= 3:
            q_cam3_img.task_done()
        if numcams >= 4:
            q_cam4_img.task_done()
        
        # Here we determine how long its taken to process the frame(s) and then add some sleep to maintain the desired framerate.
        processtime = end_time-start_time
        if processtime < frametime:
            sleeptime = frametime-processtime
            time.sleep(sleeptime)
        
        #For debugging
        #final_time = time.time()
        #print("Elapsed Time:", final_time-start_time)
