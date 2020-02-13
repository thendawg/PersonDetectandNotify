# Set Number of Cameras
numcams = 1
# Path to the frozen graph that will be used for inference, see README for more info.
model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
# RTSP Stream URL's to capture video 
capurl = "rtsp://user:pw@ip://"
capurl2 = 0
capurl3 = 0
capurl4 = 0
# Score threshold for person to be detected
threshold = 0.95
# Seperate threshold for PushBullet alerts to be sent
thresholdpb = 0.97
# How long (in seconds) the detection log will stay red after a person is last detected.
coloraftertime = 45
# How long the script will wait (after the last detection event) before creating a new directory and log entry - this prevents us from getting a bunch of log entries if someone is persistently detected coming in and out of frame.
timebetweenevents = 60
# How long each loop should take, take 1/fps will give you this number, so for instance I target 4fps, so I use .250
frametime = .250
# The below entries are related to PushBullet, pbenabled is 0 (disabled) or 1 (enabled), the rest is self explanatory.
pbenabled = 0
pbapikey = 'apiKey'
pbchannelname = 'channelname'
# URL to your webserver for the links included in the log and pushbullet (if enabled) - this should point to the same location (when accessed via browser) that the imgoutdir points to below.
url = 'https://www.yourserverurlorip.com/imgdir/'
# Filters - see documentation to understand how to configure these - leaving default values will effectively disable them - future webui will make this much easier.
# Cam 1
yfilter1cam1 = 0
xfilter1cam1 = 0
yfilter2cam1 = 9999
xfilter2cam1 = 9999
# Cam 2
yfilter1cam2 = 0
xfilter1cam2 = 0
yfilter2cam2 = 9999
xfilter2cam2 = 9999
# Cam 3
yfilter1cam3 = 0
xfilter1cam3 = 0
yfilter2cam3 = 9999
xfilter2cam3 = 9999
# Cam 4
yfilter1cam4 = 0
xfilter1cam4 = 0
yfilter2cam4 = 9999
xfilter2cam4 = 9999
# Variables below set the names for each camera that are used for the log entries, push notifications, and directory/image save names. As these are used in urls, do not put spaces or special characters other than "_", use only a-z/A-Z/_/0-9
cam1text =  'cam1'
cam2text = 'cam2'
cam3text =  'cam3'
cam4text = 'cam4'
# Directory where images are stored. Must include trailing slash (/) - Recommend using the full (absolute) path
imgoutdir = '/var/www/html/imgdir/'
# Font size for use with timestamp and bounding box score, if using higher res, you will need to increase this.
fontsize = 0.95
# The script will resize the images before writing them out, you can set this value here, keep in mind, the larger this gets, the longer the *blocking* write operation gets which can be problematic.
imgwidth = 590
imgheight = 332
# Output log path. This is the path to the log including the log name - Recommend using the full (absolute) path
logpath = '/var/www/html/imgdir/detect.log'
# Preview Mode (If True will display all images via flask, if false will display a black box unless a person is detected - saves bandwidth and resources. Font size is for the non-detect text.
previewmode = True
# Preview Mode Settings
fontsize2 = 2.0
fontsize3 = 1.25
previewtime = 10
