# PersonDetectandNotify
V2 of the Person Detection py module Ive been working on - see https://github.com/thendawg/tfhumandetectionnotify for previous documention.

More documentation will be added shortly. 

I created a new repo for this as significant changes were made to the previous module such that updating the other git didnt make since.
Everything is ran from a single py, detect.py

I also include service and shell scripts that can be used with a Linux host (py has been tested on both Windows 10 and Linux - Ubuntu 18.04)
Anything you need to config is in config.py


BASIC INSTALL (If youre using Ubuntu 18.04 I have some more helpful steps in the docs dir):


Install tensorflow, cuda (must be 10, Ive tested with 10.0 and 10.2), cudnn, nvidia driver

Install pip - use pip to install pushbullet.py, flask, and opencv-python

Install apache and php

Clone this repo to your user directory (/home/user/)

chmod +x the start.sh detect.py

Be sure to edit pdetect.service (in /servicetemplate), config.py, and start.sh (in /scripts) to match your preferences/paths (more docs on this to come, but come info can be found on the original project git here - https://github.com/thendawg/tfhumandetectionnotify)

sudo cp servicetemplate/pdetect.service /etc/systemd/system - then reload systemd (systemctl daemon-reload)

sudo cp -R html/* /var/www/html/ - youll prob have to fix perms - make sure your user can write to /var/www/html/imgdir

Make sure apache is started and php is loaded - configure additional security on Apache if you like - I recommend https and user auth unless you really trust everyone on your network :)

Start the service using sudo systemctl start pdetect.service - you can check service startup using journalctl -u pdetect.service

Once started, the process output will be logged to the processlog file in /home/user/path/logs/

2 example person detection logs are served from apache (detectlogdisplayfull.php - shows last 50 entries iirc & logdisplaygrafana.php - shows last 10 entries and is formatted for a Grafana cell)

By default these autorefresh every 1.5s - since its all text the latency/bw is extremely minimal and this provides for a fast alert - you can tweak this as you desire.

A live output is also available via flask streaming - this will be at http://IP:5000 - you can format this page however you like by editing the index.html in templates/ - I recommend reading up on flask/jinja2


This comes with no warranty and is still a work in progress. I plan to publish an installation script along with more detailed documentation that will make this much easier as well as eventually adding a web interface for configuration. In any case, referring to the doucumentation I wrote for v1 may help explain some nuances - https://github.com/thendawg/tfhumandetectionnotify


Finally, Id also eventually like to build my own app to integrate this rather than using pushbullet/http method, but Im very green to any sort of android/ios development. 




