#!/bin/bash

rm -f /home/user/path/logs/lastprocesslog.log
mv /home/user/path/logs/processlog.log /home/user/path/logs/lastprocesslog.log
rm -f /home/user/path/logs/processlog.log
/usr/bin/python3 /home/user/path/detect.py >> /home/user/path/logs/processlog.log
