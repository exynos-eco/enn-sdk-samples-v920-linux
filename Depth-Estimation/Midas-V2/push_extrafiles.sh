#/bin/bash

adb push opencv/lib* /usr/lib/
adb weston_setup.sh /data/vendor/midas/

adb push res/models/* /data/vendor/midas/
adb shell mkdir -p /data/vendor/midas/media/
adb push res/media/* /data/vendor/midas/media/
