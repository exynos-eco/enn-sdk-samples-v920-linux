#/bin/bash

adb push opencv/lib* /usr/lib/

adb weston_setup.sh /data/vendor/damoyolo/

adb push res/models/* /data/vendor/damoyolo/
adb shell mkdir -p /data/vendor/damoyolo/media/
adb push res/media/* /data/vendor/damoyolo/media/
