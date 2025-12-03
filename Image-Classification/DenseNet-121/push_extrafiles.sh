#/bin/bash

adb push opencv/lib* /usr/lib/
adb weston_setup.sh /data/vendor/densenet/

adb push res/models/* /data/vendor/densenet/
adb shell mkdir -p /data/vendor/densenet/media/
adb push res/media/* /data/vendor/densenet/media/
