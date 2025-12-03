#/bin/bash

adb push opencv/lib* /usr/lib/
adb weston_setup.sh /data/vendor/esrgan_x4plus/

adb push res/models/* /data/vendor/esrgan_x4plus/
adb shell mkdir -p /data/vendor/esrgan_x4plus/media/
adb push res/media/* /data/vendor/esrgan_x4plus/media/
