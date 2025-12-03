#!/bin/bash
unset LD_LIBRARY_PATH
SDK_PATH="/opt/ea-sdk/4.0.14/environment-setup-cortexa76-poky-linux"
source ${SDK_PATH}

rm -rf build/
mkdir build
cd build
cmake .. 
make
cd ..

echo "------------------------------------------"
echo "push the executable to /data/vendor/midas/"
adb shell mkdir -p /data/vendor/midas/
adb push build/bin/* /data/vendor/midas/

echo "DONE!"
