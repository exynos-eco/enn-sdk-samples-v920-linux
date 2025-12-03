#  Midas-V2 (Depth Estimation)
This sample application demonstrates the execution of a converted `Midas-V2` model using the ENN framework. The model is converted using **AI Studio 2.0** service.

## Functionality
This sample application estimates the depth of an input image or video file.
The generated depth map highlights the relative distance of objects in the scene, where closer regions appear brighter and farther regions appear darker.

![result.jpg](result.jpg)        

## Location
The sample is available in the `enn-sdk-samples-v920/Depth-Estimation/Midas-V2` directory within the [Github](https://github.com/exynos-eco/enn-sdk-samples-v920-linux) repository.

## Getting Started
### Build Instructions
1.	Clone or download this sample application from the repository.
2.	Install the **EA-SDK** required for building.
3.	Set the path to the **EA-SDK** inside the build.sh script.
4.  Connect the **SADK (V920)** board to the PC via a USB cable.
5.	On the device, run the following script to enable ADB:
    ```bash
    /home/root/adb.sh
    ```
6.	Execute the build script:
    ```bash
    ./build.sh
    ```
### Push Required Files to Device
1.	Connect the device to the PC via a USB cable.
2.	On the device, run the following script to enable ADB:
    ```bash
    /home/root/adb.sh
    ```
3.	On the PC, run the following script to push necessary files to the device:
    ```bash
    ./push_extrafiles.sh
    ```

### Run the Application
1. To enable screen output, run the following command on the device:

    ```bash
    /data/vendor/midas/weston_setup.sh
    ```

2. Execute the following command on the device (via UART or ADB shell):

    ```bash
    /data/vendor/midas/enn_sample_midas -m /data/vendor/midas/midas_v2_1_simplify_O2_MultiCore.nnc -i /data/vendor/midas/media/video.mp4
    ```
    ##### Command-line Options
    - `-m` : Path to the input model file (.nnc format)  
            e.g. `/data/vendor/midas/midas-v2-float_simplify_O2_MultiCore.nnc`

    - `-i` : Path to the input media file (image or video)  
            e.g. `/data/vendor/midas/media/video.mp4`