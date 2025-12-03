#  DenseNet-121 (Image Classification)
This sample application demonstrates the execution of a converted `DenseNet-121` model using the ENN framework. The model is converted using **AI Studio 2.0** service.


## Functionality
This sample application classifies the main subject of an input image/video file. The most likely category for the entire image is determined, and the predicted label and its confidence score are displayed.

![result.jpg](result.jpg)        

---

## Location
The sample is available in the `enn-sdk-samples-v920-linux/Image-Classification/DenseNet-121` directory within the [Github](https://github.com/exynos-eco/enn-sdk-samples-v920-linux) repository.

---

## Getting Started
### Build Instructions
1.	Clone or download this sample application from the repository.
2.	Install the **EA-SDK** required for building.
3.	Set the path to the **EA-SDK** inside the `build.sh` script.
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
    /data/vendor/densenet/weston_setup.sh
    ```

2. Execute the following command on the device (via UART or ADB shell):

    ```bash
    /data/vendor/densenet/enn_sample_densenet -m /data/vendor/densenet/densenet-121_simplify_O2_MultiCore.nnc -i /data/vendor/densenet/media/image.jpg
    ```
    ##### Command-line Options
    - `-m` : Path to the input model file (.nnc format)  
            e.g. `/data/vendor/densenet/densenet-121_simplify_O2_MultiCore.nnc`

    - `-i` : Path to the input media file (image or video)  
            e.g. `/data/vendor/densenet/media/image.jpg`

---

### Batch Mode Inference
You can also perform inference on all image files in a directory.

```bash
# Runs inference on all image files inside the 'media/images' folder
/data/vendor/densenet/enn_sample_densenet -m /data/vendor/densenet/densenet-121_simplify_O2_MultiCore.nnc -i /data/vendor/densenet/media/images
```

---

### Running Other Classification Models
This sample application is designed to execute various image classification models, not just DenseNet-121.
The `/data/vendor/densenet` folder includes the following 10 models:

- googlenet  
- inception-v3  
- mnasnet05  
- regnet  
- resnet18  
- resnet50  
- resnext50  
- resnext101  
- squeezenet-1_1  
- wideresnet50 

You can run these models in the same way by simply replacing the model file path specified with the `-m` option.

```bash
#running a different model
/data/vendor/densenet/enn_sample_densenet -m /data/vendor/densenet/res/another_model_name.nnc -i /data/vendor/densenet/media/image.jpg
```



