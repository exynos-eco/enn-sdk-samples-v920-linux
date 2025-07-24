# ENN SDK Samples v920 Linux

## Introduction
|Category|Sample Name| Description|
|-------------|-------------|----------------------------------------------------------------------------------------------------------|
|Pose Estimation|[HRnet](#hrnet)| Sample Linux application to demonstrate the execution of `pose_hrnet_w48_384x288` model with ENN SDK|
|Object Dectection|[DAMO-YOLO](#damo-yolo)| Sample Linux application to demonstrate the execution of `damoyolo_tinynasL25_S_460` model with ENN SDK|

## Liunx Samples
This section provides an overview of Linux sample applications.
Each sample application entry provides the details of the functionality of the sample application, its location, and instructions for running it.

***

### HRNet
This document explains Linux Sample Application operates using the [HRNet](https://github.com/HRNet/deep-high-resolution-net.pytorch) model optimized for Exynos hardware.

#### Functionality
The application accepts input from an image/video file. Then, it detects the points of a person and overlays the points and edges of a person.

#### Location
The sample is available in the `enn-sdk-samples-v920-linux/Pose-Estimation/HRNet` directory within the [Github](https://github.com/exynos-eco/enn-sdk-samples-v920-linux) repository, where you can find detailed instructions on prerequisites, build steps, and how to run the sample.

***
### DAMO-YOLO
This document explains Linux Sample Application operates using the [DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO) model optimized for Exynos hardware.

#### Functionality
This sample application identifies objects in an input image/video file.
The detected objects are highlighted with bounding boxes, and the label and score of each object are displayed.

#### Location
The sample is available in the `enn-sdk-samples-v920-linux/Object-Detection/DAMO-YOLO` directory within the [Github](https://github.com/exynos-eco/enn-sdk-samples-v920-linux) repository, where you can find detailed instructions on prerequisites, build steps, and how to run the sample.
