set(CMAKE_C_COMPILER aarch64-poky-linux-gcc)
set(CMAKE_CXX_COMPILER aarch64-poky-linux-g++)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

cmake_minimum_required(VERSION 3.16)
project(kittv2_linux)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(top_dir ${CMAKE_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wall -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -frtti")
set(ANDROID_STL "gnustl_shared")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)

include_directories(${top_dir}/include)

set(imported_lib "${top_dir}/lib64_linux/libenn_public_api_cpp.so")
add_library(enn_public_api_cpp_ SHARED IMPORTED GLOBAL)
set_target_properties(enn_public_api_cpp_ PROPERTIES IMPORTED_LOCATION "${imported_lib}")

set(imported_lib "${top_dir}/lib64_linux/libOpenCL.so")
add_library(OpenCL SHARED IMPORTED GLOBAL)
set_target_properties(OpenCL PROPERTIES IMPORTED_LOCATION ${imported_lib})
add_definitions(-DCL_TARGET_OPENCL_VERSION=120)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)
include_directories(${GST_INCLUDE_DIRS})
link_directories(${GST_LIBRARY_DIRS})

set(OPENCV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/opencv")
foreach(lib core imgcodecs imgproc videoio)
    add_library(opencv_${lib} SHARED IMPORTED GLOBAL)
    set_target_properties(opencv_${lib} PROPERTIES
        IMPORTED_LOCATION ${OPENCV_ROOT}/libopencv_${lib}.so.4.1
        INTERFACE_INCLUDE_DIRECTORIES ${OPENCV_ROOT}/include
    )
endforeach()

set(target_name enn_sample_midas)
add_executable(${target_name} enn_sample_midas.cc)

target_link_libraries(${target_name} enn_public_api_cpp_ OpenCL ${GST_LIBRARIES} opencv_core opencv_imgcodecs opencv_imgproc opencv_videoio)                       