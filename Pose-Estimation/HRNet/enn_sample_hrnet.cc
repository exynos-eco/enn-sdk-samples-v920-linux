/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system or translated into any human or
 * computer language in any form by any means, electronic, mechanical, manual or
 * otherwise or disclosed to third parties without the express written permission of
 * Samsung Electronics.
 */

#include <thread>
#include "include/enn_api-public.hpp"
#include "include/enn_sample_utils.hpp"

#include <opencv2/opencv.hpp>

#include <gst/gst.h>
#include <gst/app/app.h>

#include <CL/cl.h>
#include <CL/cl_ext.h>

enum _TEST_CASE
{
    TEST_CASE_HRNET_POSE_INFERENCE = 1,
    TEST_CASE_OUT_OF_OPTION
} TEST_CASE;

#define INPUT_WIDTH 288
#define INPUT_HEIGHT 384
#define HEATMAP_WIDTH 72
#define HEATMAP_HEIGHT 96
#define DISPLAY_WIDTH 1920
#define DISPLAY_HEIGHT 1080

bool g_enable_log = false;

volatile sig_atomic_t keep_running = 1;

void signal_handler(int signum) {
    PRINT(GREEN "[%s] Received Ctrl+C (SIGINT), exiting... " RESET, __func__);
    keep_running = 0;
}

namespace ImagePreprocessor {
    int preprocess_frame(const cv::Mat& frame, void* buffer) {
        PRINT(GREEN "[ImagePreprocessor::%s] START" RESET, __func__);

        float *input_tensor = reinterpret_cast<float *>(buffer);
        cv::Mat resized, rgb;

        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

        PRINT(BLUE "[ImagePreprocessor::%s] Resizing from %dx%d to %dx%d..." RESET, __func__, frame.cols, frame.rows, INPUT_WIDTH, INPUT_HEIGHT);
        cv::resize(rgb, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));

        PRINT(BLUE "[ImagePreprocessor::%s] Normalizing pixel values to [0,1]..." RESET, __func__);
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

        PRINT(CYAN "[ImagePreprocessor::%s] Rearranging data to CHW format..." RESET, __func__);
        int idx = 0;
        for (int c = 0; c < 3; ++c)
            for (int h = 0; h < INPUT_HEIGHT; ++h)
                for (int w = 0; w < INPUT_WIDTH; ++w)
                    input_tensor[idx++] = resized.at<cv::Vec3f>(h, w)[c];

        PRINT(GREEN "[ImagePreprocessor::%s] DONE" RESET, __func__);
        return 0;
    }

    int load_frame(cv::Mat& frame, EnnBufferPtr inBuffer, EnnModelId model_id) {
        EnnBufferInfo in_buf_info;
        enn::api::EnnGetBufferInfoByIndex(&in_buf_info, model_id, ENN_DIR_IN, 0);

        int ret = preprocess_frame(frame, inBuffer->va);

        return ret;
    }
}

namespace KeypointDrawer {
    const std::vector<std::tuple<int, int, cv::Scalar>> SKELETON = {
        {0, 1, cv::Scalar(127, 2, 240)}, {0, 2, cv::Scalar(127, 2, 240)},
        {1, 3, cv::Scalar(127, 2, 240)}, {2, 4, cv::Scalar(127, 2, 240)},
        {5, 6, cv::Scalar(142, 209, 169)}, {5, 7, cv::Scalar(142, 209, 169)}, {7, 9, cv::Scalar(142, 209, 169)},
        {6, 8, cv::Scalar(0, 255, 255)}, {8, 10, cv::Scalar(0, 255, 255)},
        {5, 11, cv::Scalar(240, 176, 0)}, {11, 13, cv::Scalar(240, 176, 0)}, {13, 15, cv::Scalar(240, 176, 0)},
        {6, 12, cv::Scalar(243, 176, 252)}, {12, 14, cv::Scalar(243, 176, 252)}, {14, 16, cv::Scalar(243, 176, 252)}
    };

    cv::Mat draw(const cv::Mat& image, const std::vector<std::pair<int, int>>& keypoints) {
        PRINT(GREEN "[KeypointDrawer::%s] START" RESET, __func__);

        float scale_x = static_cast<float>(image.cols) / HEATMAP_WIDTH;
        float scale_y = static_cast<float>(image.rows) / HEATMAP_HEIGHT;
        PRINT(YELLOW "[KeypointDrawer::%s] x_scale: %f , y_scale: %f" RESET, __func__, scale_x, scale_y);

        int base_dim = std::min(image.rows, image.cols);
        int radius = std::max(2, static_cast<int>(base_dim * 0.008f));
        int line_thickness = std::max(2, static_cast<int>(base_dim * 0.005f));

        PRINT(BLUE "[KeypointDrawer::%s] Drawing skeleton and keypoints (radius=%d, thickness=%d)..." RESET, __func__, radius, line_thickness);

        cv::Mat vis_img = image.clone();
        std::vector<cv::Point> scaled_points;

        for (size_t i = 0; i < keypoints.size(); ++i) {
            int px = static_cast<int>(keypoints[i].first * scale_x);
            int py = static_cast<int>(keypoints[i].second * scale_y);
            scaled_points.emplace_back(px, py);
            // PRINT(CYAN "[KeypointDrawer::%s] keypoint %zu: scaled (x=%d, y=%d)" RESET, __func__, i, px, py);
        }

        int num_points = static_cast<int>(scaled_points.size());
        for (const auto& [start_idx, end_idx, color] : SKELETON) {
            if (start_idx < num_points && end_idx < num_points) {
                cv::line(vis_img, scaled_points[start_idx], scaled_points[end_idx], color, line_thickness, cv::LINE_AA);
            }
        }

        for (const auto& pt : scaled_points) {
            cv::circle(vis_img, pt, radius, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }

        cv::imwrite("/tmp/nnc_hrnet_output.jpg", vis_img);
        PRINT(BLUE "[KeypointDrawer::%s] result image saved at : nnc_yolo_output.jpg \n" RESET, __func__);

        PRINT(BLUE "[KeypointDrawer::%s] Centering image in %dx%d canvas" RESET, __func__, DISPLAY_WIDTH, DISPLAY_HEIGHT);

        const int canvas_w = DISPLAY_WIDTH;
        const int canvas_h = DISPLAY_HEIGHT;

        int img_w = vis_img.cols;
        int img_h = vis_img.rows;

        float scale = std::min((float)canvas_w / img_w, (float)canvas_h / img_h);
        int new_w = static_cast<int>(img_w * scale);
        int new_h = static_cast<int>(img_h * scale);

        cv::Mat resized_img;
        cv::resize(vis_img, resized_img, cv::Size(new_w, new_h));

        cv::Mat padded_img = cv::Mat::zeros(cv::Size(canvas_w, canvas_h), vis_img.type());
        int x_offset = (canvas_w - new_w) / 2;
        int y_offset = (canvas_h - new_h) / 2;
        cv::Rect roi(x_offset, y_offset, new_w, new_h);
        resized_img.copyTo(padded_img(roi));

        PRINT(GREEN "[KeypointDrawer::%s] DONE" RESET, __func__);
        return padded_img;
    }
}

namespace GStreamerDisplay {
    GstElement* pipeline = nullptr;
    GstElement* appsrc = nullptr;

    bool initialize() {
        PRINT(GREEN "[GStreamerDisplay::%s] START" RESET, __func__ );
        gst_init(nullptr, nullptr);

        GError* error = nullptr;
        PRINT(BLUE "[GStreamerDisplay::%s] Creating GStreamer pipeline..." RESET, __func__);
        pipeline = gst_parse_launch(
            "appsrc name=mysource is-live=true block=true format=time ! "
            "videoconvert ! "
            "videoscale ! "
            "video/x-raw,width=1920,height=1080 ! "
            "waylandsink sync=false", &error);

        if (error) {
            PRINT_ERROR("[GStreamerDisplay::%s] Failed to create pipeline: %s", __func__, error->message);
            return false;
        }

        PRINT(BLUE "[GStreamerDisplay::%s] Getting appsrc element..." RESET, __func__);
        appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");

        PRINT(BLUE "[GStreamerDisplay::%s] Setting caps..." RESET, __func__);
        GstCaps* caps = gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            "width", G_TYPE_INT, DISPLAY_WIDTH,
            "height", G_TYPE_INT, DISPLAY_HEIGHT,
            "framerate", GST_TYPE_FRACTION, 30, 1, NULL);
        gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
        gst_caps_unref(caps);

        PRINT(BLUE "[GStreamerDisplay::%s] Setting pipeline to PLAYING..." RESET, __func__);
        gst_element_set_state(pipeline, GST_STATE_PLAYING);

        PRINT(GREEN "[GStreamerDisplay::%s] SUCCESS" RESET, __func__);
        return true;
    }

    void push_frame(const cv::Mat& frame, int frame_count) {
        PRINT(CYAN "[GStreamerDisplay::%s] Pushing frame %d..." RESET, __func__, frame_count);

        GstBuffer* buffer = gst_buffer_new_allocate(NULL, frame.total() * frame.elemSize(), NULL);
        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_WRITE);
        memcpy(map.data, frame.data, frame.total() * frame.elemSize());
        gst_buffer_unmap(buffer, &map);

        GST_BUFFER_PTS(buffer) = gst_util_uint64_scale(frame_count, GST_SECOND, 30);
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, 30);

        GstFlowReturn ret;
        g_signal_emit_by_name(appsrc, "push-buffer", buffer, &ret);
        gst_buffer_unref(buffer);

        PRINT(CYAN "[GStreamerDisplay::%s] Frame %d pushed. Flow return: %d" RESET, __func__, frame_count, ret);
    }

    void cleanup() {
        PRINT(YELLOW "[GStreamerDisplay::%s] START cleanup" RESET, __func__);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
        PRINT(YELLOW "[GStreamerDisplay::%s] DONE cleanup" RESET, __func__);
    }    
}

namespace OpenCLExecutor {
    enum PLATFORM { ARM = 0, AMD, UNKNOWN };

    typedef cl_mem(CL_API_CALL* clImportMemoryARM_fn)(
            cl_context,     // context
            cl_mem_flags,   // flags
            const cl_import_properties_arm *, // properties
            void *,         // memory
            size_t,         // size
            cl_int *        // errcode_ret
    ) CL_EXT_SUFFIX__VERSION_1_0;

    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue command_queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;             // copy_data_8b kernel
    cl_kernel postprocess_kernel = nullptr; // postprocess kernel
    clImportMemoryARM_fn pfn_clImportMemoryARM = nullptr;

    void printProgramBuildInfo(cl_program program, cl_device_id device) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        if (logSize > 1) {
            char *log = new char[logSize];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
            std::cout << "=== Build log: ===\n" << log << std::endl;
            delete[] log;
        }
    }

    size_t alignTo(size_t src, size_t alignment) {
        return (src + alignment - 1) / alignment * alignment;
    }

    bool initialize() {
        cl_int err;
        cl_uint num_platforms = 0;
        err = clGetPlatformIDs(0, NULL, &num_platforms);
        if (err != CL_SUCCESS) return false;

        err = clGetPlatformIDs(num_platforms, &platform, 0);
        if (err != CL_SUCCESS) return false;

        pfn_clImportMemoryARM = (clImportMemoryARM_fn)
            clGetExtensionFunctionAddressForPlatform(platform, "clImportMemoryARM");
        if (!pfn_clImportMemoryARM) {
            std::cerr << "clImportMemoryARM not supported.\n";
            return false;
        }

        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, &device, NULL);
        if (err != CL_SUCCESS) return false;

        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        if (!context || err != CL_SUCCESS) return false;

        command_queue = clCreateCommandQueue(context, device, 0, &err);
        if (!command_queue || err != CL_SUCCESS) return false;

        std::string source_str = R"CLC(
            __kernel void copy_data_8b(__global char* in, __global char* out) {
                int gid = get_global_id(0);
                out[gid] = in[gid];
            }

            __kernel void postprocess_kernel(__global const float* input,
                                             __global int2* keypoints,
                                             const int channel,
                                             const int height,
                                             const int width) {
                int ch = get_global_id(0);
                if (ch >= channel) return;

                float max_val = -1.0f;
                int max_idx = 0;
                int hw = height * width;

                for (int i = 0; i < hw; ++i) {
                    int idx = ch * hw + i;
                    float val = input[idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = i;
                    }
                }

                int y = max_idx / width;
                int x = max_idx % width;
                keypoints[ch] = (int2)(x, y);
            }        
        )CLC";

        const char* source = source_str.c_str();
        size_t source_size = source_str.size();
        program = clCreateProgramWithSource(context, 1, &source, &source_size, &err);
        if (!program || err != CL_SUCCESS) return false;

        const char options[] = "-cl-std=CL1.2 -cl-mad-enable";
        err = clBuildProgram(program, 1, &device, options, NULL, NULL);
        if (err != CL_SUCCESS) {
            printProgramBuildInfo(program, device);
            return false;
        }

        kernel = clCreateKernel(program, "copy_data_8b", &err);
        postprocess_kernel = clCreateKernel(program, "postprocess_kernel", &err);
        return kernel && postprocess_kernel && err == CL_SUCCESS;
    }

    cl_mem allocBufferWithImportMemory(const uint32_t &bytes, int &fd) {
        cl_int err = CL_SUCCESS;
        const int offset = 0;
        const cl_import_properties_arm mem_properties[] = {
            CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM,
            CL_IMPORT_TYPE_PROTECTED_ARM, CL_FALSE, 0
        };

        cl_mem buffer = pfn_clImportMemoryARM(context, CL_MEM_READ_WRITE,
                mem_properties, const_cast<int*>(&fd), offset + bytes, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "clImportMemoryARM error: " << err << std::endl;
            return nullptr;
        }

        cl_buffer_region region = {offset, bytes};
        cl_mem sub_buffer = clCreateSubBuffer(buffer, CL_MEM_READ_WRITE,
                CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "clCreateSubBuffer error: " << err << std::endl;
            return nullptr;
        }
        return sub_buffer;
    }

    cl_mem allocBufferWithHostPtr(const uint32_t &bytes, void* data) {
        cl_int err;
        cl_mem buffer = clCreateBuffer(context,
                CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, bytes, data, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "clCreateBuffer error: " << err << std::endl;
            return nullptr;
        }
        return buffer;
    }

    int run(int input_fd, size_t input_size,
            std::vector<std::pair<int, int>>& keypoints) {
        using namespace std::chrono;
        auto start = high_resolution_clock::now();
        cl_int err;

        const int channel = 17;
        const int height = HEATMAP_HEIGHT;
        const int width = HEATMAP_WIDTH;

        PRINT(GREEN "[OpenCLExecutor::%s] START" RESET, __func__);

        cl_mem input_buffer = allocBufferWithImportMemory(input_size, input_fd);
        if (!input_buffer) {
            PRINT_ERROR("[OpenCLExecutor::%s] Failed to import input buffer", __func__);
            return -1;
        }

        cl_mem keypoints_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                            sizeof(cl_int2) * channel, nullptr, &err);
        if (err != CL_SUCCESS) {
            PRINT_ERROR("[OpenCLExecutor::%s] Failed to create keypoints buffer", __func__);
            clReleaseMemObject(input_buffer);
            return -1;
        }

        err  = clSetKernelArg(postprocess_kernel, 0, sizeof(cl_mem), &input_buffer);
        err |= clSetKernelArg(postprocess_kernel, 1, sizeof(cl_mem), &keypoints_cl);
        err |= clSetKernelArg(postprocess_kernel, 2, sizeof(int), &channel);
        err |= clSetKernelArg(postprocess_kernel, 3, sizeof(int), &height);
        err |= clSetKernelArg(postprocess_kernel, 4, sizeof(int), &width);
        if (err != CL_SUCCESS) {
            PRINT_ERROR("[OpenCLExecutor::%s] Failed to set kernel args", __func__);
            clReleaseMemObject(input_buffer);
            clReleaseMemObject(keypoints_cl);
            return -1;
        }

        size_t global_work_size = channel;
        err = clEnqueueNDRangeKernel(command_queue, postprocess_kernel, 1, NULL,
                                    &global_work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            PRINT_ERROR("[OpenCLExecutor::%s] Kernel launch failed", __func__);
            clReleaseMemObject(input_buffer);
            clReleaseMemObject(keypoints_cl);
            return -1;
        }

        clFinish(command_queue);

        cl_int2 keypoints_buf[channel];
        err = clEnqueueReadBuffer(command_queue, keypoints_cl, CL_TRUE, 0,
                                sizeof(keypoints_buf), keypoints_buf, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            PRINT_ERROR("[OpenCLExecutor::%s] Read keypoints failed", __func__);
            clReleaseMemObject(input_buffer);
            clReleaseMemObject(keypoints_cl);
            return -1;
        }

        keypoints.clear();
        for (int i = 0; i < channel; ++i) {
            keypoints.emplace_back(keypoints_buf[i].s[0], keypoints_buf[i].s[1]);
            PRINT(YELLOW "[OpenCLExecutor::%s] keypoint %d: (x=%d, y=%d)" RESET, __func__, i,
                keypoints_buf[i].s[0], keypoints_buf[i].s[1]);
        }

        clReleaseMemObject(input_buffer);
        clReleaseMemObject(keypoints_cl);

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        
        PRINT(GREEN "[OpenCLExecutor::%s] DONE, elapsed = %ld us" RESET, __func__, duration);
        return 0;
    }
    
    void cleanup() {
        if (kernel) clReleaseKernel(kernel);
        if (postprocess_kernel) clReleaseKernel(postprocess_kernel);
        if (program) clReleaseProgram(program);
        if (command_queue) clReleaseCommandQueue(command_queue);
        if (context) clReleaseContext(context);        
    }
}

namespace EnnSession {
    EnnModelId model_id;
    EnnBufferPtr* npu_in_buffers;
    EnnBufferPtr* npu_out_buffers;
    size_t* npu_out_buffers_size;
    uint32_t n_in_buf;
    uint32_t n_out_buf;
    enn::sample_utils::DmaAllocator dma_allocator;
    uint32_t *npu_in_fds;
    uint32_t *npu_out_fds;

    bool initializeSession(const std::string& model_file) {
        PRINT(GREEN "[EnnSession::%s] START" RESET, __func__);

        if (enn::api::EnnInitialize()) return false;

        if (enn::api::EnnOpenModel(model_file.c_str(), &model_id)) return false;

        NumberOfBuffersInfo buf_info;
        if (enn::api::EnnGetBuffersInfo(&buf_info, model_id)) return false;

        n_in_buf = buf_info.n_in_buf;
        n_out_buf = buf_info.n_out_buf;

        npu_in_buffers = new EnnBufferPtr[n_in_buf];
        npu_out_buffers = new EnnBufferPtr[n_out_buf];
        npu_out_buffers_size = new size_t[n_out_buf];

        EnnBufferInfo tmp_buf_info;

        npu_in_fds = new uint32_t[n_in_buf];
        npu_out_fds = new uint32_t[n_out_buf];

        for (uint32_t i = 0; i < n_in_buf; ++i) {
            if (enn::api::EnnGetBufferInfoByIndex(&tmp_buf_info, model_id, ENN_DIR_IN, i)) return false;

            // create dma buffer for npu input
            int fd = dma_allocator.allocate_dma_buffer(tmp_buf_info.size);
            if (fd < 0) return false;

            npu_in_fds[i] = fd;

            // import the dma buffer as ENN Input buffer
            if (enn::api::EnnCreateBufferFromFd(&(npu_in_buffers[i]), fd, tmp_buf_info.size)) return false;
            if (enn::api::EnnSetBufferByIndex(model_id, ENN_DIR_IN, i, npu_in_buffers[i])) return false;
        }

        for (uint32_t i = 0; i < n_out_buf; ++i) {
            if (enn::api::EnnGetBufferInfoByIndex(&tmp_buf_info, model_id, ENN_DIR_OUT, i)) return false;

            // create dma buffer for npu output & gpu input
            int fd = dma_allocator.allocate_dma_buffer(tmp_buf_info.size);
            if (fd < 0) return false;

            npu_out_fds[i] = fd;

            // import the dma buffer as ENN output buffer
            if (enn::api::EnnCreateBufferFromFd(&(npu_out_buffers[i]), fd, tmp_buf_info.size)) return false;
            if (enn::api::EnnSetBufferByIndex(model_id, ENN_DIR_OUT, i, npu_out_buffers[i])) return false;

            // NPU out buffer size
            npu_out_buffers_size[i] = tmp_buf_info.size;
        }    

        return true;
    }

    void run(cv::Mat& frame, std::vector<std::pair<int, int>>& keypoints) {
        ImagePreprocessor::load_frame(frame, npu_in_buffers[0], model_id);

        enn::api::EnnBufferCommit(model_id);
        enn::api::EnnExecuteModel(model_id);
        
        for (uint32_t i = 0; i < n_out_buf; ++i) {
            int output_fd;
            enn::api::EnnGetFileDescriptorFromEnnBuffer(npu_out_buffers[i], &output_fd);
            int ret = OpenCLExecutor::run(output_fd, npu_out_buffers_size[i], keypoints);                 
            if (ret != 0) {
                PRINT_ERROR("OpenCLExecutor run failed for output[%d]", i);
            }
        }
    }

    void cleanup() {
        for (uint32_t i = 0; i < n_in_buf; ++i) {
            dma_allocator.release_dma_buffer(npu_in_fds[i]);
        }
        for (uint32_t i = 0; i < n_out_buf; ++i) {
            dma_allocator.release_dma_buffer(npu_out_fds[i]);
        }

        delete[] npu_in_buffers;
        delete[] npu_out_buffers;
        delete[] npu_out_buffers_size;

        enn::api::EnnCloseModel(model_id);
        enn::api::EnnDeinitialize();
    }
}

int run_hrnet_pose_estimation(const std::string& model_file, const std::string& input_file) {
    PRINT(GREEN "[%s] START " RESET, __func__);

    setenv("WAYLAND_DISPLAY", "wayland-1", 1);
    signal(SIGINT, signal_handler);

    int input_type = 0; // 1: image, 2: video
    cv::Mat img_frame = cv::imread(input_file);
    cv::VideoCapture cap(input_file);

    if (!img_frame.empty()) {
        input_type = 1;
    } else if (cap.isOpened()) {
        input_type = 2;
    } else {
        PRINT_ERROR("Failed to open input file: %s", input_file.c_str());
        return 0;
    }
    
    PRINT(BLUE "[%s] Successfully opened input file: %s" RESET, __func__, input_file.c_str());

    if (!EnnSession::initializeSession(model_file)) {
        PRINT_ERROR("Failed to initialize ENN Session.");
        EnnSession::cleanup();
        return 0;
    }

    if (!OpenCLExecutor::initialize()) {
        PRINT_ERROR("Failed to initialize OpenCLExecutor");
        OpenCLExecutor::cleanup();
        return 0;
    }    

    if (!GStreamerDisplay::initialize()) {
        PRINT_ERROR("Failed to initialize GStreamer.");
        GStreamerDisplay::cleanup();
        return 0;
    }

    int frame_count = 0;
    cv::Mat last_frame;

    if (input_type == 1) {
        std::vector<std::pair<int, int>> keypoints;

        auto start = std::chrono::high_resolution_clock::now();
        EnnSession::run(img_frame, keypoints);
        auto end = std::chrono::high_resolution_clock::now();
        
        cv::Mat vis = KeypointDrawer::draw(img_frame, keypoints);
        last_frame = vis.clone();

        GStreamerDisplay::push_frame(vis, frame_count++);
        
        std::chrono::duration<double, std::milli> duration = end - start;
        PRINT(YELLOW "[%s] Inference + Postprocess Time: %.3f ms" RESET, __func__, duration.count());     
    } else {
        double total_duration_ms = 0.0;

        while (keep_running) {
            cv::Mat frame;
            if (!cap.read(frame)) break;

            std::vector<std::pair<int, int>> keypoints;

            auto start = std::chrono::high_resolution_clock::now();
            EnnSession::run(frame, keypoints);
            auto end = std::chrono::high_resolution_clock::now();
            
            cv::Mat vis = KeypointDrawer::draw(frame, keypoints);

            GStreamerDisplay::push_frame(vis, frame_count++);
            last_frame = vis.clone();

            std::chrono::duration<double, std::milli> duration = end - start;
            total_duration_ms += duration.count();

            PRINT(YELLOW "[%s] Inference + Postprocess Time: %.3f ms" RESET, __func__, duration.count());
        }

        if (frame_count > 0) {
            double avg_time = total_duration_ms / frame_count;
            PRINT(YELLOW "[%s] Average Inference Time per Frame: %.3f ms (Total: %.1f ms, Frames: %d)" RESET,
                  __func__, avg_time, total_duration_ms, frame_count);
        } else {
            PRINT_ERROR("No valid frames processed.");
        }
    }

    int tick_count = 0;
    while (keep_running) {
        if (tick_count % 10 == 0) {
            PRINT(CYAN "[main] Waiting for Ctrl+C to exit..." RESET);
        }        

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        tick_count++;
    }

    GStreamerDisplay::cleanup();
    OpenCLExecutor::cleanup();
    EnnSession::cleanup();
    PRINT(GREEN "[%s] Finished" RESET, __func__);
    return 0;
}

void print_help(char** argv) {
    printf(
        "\n"
        "Usage: %s [OPTION]\n"
        "[REQUIRED]\n"
        "   -m         modle file path\n"
        "   -i         input file path\n\n"
        "   -c         test case number\n"
        "              1: [sample] run hrnet with ENN API\n"
        "              [default : 1]\n\n"
        , argv[0]
    );

}

int main(int argc, char** argv ) {
    int opt;
    std::string arg_model = TEST_MODEL_PATH_POSE_HRNET;
    std::string arg_input = TEST_INPUT_FILE_PATH;
    int test_case = TEST_CASE_HRNET_POSE_INFERENCE;

    while ((opt = getopt(argc, argv, "m:i:c:l:")) != -1) {
        switch (opt) {
            case 'm':
                arg_model = optarg;
                break;
            case 'i':
                arg_input = optarg;
                break;
            case 'c':
                test_case = atoi(optarg);
                break;           
            case 'l':
                g_enable_log = atoi(optarg) != 0;
                break;                
            default:
                print_help(argv);
                return 1;
        }
    }

    switch(test_case)
    {
        case TEST_CASE_HRNET_POSE_INFERENCE:
            run_hrnet_pose_estimation(arg_model, arg_input);
            break;

        default:
            break;
    }

    return 0;
}
