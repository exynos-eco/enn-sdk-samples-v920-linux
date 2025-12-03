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

enum _TEST_CASE
{
    TEST_CASE_ENN_MIDAS = 1,
    TEST_CASE_OUT_OF_OPTION
} TEST_CASE;

#define DISPLAY_WIDTH 1920
#define DISPLAY_HEIGHT 1080

int g_input_width = 256;
int g_input_height = 256;
int g_input_channel = 3;

bool g_enable_log = false;

volatile sig_atomic_t keep_running = 1;

void signal_handler(int signum) {
    PRINT(GREEN "[%s] Received Ctrl+C (SIGINT), exiting... " RESET, __func__);
    keep_running = 0;
}

namespace ImageProcessor {
    float resize_scale_x;
    float resize_scale_y;

    int preprocess_frame(const cv::Mat& frame, void* buffer) {
        PRINT(GREEN "[ImageProcessor::%s] START" RESET, __func__);

        float *input_tensor = reinterpret_cast<float *>(buffer);
        cv::Mat resized, rgb;

        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

        PRINT(BLUE "[ImageProcessor::%s] Resizing from %dx%d to %dx%d..." RESET, __func__, frame.cols, frame.rows, g_input_width, g_input_height);
        cv::resize(rgb, resized, cv::Size(g_input_width, g_input_height));

        int orig_w = frame.cols;
        int orig_h = frame.rows;

        resize_scale_x = static_cast<float>(orig_w) / g_input_width;
        resize_scale_y = static_cast<float>(orig_h) / g_input_height;

        PRINT(BLUE "[ImageProcessor::%s] Normalizing pixel values to [0,1]..." RESET, __func__);
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std[3]  = {0.229f, 0.224f, 0.225f};      

        PRINT(CYAN "[ImageProcessor::%s] Rearranging data to CHW format..." RESET, __func__);

        int idx = 0;
        for (int c = 0; c < g_input_channel; ++c) {
            for (int h = 0; h < g_input_height; ++h) {
                for (int w = 0; w < g_input_width; ++w) {

                    float value = resized.at<cv::Vec3f>(h, w)[c];
                    value = (value - mean[c]) / std[c];
                    input_tensor[idx++] = value;
                }
            }
        }

        PRINT(GREEN "[ImageProcessor::%s] DONE" RESET, __func__);
        return 0;
    }

    int load_frame(cv::Mat& frame, EnnBufferPtr inBuffer, EnnModelId model_id) {
        EnnBufferInfo in_buf_info;
        enn::api::EnnGetBufferInfoByIndex(&in_buf_info, model_id, ENN_DIR_IN, 0);

        int ret = preprocess_frame(frame, inBuffer->va);

        return ret;
    }

    int postprocess(EnnBufferPtr outBuffer, EnnModelId model_id, cv::Mat& depth_color_map) {
        PRINT(YELLOW "[%s] START" RESET, __func__);

        float *output_p = reinterpret_cast<float *>(outBuffer->va);
        EnnBufferInfo out_buf_info;
        enn::api::EnnGetBufferInfoByIndex(&out_buf_info, model_id, ENN_DIR_OUT, 0);

        int channel = out_buf_info.channel;
        int h = out_buf_info.height;
        int w = out_buf_info.width;

        PRINT(BLUE "[InferenceProcessor::%s] Output shape: channel=%d, height=%d, width=%d" RESET, __func__, channel, h, w);
       
        cv::Mat depth_map(h, w, CV_32F, output_p);

        cv::Mat depth_resized;
        int orig_width = static_cast<int>(w * resize_scale_x);
        int orig_height = static_cast<int>(h * resize_scale_y);
        cv::resize(depth_map, depth_resized, cv::Size(orig_width, orig_height), 0, 0, cv::INTER_LINEAR);

        cv::Mat depth_norm;
        cv::normalize(depth_resized, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::applyColorMap(depth_norm, depth_color_map, cv::COLORMAP_INFERNO);

        PRINT(GREEN "[%s] DONE" RESET, __func__);

        return 0;
    }  
}

namespace ResultDrawer {
    cv::Mat draw(const cv::Mat& orig_image, const cv::Mat& depth_color_map) {
        PRINT(GREEN "[ResultDrawer::%s] START" RESET, __func__);

        const int canvas_w = DISPLAY_WIDTH;
        const int canvas_h = DISPLAY_HEIGHT;

        const int SPACE = 60;
        const int LABEL_H = 50;
        const int MARGIN = 150;

        const int max_img_w = (canvas_w - 2 * MARGIN - SPACE) / 2;
        const int max_img_h = canvas_h - 2 * MARGIN - LABEL_H;

        auto resize_with_aspect = [&](const cv::Mat& img) {
            int img_w = img.cols;
            int img_h = img.rows;
            float scale = std::min((float)max_img_w / img_w, (float)max_img_h / img_h);
            int new_w = static_cast<int>(img_w * scale);
            int new_h = static_cast<int>(img_h * scale);
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(new_w, new_h));
            return resized;
        };

        cv::Mat orig_resized = resize_with_aspect(orig_image);
        cv::Mat result_resized = resize_with_aspect(depth_color_map);

        int combined_w = orig_resized.cols + SPACE + result_resized.cols;
        int combined_h = std::max(orig_resized.rows, result_resized.rows) + LABEL_H;
        cv::Mat combined = cv::Mat::zeros(combined_h, combined_w, orig_image.type());

        orig_resized.copyTo(combined(cv::Rect(0, LABEL_H, orig_resized.cols, orig_resized.rows)));

        int result_x = orig_resized.cols + SPACE;
        result_resized.copyTo(combined(cv::Rect(result_x, LABEL_H, result_resized.cols, result_resized.rows)));

        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 1.0;
        int thickness = 2;
        cv::Scalar text_color(255, 255, 255);

        std::string text1 = "Original Image";
        cv::Size text_size1 = cv::getTextSize(text1, font_face, font_scale, thickness, nullptr);
        int x1 = (orig_resized.cols - text_size1.width) / 2;
        int y1 = (LABEL_H + text_size1.height) / 2;
        cv::putText(combined, text1, cv::Point(x1, y1), font_face, font_scale,
                    text_color, thickness, cv::LINE_AA);

        std::string text2 = "Midas-v2";
        cv::Size text_size2 = cv::getTextSize(text2, font_face, font_scale, thickness, nullptr);
        int x2 = result_x + (result_resized.cols - text_size2.width) / 2;
        int y2 = (LABEL_H + text_size2.height) / 2;
        cv::putText(combined, text2, cv::Point(x2, y2), font_face, font_scale,
                    text_color, thickness, cv::LINE_AA);

        cv::Mat padded_img = cv::Mat::zeros(cv::Size(canvas_w, canvas_h), orig_image.type());
        int x_offset = (canvas_w - combined.cols) / 2;
        int y_offset = (canvas_h - combined.rows) / 2;
        cv::Rect roi(x_offset, y_offset, combined.cols, combined.rows);
        combined.copyTo(padded_img(roi));

        PRINT(GREEN "[ResultDrawer::%s] DONE" RESET, __func__);
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

namespace EnnSession {
    EnnModelId model_id;
    EnnBufferPtr* buffer_set;
    uint32_t n_in_buf;
    uint32_t n_out_buf;

    bool initializeSession(const std::string& model_file) {
        PRINT(GREEN "[EnnSession::%s] START" RESET, __func__);

        PRINT(BLUE "[EnnSession::%s] Initializing ENN..." RESET, __func__);
        enn::api::EnnInitialize();

        PRINT(BLUE "[EnnSession::%s] Opening model from memory: %s" RESET, __func__, model_file.c_str());
        uint32_t model_size = enn::sample_utils::get_file_size(model_file.c_str());
        if (model_size <= 0) {
            PRINT_ERROR("Invalid File: %s", model_file.c_str());
            return false;
        }
        std::vector<char> model_vec(model_size);
        if(enn::sample_utils::import_file_to_mem(model_file.c_str(), model_vec.data()) < 0) {
            PRINT_ERROR("File read failure: %s", model_file.c_str());
            return false;
        }

        // Open model from memory and get model_id
        if (enn::api::EnnOpenModelFromMemory(model_vec.data(), model_size, &model_id)) {
            PRINT_ERROR("Open Model Failed: %s", model_file.c_str());
            return false;
        } 
        
        // after open, model data from user could be released
        model_vec.clear();

        PRINT(BLUE "[EnnSession::%s] Allocating buffers..." RESET, __func__);
        NumberOfBuffersInfo num_buffers;

        enn::api::EnnAllocateAllBuffers(model_id, &buffer_set, &num_buffers);

        EnnBufferInfo in_buf_info;
        enn::api::EnnGetBufferInfoByIndex(&in_buf_info, model_id, ENN_DIR_IN, 0);
        PRINT(BLUE "[EnnSession::%s] Model input NCHW: %d x %d x %d x %d" RESET, __func__, 
            in_buf_info.n, in_buf_info.channel, in_buf_info.height, in_buf_info.width);

        g_input_width = in_buf_info.width;
        g_input_height = in_buf_info.height;
        g_input_channel = in_buf_info.channel;

        n_in_buf = num_buffers.n_in_buf;
        n_out_buf = num_buffers.n_out_buf;
        PRINT(YELLOW "[EnnSession::%s] input buffers: [%u], output buffers: [%u]" RESET, __func__, n_in_buf, n_out_buf);

        PRINT(GREEN "[EnnSession::%s] DONE" RESET, __func__);
        return true;
    }

    void run(cv::Mat& frame, cv::Mat& depth_color_map) {
        PRINT(GREEN "[EnnSession::%s] START" RESET, __func__);

        ImageProcessor::load_frame(frame, buffer_set[0], model_id);

        PRINT(BLUE "[EnnSession::%s] EnnExcute model inference..." RESET, __func__);
        enn::api::EnnExecuteModel(model_id);

        ImageProcessor::postprocess(buffer_set[1], model_id, depth_color_map);

        PRINT(GREEN "[EnnSession::%s] DONE" RESET, __func__);
    }

    void cleanup() {
        PRINT(GREEN "[EnnSession::%s] START cleanup" RESET, __func__);

        PRINT(BLUE "[EnnSession::%s] Releasing buffers..." RESET, __func__);
        enn::api::EnnReleaseBuffers(buffer_set, n_in_buf + n_out_buf);

        PRINT(BLUE "[EnnSession::%s] Closing model..." RESET, __func__);
        enn::api::EnnCloseModel(model_id);

        PRINT(BLUE "[EnnSession::%s] Deinitializing ENN..." RESET, __func__);
        enn::api::EnnDeinitialize();

        PRINT(GREEN "[EnnSession::%s] DONE cleanup" RESET, __func__);
    }
}


int run_midas_depth_estimation(const std::string& model_file, const std::string& input_file) {
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

    if (!GStreamerDisplay::initialize()) {
        PRINT_ERROR("Failed to initialize GStreamer.");
        GStreamerDisplay::cleanup();
        return 0;
    }

    int frame_count = 0;

    if (input_type == 1) {
        cv::Mat depth_color_map;

        auto start = std::chrono::high_resolution_clock::now();
        EnnSession::run(img_frame, depth_color_map);
        auto end = std::chrono::high_resolution_clock::now();

        cv::Mat vis = ResultDrawer::draw(img_frame, depth_color_map);

        GStreamerDisplay::push_frame(vis, frame_count++);

        std::chrono::duration<double, std::milli> duration = end - start;
        PRINT(YELLOW "[%s] Inference + Postprocess Time: %.3f ms" RESET, __func__, duration.count());     
    } else {
        double total_duration_ms = 0.0;

        while (keep_running) {
            cv::Mat frame;
            if (!cap.read(frame)) break;

            cv::Mat depth_color_map;

            auto start = std::chrono::high_resolution_clock::now();
            EnnSession::run(frame, depth_color_map);
            auto end = std::chrono::high_resolution_clock::now();

            cv::Mat vis = ResultDrawer::draw(frame, depth_color_map);

            GStreamerDisplay::push_frame(vis, frame_count++);

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
        "   -i         input file path\n"
        "   -c         test case number\n\n"
        , argv[0]
    );

}

int main(int argc, char** argv ) {
    int opt;
    std::string arg_model = TEST_MODEL_PATH_MIDAS;
    std::string arg_input = TEST_INPUT_FILE_PATH;
    int test_case = TEST_CASE_ENN_MIDAS;

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
        case TEST_CASE_ENN_MIDAS:
            if(!arg_model.empty())
                run_midas_depth_estimation(arg_model, arg_input);
            break;            
        default:
            break;
    }

    return 0;
}
