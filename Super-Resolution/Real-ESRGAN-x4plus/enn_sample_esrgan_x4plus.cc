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

#include <filesystem>
namespace fs = std::filesystem;

enum _TEST_CASE
{
    TEST_CASE_ENN_ESRGAN_X4PLUS = 1,
    TEST_CASE_OUT_OF_OPTION
} TEST_CASE;

typedef struct _PRED_RESULT {
    int index;
    float score;
    float prob;
} PRED_RESULT;

#define INPUT_WIDTH 128
#define INPUT_HEIGHT 128
#define INPUT_CHANNEL 3

#define DISPLAY_WIDTH 1920
#define DISPLAY_HEIGHT 1080

bool g_enable_log = false;

volatile sig_atomic_t keep_running = 1;

struct ImageCropInfo {
    std::string filename;
    int start_x;
    int start_y;
};

// temp
std::unordered_map<std::string, std::pair<int,int>> g_crop_coords = {
    {"image.jpg", {550, 400}},
    {"image_1.jpg", {470, 180}},
    {"image_2.jpg", {70, 230}},
    {"image_3.jpg", {380, 260}}
};

void signal_handler(int signum) {
    PRINT(GREEN "[%s] Received Ctrl+C (SIGINT), exiting... " RESET, __func__);
    keep_running = 0;
}

namespace ImageProcessor {
    int preprocess_frame(const cv::Mat& frame, void* buffer, cv::Mat& crop_frame, const std::string& file_path) { 
        PRINT(GREEN "[ImageProcessor::%s] START" RESET, __func__);

        float *input_tensor = reinterpret_cast<float *>(buffer);
        cv::Mat rgb;

        // BGR -> RGB
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

        int crop_w = INPUT_WIDTH;
        int crop_h = INPUT_HEIGHT;

        int start_x = 0;
        int start_y = 0;

        std::string filename = std::filesystem::path(file_path).filename().string();
        auto it = g_crop_coords.find(filename);
        if (it != g_crop_coords.end()) {
            start_x = it->second.first;
            start_y = it->second.second;
        } else {
            PRINT_ERROR("Crop coordinates not found for file: %s", file_path.c_str());
            start_x = 0;
            start_y = 0;
        }

        cv::Mat cropped_rgb = rgb(cv::Rect(start_x, start_y, crop_w, crop_h)).clone();
        cv::cvtColor(cropped_rgb, crop_frame, cv::COLOR_RGB2BGR);

        cv::Mat cropped_float;
        cropped_rgb.convertTo(cropped_float, CV_32FC3, 1.0 / 255.0);

        int idx = 0;
        for (int c = 0; c < INPUT_CHANNEL; ++c)
            for (int h = 0; h < INPUT_HEIGHT; ++h)
                for (int w = 0; w < INPUT_WIDTH; ++w)
                    input_tensor[idx++] = cropped_float.at<cv::Vec3f>(h, w)[c];

        PRINT(GREEN "[ImageProcessor::%s] DONE" RESET, __func__);
        return 0;
    }    
    
    int load_frame(cv::Mat& frame, EnnBufferPtr inBuffer, EnnModelId model_id, cv::Mat& resize_frame, const std::string& file) {
        EnnBufferInfo in_buf_info;
        enn::api::EnnGetBufferInfoByIndex(&in_buf_info, model_id, ENN_DIR_IN, 0);

        int ret = preprocess_frame(frame, inBuffer->va, resize_frame, file);

        return ret;
    }

    int postprocess(EnnBufferPtr outBuffer, EnnModelId model_id, cv::Mat& upscale_frame) {
        PRINT(YELLOW "[%s] START" RESET, __func__);

        float *output_p = reinterpret_cast<float *>(outBuffer->va);
        EnnBufferInfo out_buf_info;
        enn::api::EnnGetBufferInfoByIndex(&out_buf_info, model_id, ENN_DIR_OUT, 0);

        int channel = out_buf_info.channel;
        int h = (out_buf_info.height > 0) ? out_buf_info.height : 1;
        int w = (out_buf_info.width > 0) ? out_buf_info.width : 1;

        PRINT(BLUE "[InferenceProcessor::%s] Output shape: channel=%d, height=%d, width=%d" RESET,
            __func__, channel, h, w);

        cv::Mat temp_frame(h, w, CV_8UC3);
        for (int c = 0; c < channel; c++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int idx = c * h * w + y * w + x;
                    float val = output_p[idx];
                    val = std::max(0.0f, std::min(1.0f, val));
                    unsigned char pix = static_cast<unsigned char>(val * 255.0f);

                    int cv_c = (c == 0) ? 2 : (c == 2) ? 0 : 1;
                    temp_frame.at<cv::Vec3b>(y, x)[cv_c] = pix;
                }
            }
        }

        upscale_frame = temp_frame.clone();

        PRINT(GREEN "[%s] DONE" RESET, __func__);
        return 0;
    }
}

namespace ResultDrawer {
    cv::Mat draw(const cv::Mat& orig_image,
                const cv::Mat& crop_image,
                const cv::Mat& upscale_crop_image) {
        PRINT(GREEN "[ResultDrawer::%s] START" RESET, __func__);

        const int canvas_w = DISPLAY_WIDTH;
        const int canvas_h = DISPLAY_HEIGHT;

        const int SPACE = 100;
        const int LABEL_H = 50;
        const int MARGIN = 10;

        const int max_left_w = (canvas_w - 2 * MARGIN - SPACE) / 2;
        const int max_left_h = canvas_h - 2 * MARGIN - LABEL_H;

        const int max_right_w = (canvas_w - 2 * MARGIN - SPACE) / 2;
        const int max_right_h = (canvas_h - 2 * MARGIN - LABEL_H) / 2 - SPACE / 2;

        auto resize_with_aspect = [](const cv::Mat& img, int max_w, int max_h) {
            int img_w = img.cols;
            int img_h = img.rows;
            float scale = std::min((float)max_w / img_w, (float)max_h / img_h);
            int new_w = static_cast<int>(img_w * scale);
            int new_h = static_cast<int>(img_h * scale);
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(new_w, new_h));
            return resized;
        };

        cv::Mat orig_resized = resize_with_aspect(orig_image, max_left_w, max_left_h);
        cv::Mat crop_resized = resize_with_aspect(crop_image, max_right_w, max_right_h);
        cv::Mat upscale_resized = resize_with_aspect(upscale_crop_image, max_right_w, max_right_h);

        int combined_w = orig_resized.cols + SPACE + std::max(crop_resized.cols, upscale_resized.cols);
        int combined_h = std::max(orig_resized.rows, crop_resized.rows + upscale_resized.rows + SPACE);

        cv::Mat combined = cv::Mat::zeros(combined_h + LABEL_H, combined_w, orig_image.type());

        int orig_y = LABEL_H + (combined_h - orig_resized.rows) / 2;
        orig_resized.copyTo(combined(cv::Rect(0, orig_y, orig_resized.cols, orig_resized.rows)));

        int right_x = orig_resized.cols + SPACE;
        crop_resized.copyTo(combined(cv::Rect(right_x, LABEL_H, crop_resized.cols, crop_resized.rows)));
        int bottom_y = LABEL_H + crop_resized.rows + SPACE / 2;
        upscale_resized.copyTo(combined(cv::Rect(right_x, bottom_y, upscale_resized.cols, upscale_resized.rows)));

        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 1.0;
        int thickness = 2;
        cv::Scalar text_color(255, 255, 255);

        std::string text1 = "Original Image";
        cv::Size text_size1 = cv::getTextSize(text1, font_face, font_scale, thickness, nullptr);
        int x1 = (orig_resized.cols - text_size1.width) / 2;
        int y1 = orig_y - text_size1.height ;
        cv::putText(combined, text1, cv::Point(x1, y1), font_face, font_scale,
                    text_color, thickness, cv::LINE_AA);

        std::string text2 = "Crop";
        cv::Size text_size2 = cv::getTextSize(text2, font_face, font_scale, thickness, nullptr);
        int x2 = right_x + (crop_resized.cols - text_size2.width) / 2;
        int y2 = LABEL_H / 2 + text_size2.height / 2;
        cv::putText(combined, text2, cv::Point(x2, y2), font_face, font_scale,
                    text_color, thickness, cv::LINE_AA);

        std::string text3 = "Upscaled Crop";
        cv::Size text_size3 = cv::getTextSize(text3, font_face, font_scale, thickness, nullptr);
        int x3 = right_x + (upscale_resized.cols - text_size3.width) / 2;
        int y3 = bottom_y - text_size3.height + 5;
        cv::putText(combined, text3, cv::Point(x3, y3), font_face, font_scale,
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

        PRINT(BLUE "[EnnSession::%s] Opening model: %s" RESET, __func__, model_file.c_str());
        enn::api::EnnOpenModel(model_file.c_str(), &model_id);

        PRINT(BLUE "[EnnSession::%s] Allocating buffers..." RESET, __func__);
        NumberOfBuffersInfo num_buffers;

        enn::api::EnnAllocateAllBuffers(model_id, &buffer_set, &num_buffers);

        n_in_buf = num_buffers.n_in_buf;
        n_out_buf = num_buffers.n_out_buf;
        PRINT(YELLOW "[EnnSession::%s] input buffers: [%u], output buffers: [%u]" RESET, __func__, n_in_buf, n_out_buf);

        PRINT(GREEN "[EnnSession::%s] DONE" RESET, __func__);
        return true;
    }

    void run(cv::Mat& frame, cv::Mat& resize_frame, cv::Mat& upscale_frame, const std::string& file) {
        PRINT(GREEN "[EnnSession::%s] START" RESET, __func__);

        ImageProcessor::load_frame(frame, buffer_set[0], model_id, resize_frame, file);

        PRINT(BLUE "[EnnSession::%s] EnnExcute model inference..." RESET, __func__);
        enn::api::EnnExecuteModel(model_id);

        ImageProcessor::postprocess(buffer_set[1], model_id, upscale_frame);

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

int run_esrgan_x4plus_upscale(const std::string& model_file, const std::string& input_path) {
    PRINT(GREEN "[%s] START " RESET, __func__);

    setenv("WAYLAND_DISPLAY", "wayland-1", 1);
    signal(SIGINT, signal_handler);

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

    std::vector<std::string> files_to_process;
    namespace fs = std::filesystem;

    fs::path path(input_path);
    if (fs::is_regular_file(path)) {
        files_to_process.push_back(input_path);
    } else if (fs::is_directory(path)) {
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                    files_to_process.push_back(entry.path().string());
                }
            }
        }
        if (files_to_process.empty()) {
            PRINT_ERROR("No jpg files found in directory: %s", input_path.c_str());
            GStreamerDisplay::cleanup();
            EnnSession::cleanup();
            return 0;
        }
    } else {
        PRINT_ERROR("Invalid input path: %s", input_path.c_str());
        GStreamerDisplay::cleanup();
        EnnSession::cleanup();
        return 0;
    }

    int frame_count = 0;

    for (const auto& file : files_to_process) {
        cv::Mat frame = cv::imread(file);
        if (frame.empty()) {
            PRINT_ERROR("Failed to open file: %s", file.c_str());
            continue;
        }

        PRINT(BLUE "[%s] Processing file: %s" RESET, __func__, file.c_str());

        cv::Mat upscale_frame = frame.clone();
        cv::Mat resize_frame = frame.clone();

        auto start = std::chrono::high_resolution_clock::now();
        EnnSession::run(frame, resize_frame, upscale_frame, file);
        auto end = std::chrono::high_resolution_clock::now();

        // cv::Mat vis = ResultDrawer::draw(resize_frame, upscale_frame);
        cv::Mat vis = ResultDrawer::draw(frame, resize_frame, upscale_frame);
        GStreamerDisplay::push_frame(vis, frame_count++);

        std::chrono::duration<double, std::milli> duration = end - start;
        PRINT(YELLOW "[%s] Inference + Postprocess Time: %.3f ms" RESET, __func__, duration.count());

        std::this_thread::sleep_for(std::chrono::seconds(3));

        if (!keep_running) break;
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
    std::string arg_model = TEST_MODEL_PATH_ESRGAN_X4PLUS;
    std::string arg_input = TEST_INPUT_FILE_PATH;
    int test_case = TEST_CASE_ENN_ESRGAN_X4PLUS;

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
        case TEST_CASE_ENN_ESRGAN_X4PLUS:
            if(!arg_model.empty()) {
                run_esrgan_x4plus_upscale(arg_model, arg_input);
            }
            break;            
        default:
            break;
    }

    return 0;
}
