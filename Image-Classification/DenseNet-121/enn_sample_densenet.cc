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

#include "include/class_labels.h" 

#include <filesystem>
namespace fs = std::filesystem;

enum _TEST_CASE
{
    TEST_CASE_ENN_DENSENET = 1,
    TEST_CASE_OUT_OF_OPTION
} TEST_CASE;

typedef struct _PRED_RESULT {
    int index;
    float score;
    float prob;
} PRED_RESULT;

#define DISPLAY_WIDTH 1920
#define DISPLAY_HEIGHT 1080

int g_input_width = 224;
int g_input_height = 224;
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

    int postprocess(EnnBatchInferenceBuffer& outBuffer, EnnModelId model_id, std::vector<PRED_RESULT>& pred_results) {
        PRINT(YELLOW "[%s] START" RESET, __func__);

        float *output_p = reinterpret_cast<float *>(outBuffer.va);
        EnnBufferInfo out_buf_info;
        enn::api::EnnGetBufferInfoByIndex(&out_buf_info, model_id, ENN_DIR_OUT, 0);

        int channel = out_buf_info.channel;
        int h = (out_buf_info.height > 0) ? out_buf_info.height : 1;
        int w = (out_buf_info.width > 0) ? out_buf_info.width : 1;
        int size = channel * h * w;

        PRINT(BLUE "[InferenceProcessor::%s] Output shape: channel=%d, height=%d, width=%d" RESET,
            __func__, channel, h, w);

        pred_results.clear();
        pred_results.reserve(size);

        float max_val = output_p[0];
        for (int i = 1; i < size; i++) {
            if (output_p[i] > max_val) max_val = output_p[i];
        }

        std::vector<float> exp_vals(size);
        float sum_exp = 0.0f;
        for (int i = 0; i < size; i++) {
            exp_vals[i] = std::exp(output_p[i] - max_val);
            sum_exp += exp_vals[i];
        }

        for (int i = 0; i < size; i++) {
            float prob = (exp_vals[i] / sum_exp) * 100.0f;
            pred_results.push_back({i, output_p[i], prob});
        }

        std::partial_sort(pred_results.begin(), pred_results.begin() + std::min(5, size), pred_results.end(),
                        [](const PRED_RESULT &a, const PRED_RESULT &b) {
                            return a.score > b.score;
                        });

        int topk = std::min(5, size);
        PRINT(GREEN "Top-%d Predictions:" RESET, topk);
        for (int i = 0; i < topk; i++) {
            int idx = pred_results[i].index;
            float prob = pred_results[i].prob;
            const char* label = (idx < class_labels_count) ? class_labels[idx] : "Unknown";
            PRINT("  %d: %s (%.2f%%)", idx, label, prob);
        }

        PRINT(GREEN "[%s] DONE" RESET, __func__);
        return 0;
    }
}

namespace ResultDrawer {
    cv::Mat draw(const cv::Mat& image, std::vector<PRED_RESULT>& pred_results) {
        PRINT(GREEN "[ResultDrawer::%s] START" RESET, __func__);
        cv::Mat img = image.clone();

        PRINT(BLUE "[ResultDrawer::%s] Centering image in %dx%d canvas" RESET, __func__, DISPLAY_WIDTH, DISPLAY_HEIGHT);

        const int canvas_w = DISPLAY_WIDTH;
        const int canvas_h = DISPLAY_HEIGHT;

        const int max_img_w = 1024;
        const int max_img_h = 768;

        int img_w = img.cols;
        int img_h = img.rows;

        float scale = std::min((float)max_img_w / img_w, (float)max_img_h / img_h);
        int new_w = static_cast<int>(img_w * scale);
        int new_h = static_cast<int>(img_h * scale);

        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(new_w, new_h));

        cv::Mat padded_img = cv::Mat::zeros(cv::Size(canvas_w, canvas_h), img.type());
        int x_offset = (canvas_w - new_w) / 2;
        int y_offset = (canvas_h - new_h) / 2;
        cv::Rect roi(x_offset, y_offset, new_w, new_h);
        resized_img.copyTo(padded_img(roi));

        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.9;
        int thickness = 2;
        int line_height = 30;

        cv::putText(padded_img, "Top-5 Predictions", cv::Point(10, 30),
                    font_face, font_scale, cv::Scalar(0, 255, 255), thickness);

        for (size_t i = 0; i < pred_results.size() && i < 5; i++) {
            int idx = pred_results[i].index;
            float prob = pred_results[i].prob;
            const char* label = (idx < class_labels_count) ? class_labels[idx] : "Unknown";

            std::ostringstream text;
            text << std::fixed << std::setprecision(2)
                << i + 1 << ": " << label << " (" << prob << "%)";

            cv::Point org(10, 30 + (i + 1) * line_height);
            cv::putText(padded_img, text.str(), org, font_face, 0.7,
                        cv::Scalar(0, 255, 0), 2);
        }         

        cv::imwrite("result_image.jpg", padded_img);

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

#define BATCH_SIZE 10

namespace EnnSession {
    EnnModelId model_id;
    uint32_t n_in_buf;
    uint32_t n_out_buf;

    EnnBatchInferenceBuffer* in_buffers  = nullptr;
    EnnBatchInferenceBuffer* out_buffers = nullptr;
    EnnBatchBuffers* batchBufsForN       = nullptr;

    bool initializeSession(const std::string& model_file, uint32_t batch_size = BATCH_SIZE) {
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
        if (enn::api::EnnPrepareBatch(model_id, &num_buffers)) return false;

        EnnBufferInfo in_buf_info;
        enn::api::EnnGetBufferInfoByIndex(&in_buf_info, model_id, ENN_DIR_IN, 0);
        PRINT(BLUE "[EnnSession::%s] Model input NCHW: %d x %d x %d x %d" RESET, __func__, 
            in_buf_info.n, in_buf_info.channel, in_buf_info.height, in_buf_info.width);

        g_input_width = in_buf_info.width;
        g_input_height = in_buf_info.height;
        g_input_channel = in_buf_info.channel;

        n_in_buf  = num_buffers.n_in_buf;
        n_out_buf = num_buffers.n_out_buf;

        size_t in_size = g_input_width * g_input_height * g_input_channel * sizeof(float);
        size_t out_size = 1000 * sizeof(float);

        in_buffers  = new EnnBatchInferenceBuffer[batch_size * n_in_buf];
        out_buffers = new EnnBatchInferenceBuffer[batch_size * n_out_buf];
        batchBufsForN = new EnnBatchBuffers[batch_size];

        enn::sample_utils::prepare_user_batch_buffers(in_size, batch_size, n_in_buf, in_buffers, true);
        enn::sample_utils::prepare_user_batch_buffers(out_size, batch_size, n_out_buf, out_buffers, true);

        for (uint32_t i = 0; i < batch_size; i++) {
            batchBufsForN[i].inputs = &in_buffers[i * n_in_buf];
            batchBufsForN[i].outputs = &out_buffers[i * n_out_buf];
        }

        EnnBatchBufferInfo batchInfo;
        batchInfo.batch_buffers = batchBufsForN;
        enn::api::EnnUpdateBatchInfo(model_id, &batchInfo, batch_size, true);

        PRINT(GREEN "[EnnSession::%s] DONE" RESET, __func__);
        return true;
    }

    void runBatch(std::vector<cv::Mat>& frames, std::vector<std::vector<PRED_RESULT>>& pred_results, uint32_t batch_size = BATCH_SIZE) {
        PRINT(GREEN "[EnnSession::%s] START Batch" RESET, __func__);

        if (frames.size() != batch_size) {
            PRINT_ERROR("[EnnSession::%s] frames.size (%zu) != batch_size (%u)", __func__, frames.size(), batch_size);
            return;
        }

        size_t in_size = g_input_width * g_input_height * g_input_channel * sizeof(float);
        uint8_t* in_ptr = static_cast<uint8_t*>(in_buffers[0].va);

        for (uint32_t i = 0; i < batch_size; i++) {
            ImageProcessor::preprocess_frame(frames[i], in_ptr + in_size * i);
        }

        if (enn::api::EnnExecuteModelBatch(model_id)) {
            PRINT_ERROR("[EnnSession::%s] EnnExecuteModelBatch failed", __func__);
            return;
        }

        size_t out_size = 1000 * sizeof(float);
        uint8_t* out_ptr = static_cast<uint8_t*>(out_buffers[0].va);

        pred_results.resize(batch_size);
        for (uint32_t i = 0; i < batch_size; i++) {
            EnnBatchInferenceBuffer buf;
            buf.va   = out_ptr + out_size * i;
            buf.size = out_size;
            ImageProcessor::postprocess(buf, model_id, pred_results[i]);
        }

        PRINT(GREEN "[EnnSession::%s] DONE Batch" RESET, __func__);
    }

    void cleanup() {
        PRINT(GREEN "[EnnSession::%s] START cleanup" RESET, __func__);

        delete[] in_buffers;
        delete[] out_buffers;
        delete[] batchBufsForN;

        PRINT(BLUE "[EnnSession::%s] Closing model..." RESET, __func__);
        enn::api::EnnCloseModel(model_id);

        PRINT(BLUE "[EnnSession::%s] Deinitializing ENN..." RESET, __func__);
        enn::api::EnnDeinitialize();

        PRINT(GREEN "[EnnSession::%s] DONE cleanup" RESET, __func__);
    }
}

int run_densenet_classification_batch_dir(const std::string& model_file, const std::string& input_dir) {
    PRINT(GREEN "[%s] START " RESET, __func__);

    setenv("WAYLAND_DISPLAY", "wayland-1", 1);
    signal(SIGINT, signal_handler);

    std::vector<std::string> jpg_files;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg") {
                jpg_files.push_back(entry.path().string());
            }
        }
    }

    if (jpg_files.empty()) {
        PRINT(RED "[%s] No jpg files found in directory: %s" RESET, __func__, input_dir.c_str());
        return -1;
    }

    PRINT(CYAN "[%s] Found %zu jpg files in directory: %s" RESET, 
          __func__, jpg_files.size(), input_dir.c_str());


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
    
    double total_duration_ms = 0.0;
    const int batch_size = 10;
    std::vector<cv::Mat> frame_batch;
    size_t file_idx = 0;
    int frame_count = 0;

    while (keep_running && file_idx < jpg_files.size()) {
        frame_batch.clear();
        size_t before_idx = file_idx;

        for (int i = 0; i < batch_size && file_idx < jpg_files.size(); ++i, ++file_idx) {
            cv::Mat frame = cv::imread(jpg_files[file_idx]);
            if (!frame.empty()) {
                frame_batch.push_back(frame);
            } else {
                PRINT(RED "[%s] Failed to read image: %s" RESET, 
                      __func__, jpg_files[file_idx].c_str());
            }
        }

        PRINT(CYAN "[%s] Preparing batch: %zu images", __func__, frame_batch.size());
        for (size_t i = 0; i < frame_batch.size(); i++) {
            PRINT(CYAN "  [%zu] %s (empty=%d, size=%dx%d)", 
                i, jpg_files[file_idx - frame_batch.size() + i].c_str(),
                frame_batch[i].empty(),
                frame_batch[i].cols, frame_batch[i].rows);
        }

        size_t read_count = file_idx - before_idx;
        PRINT(BLUE "[%s] Read %zu images for this batch (total processed: %zu/%zu)" RESET,
              __func__, read_count, file_idx, jpg_files.size());

        if (frame_batch.empty()) break;

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<PRED_RESULT>> pred_results_batch;
        EnnSession::runBatch(frame_batch, pred_results_batch, frame_batch.size());
        auto end = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < frame_batch.size(); i++) {
            cv::Mat vis = ResultDrawer::draw(frame_batch[i], pred_results_batch[i]);
            GStreamerDisplay::push_frame(vis, frame_count++);
            std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        }

        std::chrono::duration<double, std::milli> duration = end - start;
        total_duration_ms += duration.count();
        PRINT(YELLOW "[%s] Batch Inference + Postprocess Time (batch=%ld): %.3f ms (avg=%.3f ms/frame)" RESET,
              __func__, frame_batch.size(), duration.count(), duration.count() / frame_batch.size());
    }

    PRINT(GREEN "[%s] Finished. Total frames: %d, Average per frame: %.3f ms" RESET, 
          __func__, frame_count, total_duration_ms / (frame_count ? frame_count : 1));

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


int run_densenet_classification_file(const std::string& model_file, const std::string& input_file) {
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
        std::vector<cv::Mat> frame_batch;
        std::vector<std::vector<PRED_RESULT>> pred_results_batch;

        frame_batch.push_back(img_frame);

        auto start = std::chrono::high_resolution_clock::now();
        EnnSession::runBatch(frame_batch, pred_results_batch, frame_batch.size());
        auto end = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < static_cast<int>(frame_batch.size()); i++) {
            cv::Mat vis = ResultDrawer::draw(frame_batch[i], pred_results_batch[i]);
            GStreamerDisplay::push_frame(vis, frame_count++);
        }

        std::chrono::duration<double, std::milli> duration = end - start;
        PRINT(YELLOW "[%s] Inference + Postprocess Time: %.3f ms" RESET, __func__, duration.count());       
        
    } else {
        double total_duration_ms = 0.0;
        const int batch_size = BATCH_SIZE;
        std::vector<cv::Mat> frame_batch;

        while (keep_running) {
            cv::Mat frame;

            for (int i = 0; i < batch_size; ++i) {
                if (!cap.read(frame)) {
                    break;
                }
                frame_batch.push_back(frame);
            }

            if (!frame_batch.size()) break;

            auto start = std::chrono::high_resolution_clock::now();

            std::vector<std::vector<PRED_RESULT>> pred_results_batch;
            EnnSession::runBatch(frame_batch, pred_results_batch, frame_batch.size());

            auto end = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < static_cast<int>(frame_batch.size()); i++) {
                cv::Mat vis = ResultDrawer::draw(frame_batch[i], pred_results_batch[i]);
                GStreamerDisplay::push_frame(vis, frame_count++);
            }

            std::chrono::duration<double, std::milli> duration = end - start;
            total_duration_ms += duration.count();
            PRINT(YELLOW "[%s] Batch Inference + Postprocess Time (batch=%ld): %.3f ms (avg=%.3f ms/frame)" RESET,
                __func__, frame_batch.size(), duration.count(), duration.count() / frame_batch.size());

            frame_batch.clear();
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
    std::string arg_model = TEST_MODEL_PATH_DENSENET;
    std::string arg_input = TEST_INPUT_FILE_PATH;
    int test_case = TEST_CASE_ENN_DENSENET;

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
        case TEST_CASE_ENN_DENSENET:
            if(!arg_model.empty()) {
                if (fs::exists(arg_input) && fs::is_directory(arg_input)) {
                    run_densenet_classification_batch_dir(arg_model, arg_input);
                } else {
                    run_densenet_classification_file(arg_model, arg_input);
                }
            }
            break;            
        default:
            break;
    }

    return 0;
}
