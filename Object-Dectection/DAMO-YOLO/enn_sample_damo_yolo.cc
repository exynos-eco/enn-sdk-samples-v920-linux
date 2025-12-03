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
    TEST_CASE_ENN_DAMO_YOLO = 1,
    TEST_CASE_OUT_OF_OPTION
} TEST_CASE;

typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
} DL_RESULT;

#define INPUT_WIDTH 640
#define INPUT_HEIGHT 640

#define DISPLAY_WIDTH 1920
#define DISPLAY_HEIGHT 1080

bool g_enable_log = false;

volatile sig_atomic_t keep_running = 1;

void signal_handler(int signum) {
    PRINT(GREEN "[%s] Received Ctrl+C (SIGINT), exiting... " RESET, __func__);
    keep_running = 0;
}

namespace ImageProcessor {
    float resize_scale_x;
    float resize_scale_y;

    int preprocess_frame(cv::Mat& input_img, const cv::Size& input_size, void* buffer)
    {
        int orig_w = input_img.cols;
        int orig_h = input_img.rows;
        int target_w = input_size.width;   // e.g. 640
        int target_h = input_size.height;  // e.g. 640

        // 1. Resize to target size directly (no aspect ratio preservation, no padding)
        cv::Mat resized;
        cv::resize(input_img, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

        // 2. Save resize scale (used later to rescale output bboxes back to original image)
        resize_scale_x = static_cast<float>(orig_w) / target_w;
        resize_scale_y = static_cast<float>(orig_h) / target_h;

        // 3. Convert resized image to float32 CHW (no normalization)
        float* output_tensor = reinterpret_cast<float *>(buffer);
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < target_h; ++h) {
                for (int w = 0; w < target_w; ++w) {
                    float val = static_cast<float>(resized.at<cv::Vec3b>(h, w)[c]);
                    output_tensor[c * target_h * target_w + h * target_w + w] = val;
                }
            }
        }

        return 0;
    }

    int load_frame(cv::Mat& frame, EnnBufferPtr inBuffer, EnnModelId model_id) {
        EnnBufferInfo in_buf_info;
        enn::api::EnnGetBufferInfoByIndex(&in_buf_info, model_id, ENN_DIR_IN, 0);

        int ret = preprocess_frame(frame, {(int)(in_buf_info.height), (int)(in_buf_info.width)}, inBuffer->va);

        return ret;
    }

    int postprocess(EnnBufferPtr buffers, EnnBufferPtr buffers_2, EnnModelId model_id, std::vector<DL_RESULT>& oResult) {
        PRINT(YELLOW "[%s] START" RESET, __func__);

        float* output_p    = reinterpret_cast<float*>(buffers->va);       // [1, 8400, 80]
        float* output_p_2  = reinterpret_cast<float*>(buffers_2->va);     // [1, 8400, 4]

        EnnBufferInfo out_buf_info, out_buf_info_2;

        float conf_thresh = 0.45f;
        float iou_thresh  = 0.5f;   

        if (enn::api::EnnGetBufferInfoByIndex(&out_buf_info, model_id, ENN_DIR_OUT, 0) ||
            enn::api::EnnGetBufferInfoByIndex(&out_buf_info_2, model_id, ENN_DIR_OUT, 1)) {
            PRINT_ERROR("[%s] Failed to get output buffer info", __func__);
            return -1;
        }

        int num_classes = out_buf_info.height;   // 80
        int num_boxes   = out_buf_info.channel;  // 8400

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        PRINT(MAGENTA "[%s] resize_info.scale [%f][%f]", __func__, resize_scale_x, resize_scale_y);

        for (int i = 0; i < num_boxes; ++i) {
            float* class_scores = output_p + i * num_classes;

            float max_score = 0.f;
            int max_class_id = -1;

            for (int c = 0; c < num_classes; ++c) {
                float score = class_scores[c];
                if (score > max_score) {
                    max_score = score;
                    max_class_id = c;
                }
            }

            if (max_score > conf_thresh) {
                float x_min = output_p_2[i * 4 + 0] * resize_scale_x;
                float y_min = output_p_2[i * 4 + 1] * resize_scale_y;
                float x_max  = output_p_2[i * 4 + 2] * resize_scale_x;
                float y_max  = output_p_2[i * 4 + 3] * resize_scale_y;

                int left   = static_cast<int>(x_min);
                int top    = static_cast<int>(y_min);
                int width  = static_cast<int>(x_max - x_min);
                int height = static_cast<int>(y_max - y_min);


                if (width > 0 && height > 0) {
                    boxes.emplace_back(left, top, width, height);
                    confidences.emplace_back(max_score);
                    class_ids.emplace_back(max_class_id);
                }
            }
        }

        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, conf_thresh, iou_thresh, nms_result);

        for (int idx : nms_result) {
            DL_RESULT result;
            result.classId    = class_ids[idx];
            result.confidence = confidences[idx];
            result.box        = boxes[idx];
            oResult.push_back(result);
        }

        PRINT(GREEN "[%s] DONE: %zu objects detected after NMS" RESET, __func__, oResult.size());
        return 0;
    }  
}

namespace DetectListDrawer {
    std::vector<std::string> classes = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    cv::Mat draw(const cv::Mat& image, const std::vector<DL_RESULT>& oResult) {
        PRINT(GREEN "[DetectListDrawer::%s] START" RESET, __func__);
        cv::Mat img = image.clone();

        if(oResult.size() > 0 ) {
            PRINT(BLUE "[DetectListDrawer::%s] Draw Result Rects : %zu" RESET, __func__, oResult.size());

            int base_dim = std::min(img.cols, img.rows);
            float scale_factor = base_dim / 640.0f;

            int box_thickness = std::max(1, static_cast<int>(2 * scale_factor));
            double font_scale = std::max(0.5, 0.5 * scale_factor);
            int font_thickness = std::max(1, static_cast<int>(1 * scale_factor));

            for (auto& re : oResult)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                cv::rectangle(img, re.box, color, box_thickness);

                float confidence = floor(100 * re.confidence) / 100;
                std::string label = classes[re.classId] + " " +
                                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                int baseline = 0;
                cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
                int label_width = text_size.width;
                int label_height = text_size.height + baseline;

                int top_y = std::max(re.box.y - label_height - 5, 0);

                cv::rectangle(
                    img,
                    cv::Point(re.box.x, top_y),
                    cv::Point(re.box.x + label_width, top_y + label_height + 5),
                    color,
                    cv::FILLED
                );

                cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, top_y + label_height),
                    cv::FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    cv::Scalar(0, 0, 0),
                    font_thickness
                );
            }
        }
        else
        {
            PRINT(BLUE "[KeypointDrawer::%s] No Detection Result\n" RESET, __func__);
        } 


        PRINT(BLUE "[KeypointDrawer::%s] Centering image in %dx%d canvas" RESET, __func__, DISPLAY_WIDTH, DISPLAY_HEIGHT);

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

#define SESSION_NUM 4

namespace EnnSession {
    EnnModelId model_id;

    EnnBufferPtr *buffer_set[SESSION_NUM];
    NumberOfBuffersInfo buffers_info[SESSION_NUM];
    uint32_t n_in_buf[SESSION_NUM];
    uint32_t n_out_buf[SESSION_NUM];

    bool initializeSession(const std::string& model_file) {
        PRINT(GREEN "[EnnSession::%s] START" RESET, __func__);

        PRINT(BLUE "[EnnSession::%s] Initializing ENN..." RESET, __func__);
        enn::api::EnnInitialize();

        PRINT(BLUE "[EnnSession::%s] Opening model: %s" RESET, __func__, model_file.c_str());
        enn::api::EnnOpenModel(model_file.c_str(), &model_id);

        for (uint32_t session_id = 0; session_id < SESSION_NUM; ++session_id) {
            PRINT(BLUE "[EnnSession::%s]  Session[%d] allocating buffers..." RESET, __func__, session_id);
            enn::api::EnnAllocateAllBuffers(model_id, &buffer_set[session_id], &buffers_info[session_id], session_id, false);
            enn::api::EnnBufferCommit(model_id, session_id);

            n_in_buf[session_id] = buffers_info[session_id].n_in_buf;
            n_out_buf[session_id] = buffers_info[session_id].n_out_buf;
        }

        PRINT(GREEN "[EnnSession::%s] DONE" RESET, __func__);
        return true;
    }

    void run(cv::Mat& frame, std::vector<DL_RESULT>& detect_list) {
        PRINT(GREEN "[EnnSession::%s] START" RESET, __func__);
        
        uint32_t session_id = 0;

        PRINT(BLUE "[EnnSession::%s] EnnExcute model inference..." RESET, __func__);
        ImageProcessor::load_frame(frame, buffer_set[session_id][0], model_id);

        enn::api::EnnExecuteModel(model_id);

        ImageProcessor::postprocess(buffer_set[session_id][1], buffer_set[session_id][2], model_id, detect_list);

        PRINT(GREEN "[EnnSession::%s] DONE" RESET, __func__);
    }

    void run_multi_session(std::array<cv::Mat, SESSION_NUM>& frames, std::array<std::vector<DL_RESULT>, SESSION_NUM>& detect_lists, uint32_t session_n) {
        PRINT(GREEN "[EnnSession::%s] START" RESET, __func__);

        PRINT(BLUE "[EnnSession::%s] EnnExcute model inference..." RESET, __func__);
        for (uint32_t session_id = 0; session_id < session_n; ++session_id) {
            ImageProcessor::load_frame(frames[session_id], buffer_set[session_id][0], model_id);
            enn::api::EnnExecuteModelAsync(model_id, session_id);
        }

        for (uint32_t session_id = 0; session_id < session_n; ++session_id) {
            if (enn::api::EnnExecuteModelWait(model_id, session_id)) {
                PRINT_ERROR("Failed while waiting for model");
            }
        }        

        for (uint32_t session_id = 0; session_id < session_n; ++session_id) {
            ImageProcessor::postprocess(buffer_set[session_id][1], buffer_set[session_id][2], model_id, detect_lists[session_id]);
        }

        PRINT(GREEN "[EnnSession::%s] DONE" RESET, __func__);
    }

    void cleanup() {
        PRINT(GREEN "[EnnSession::%s] START cleanup" RESET, __func__);

        PRINT(BLUE "[EnnSession::%s] Releasing buffers..." RESET, __func__);
        for (uint32_t session_id = 0; session_id < SESSION_NUM; ++session_id) {
            enn::api::EnnReleaseBuffers(buffer_set[session_id], n_in_buf[session_id] + n_out_buf[session_id]);
        }
        PRINT(BLUE "[EnnSession::%s] Closing model..." RESET, __func__);
        enn::api::EnnCloseModel(model_id);

        PRINT(BLUE "[EnnSession::%s] Deinitializing ENN..." RESET, __func__);
        enn::api::EnnDeinitialize();

        PRINT(GREEN "[EnnSession::%s] DONE cleanup" RESET, __func__);
    }
}


int run_damo_yolo_detection(const std::string& model_file, const std::string& input_file) {
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
        std::vector<DL_RESULT> detect_list; 

        auto start = std::chrono::high_resolution_clock::now();
        EnnSession::run(img_frame, detect_list);
        auto end = std::chrono::high_resolution_clock::now();

        cv::Mat vis = DetectListDrawer::draw(img_frame, detect_list);

        GStreamerDisplay::push_frame(vis, frame_count++);

        std::chrono::duration<double, std::milli> duration = end - start;
        PRINT(YELLOW "[%s] Inference + Postprocess Time: %.3f ms" RESET, __func__, duration.count());   
    } else {
        double total_duration_ms = 0.0;

        while (keep_running) {
            std::array<cv::Mat, SESSION_NUM> frames;
            std::array<std::vector<DL_RESULT>, SESSION_NUM> detect_lists;

            int num_read_frames = 0;
            for (int i = 0; i < SESSION_NUM; ++i) {
                if (!cap.read(frames[i])) {
                    break;
                }
                 num_read_frames++;
            }

            if (!num_read_frames) break;

            auto start = std::chrono::high_resolution_clock::now();
            EnnSession::run_multi_session(frames, detect_lists, num_read_frames);
            auto end = std::chrono::high_resolution_clock::now();


            for (int i = 0; i < num_read_frames; ++i) {
                cv::Mat vis = DetectListDrawer::draw(frames[i], detect_lists[i]);
                GStreamerDisplay::push_frame(vis, frame_count++);
            }

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
        "   -c         test case number\n"
        "              1: [sample] run damo_yolo with ENN API\n"   
        "              [default : 1]\n\n"
        , argv[0]
    );

}

int main(int argc, char** argv ) {
    int opt;
    std::string arg_model = TEST_MODEL_PATH_DAMO_YOLO;
    std::string arg_input = TEST_INPUT_FILE_PATH;
    int test_case = TEST_CASE_ENN_DAMO_YOLO;

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
        case TEST_CASE_ENN_DAMO_YOLO:
            if(!arg_model.empty())
                run_damo_yolo_detection(arg_model, arg_input);
            break;            
        default:
            break;
    }

    return 0;
}
