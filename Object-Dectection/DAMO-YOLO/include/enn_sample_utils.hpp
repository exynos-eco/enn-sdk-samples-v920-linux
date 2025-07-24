#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>

#include <sys/stat.h>
#include <gst/gst.h>
#include <gst/app/app.h>

/* Debug Option */
#define DEF_ENN_BENCHMARK

#define RED         "\x1b[31m"
#define GREEN         "\x1b[32m"
#define YELLOW         "\x1b[33m"
#define BLUE         "\x1b[34m"
#define MAGENTA        "\x1b[35m"
#define CYAN         "\x1b[36m"
#define RESET         "\x1b[0m"

#define TEST_MODEL_PATH_DAMO_YOLO "/data/vendor/damoyolo/damoyolo_tinynasL25_S_460_simplify_O2_MultiCore.nnc"
#define TEST_INPUT_FILE_PATH "/data/vendor/damoyolo/media/image.jpg"

extern bool g_enable_log; 

#define PRINT_ERROR(message, ...) \
    if (g_enable_log) { \
        printf(RED "TEST: " message, ##__VA_ARGS__); \
        printf("\n" RESET); \
    }

#define PRINT(message, ...) \
    if (g_enable_log) { \
        printf("TEST: " message, ##__VA_ARGS__); \
        printf("\n"); \
    }

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

namespace enn {
namespace sample_utils {
int export_mem_to_file(const char *filename, const void *va, uint32_t size) {
    size_t ret_cnt;

    PRINT("DEBUG:: Export memory to file: name(%s) va(%p), size(%d)", filename, va, size);

    FILE *fp = fopen(filename, "wb");

    ret_cnt = fwrite(va, size, 1, fp);
    if (ret_cnt <= 0) {
        PRINT("FileWrite Failed!!(%zu)", ret_cnt);
        fclose(fp);
        return ENN_RET_INVAL;
    }

    PRINT("DEBUG:: File Save Completed.");
    fclose(fp);

    return ENN_RET_SUCCESS;
}

void show_raw_memory_to_hex(uint8_t *va, uint32_t size, const int line_max, const int size_max) {
    char line_tmp[100] = {0,};
    int int_size = static_cast<int>(size);
    int max = (size_max == 0 ? int_size : (int_size < size_max ? int_size : size_max));
    int idx = sprintf(line_tmp, "[%p] ", va);  // prefix of line
    int i = 0;                                 // idx records current location of print line
    for (; i < max; ++i) {
        idx += sprintf(&(line_tmp[idx]), "%02X ", va[i]);
        if (i % line_max == (line_max - 1)) {
            // if new line is required, flush print --> and record prefix print
            line_tmp[idx] = 0;
            std::cout << line_tmp << std::endl;
            idx = 0;
            idx = sprintf(line_tmp, "[%p] ", &(va[i]));
        }
    }
    if (i % line_max != 0) {
        std::cout << line_tmp << std::endl;
    }
}

void print_buffer_hex(uint8_t *ptr, const int length, int line_max)
{
    printf( " [ %s , %d, %d] ", __func__, length, line_max);

    for(int i = 0 ; i < length ; i++)
    {
        if( i == 0 || (i%line_max) == 0 )
        {
            printf("\n[%p] index(%4d) -" , ptr, i);
        }
        printf(" %02x,", *ptr);
        ptr++;
    }
    printf("\n " );
}

void print_buffer_info(EnnBufferInfo info_p, const enn_buf_dir_e direction)
{
    if(direction == ENN_DIR_IN)
    {
        printf(YELLOW);
        printf( " >>>>>>>>>>>> Input Buffer information  \n" );
        printf( " is_able_to_update : %d \n" , info_p.is_able_to_update);
        printf( " batch size        : %d \n" , info_p.n);
        printf( " width             : %d \n" , info_p.width);
        printf( " height            : %d \n" , info_p.height);
        printf( " channel           : %d \n" , info_p.channel);
        printf( " size              : %d \n" , info_p.size);
        printf( " buffer type       : %d \n" , info_p.buffer_type);
        printf( " label             : %s \n" , info_p.label);
        printf("\n" RESET);
    }
    else if(direction == ENN_DIR_OUT)
    {
        printf(YELLOW);
        printf( " <<<<<<<<<<<< Output Buffer information  \n" );
        printf( " is_able_to_update : %d \n" , info_p.is_able_to_update);
        printf( " batch size        : %d \n" , info_p.n);
        printf( " width             : %d \n" , info_p.width);
        printf( " height            : %d \n" , info_p.height);
        printf( " channel           : %d \n" , info_p.channel);
        printf( " size              : %d \n" , info_p.size);
        printf( " buffer type       : %d \n" , info_p.buffer_type);
        printf( " label             : %s \n" , info_p.label);
        printf("\n" RESET);
    }
    return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct InferenceProcessor {
    inline static std::vector<std::string> classes;
    inline static float resize_scale_x;
    inline static float resize_scale_y;

    static int preprocess_frame(cv::Mat& input_img, const cv::Size& input_size, void* buffer)
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

    static int load_frame(cv::Mat& frame, EnnBufferPtr inBuffer, EnnModelId model_id) {
        EnnBufferInfo in_buf_info;
        enn::api::EnnGetBufferInfoByIndex(&in_buf_info, model_id, ENN_DIR_IN, 0);

        int ret = preprocess_frame(frame, {(int)(in_buf_info.height), (int)(in_buf_info.width)}, inBuffer->va);

        return ret;
    }

    static int postprocess(EnnBufferPtr buffers, EnnBufferPtr buffers_2, EnnModelId model_id, std::vector<DL_RESULT>& oResult) {
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
};


struct DetectListDrawer {
    inline static std::vector<std::string> classes = {
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

    static cv::Mat draw(const cv::Mat& image, const std::vector<DL_RESULT>& oResult) {
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
                                              
            // for (auto& re : oResult)
            // {
            //     cv::RNG rng(cv::getTickCount());
            //     cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

            //     cv::rectangle(img, re.box, color, 3);

            //     float confidence = floor(100 * re.confidence) / 100;
            //     std::cout << std::fixed << std::setprecision(2);

            //     std::string label = classes[re.classId] + " " +
            //         std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

            //     cv::rectangle(
            //         img,
            //         cv::Point(re.box.x, re.box.y - 25),
            //         cv::Point(re.box.x + label.length() * 15, re.box.y),
            //         color,
            //         cv::FILLED
            //     );

            //     cv::putText(
            //         img,
            //         label,
            //         cv::Point(re.box.x, re.box.y - 5),
            //         cv::FONT_HERSHEY_SIMPLEX,
            //         0.75,
            //         cv::Scalar(0, 0, 0),
            //         2
            //     );
            // }

            cv::imwrite("/tmp/nnc_yolo_output.jpg", img);
            PRINT(BLUE "[KeypointDrawer::%s] result image saved at : nnc_yolo_output.jpg \n" RESET, __func__);
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
};


struct GStreamerDisplay {
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
            "autovideosink sync=false", &error);

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
};


struct EnnSession {
    EnnModelId model_id;
    EnnBufferPtr* buffer_set;
    EnnBufferInfo* in_bufs;
    EnnBufferInfo* out_bufs;
    uint32_t n_in_buf;
    uint32_t n_out_buf;

    bool initialize(const std::string& model_file) {
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
        PRINT(YELLOW "[EnnSession::%s] Number of input buffers: %u" RESET, __func__, n_in_buf);
        PRINT(YELLOW "[EnnSession::%s] Number of output buffers: %u" RESET, __func__, n_out_buf);

        in_bufs = new EnnBufferInfo[n_in_buf];
        out_bufs = new EnnBufferInfo[n_out_buf];

        for (uint32_t i = 0; i < n_in_buf; ++i)
            enn::api::EnnGetBufferInfoByIndex(&in_bufs[i], model_id, ENN_DIR_IN, i);

        for (uint32_t i = 0; i < n_out_buf; ++i)
            enn::api::EnnGetBufferInfoByIndex(&out_bufs[i], model_id, ENN_DIR_OUT, i);

        PRINT(GREEN "[EnnSession::%s] DONE" RESET, __func__);
        return true;
    }

    void run(cv::Mat& frame, std::vector<DL_RESULT>& detect_list) {
        PRINT(GREEN "[EnnSession::%s] START" RESET, __func__);

        InferenceProcessor::load_frame(frame, buffer_set[0], model_id);

        PRINT(BLUE "[EnnSession::%s] EnnExcute model inference..." RESET, __func__);
        enn::api::EnnExecuteModel(model_id);

        InferenceProcessor::postprocess(buffer_set[1], buffer_set[2], model_id, detect_list);

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

        delete[] in_bufs;
        delete[] out_bufs;

        PRINT(GREEN "[EnnSession::%s] DONE cleanup" RESET, __func__);
    }
};


}  // namespace sample_utils
}  // namespace enn
