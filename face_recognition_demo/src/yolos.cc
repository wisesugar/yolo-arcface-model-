#include "yolos.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <algorithm>

YoloFaceDetector::YoloFaceDetector()
{
    ctx_ = 0;
    inited_ = false;
    candidate_num_ = 25200;

    memset(&input_attr_, 0, sizeof(input_attr_));
    memset(&output_attr_, 0, sizeof(output_attr_));
}

YoloFaceDetector::~YoloFaceDetector()
{
    release();
}

bool YoloFaceDetector::is_inited() const
{
    return inited_;
}

unsigned char* YoloFaceDetector::load_model(const char* filename, int* model_size)
{
    FILE* fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        printf("fopen %s fail!\n", filename);
        return nullptr;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char* data = (unsigned char*)malloc(size);
    if (data == nullptr)
    {
        fclose(fp);
        return nullptr;
    }

    if (fread(data, 1, size, fp) != (size_t)size)
    {
        printf("read model fail!\n");
        free(data);
        fclose(fp);
        return nullptr;
    }

    fclose(fp);
    *model_size = size;
    return data;
}

int YoloFaceDetector::init(const char* model_path)
{
    if (inited_)
    {
        release();
    }

    printf("[YOLO] model_path: %s\n", model_path);

    int model_size = 0;
    unsigned char* model_data = load_model(model_path, &model_size);
    if (model_data == nullptr)
    {
        printf("[YOLO] load model failed\n");
        return -1;
    }

    int ret = rknn_init(&ctx_, model_data, model_size, 0, nullptr);
    free(model_data);

    if (ret < 0)
    {
        printf("[YOLO] rknn_init failed! ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
    if (ret == RKNN_SUCC)
    {
        printf("[YOLO] sdk version: %s driver version: %s\n",
               version.api_version,
               version.drv_version);
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("[YOLO] rknn_query RKNN_QUERY_IN_OUT_NUM failed! ret=%d\n", ret);
        release();
        return -1;
    }

    printf("[YOLO] model input num: %d, output num: %d\n",
           io_num.n_input,
           io_num.n_output);

    if (io_num.n_input != 1 || io_num.n_output != 1)
    {
        printf("[YOLO] ERROR: this code expects 1 input and 1 decoded output.\n");
        release();
        return -1;
    }

    memset(&input_attr_, 0, sizeof(input_attr_));
    input_attr_.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));
    if (ret != RKNN_SUCC)
    {
        printf("[YOLO] query input attr failed! ret=%d\n", ret);
        release();
        return -1;
    }

    printf("[YOLO] input dims: [%d %d %d %d], fmt=%d, type=%d, size=%d, qnt_type=%d, zp=%d, scale=%f\n",
           input_attr_.dims[0],
           input_attr_.dims[1],
           input_attr_.dims[2],
           input_attr_.dims[3],
           input_attr_.fmt,
           input_attr_.type,
           input_attr_.size,
           input_attr_.qnt_type,
           input_attr_.zp,
           input_attr_.scale);

    memset(&output_attr_, 0, sizeof(output_attr_));
    output_attr_.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attr_, sizeof(output_attr_));
    if (ret != RKNN_SUCC)
    {
        printf("[YOLO] query output attr failed! ret=%d\n", ret);
        release();
        return -1;
    }

    printf("[YOLO] output dims: [%d %d %d %d], n_dims=%d, fmt=%d, type=%d, size=%d, qnt_type=%d, zp=%d, scale=%f\n",
           output_attr_.dims[0],
           output_attr_.dims[1],
           output_attr_.dims[2],
           output_attr_.dims[3],
           output_attr_.n_dims,
           output_attr_.fmt,
           output_attr_.type,
           output_attr_.size,
           output_attr_.qnt_type,
           output_attr_.zp,
           output_attr_.scale);

    candidate_num_ = get_candidate_num_from_attr(output_attr_);
    printf("[YOLO] candidate_num=%d\n", candidate_num_);

    input_int8_buf_.resize(YOLO_INPUT_W * YOLO_INPUT_H * YOLO_INPUT_C);

    inited_ = true;
    return 0;
}

void YoloFaceDetector::release()
{
    if (inited_)
    {
        rknn_destroy(ctx_);
        ctx_ = 0;
        inited_ = false;
    }
}

int YoloFaceDetector::get_candidate_num_from_attr(const rknn_tensor_attr& output_attr)
{
    /*
     * 你的 YOLOv5-face FP16 decoded 模型一般是：
     *   [1, 25200, 16, 1]
     * 或：
     *   [1, 25200, 16]
     */
    if (output_attr.n_dims >= 3)
    {
        if (output_attr.dims[2] == 16)
        {
            return output_attr.dims[1];
        }

        if (output_attr.dims[1] == 16)
        {
            return output_attr.dims[2];
        }
    }

    return 25200;
}

float YoloFaceDetector::iou(const cv::Rect2f& a, const cv::Rect2f& b)
{
    float inter_x1 = std::max(a.x, b.x);
    float inter_y1 = std::max(a.y, b.y);
    float inter_x2 = std::min(a.x + a.width, b.x + b.width);
    float inter_y2 = std::min(a.y + a.height, b.y + b.height);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float union_area = a.width * a.height + b.width * b.height - inter_area;
    if (union_area <= 0.0f)
    {
        return 0.0f;
    }

    return inter_area / union_area;
}

void YoloFaceDetector::nms(std::vector<FaceObject>& faces, float nms_thresh)
{
    std::sort(faces.begin(), faces.end(),
              [](const FaceObject& a, const FaceObject& b)
              {
                  return a.score > b.score;
              });

    std::vector<FaceObject> result;
    std::vector<int> removed(faces.size(), 0);

    for (size_t i = 0; i < faces.size(); ++i)
    {
        if (removed[i])
        {
            continue;
        }

        result.push_back(faces[i]);

        for (size_t j = i + 1; j < faces.size(); ++j)
        {
            if (removed[j])
            {
                continue;
            }

            if (iou(faces[i].box, faces[j].box) > nms_thresh)
            {
                removed[j] = 1;
            }
        }
    }

    faces.swap(result);
}

void YoloFaceDetector::postprocess_yolov5_face(
    const float* output,
    int candidate_num,
    int frame_w,
    int frame_h,
    float conf_thresh,
    float nms_thresh,
    std::vector<FaceObject>& faces)
{
    faces.clear();

    float scale_x = (float)frame_w / YOLO_INPUT_W;
    float scale_y = (float)frame_h / YOLO_INPUT_H;

    float max_obj = -999.0f;
    float max_cls = -999.0f;
    float max_score = -999.0f;
    int pass_count = 0;

    for (int i = 0; i < candidate_num; ++i)
    {
        const float* p = output + i * 16;

        float cx = p[0];
        float cy = p[1];
        float w = p[2];
        float h = p[3];

        float obj_conf = p[4];
        float cls_conf = p[15];

        /*
         * 你的 decoded FP16 模型这里 objectness 已经是 0~1。
         * 单类别 face 模型，直接用 obj_conf 更稳定。
         */
        float score = obj_conf;

        if (obj_conf > max_obj)
        {
            max_obj = obj_conf;
        }

        if (cls_conf > max_cls)
        {
            max_cls = cls_conf;
        }

        if (score > max_score)
        {
            max_score = score;
        }

        if (score < conf_thresh)
        {
            continue;
        }

        pass_count++;

        float x1 = (cx - w * 0.5f) * scale_x;
        float y1 = (cy - h * 0.5f) * scale_y;
        float x2 = (cx + w * 0.5f) * scale_x;
        float y2 = (cy + h * 0.5f) * scale_y;

        x1 = std::max(0.0f, std::min(x1, (float)(frame_w - 1)));
        y1 = std::max(0.0f, std::min(y1, (float)(frame_h - 1)));
        x2 = std::max(0.0f, std::min(x2, (float)(frame_w - 1)));
        y2 = std::max(0.0f, std::min(y2, (float)(frame_h - 1)));

        if (x2 <= x1 || y2 <= y1)
        {
            continue;
        }

        FaceObject face;
        face.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        face.score = score;

        for (int k = 0; k < 5; ++k)
        {
            float lx = p[5 + k * 2 + 0] * scale_x;
            float ly = p[5 + k * 2 + 1] * scale_y;

            lx = std::max(0.0f, std::min(lx, (float)(frame_w - 1)));
            ly = std::max(0.0f, std::min(ly, (float)(frame_h - 1)));

            face.landmark[k] = cv::Point2f(lx, ly);
        }

        faces.push_back(face);
    }

    nms(faces, nms_thresh);

    static int debug_count = 0;
    debug_count++;
    if (debug_count % 30 == 0)
    {
        printf("[YOLO] debug: max_obj=%f max_cls=%f max_score=%f pass=%d faces=%zu\n",
               max_obj,
               max_cls,
               max_score,
               pass_count,
               faces.size());
    }
}

int YoloFaceDetector::detect(const cv::Mat& bgr_frame, std::vector<FaceObject>& faces)
{
    faces.clear();

    if (!inited_)
    {
        printf("[YOLO] detector not inited\n");
        return -1;
    }

    if (bgr_frame.empty())
    {
        printf("[YOLO] empty frame\n");
        return -1;
    }

    int frame_w = bgr_frame.cols;
    int frame_h = bgr_frame.rows;

    cv::Mat resized;
    cv::resize(bgr_frame, resized, cv::Size(YOLO_INPUT_W, YOLO_INPUT_H));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    if (!rgb.isContinuous())
    {
        rgb = rgb.clone();
    }

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));

    inputs[0].index = 0;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = YOLO_INPUT_W * YOLO_INPUT_H * YOLO_INPUT_C;

    /*
     * FP16 模型：
     *   使用 UINT8 + pass_through=0，让 RKNN runtime 按模型配置处理。
     *
     * INT8 模型：
     *   这里保留手动量化分支。
     */
    if (input_attr_.type == RKNN_TENSOR_INT8)
    {
        float in_scale = input_attr_.scale;
        int in_zp = input_attr_.zp;

        if (in_scale <= 0.0f)
        {
            printf("[YOLO] invalid input scale: %f\n", in_scale);
            return -1;
        }

        unsigned char* src = rgb.data;
        int total = YOLO_INPUT_W * YOLO_INPUT_H * YOLO_INPUT_C;

        for (int i = 0; i < total; ++i)
        {
            float real_value = src[i] / 255.0f;
            int q = (int)roundf(real_value / in_scale + in_zp);

            if (q > 127)
            {
                q = 127;
            }

            if (q < -128)
            {
                q = -128;
            }

            input_int8_buf_[i] = (int8_t)q;
        }

        inputs[0].type = RKNN_TENSOR_INT8;
        inputs[0].buf = input_int8_buf_.data();
        inputs[0].pass_through = 1;
    }
    else
    {
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].buf = rgb.data;
        inputs[0].pass_through = 0;
    }

    int ret = rknn_inputs_set(ctx_, 1, inputs);
    if (ret < 0)
    {
        printf("[YOLO] rknn_inputs_set failed! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(ctx_, nullptr);
    if (ret < 0)
    {
        printf("[YOLO] rknn_run failed! ret=%d\n", ret);
        return -1;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));

    outputs[0].want_float = 1;
    outputs[0].is_prealloc = 0;

    ret = rknn_outputs_get(ctx_, 1, outputs, nullptr);
    if (ret < 0)
    {
        printf("[YOLO] rknn_outputs_get failed! ret=%d\n", ret);
        return -1;
    }

    float* out = (float*)outputs[0].buf;

    const float conf_thresh = 0.10f;
    const float nms_thresh = 0.45f;

    postprocess_yolov5_face(
        out,
        candidate_num_,
        frame_w,
        frame_h,
        conf_thresh,
        nms_thresh,
        faces
    );

    rknn_outputs_release(ctx_, 1, outputs);

    return 0;
}