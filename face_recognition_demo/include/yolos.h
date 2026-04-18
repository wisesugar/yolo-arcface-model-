#ifndef YOLOS_H
#define YOLOS_H

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "rknn_api.h"

#define YOLO_INPUT_W 640
#define YOLO_INPUT_H 640
#define YOLO_INPUT_C 3

struct FaceObject
{
    cv::Rect2f box;
    float score;
    cv::Point2f landmark[5];
};

class YoloFaceDetector
{
public:
    YoloFaceDetector();
    ~YoloFaceDetector();

    int init(const char* model_path);
    int detect(const cv::Mat& bgr_frame, std::vector<FaceObject>& faces);
    void release();

    bool is_inited() const;

private:
    unsigned char* load_model(const char* filename, int* model_size);

    float iou(const cv::Rect2f& a, const cv::Rect2f& b);
    void nms(std::vector<FaceObject>& faces, float nms_thresh);

    int get_candidate_num_from_attr(const rknn_tensor_attr& output_attr);

    void postprocess_yolov5_face(
        const float* output,
        int candidate_num,
        int frame_w,
        int frame_h,
        float conf_thresh,
        float nms_thresh,
        std::vector<FaceObject>& faces
    );

private:
    rknn_context ctx_;
    bool inited_;

    rknn_tensor_attr input_attr_;
    rknn_tensor_attr output_attr_;

    int candidate_num_;

    std::vector<int8_t> input_int8_buf_;
};

#endif