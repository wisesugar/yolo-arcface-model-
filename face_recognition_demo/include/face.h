#ifndef FACE_H
#define FACE_H

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "rknn_api.h"
#include "yolos.h"

#define ARCFACE_INPUT_W 112
#define ARCFACE_INPUT_H 112
#define ARCFACE_INPUT_C 3

struct GalleryItem
{
    std::string name;
    std::string image_path;
    std::vector<float> feature;
};

class ArcFaceRecognizer
{
public:
    ArcFaceRecognizer();
    ~ArcFaceRecognizer();

    int init(const char* model_path);
    void release();

    bool is_inited() const;

    int align_face(
        const cv::Mat& bgr_frame,
        const cv::Point2f landmark[5],
        cv::Mat& aligned_bgr
    );

    int extract_feature_from_aligned(
        const cv::Mat& aligned_bgr,
        std::vector<float>& feature
    );

    int extract_feature(
        const cv::Mat& bgr_frame,
        const cv::Point2f landmark[5],
        std::vector<float>& feature
    );

    float cosine_similarity(
        const std::vector<float>& a,
        const std::vector<float>& b
    );

    int load_gallery(
        const std::string& gallery_dir,
        std::vector<GalleryItem>& gallery
    );

    int register_face(
        const cv::Mat& bgr_frame,
        const cv::Point2f landmark[5],
        const std::string& gallery_dir,
        std::vector<GalleryItem>& gallery,
        std::string& registered_name
    );

    std::string recognize(
        const std::vector<float>& feature,
        const std::vector<GalleryItem>& gallery,
        float threshold,
        float& best_score
    );

    int get_largest_face_index(const std::vector<FaceObject>& faces);

private:
    unsigned char* load_model(const char* filename, int* model_size);

    void l2_normalize(std::vector<float>& feat);

    bool file_exists(const std::string& path);
    bool make_dir_if_not_exists(const std::string& dir);
    std::string make_next_user_name(const std::vector<GalleryItem>& gallery);
    std::string path_join(const std::string& dir, const std::string& name);

private:
    rknn_context ctx_;
    bool inited_;

    rknn_tensor_attr input_attr_;
    rknn_tensor_attr output_attr_;
};

#endif