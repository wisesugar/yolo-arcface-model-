#include "face.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

ArcFaceRecognizer::ArcFaceRecognizer()
{
    ctx_ = 0;
    inited_ = false;

    memset(&input_attr_, 0, sizeof(input_attr_));
    memset(&output_attr_, 0, sizeof(output_attr_));
}

ArcFaceRecognizer::~ArcFaceRecognizer()
{
    release();
}

bool ArcFaceRecognizer::is_inited() const
{
    return inited_;
}

unsigned char* ArcFaceRecognizer::load_model(const char* filename, int* model_size)
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

int ArcFaceRecognizer::init(const char* model_path)
{
    if (inited_)
    {
        release();
    }

    printf("[ArcFace] model_path: %s\n", model_path);

    int model_size = 0;
    unsigned char* model_data = load_model(model_path, &model_size);
    if (model_data == nullptr)
    {
        printf("[ArcFace] load model failed\n");
        return -1;
    }

    int ret = rknn_init(&ctx_, model_data, model_size, 0, nullptr);
    free(model_data);

    if (ret < 0)
    {
        printf("[ArcFace] rknn_init failed! ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
    if (ret == RKNN_SUCC)
    {
        printf("[ArcFace] sdk version: %s driver version: %s\n",
               version.api_version,
               version.drv_version);
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("[ArcFace] rknn_query RKNN_QUERY_IN_OUT_NUM failed! ret=%d\n", ret);
        release();
        return -1;
    }

    printf("[ArcFace] model input num: %d, output num: %d\n",
           io_num.n_input,
           io_num.n_output);

    if (io_num.n_input != 1 || io_num.n_output != 1)
    {
        printf("[ArcFace] ERROR: this code expects 1 input and 1 output.\n");
        release();
        return -1;
    }

    memset(&input_attr_, 0, sizeof(input_attr_));
    input_attr_.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));
    if (ret != RKNN_SUCC)
    {
        printf("[ArcFace] query input attr failed! ret=%d\n", ret);
        release();
        return -1;
    }

    printf("[ArcFace] input dims: [%d %d %d %d], fmt=%d, type=%d, size=%d, qnt_type=%d, zp=%d, scale=%f\n",
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
        printf("[ArcFace] query output attr failed! ret=%d\n", ret);
        release();
        return -1;
    }

    printf("[ArcFace] output dims: [%d %d %d %d], fmt=%d, type=%d, size=%d, qnt_type=%d, zp=%d, scale=%f\n",
           output_attr_.dims[0],
           output_attr_.dims[1],
           output_attr_.dims[2],
           output_attr_.dims[3],
           output_attr_.fmt,
           output_attr_.type,
           output_attr_.size,
           output_attr_.qnt_type,
           output_attr_.zp,
           output_attr_.scale);

    inited_ = true;
    return 0;
}

void ArcFaceRecognizer::release()
{
    if (inited_)
    {
        rknn_destroy(ctx_);
        ctx_ = 0;
        inited_ = false;
    }
}

void ArcFaceRecognizer::l2_normalize(std::vector<float>& feat)
{
    double sum = 0.0;

    for (size_t i = 0; i < feat.size(); ++i)
    {
        sum += feat[i] * feat[i];
    }

    double norm = sqrt(sum);
    if (norm < 1e-12)
    {
        return;
    }

    for (size_t i = 0; i < feat.size(); ++i)
    {
        feat[i] = (float)(feat[i] / norm);
    }
}

float ArcFaceRecognizer::cosine_similarity(
    const std::vector<float>& a,
    const std::vector<float>& b)
{
    if (a.size() != b.size() || a.empty())
    {
        return -999.0f;
    }

    float dot = 0.0f;

    for (size_t i = 0; i < a.size(); ++i)
    {
        dot += a[i] * b[i];
    }

    return dot;
}

int ArcFaceRecognizer::align_face(
    const cv::Mat& bgr_frame,
    const cv::Point2f landmark[5],
    cv::Mat& aligned_bgr)
{
    aligned_bgr.release();

    if (bgr_frame.empty())
    {
        printf("[ArcFace] align_face empty frame\n");
        return -1;
    }

    /*
     * ArcFace 常用 112x112 五点模板。
     * 点顺序：
     *   0 左眼
     *   1 右眼
     *   2 鼻尖
     *   3 左嘴角
     *   4 右嘴角
     */
    std::vector<cv::Point2f> src(5);
    std::vector<cv::Point2f> dst(5);

    for (int i = 0; i < 5; ++i)
    {
        src[i] = landmark[i];
    }

    dst[0] = cv::Point2f(38.2946f, 51.6963f);
    dst[1] = cv::Point2f(73.5318f, 51.5014f);
    dst[2] = cv::Point2f(56.0252f, 71.7366f);
    dst[3] = cv::Point2f(41.5493f, 92.3655f);
    dst[4] = cv::Point2f(70.7299f, 92.2041f);

    cv::Mat M = cv::estimateAffinePartial2D(src, dst);
    if (M.empty())
    {
        printf("[ArcFace] estimateAffinePartial2D failed\n");
        return -1;
    }

    cv::warpAffine(
        bgr_frame,
        aligned_bgr,
        M,
        cv::Size(ARCFACE_INPUT_W, ARCFACE_INPUT_H),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0)
    );

    if (aligned_bgr.empty())
    {
        printf("[ArcFace] warpAffine failed\n");
        return -1;
    }

    return 0;
}

int ArcFaceRecognizer::extract_feature_from_aligned(
    const cv::Mat& aligned_bgr,
    std::vector<float>& feature)
{
    feature.clear();

    if (!inited_)
    {
        printf("[ArcFace] recognizer not inited\n");
        return -1;
    }

    if (aligned_bgr.empty())
    {
        printf("[ArcFace] empty aligned face\n");
        return -1;
    }

    cv::Mat resized;
    if (aligned_bgr.cols != ARCFACE_INPUT_W || aligned_bgr.rows != ARCFACE_INPUT_H)
    {
        cv::resize(aligned_bgr, resized, cv::Size(ARCFACE_INPUT_W, ARCFACE_INPUT_H));
    }
    else
    {
        resized = aligned_bgr;
    }

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    if (!rgb.isContinuous())
    {
        rgb = rgb.clone();
    }

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = ARCFACE_INPUT_W * ARCFACE_INPUT_H * ARCFACE_INPUT_C;
    inputs[0].buf = rgb.data;
    inputs[0].pass_through = 0;

    int ret = rknn_inputs_set(ctx_, 1, inputs);
    if (ret < 0)
    {
        printf("[ArcFace] rknn_inputs_set failed! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(ctx_, nullptr);
    if (ret < 0)
    {
        printf("[ArcFace] rknn_run failed! ret=%d\n", ret);
        return -1;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));

    outputs[0].want_float = 1;
    outputs[0].is_prealloc = 0;

    ret = rknn_outputs_get(ctx_, 1, outputs, nullptr);
    if (ret < 0)
    {
        printf("[ArcFace] rknn_outputs_get failed! ret=%d\n", ret);
        return -1;
    }

    float* out = (float*)outputs[0].buf;
    int float_count = outputs[0].size / sizeof(float);

    feature.resize(float_count);
    for (int i = 0; i < float_count; ++i)
    {
        feature[i] = out[i];
    }

    rknn_outputs_release(ctx_, 1, outputs);

    if (feature.empty())
    {
        printf("[ArcFace] empty feature\n");
        return -1;
    }

    l2_normalize(feature);

    return 0;
}

int ArcFaceRecognizer::extract_feature(
    const cv::Mat& bgr_frame,
    const cv::Point2f landmark[5],
    std::vector<float>& feature)
{
    feature.clear();

    cv::Mat aligned_bgr;
    int ret = align_face(bgr_frame, landmark, aligned_bgr);
    if (ret != 0)
    {
        return -1;
    }

    return extract_feature_from_aligned(aligned_bgr, feature);
}

bool ArcFaceRecognizer::file_exists(const std::string& path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

bool ArcFaceRecognizer::make_dir_if_not_exists(const std::string& dir)
{
    struct stat st;

    if (stat(dir.c_str(), &st) == 0)
    {
        if (S_ISDIR(st.st_mode))
        {
            return true;
        }

        printf("[Gallery] path exists but not dir: %s\n", dir.c_str());
        return false;
    }

    int ret = mkdir(dir.c_str(), 0755);
    if (ret != 0)
    {
        printf("[Gallery] mkdir failed: %s\n", dir.c_str());
        return false;
    }

    return true;
}

std::string ArcFaceRecognizer::path_join(
    const std::string& dir,
    const std::string& name)
{
    if (dir.empty())
    {
        return name;
    }

    if (dir[dir.size() - 1] == '/')
    {
        return dir + name;
    }

    return dir + "/" + name;
}

std::string ArcFaceRecognizer::make_next_user_name(
    const std::vector<GalleryItem>& gallery)
{
    int next_id = (int)gallery.size() + 1;

    char name[64];
    snprintf(name, sizeof(name), "user_%03d", next_id);

    return std::string(name);
}

int ArcFaceRecognizer::load_gallery(
    const std::string& gallery_dir,
    std::vector<GalleryItem>& gallery)
{
    gallery.clear();

    if (!make_dir_if_not_exists(gallery_dir))
    {
        return -1;
    }

    DIR* dir = opendir(gallery_dir.c_str());
    if (dir == nullptr)
    {
        printf("[Gallery] opendir failed: %s\n", gallery_dir.c_str());
        return -1;
    }

    struct dirent* ent = nullptr;

    while ((ent = readdir(dir)) != nullptr)
    {
        std::string filename = ent->d_name;

        if (filename == "." || filename == "..")
        {
            continue;
        }

        std::string lower = filename;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        bool is_image = false;
        if (lower.size() >= 4)
        {
            if (lower.find(".jpg") != std::string::npos ||
                lower.find(".png") != std::string::npos ||
                lower.find(".bmp") != std::string::npos ||
                lower.find(".jpeg") != std::string::npos)
            {
                is_image = true;
            }
        }

        if (!is_image)
        {
            continue;
        }

        std::string path = path_join(gallery_dir, filename);

        cv::Mat img = cv::imread(path);
        if (img.empty())
        {
            printf("[Gallery] read image failed: %s\n", path.c_str());
            continue;
        }

        /*
         * gallery 目录里建议保存 112x112 对齐后的人脸图。
         * 所以这里直接送 ArcFace，不再跑 YOLO。
         */
        std::vector<float> feat;
        int ret = extract_feature_from_aligned(img, feat);
        if (ret != 0)
        {
            printf("[Gallery] extract feature failed: %s\n", path.c_str());
            continue;
        }

        GalleryItem item;

        size_t dot_pos = filename.find_last_of('.');
        if (dot_pos == std::string::npos)
        {
            item.name = filename;
        }
        else
        {
            item.name = filename.substr(0, dot_pos);
        }

        item.image_path = path;
        item.feature = feat;

        gallery.push_back(item);

        printf("[Gallery] loaded: %s, feature_dim=%zu\n",
               item.name.c_str(),
               item.feature.size());
    }

    closedir(dir);

    printf("[Gallery] total loaded: %zu\n", gallery.size());

    return 0;
}

int ArcFaceRecognizer::register_face(
    const cv::Mat& bgr_frame,
    const cv::Point2f landmark[5],
    const std::string& gallery_dir,
    std::vector<GalleryItem>& gallery,
    std::string& registered_name)
{
    registered_name.clear();

    if (!make_dir_if_not_exists(gallery_dir))
    {
        return -1;
    }

    cv::Mat aligned_bgr;
    int ret = align_face(bgr_frame, landmark, aligned_bgr);
    if (ret != 0)
    {
        printf("[Gallery] align face failed, register failed\n");
        return -1;
    }

    std::vector<float> feat;
    ret = extract_feature_from_aligned(aligned_bgr, feat);
    if (ret != 0)
    {
        printf("[Gallery] extract feature failed, register failed\n");
        return -1;
    }

    std::string name = make_next_user_name(gallery);
    std::string filename = name + ".jpg";
    std::string save_path = path_join(gallery_dir, filename);

    /*
     * 如果 user_001.jpg 已经存在，则继续往后找。
     */
    int guard = 0;
    while (file_exists(save_path) && guard < 10000)
    {
        GalleryItem dummy;
        std::vector<GalleryItem> temp = gallery;
        temp.resize(temp.size() + 1 + guard);

        name = make_next_user_name(temp);
        filename = name + ".jpg";
        save_path = path_join(gallery_dir, filename);

        guard++;
    }

    bool ok = cv::imwrite(save_path, aligned_bgr);
    if (!ok)
    {
        printf("[Gallery] save image failed: %s\n", save_path.c_str());
        return -1;
    }

    GalleryItem item;
    item.name = name;
    item.image_path = save_path;
    item.feature = feat;

    gallery.push_back(item);

    registered_name = name;

    printf("[Gallery] registered: %s -> %s\n",
           registered_name.c_str(),
           save_path.c_str());

    return 0;
}

std::string ArcFaceRecognizer::recognize(
    const std::vector<float>& feature,
    const std::vector<GalleryItem>& gallery,
    float threshold,
    float& best_score)
{
    best_score = -999.0f;

    if (feature.empty() || gallery.empty())
    {
        return "unknown";
    }

    std::string best_name = "unknown";

    for (size_t i = 0; i < gallery.size(); ++i)
    {
        float sim = cosine_similarity(feature, gallery[i].feature);

        if (sim > best_score)
        {
            best_score = sim;
            best_name = gallery[i].name;
        }
    }

    if (best_score < threshold)
    {
        return "unknown";
    }

    return best_name;
}

int ArcFaceRecognizer::get_largest_face_index(const std::vector<FaceObject>& faces)
{
    if (faces.empty())
    {
        return -1;
    }

    int best_idx = 0;
    float best_area = faces[0].box.width * faces[0].box.height;

    for (size_t i = 1; i < faces.size(); ++i)
    {
        float area = faces[i].box.width * faces[i].box.height;
        if (area > best_area)
        {
            best_area = area;
            best_idx = (int)i;
        }
    }

    return best_idx;
}