#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>
#include <string>
#include <atomic>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "yolos.h"
#include "face.h"

static std::atomic<bool> g_register_request(false);
static cv::Rect g_register_btn;

static void on_mouse(int event, int x, int y, int flags, void* userdata)
{
    (void)flags;
    (void)userdata;

    if (event == cv::EVENT_LBUTTONDOWN)
    {
        if (g_register_btn.contains(cv::Point(x, y)))
        {
            g_register_request.store(true);
            printf("[UI] register button clicked: x=%d y=%d\n", x, y);
        }
    }
}

static void draw_face_info(
    cv::Mat& frame,
    const FaceObject& face,
    const std::string& name,
    float recog_score)
{
    cv::rectangle(frame, face.box, cv::Scalar(0, 255, 0), 2);

    char text[128];
    snprintf(text, sizeof(text), "%s %.3f", name.c_str(), recog_score);

    int text_y = (int)face.box.y - 8;
    if (text_y < 20)
    {
        text_y = (int)face.box.y + 20;
    }

    cv::putText(frame,
                text,
                cv::Point((int)face.box.x, text_y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(0, 255, 0),
                2);

    for (int k = 0; k < 5; ++k)
    {
        cv::circle(frame, face.landmark[k], 3, cv::Scalar(0, 0, 255), -1);

        char lm_text[8];
        snprintf(lm_text, sizeof(lm_text), "%d", k);
        cv::putText(frame,
                    lm_text,
                    cv::Point((int)face.landmark[k].x + 4, (int)face.landmark[k].y - 4),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.45,
                    cv::Scalar(255, 0, 0),
                    1);
    }
}

static void draw_register_button(cv::Mat& frame)
{
    int btn_w = 180;
    int btn_h = 50;
    int btn_x = 20;
    int btn_y = frame.rows - btn_h - 20;

    g_register_btn = cv::Rect(btn_x, btn_y, btn_w, btn_h);

    cv::rectangle(frame, g_register_btn, cv::Scalar(0, 180, 255), -1);
    cv::rectangle(frame, g_register_btn, cv::Scalar(255, 255, 255), 2);

    cv::putText(frame,
                "REGISTER",
                cv::Point(btn_x + 18, btn_y + 33),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(255, 255, 255),
                2);
}

int main(int argc, char** argv)
{
    const char* yolo_model_path = "model/yolov5s_face_fp16.rknn";
    const char* arcface_model_path = "model/arcface_no_l2_int8.rknn";
    const char* camera_path = "/dev/video9";
    const char* gallery_dir = "gallery";

    if (argc >= 2)
    {
        yolo_model_path = argv[1];
    }
    if (argc >= 3)
    {
        arcface_model_path = argv[2];
    }
    if (argc >= 4)
    {
        camera_path = argv[3];
    }
    if (argc >= 5)
    {
        gallery_dir = argv[4];
    }

    printf("========== Face Recognition Demo ==========\n");
    printf("yolo_model_path    : %s\n", yolo_model_path);
    printf("arcface_model_path : %s\n", arcface_model_path);
    printf("camera_path        : %s\n", camera_path);
    printf("gallery_dir        : %s\n", gallery_dir);
    printf("===========================================\n");

    YoloFaceDetector detector;
    ArcFaceRecognizer recognizer;

    if (detector.init(yolo_model_path) != 0)
    {
        printf("init yolo detector failed\n");
        return -1;
    }

    if (recognizer.init(arcface_model_path) != 0)
    {
        printf("init arcface recognizer failed\n");
        return -1;
    }

    std::vector<GalleryItem> gallery;
    if (recognizer.load_gallery(gallery_dir, gallery) != 0)
    {
        printf("load gallery failed, continue with empty gallery\n");
    }

    cv::VideoCapture cap(camera_path, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
        printf("open camera failed: %s\n", camera_path);
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    printf("camera opened\n");

    const char* window_name = "Face Recognition";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::setMouseCallback(window_name, on_mouse, nullptr);

    int frame_count = 0;
    double display_fps = 0.0;
    auto fps_time = std::chrono::steady_clock::now();

    while (true)
    {
        cv::Mat frame;
        cap.read(frame);

        if (frame.empty())
        {
            printf("read frame failed\n");
            continue;
        }

        /*
         * 如果画面方向不对，按需启用
         */
        // cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        // cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);
        // cv::rotate(frame, frame, cv::ROTATE_180);

        std::vector<FaceObject> faces;
        int ret = detector.detect(frame, faces);
        if (ret != 0)
        {
            printf("detect failed\n");
        }

        for (size_t i = 0; i < faces.size(); ++i)
        {
            std::vector<float> feat;
            float recog_score = -999.0f;
            std::string name = "unknown";

            int feat_ret = recognizer.extract_feature(frame, faces[i].landmark, feat);
            if (feat_ret == 0)
            {
                name = recognizer.recognize(feat, gallery, 0.35f, recog_score);
            }

            draw_face_info(frame, faces[i], name, recog_score);
        }

        if (g_register_request.exchange(false))
        {
            if (faces.empty())
            {
                printf("[Register] no face detected, register failed\n");
            }
            else
            {
                int idx = recognizer.get_largest_face_index(faces);
                if (idx >= 0)
                {
                    std::string reg_name;
                    int reg_ret = recognizer.register_face(
                        frame,
                        faces[idx].landmark,
                        gallery_dir,
                        gallery,
                        reg_name
                    );

                    if (reg_ret == 0)
                    {
                        printf("[Register] success: %s\n", reg_name.c_str());
                    }
                    else
                    {
                        printf("[Register] failed\n");
                    }
                }
            }
        }

        draw_register_button(frame);

        char info_text[128];
        snprintf(info_text, sizeof(info_text),
                 "faces: %zu  gallery: %zu",
                 faces.size(),
                 gallery.size());

        cv::putText(frame,
                    info_text,
                    cv::Point(10, 28),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(255, 255, 0),
                    2);

        frame_count++;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - fps_time).count();
        if (elapsed >= 1.0)
        {
            display_fps = frame_count / elapsed;
            frame_count = 0;
            fps_time = now;
        }

        char fps_text[64];
        snprintf(fps_text, sizeof(fps_text), "FPS %.1f", display_fps);

        cv::putText(frame,
                    fps_text,
                    cv::Point(10, 58),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(255, 255, 0),
                    2);

        cv::imshow(window_name, frame);

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}