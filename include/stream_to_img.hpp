#ifndef _STREAM_TO_IMG_HPP_
#define _STREAM_TO_IMG_HPP_

#include <opencv2/opencv.hpp>
#include <string>

#define IMAGE 0
#define VIDEO 1
#define RTSP 2

class stream_to_img
{
    private:
        cv::VideoCapture cap; // camera object
        cv::Mat frame; // current frame of the camera
        std::string source; // source path, image, video or rstp stream
        bool is_opened; // whether the camera is opened
        int type; // type of the source, image, video or rstp stream

    public:
        stream_to_img(std::string const &source);
        ~stream_to_img();

        bool get_frame(cv::Mat &frame);
        bool is_open();
};

#endif