// #ifndef _STREAM_HPP_
// #define _STREAM_HPP_

// #include <opencv2/opencv.hpp>
// #include <string>
// #include <queue>

// #define IMAGE 0
// #define VIDEO 1
// #define RTSP 2

// class Stream
// {
//     private:
//         cv::VideoCapture cap; // camera object
//         cv::Mat frame; // current frame of the camera
//         std::string source; // source path, image, video or rstp stream
//         bool is_opened; // whether the camera is opened
//         int type; // type of the source, image, video or rstp stream
//         int buffer_size_; // buffer size for rtsp stream
//         std::queue<cv::Mat> frame_buffer; // frame queue for rtsp stream

//         void RTSPCapture(cv::Mat &frame);
//         void VideoCapture(cv::Mat &frame);
//         void ImageCapture(cv::Mat &frame);

//     public:
//         Stream(std::string const &source, int buffer_size = 10);
//         ~Stream();

//         void start();
//         bool get_frame(cv::Mat &frame);
//         bool is_open();
// };

// #endif