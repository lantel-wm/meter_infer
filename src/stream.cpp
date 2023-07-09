// #include "stream.hpp"
// #include "glog/logging.h"

// bool endsWith(std::string const &str, std::string const &suffix) {
//     if (str.length() < suffix.length()) {
//         return false;
//     }
//     return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
// }

// bool startsWith(std::string const &str, std::string const &prefix) {
//     if (str.length() < prefix.length()) {
//         return false;
//     }
//     return str.compare(0, prefix.length(), prefix) == 0;
// }

// Stream::Stream(std::string const &source, int buffer_size = 10)
// {
//     this->source = source;
//     this->is_opened = false;
//     this->buffer_size_ = buffer_size;

//     if (endsWith(source, ".jpg") || endsWith(source, ".png") || endsWith(source, ".jpeg")) {
//         this->is_opened = true;
//         this->type = IMAGE;
//     }

//     if (endsWith(source, ".mp4") || endsWith(source, ".avi") || endsWith(source, ".mkv")) {
//         this->cap.open(source);
//         this->is_opened = this->cap.isOpened();
//         this->type = VIDEO;
//     }

//     if (startsWith(source, "rtsp://")) {
//         this->cap.open(source);
//         this->is_opened = this->cap.isOpened();
//         this->type = RTSP;
//     }
// }

// Stream::~Stream()
// {

// }

// void Stream::RTSPCapture()
// {
//     this->cap(this->source);
//     this->cap.set(cv::CAP_PROP_BUFFERSIZE, 5);
//     this->is_opened = true;
//     cv::Mat frame;

//     while (true)
//     {
//         if (!this->cap.open(this->source))
//         {
//             LOG(WARNING) << "Cannot open RTSP stream, retry in 1s...";
//             cv::waitKey(1000);
//             continue;
//         }

//         LOG(INFO) << "RTSP stream opened" << std::endl;

//         while (this->cap.read(frame))
//         {
            
//         }

//         LOG(WARNING) << "Connection lost, retry in 1s...";
//         this->cap.release();
//         cv::waitKey(1000);
//     }

//     this->cap.release();
//     this->is_opened = false;
// }

// void Stream::VideoCapture()
// {
//     this->cap.open(this->source);
//     this->is_opened = true;
//     cv::Mat frame;

//     while (this->cap.read(frame))
//     {
//         frame_buffer.push(frame);
//         while (frame_buffer.size() > buffer_size_)
//         {
//             frame_buffer.pop();
//         }
//     }
//     this->cap.release();
//     this->is_opened = false;
// }

// void Stream::ImageCapture()
// {
//     cv::Mat frame = cv::imread(this->source);
//     this->is_opened = true;
//     frame_buffer.push(frame);
// }

// bool Stream::get_frame(cv::Mat &frame)
// {
//     if (this->type == RTSP)
//     {
//         RTSPCapture(frame);
//     }
// }

// bool Stream::is_open()
// {
//     return this->is_opened;
// }