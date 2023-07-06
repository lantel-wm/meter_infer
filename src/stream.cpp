#include "stream.hpp"
#include "glog/logging.h"

bool endsWith(std::string const &str, std::string const &suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

bool startsWith(std::string const &str, std::string const &prefix) {
    if (str.length() < prefix.length()) {
        return false;
    }
    return str.compare(0, prefix.length(), prefix) == 0;
}

Stream::Stream(std::string const &source)
{
    this->source = source;
    this->is_opened = false;
    if (endsWith(source, ".jpg") || endsWith(source, ".png") || endsWith(source, ".jpeg")) {
        this->is_opened = true;
        this->type = IMAGE;
    }

    if (endsWith(source, ".mp4") || endsWith(source, ".avi") || endsWith(source, ".mkv")) {
        this->cap.open(source);
        this->is_opened = this->cap.isOpened();
        this->type = VIDEO;
    }

    if (startsWith(source, "rtsp://")) {
        this->cap.open(source);
        this->is_opened = this->cap.isOpened();
        this->type = RTSP;
    }
}

Stream::~Stream()
{
    if (this->is_opened) {
        this->cap.release();
    }
}

bool Stream::get_frame(cv::Mat &frame)
{
    if (this->is_opened) {
        switch (this->type) {
            case IMAGE:
                frame = cv::imread(this->source);
                this->is_opened = false;
                break;
            case VIDEO:
                this->is_opened = this->cap.isOpened();
                if (this->is_opened) {
                    this->cap >> frame;
                }
                if (frame.empty()) {
                    this->is_opened = false;
                }
                break;
            case RTSP:
            default:
                break;
        }
    }
    return this->is_opened;
}

bool Stream::is_open()
{
    return this->is_opened;
}