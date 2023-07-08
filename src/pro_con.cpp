#include <opencv2/opencv.hpp>
#include "common.hpp"
#include "pro_con.hpp"
#include "meter_reader.hpp"
#include "yolo.hpp"
#include "config.hpp"

void ProducerThread(ProducerConsumer<cv::Mat>& pc, const std::string& stream_url) {
    cv::VideoCapture cap(stream_url);
    if (!cap.isOpened()) {
        std::cout << "Failed to open stream: " << stream_url << std::endl;
        return;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        pc.Produce(frame);
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
    pc.Stop();
}

void ConsumerThread(ProducerConsumer<cv::Mat>& pc, int &result) {
    int num = 0;
    while (true) {
        cv::Mat frame = pc.Consume();
        if (frame.empty()) {
            break;
        }
        // do something with frame

        result = num++;
        num = num % 100;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void DisplayThread(ProducerConsumer<cv::Mat>& pc, int &result) {
    cv::namedWindow("Display", cv::WINDOW_NORMAL);
    while (true) {
        cv::Mat frame = pc.Peek().clone();
        if (frame.empty()) {
            break;
        }

        std::unique_lock<std::mutex> lock(pc.GetMutex());
        cv::putText(frame, std::to_string(result), cv::Point(100, 500), cv::FONT_HERSHEY_SIMPLEX, 10.0, cv::Scalar(0, 0, 255), 2);
        cv::imshow("Display", frame);
        cv::waitKey(1000 / 30);
        lock.unlock();
    }   
    cv::destroyAllWindows();
}

void run(int num_cam, int capacity, std::vector<std::string> stream_urls) {
    std::vector<MeterInfo> meters; // result

    std::vector<std::string> stream_urls = {
        "/home/zzy/cublas_test/data/201.mp4"
    };
    
    ProducerConsumer<cv::Mat> pc(capacity);

    std::vector<std::thread> producers;

    for (const auto& stream_url: stream_urls) 
    {
        producers.emplace_back(ProducerThread, std::ref(pc), stream_url);
    }

    std::thread consumer(ConsumerThread, std::ref(pc), std::ref(result));
    std::thread display(DisplayThread, std::ref(pc), std::ref(result));
    
    for (auto& producer : producers) {
        producer.join();
    }
    consumer.join();
    display.join();

    return 0;
}
