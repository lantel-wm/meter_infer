#include <opencv2/opencv.hpp>
#include "common.hpp"
#include "pro_con.hpp"
#include "meter_reader.hpp"
#include "yolo.hpp"
#include "config.hpp"

void merge_frames(std::vector<cv::Mat> frames, cv::Mat &display_frame)
{
    if (frames.size() == 2)
    {
        cv::hconcat(frames[0], frames[1], display_frame);
    }
    else if (frames.size() == 3)
    {
        cv::Mat tmp;
        cv::hconcat(frames[0], frames[1], tmp);
        cv::hconcat(tmp, frames[2], display_frame);
    }
    else if (frames.size() == 4)
    {
        cv::Mat tmp1, tmp2;
        cv::hconcat(frames[0], frames[1], tmp1);
        cv::hconcat(frames[2], frames[3], tmp2);
        cv::vconcat(tmp1, tmp2, display_frame);
    }
    else if (frames.size() == 5)
    {
        cv::Mat tmp1, tmp2, tmp3;
        cv::hconcat(frames[0], frames[1], tmp1);
        cv::hconcat(frames[2], frames[3], tmp2);
        cv::hconcat(tmp1, frames[4], tmp3);
        cv::vconcat(tmp3, tmp2, display_frame);
    }
    else if (frames.size() == 6)
    {
        cv::Mat tmp1, tmp2, tmp3;
        cv::hconcat(frames[0], frames[1], tmp1);
        cv::hconcat(frames[2], frames[3], tmp2);
        cv::hconcat(frames[4], frames[5], tmp3);
        cv::vconcat(tmp1, tmp2, display_frame);
        cv::vconcat(display_frame, tmp3, display_frame);
    }
    else if (frames.size() == 7)
    {
        cv::Mat tmp1, tmp2, tmp3, tmp4;
        cv::hconcat(frames[0], frames[1], tmp1);
        cv::hconcat(frames[2], frames[3], tmp2);
        cv::hconcat(frames[4], frames[5], tmp3);
        cv::hconcat(tmp1, frames[6], tmp4);
        cv::vconcat(tmp4, tmp2, display_frame);
        cv::vconcat(display_frame, tmp3, display_frame);
    }
    else if (frames.size() == 8)
    {
        // 4 * 2
        cv::Mat tmp1, tmp2, tmp3, tmp4, tmp5;
        cv::hconcat(frames[0], frames[1], tmp1);
        cv::hconcat(frames[2], frames[3], tmp2);
        cv::hconcat(frames[4], frames[5], tmp3);
        cv::hconcat(frames[6], frames[7], tmp4);
        cv::hconcat(tmp1, tmp2, display_frame);
        cv::hconcat(tmp3, tmp4, tmp5);
        cv::vconcat(display_frame, tmp5, display_frame);
    }
    else if (frames.size() == 9)
    {
        cv::Mat tmp1, tmp2, tmp3, tmp4, tmp5;
        cv::hconcat(frames[0], frames[1], tmp1);
        cv::hconcat(frames[2], frames[3], tmp2);
        cv::hconcat(frames[4], frames[5], tmp3);
        cv::hconcat(frames[6], frames[7], tmp4);
        cv::hconcat(tmp1, frames[8], tmp5);
        cv::vconcat(tmp5, tmp2, display_frame);
        cv::vconcat(display_frame, tmp3, display_frame);
        cv::vconcat(display_frame, tmp4, display_frame);
    }
    else if (frames.size() == 10)
    {
        cv::Mat tmp1, tmp2, tmp3, tmp4, tmp5;
        cv::hconcat(frames[0], frames[1], tmp1);
        cv::hconcat(frames[2], frames[3], tmp2);
        cv::hconcat(frames[4], frames[5], tmp3);
        cv::hconcat(frames[6], frames[7], tmp4);
        cv::hconcat(frames[8], frames[9], tmp5);
        cv::vconcat(tmp1, tmp2, display_frame);
        cv::vconcat(display_frame, tmp3, display_frame);
        cv::vconcat(display_frame, tmp4, display_frame);
        cv::vconcat(display_frame, tmp5, display_frame);
    }
    else
    {
        LOG(ERROR) << "frames size is " << frames.size() << ", not supported";
    }
}

void draw_boxes()
{
    std::string display_text = meter_info.class_name + " " + meter_info.meter_reading;
    cv::Scalar color = COLORS[meter_info.class_id];
    cv::rectangle(frame_info.frame, meter_info.rect, color, 2);
    cv::putText(frame_info.frame, display_text, cv::Point(meter_info.rect.x, meter_info.rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
}

void ProducerThread(ProducerConsumer<FrameInfo>& pc, const std::string& stream_url, int thread_id) {
    cv::VideoCapture cap(stream_url);
    int retry = 5;

    // retry 5 times
    while (!cap.isOpened() && retry > 0) {
        LOG(WARNING) << "Failed to open stream: " << stream_url << ", retrying in 1s ..." << std::endl;
        cap.open(stream_url);
        retry--;
    }

    if (!cap.isOpened()) {
        LOG(WARNING) << "Failed to open stream: " << stream_url << "after retrying " 
            << retry << " times, exiting thread " << thread_id << " ...";
        return;
    }

    cv::Mat frame;
    float fps = cap.get(cv::CAP_PROP_FPS);
    std::chrono::milliseconds frameInterval(static_cast<int>(1000.0 / fps));
    std::chrono::steady_clock::time_point lastFrameTime = std::chrono::steady_clock::now();
    std::vector<cv::Mat> frames(pc.GetNumBuffer());

    while (cap.read(frame))
    {
        FrameInfo frame_info;
        frame_info.frame = frame;
        frame_info.info = stream_url;
        frame_info.thread_id = thread_id;
        pc.Produce(frame_info);
        pc.Write(frame_info, thread_id);
        
        // frame rate control
        std::chrono::steady_clock::time_point currentFrameTime = std::chrono::steady_clock::now();
        std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentFrameTime - lastFrameTime);
        std::chrono::milliseconds remaining = frameInterval - elapsed;
        if (remaining > std::chrono::milliseconds::zero()) {
            std::this_thread::sleep_for(remaining);
        }

        // update last frame time
        lastFrameTime = currentFrameTime;
    }
    pc.Stop();
}

void ConsumerThread(ProducerConsumer<FrameInfo>& pc, std::vector<MeterInfo> &meters, 
    int det_batch, int seg_batch, std::string det_model, std::string seg_model) 
{
    meterReader meter_reader(det_model, seg_model);
    while (true) 
    {
        std::vector<FrameInfo> frame_batch;
        for (int ibatch = 0; ibatch < det_batch; ibatch++)
        {
            FrameInfo frame_info = pc.Consume();
            if (frame_info.frame.empty()) {
                continue;
            }
            frame_batch.push_back(frame_info);
        }

        if (frame_batch.empty()) {
            break;
        }

        // do something with frame
        meter_reader.read(frame_batch, meters);
    }
}

void DisplayThread(ProducerConsumer<FrameInfo>& pc, std::vector<MeterInfo> meters
) {
    cv::namedWindow("Display", cv::WINDOW_NORMAL);

    int fps = 30;
    std::chrono::milliseconds frameInterval(static_cast<int>(1000.0 / fps));
    std::chrono::steady_clock::time_point lastFrameTime = std::chrono::steady_clock::now();
    std::vector<cv::Mat> frames(pc.GetNumBuffer());

    std::vector<int> display_result;

    while (true) 
    {
        FrameInfo frame_info;
        bool all_empty = true;
        for (int thread_id = 0; thread_id < pc.GetNumBuffer(); thread_id++) 
        {
            if (!pc.Read(frame_info, thread_id))
            {
                continue;
            }
            all_empty = false;
            frames[thread_id] = frame_info.frame.clone();
        }
        

        if (all_empty) {
            std::cout << "Display thread exit" << std::endl;
            break;
        }

        std::unique_lock<std::mutex> lock(pc.GetMutex());
        display_result = meters;
        lock.unlock();

        for (int thread_id = 0; thread_id < pc.GetNumBuffer(); thread_id++) 
        {
            draw_boxes(frames[thread_id], display_result[thread_id]);
        }

        cv::Mat display_frame;
        merge_frames(frames, display_frame);

        cv::imshow("Display", display_frame);
        cv::waitKey(1);

        std::chrono::steady_clock::time_point currentFrameTime = std::chrono::steady_clock::now();
        std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentFrameTime - lastFrameTime);
        std::chrono::milliseconds remaining = frameInterval - elapsed;
        if (remaining > std::chrono::milliseconds::zero()) {
            std::this_thread::sleep_for(remaining);
        }

        // update last frame time
        lastFrameTime = currentFrameTime;
    }   
    cv::destroyAllWindows();
}

void run(int num_cam, int capacity, std::vector<std::string> stream_urls, int det_batch, int seg_batch) 
{
    std::vector<MeterInfo> meters(num_cam);

    std::vector<std::string> stream_urls = {
        "/home/zzy/cublas_test/data/201.mp4", 
        // "/home/zzy/cublas_test/data/201.mp4",
        // "/home/zzy/cublas_test/data/201.mp4", 
        // "/home/zzy/cublas_test/data/201.mp4",
        // "/home/zzy/cublas_test/data/201.mp4", 
        // "/home/zzy/cublas_test/data/201.mp4",
        // "/home/zzy/cublas_test/data/201.mp4", 
        // "/home/zzy/cublas_test/data/201.mp4"
    };
    
    ProducerConsumer<FrameInfo> pc(capacity, num_cam);

    std::vector<std::thread> producers;

    // create threads for each camera
    for (int thread_id = 0; thread_id < num_cam; thread_id++) 
    {
        producers.emplace_back(ProducerThread, std::ref(pc), stream_urls[thread_id], thread_id);
    }

    std::thread consumer(ConsumerThread, std::ref(pc), std::ref(meters), int det_batch, int seg_batch);
    std::thread display(DisplayThread, std::ref(pc), std::ref(meters));
    
    for (auto& producer : producers) 
    {
        producer.join();
    }

    consumer.join();
    display.join();
}
