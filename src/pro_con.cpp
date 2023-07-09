#include <opencv2/opencv.hpp>
#include "yolo.hpp"
#include "common.hpp"
#include "pro_con.hpp"
#include "meter_reader.hpp"

#include "config.hpp"

void merge_frames(std::vector<cv::Mat> frames, cv::Mat &display_frame)
{
    if (frames.size() == 1)
    {
        display_frame = frames[0];
    }
    else if (frames.size() == 2)
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

void draw_boxes(std::vector<cv::Mat> &frames, std::vector<MeterInfo> meters)
{
    LOG(INFO) << "displaying " << meters.size() << " meters";
    for (auto &meter_info: meters)
    {
        LOG(INFO) << meter_info.class_name << " " << meter_info.meter_reading << " " << meter_info.rect.x << " " << meter_info.rect.y << " " << meter_info.rect.width << " " << meter_info.rect.height;
    }
    
    for (int ibatch = 0; ibatch < frames.size(); ibatch++)
    {   
        for (auto &meter_info: meters)
        {
            if (meter_info.frame_batch_id != ibatch)
                continue;
            
            std::string display_text = meter_info.meter_reading;
            cv::Scalar color = COLORS[meter_info.class_id];
            cv::rectangle(frames[ibatch], meter_info.rect, color, 4);
            cv::putText(frames[ibatch], display_text, cv::Point(meter_info.rect.x, meter_info.rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
        }
    }
}

void ProducerThread(ProducerConsumer<FrameInfo>& pc, const std::string& stream_url, int thread_id) {
    cv::VideoCapture cap(stream_url);
    int retry = 5;

    // retry 5 times
    while (!cap.isOpened() && retry > 0) 
    {
        LOG(WARNING) << "Failed to open stream: " << stream_url << ", retrying in 1s ..." << std::endl;
        cap.open(stream_url);
        retry--;
    }

    if (!cap.isOpened()) 
    {
        LOG(WARNING) << "Failed to open stream: " << stream_url << "after retrying " 
            << retry << " times, exiting thread " << thread_id << " ...";
        return;
    }

    int warmup_frames = 10;
    while (warmup_frames > 0)
    {
        cv::Mat frame;
        if (cap.read(frame))
        {
            warmup_frames--;
        }
    }

    cv::Mat frame;
    float fps = cap.get(cv::CAP_PROP_FPS);
    std::chrono::milliseconds frameInterval(static_cast<int>(1000.0 / fps));
    std::chrono::steady_clock::time_point lastFrameTime = std::chrono::steady_clock::now();
    std::vector<cv::Mat> frames(pc.GetNumBuffer());

    LOG(INFO) << "Thread " << thread_id << " started, stream url: " << stream_url << ", fps: " << fps;

    while (true)
    {
        if (!cap.open(stream_url))
        {
            LOG(WARNING) << "Thread " << thread_id << " cannot open the video file, retry in 1s...";
            cv::waitKey(1000);
            continue;
        }
        
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
            if (remaining > std::chrono::milliseconds::zero())
            {
                std::this_thread::sleep_for(remaining);
            }

            // update last frame time
            lastFrameTime = currentFrameTime;
        }

        LOG(WARNING) << "Thread " << thread_id << "connection lost, retry in 1s...";
        cap.release();
        cv::waitKey(1000);
    }
    pc.Stop(thread_id);
}

void ConsumerThread(ProducerConsumer<FrameInfo>& pc, std::vector<MeterInfo> &meters_buffer, 
    int det_batch, int seg_batch, meterReader &meter_reader)
{
    // std::this_thread::sleep_for(std::chrono::seconds(3));
    while (true) 
    {
        if (pc.IsStopped()) 
        {
            LOG(WARNING) << "meter reader thread stopped";
            break;
        }
        std::vector<FrameInfo> frame_batch;
        for (int ibatch = 0; ibatch < det_batch; ibatch++)
        {
            FrameInfo frame_info = pc.Consume();
            if (frame_info.frame.empty()) {
                continue;
            }
            frame_batch.push_back(frame_info);
        }

        // do something with frame
        std::vector<MeterInfo> meters;
        auto t1 = std::chrono::high_resolution_clock::now();

        bool no_meter_detected = meter_reader.read(frame_batch, meters);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        
        LOG(WARNING) << "meter reading time: " << duration << " ms";

        if (no_meter_detected) // no meter detected
        {
            LOG(WARNING) << "no meter detected";   
        }
        else // meters detected
        {
            LOG_ASSERT(0) << " stop here";
            std::unique_lock<std::mutex> lock(pc.GetMutex());
            meters_buffer = meters;
            lock.unlock();
        }

        LOG(INFO) <<  meters_buffer.size() << " meters detected";

        for (auto &meter_info: meters_buffer)
        {
            LOG(INFO) << meter_info.class_name << " " << meter_info.meter_reading << " " << meter_info.rect.x << " " << meter_info.rect.y << " " << meter_info.rect.width << " " << meter_info.rect.height;
        }

        // LOG_ASSERT(0) << " stop here";
    }
}

void DisplayThread(ProducerConsumer<FrameInfo>& pc, std::vector<MeterInfo> &meters_buffer) 
{
    cv::namedWindow("Display", cv::WINDOW_NORMAL);

    int fps = 30;
    std::chrono::milliseconds frameInterval(static_cast<int>(1000.0 / fps));
    std::chrono::steady_clock::time_point lastFrameTime = std::chrono::steady_clock::now();
    std::vector<cv::Mat> frames(pc.GetNumBuffer());

    std::vector<MeterInfo> display_result;

    while (true) 
    {
        FrameInfo frame_info;
        bool all_empty = true;
        // read all frames in sequence of thread_id
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
            LOG(WARNING) << "all frames are empty, exiting display thread ...";
            break;
        }

        // LOG_ASSERT(frames.size() > 0) << " frames size is 0";

        // update display result
        std::unique_lock<std::mutex> lock(pc.GetMutex());
        display_result = meters_buffer;
        lock.unlock();

        LOG(INFO) << "meters in display thread: "  << meters_buffer.size();
        for (auto &meter_info: meters_buffer)
        {
            LOG(INFO) << meter_info.class_name << " " << meter_info.meter_reading << " " << meter_info.rect.x << " " << meter_info.rect.y << " " << meter_info.rect.width << " " << meter_info.rect.height;
        }

        // draw all boxes in all frames
        draw_boxes(frames, display_result);
        
        cv::Mat display_frame;
        merge_frames(frames, display_frame);

        cv::imshow("Display", display_frame);
        cv::waitKey(1);

        // LOG_ASSERT(0) << " stop here";
        

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

// num_cam: number of cameras
// capacity: capacity of the buffer
// stream_urls: vector of stream urls
// det_batch: batch size for detection
// seg_batch: batch size for segmentation
// det_model: path to detection model
// seg_model: path to segmentation model
void run(int num_cam, int capacity, std::vector<std::string> stream_urls, int det_batch, int seg_batch, std::string det_model, std::string seg_model) 
{
    meterReader meter_reader(det_model, seg_model);
    
    std::vector<MeterInfo> meters_buffer(num_cam);
    
    ProducerConsumer<FrameInfo> pc(capacity, num_cam);

    std::vector<std::thread> producers;

    // create threads for each camera
    for (int thread_id = 0; thread_id < num_cam; thread_id++) 
    {
        producers.emplace_back(ProducerThread, std::ref(pc), stream_urls[thread_id], thread_id);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::thread consumer(ConsumerThread, std::ref(pc), std::ref(meters_buffer), det_batch, seg_batch, std::ref(meter_reader));
    std::thread display(DisplayThread, std::ref(pc), std::ref(meters_buffer));
    
    for (auto& producer : producers) 
    {
        producer.join();
    }

    consumer.join();
    display.join();
}
