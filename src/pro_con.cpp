/*
An implementation of producer-consumer pattern, the resources are frames from RTSP server,
producer produces frames, consumer consumes frames. 

More precisely, the producer thread reads frames from RTSP server and put them into the buffer,
the consumer thread reads frames from the buffer and do meter reading. Additionally, there is
a display thread which reads frames from the buffer and display them on the screen.

There is a little difference between this implementation and traditional producer-consumer pattern.
In this implementation, there are two buffers. The first buffer, called queue_, is a queue. It is 
the traditionalbuffer in producer-consumer pattern which stores the resources for producers to 
produce and consumers to consume. When the buffer is full, the producer thread will be blocked 
until the buffer is not full. When the buffer is empty, the consumer thread will be blocked until
the buffer is not empty.

The second buffer, called buffers_, is a vector of queues, each queue in the vector is
a buffer for a camera. This buffer is used to display the real-time frames. Therefore, 
when one of the second buffers is full, the producer thread will not be blocked, it will
continue to produce frames and overwrite the oldest frame in the buffer. When one of the
second buffers is empty, the display thread will not be blocked, it will continue to display
the last frame in the buffer.

In this implementation, "T Consume()" and "void Produce(const T& item)" are used to manipulate
the first buffer, "T Read(int thread_id)" and "void Write(const T& item, int thread_id)" are
used to manipulate the second buffer.

*/
#include <opencv2/opencv.hpp>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <thread>
#include <chrono>

#include "yolo.hpp"
#include "common.hpp"
#include "pro_con.hpp"
#include "meter_reader.hpp"
#include "mysql.hpp"
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

// draw bounding boxes of detected meters, and display the meter reading on the top of the bounding box
void draw_boxes(std::vector<cv::Mat> &frames, std::vector<MeterInfo> meters)
{
    LOG(INFO) << "displaying " << meters.size() << " meters";
    for (auto &meter_info: meters)
    {
        LOG(INFO) << meter_info.class_name << " " << meter_info.meter_reading << " " << meter_info.rect.x << " " << meter_info.rect.y << " " << meter_info.rect.width << " " << meter_info.rect.height;
    }
    
    for (int thread_id = 0; thread_id < frames.size(); thread_id++)
    {   
        // put camera_id on the top left corner
        std::string display_text0 = "camera_id: " + std::to_string(thread_id);
        cv::Scalar color = cv::Scalar(0, 0, 0);
        cv::putText(frames[thread_id], display_text0, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
    }

    for (int instrument_id = 0; instrument_id < meters.size(); instrument_id++)
    {
        MeterInfo meter_info = meters[instrument_id];
        std::string display_text = std::to_string(meter_info.instrument_id) + " " +  meter_info.meter_reading;
        cv::Scalar color = COLORS[meter_info.class_id];
        cv::rectangle(frames[meter_info.camera_id], meter_info.rect, color, 4);
        cv::putText(frames[meter_info.camera_id], display_text, cv::Point(meter_info.rect.x, meter_info.rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
    }
}

// producer thread, read frames from RTSP server and put them into the buffer
void ProducerThread(ProducerConsumer<FrameInfo>& pc, const std::string& stream_url, int thread_id) {
    cv::VideoCapture cap(stream_url);
    cv::Mat frame;

    float fps = cap.get(cv::CAP_PROP_FPS);

    // record time interval between two frames to control frame rate
    std::chrono::milliseconds frameInterval(static_cast<int>(1000.0 / fps));
    std::chrono::steady_clock::time_point lastFrameTime = std::chrono::steady_clock::now();
    std::vector<cv::Mat> frames(pc.GetNumBuffer());

    LOG(INFO) << "Thread " << thread_id << " started, stream url: " << stream_url << ", fps: " << fps;

    while (true)
    {
        if (!cap.open(stream_url))
        {
            pc.SetInactive(thread_id);
            LOG(ERROR) << "Thread " << thread_id << " cannot open the camera, retry in 1s...";
            cv::waitKey(1000);
            continue;
        }
        
        while (cap.read(frame))
        {
            pc.SetActive(thread_id);

            FrameInfo frame_info;
            frame_info.frame = frame;
            frame_info.info = stream_url;
            frame_info.camera_id = thread_id;
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

        LOG(ERROR) << "Thread " << thread_id << "connection lost, retry in 1s...";
        cap.release();
        cv::waitKey(1000);
    }
    pc.Stop(thread_id);
}

// consumer thread, read frames from the buffer and do meter reading
void ConsumerThread(ProducerConsumer<FrameInfo>& pc, std::vector<MeterInfo> &display_result,
    int det_batch, int seg_batch, meterReader &meter_reader, mysqlServer &mysql_server)
{
    // save the readings to database every 1 seconds for each camera
    // maintain a vector of last save time for each camera
    // std::chrono::milliseconds save_interval(static_cast<int>(1000.0));
    // std::vector<std::chrono::steady_clock::time_point> last_save_times(meter_reader.get_instrument_num());
    // for (int instrument_id = 0; instrument_id < meter_reader.get_instrument_num(); instrument_id++)
    // {
    //     last_save_times[instrument_id] = std::chrono::steady_clock::now();
    // }

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
            // cv::imwrite("frame.png", frame_info.frame);
            frame_batch.push_back(frame_info);
        }

        // do something with frame
        std::vector<MeterInfo> meters;
        auto t1 = std::chrono::high_resolution_clock::now();

        bool read_error = meter_reader.read(frame_batch, meters);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        
        LOG(WARNING) << "meter reading time: " << duration << " ms";

        if (read_error) // no meter detected, skip update
        {
            LOG(WARNING) << "read error, skip update";   
        }
        else // meters detected
        {
            std::unique_lock<std::mutex> lock(pc.GetMutex());
            for (auto &meter_info: meters)
            {
                display_result[meter_info.instrument_id] = meter_info;
            }
            lock.unlock();
        }

        LOG(INFO) <<  meters.size() << " meters detected";

        for (auto &meter_info: meters)
        {
            LOG(INFO) << meter_info.class_name << " " << meter_info.meter_reading << " " << meter_info.rect.x << " " << meter_info.rect.y << " " << meter_info.rect.width << " " << meter_info.rect.height;
        }

        // mysql_server.insert_readings(meters, last_save_times);

        // saveReadings(meters);

        // LOG_ASSERT(0) << " stop here";
    }
}


// display thread
void DisplayThread(ProducerConsumer<FrameInfo>& pc, std::vector<MeterInfo> &display_result)
{
    cv::namedWindow("Display", cv::WINDOW_NORMAL);

    int fps = 30;
    std::chrono::milliseconds frameInterval(static_cast<int>(1000.0 / fps));
    std::chrono::steady_clock::time_point lastFrameTime = std::chrono::steady_clock::now();
    std::vector<cv::Mat> frames(pc.GetNumBuffer());
    // std::vector<FrameInfo> frame_batch(pc.GetNumBuffer());

    std::vector<MeterInfo> display_result_copy;

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
            // frame_batch[thread_id] = frame_info;
        }
        

        if (all_empty) {
            LOG(WARNING) << "all frames are empty, exiting display thread ...";
            break;
        }

        std::unique_lock<std::mutex> lock(pc.GetMutex());
        display_result_copy = display_result;
        lock.unlock();
        
        // draw all boxes in all frames
        cv::Mat display_frame;
        draw_boxes(frames, display_result_copy);
        merge_frames(frames, display_frame);
        cv::imshow("Display", display_frame);

        if (cv::waitKey(1) == 27) 
        {
            LOG(WARNING) << "esc key is pressed by user, exiting display thread ...";
            break;
        }

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
    // pc.StopAll();
}


// send a RTSP GET_PARAMETER request to the RTSP server
void sendGetParameter(std::string rtsp_server_url)
{
    // create a socket
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
    {
        LOG(ERROR) << "ERROR opening socket";
        return;
    }

    // parse the rtsp url to extract rtsp server ip and port 
    char* rtsp_server_ip_cstr = new char[rtsp_server_url.length() + 1];
    int rtsp_server_port;

    if (sscanf(rtsp_server_url.c_str(), "rtsp://%*[^:]:%*[^@]@%[^:]:%d", rtsp_server_ip_cstr, &rtsp_server_port) != 2)
    {
        LOG(ERROR) << "ERROR parsing RTSP url: " << rtsp_server_url;
        close(sockfd);
        return;
    }

    std::string rtsp_server_ip(rtsp_server_ip_cstr);
    delete[] rtsp_server_ip_cstr;

    LOG(INFO) << "rtsp server ip: " << rtsp_server_ip;
    LOG(INFO) << "rtsp server port: " << rtsp_server_port;

    // connect to the RTSP server
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(rtsp_server_port);
    inet_pton(AF_INET, rtsp_server_ip.c_str(), &(server_addr.sin_addr));

    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0)
    {
        LOG(ERROR) << "ERROR connecting to RTSP server";
        close(sockfd);
        return;
    }

    // build the GET_PARAMETER request
    std::string request = "GET_PARAMETER " + rtsp_server_url + " RTSP/1.0\r\n";
    request += "CSeq: 1\r\n";
    request += "\r\n";

    // send the GET_PARAMETER request
    if (send(sockfd, request.c_str(), request.length(), 0) < 0)
    {
        LOG(ERROR) << "ERROR sending GET_PARAMETER request";
        close(sockfd);
        return;
    }

    LOG(INFO) << "GET_PARAMETER request sent";

    // close the socket
    close(sockfd);
}

// send a RTSP GET_PARAMETER request to the RTSP server every interval_seconds
void HeartbeatThread(std::vector<std::string> stream_urls, int interval_seconds)
{
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(interval_seconds));
        for (auto &stream_url: stream_urls)
        {
            sendGetParameter(stream_url);
            break;
        }
    }
}

void draw_instrument_id(std::vector<FrameInfo> &frame_batch)
{
    for (auto &frame_info: frame_batch)
    {
        cv::Mat frame = frame_info.frame;
        std::vector<DetObject> objs = frame_info.det_objs;
        
        // put camera_id on the top left corner
        std::string display_text0 = "camera_id: " + std::to_string(frame_info.camera_id);
        cv::Scalar color0 = cv::Scalar(0, 0, 0);
        cv::putText(frame, display_text0, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, color0, 2);

        for (auto &obj: objs)
        {
            std::string display_text = "instrument_id: " + std::to_string(obj.instrument_id);
            cv::Scalar color = COLORS[obj.class_id];
            cv::rectangle(frame, obj.rect, color, 2);
            cv::putText(frame, display_text, cv::Point(obj.rect.x, obj.rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
        }
    }
}

void setupCameraInstrumentMapping(
    ProducerConsumer<FrameInfo> &pc, 
    std::vector<std::string> stream_urls, 
    meterReader &meter_reader,
    mysqlServer &mysql_server
    )
{
    while (true)
    {
        if (!pc.IsAllNotEmpty()) // not (any not empty) => exists one is empty => not ready
        {
            std::cout << "Waiting for all cameras to be ready..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        
        // get one frame from each camera
        int num_cam = pc.GetNumBuffer();
        std::vector<FrameInfo> frame_batch(num_cam);
        for (int thread_id = 0; thread_id < num_cam; thread_id++)
        {
            pc.Read(frame_batch[thread_id], thread_id);
        }
        // do detection
        meter_reader.recognize(frame_batch);

        int instrument_id = 0;
        for (int camera_id = 0; camera_id < stream_urls.size(); camera_id++)
        {
            // TODO:set mysql
            // table name: Cameras
            // column name: camera_id, camera_url
            
            for (int iobj = 0; iobj < frame_batch[camera_id].det_objs.size(); iobj++)
            {
                // TODO:set mysql
                // table name: Instruments
                // column name: instrument_id, instrument_type, instrument_unit, camera_id
                frame_batch[camera_id].det_objs[iobj].instrument_id = instrument_id;
                // std::cout << obj.instrument_id << std::endl;
                instrument_id++;
            }
        }

        std::cout << "All instrument_id set, showing the result..." << std::endl;

        for (auto &frame_info: frame_batch)
        {
            for (auto &obj: frame_info.det_objs)
            {
                std::cout << obj.instrument_id << std::endl;
            }
        }

        draw_instrument_id(frame_batch);

        std::vector<cv::Mat> frames;
        cv::Mat display_frame;
        for (auto &frame_info: frame_batch)
        {
            frames.push_back(frame_info.frame);
        }

        merge_frames(frames, display_frame);
        cv::resize(display_frame, display_frame, cv::Size(1920, 1080 / 2), 0, 0, cv::INTER_CUBIC);

        // cv::namedWindow("Display instrument_id", cv::WINDOW_NORMAL);
        // cv::resizeWindow("Display instrument_id", 1920, 1080 / 2);
        cv::imshow("Display instrument_id", display_frame);
        cv::imwrite("instrument_id.png", display_frame);
        cv::waitKey(0);
        // cv::destroyAllWindows();
        
        char c;
        while (true)
        {

            std::cout << "Do you want to save the result? (y/n)" << std::endl;
            std::cin >> c;

            if (c == 'y')
            {
                meter_reader.set_camera_instrument_id(frame_batch);
                mysql_server.init_camera_instruments(frame_batch, stream_urls);
                std::cout << "Result saved." << std::endl;
                break;
            }
            else if (c == 'n')
            {
                std::cout << "Resetting instrument_id in 1s..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                break;
            }
            else
            {
                std::cout << "Invalid input, please input again." << std::endl;
                continue;
            }
        }

        if (c == 'y')
        {
            break;
        }
    }

}

void InsertReadingThread(ProducerConsumer<FrameInfo> &pc, std::vector<MeterInfo> &meters, mysqlServer &mysql_server, int interval_seconds)
{
    auto next_time = std::chrono::system_clock::now() + std::chrono::seconds(interval_seconds);
    std::vector<MeterInfo> meters_copy;
    while(true)
    {
        std::unique_lock<std::mutex> lock(pc.GetMutex());
        meters_copy = meters;
        lock.unlock();
        mysql_server.insert_readings(meters_copy);
        std::this_thread::sleep_until(next_time);
        next_time += std::chrono::seconds(interval_seconds);
    }
}

// num_cam: number of cameras
// capacity: capacity of the buffer
// stream_urls: vector of stream urls
// det_batch: batch size for detection
// seg_batch: batch size for segmentation
// det_model: path to detection model
// seg_model: path to segmentation model
void run(int num_cam, int capacity, std::vector<std::string> stream_urls, 
    int det_batch, int seg_batch, std::string det_model, std::string seg_model,
    int debug_on) 
{
    meterReader meter_reader(det_model, seg_model, det_batch, seg_batch);

    mysqlServer mysql_server(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWD, MYSQL_DB, debug_on);
    
    // std::vector<MeterInfo> meters_buffer(num_cam);

    ProducerConsumer<FrameInfo> pc(capacity, num_cam);

    std::vector<std::thread> producers;

    // create threads for each camera
    for (int thread_id = 0; thread_id < num_cam; thread_id++) 
    {
        producers.emplace_back(ProducerThread, std::ref(pc), stream_urls[thread_id], thread_id);
    }

    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    setupCameraInstrumentMapping(std::ref(pc), stream_urls, std::ref(meter_reader), std::ref(mysql_server));
    std::vector<MeterInfo> display_result(meter_reader.get_instrument_num()); // display result
    // init display result
    for (int instrument_id = 0; instrument_id < meter_reader.get_instrument_num(); instrument_id++)
    {
        display_result[instrument_id].instrument_id = instrument_id;
        display_result[instrument_id].meter_reading = "0";
        display_result[instrument_id].rect = cv::Rect(0, 0, 0, 0);
        display_result[instrument_id].class_name = "N/A";
    }

    std::thread consumer(ConsumerThread, std::ref(pc), std::ref(display_result), det_batch, seg_batch, std::ref(meter_reader), std::ref(mysql_server));
    std::thread display(DisplayThread, std::ref(pc), std::ref(display_result));
    // std::thread heartbeat(HeartbeatThread, stream_urls, 60);
    std::thread insert_reading(InsertReadingThread, std::ref(pc), std::ref(display_result), std::ref(mysql_server), 1);
    
    for (auto& producer : producers) 
    {
        producer.join();
    }

    consumer.join();
    display.join();
    // heartbeat.join();
    insert_reading.join();
}
