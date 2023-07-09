// Producer-Consumer pattern implementation
#ifndef _PRO_CON_HPP_
#define _PRO_CON_HPP_

#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "yolo.hpp"
#include "meter_reader.hpp"

void run(int num_cam, int capacity, std::vector<std::string> stream_urls, int det_batch, int seg_batch, std::string det_model, std::string seg_model);

template<typename T>
class ProducerConsumer 
{
    public:
        ProducerConsumer(int capacity, int num_cam)
        {
            capacity_ = capacity;
            num_buffer_ = num_cam;
            stop_ = false;
            
            if (capacity_ < num_buffer_)
            {
                capacity_ = num_buffer_;
            }

            buffer_size_ = capacity_ / num_buffer_;

            for (int i = 0; i < num_buffer_; i++)
            {
                buffers_.push_back(std::queue<T>());
            }
        }

        void Produce(const T& item) 
        {
            std::unique_lock<std::mutex> lock(mutex_);
            // notFull_.wait(lock, [this] { return queue_.size() < capacity_ || stop_; });

            if (!stop_) 
            {
                if (queue_.size() < capacity_)
                {
                    queue_.push(item);
                    // std::cout << "Produced: " << item << std::endl;
                    notEmpty_.notify_all();
                }
            }

            lock.unlock();
        }

        T Consume() 
        {
            std::unique_lock<std::mutex> lock(mutex_);
            notEmpty_.wait(lock, [this] { return !queue_.empty() || stop_; });

            T item;
            if (!stop_) 
            {
                item = queue_.front();
                queue_.pop();
                // std::cout << "Consumed: " << item << std::endl;
                notFull_.notify_all();
            }

            return item;
        }

        void Write(const T& item, int thread_id) 
        {
            std::unique_lock<std::mutex> lock(mutex_);

            if (!stop_) 
            {
                buffers_[thread_id].push(item);
                if (buffers_[thread_id].size() > capacity_)
                {
                    buffers_[thread_id].pop();
                }
            }

            lock.unlock();
        }

        bool Read(T &item, int thread_id)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            notEmpty_.wait(lock, [this, thread_id] { return !buffers_[thread_id].empty() || stop_; });

            if (!stop_) 
            {
                item = buffers_[thread_id].front();
                return true;
            }
            return false;
        }

        void Stop() 
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            notEmpty_.notify_all();
            notFull_.notify_all();
        }

        std::mutex& GetMutex() 
        {
            return mutex_;
        }

        int GetNumBuffer() 
        {
            return num_buffer_;
        }

    private:
        std::queue<T> queue_;
        std::vector<std::queue<T> > buffers_;
        std::mutex mutex_;
        std::condition_variable notEmpty_;
        std::condition_variable notFull_;
        int capacity_;
        int buffer_size_;
        int num_buffer_;
        bool stop_;
};

#endif  // _PRO_CON_HPP_