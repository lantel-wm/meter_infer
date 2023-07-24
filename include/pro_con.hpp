// Producer-Consumer pattern implementation
#ifndef _PRO_CON_HPP_
#define _PRO_CON_HPP_

#include <iostream>
#include <queue>
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
            
            if (capacity_ < num_buffer_)
            {
                capacity_ = num_buffer_;
            }

            buffer_size_ = capacity_ / num_buffer_;

            for (int i = 0; i < num_buffer_; i++)
            {
                buffers_.push_back(std::queue<T>());
                stop_.push_back(false);
                active_.push_back(false);
            }
        }

        void Produce(const T& item) 
        {
            std::unique_lock<std::mutex> lock(mutex_);
            // notFull_.wait(lock, [this] { return queue_.size() < capacity_ || stop_; });

            if (!IsStopped()) 
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
            notEmpty_.wait(lock, [this] { return !queue_.empty() || IsStopped(); });

            T item;
            if (!IsStopped()) 
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

            if (!IsStopped()) 
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
            notEmpty_.wait(lock, [this, thread_id] { return !buffers_[thread_id].empty() || IsStopped(); });

            if (!IsStopped()) 
            {
                item = buffers_[thread_id].front();
                return true;
            }
            return false;
        }

        void Stop(int thread_id) 
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_[thread_id] = true;
            notEmpty_.notify_all();
            notFull_.notify_all();
        }

        void StopAll() 
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (int i = 0; i < num_buffer_; i++)
            {
                stop_[i] = true;
            }
            notEmpty_.notify_all();
            notFull_.notify_all();
        }

        void getBatch(std::vector<T> &batch)
        {
            std::unique_lock<std::mutex> lock(mutex_);

            if (!IsStopped()) 
            {
                for (int i = 0; i < num_buffer_; i++)
                {
                    batch[i] = buffers_[i].front();
                }
            }

            lock.unlock();
        }

        void SetActive(int thread_id)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            active_[thread_id] = true;
            lock.unlock();
        }

        void SetInactive(int thread_id)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            active_[thread_id] = false;
            lock.unlock();
        }

        bool IsActive(int thread_id)
        {
            return active_[thread_id];
        }

        bool IsAllActive()
        {
            for (int i = 0; i < num_buffer_; i++)
            {
                if (!active_[i])
                {
                    return false;
                }
            }
            return true;
        }

        std::mutex& GetMutex() 
        {
            return mutex_;
        }

        int GetNumBuffer() 
        {
            return num_buffer_;
        }

        bool IsStopped() 
        {
            for (int i = 0; i < num_buffer_; i++)
            {
                if (!stop_[i])
                {
                    return false;
                }
            }
            return true;
        }
        bool IsAllNotEmpty()
        {
            for (int i = 0; i < num_buffer_; i++)
            {
                if (buffers_[i].empty())
                {
                    return false;
                }
            }
            return true;
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
        std::vector<bool> stop_;
        std::vector<bool> active_;
        // bool stop_;
};

#endif  // _PRO_CON_HPP_