// Producer-Consumer pattern implementation
#ifndef _PRO_CON_HPP_
#define _PRO_CON_HPP_

#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

template<typename T>
class ProducerConsumer {
public:
    ProducerConsumer(int capacity) : capacity_(capacity), stop_(false) {}

    void Produce(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        notFull_.wait(lock, [this] { return queue_.size() < capacity_ || stop_; });

        if (!stop_) {
            queue_.push(item);
            // std::cout << "Produced: " << item << std::endl;
            notEmpty_.notify_all();
        }
    }

    T Consume() {
        std::unique_lock<std::mutex> lock(mutex_);
        notEmpty_.wait(lock, [this] { return !queue_.empty() || stop_; });

        T item;
        if (!stop_) {
            item = queue_.front();
            queue_.pop();
            // std::cout << "Consumed: " << item << std::endl;
            notFull_.notify_all();
        }

        return item;
    }

    T Peek() {
        std::unique_lock<std::mutex> lock(mutex_);
        notEmpty_.wait(lock, [this] { return !queue_.empty() || stop_; });

        T item;
        if (!stop_) {
            item = queue_.front();
        }

        return item;
    }

    void Stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

    std::mutex& GetMutex() {
        return mutex_;
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::condition_variable notFull_;
    int capacity_;
    bool stop_;
};

#endif // _PRO_CON_HPP_