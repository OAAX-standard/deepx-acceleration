// Copyright (c) 2022 DEEPX Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class ConcurrentQueue {
public:
    explicit ConcurrentQueue(size_t max_size) : _max_size(max_size) {}

    ConcurrentQueue(const ConcurrentQueue&) = delete;
    ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;

    // push data (wait if the queue is full)
    void push(const T& value) {
        std::unique_lock<std::mutex> lock(_mtx);
        _cv_push.wait(lock, [this]() { return _q.size() < _max_size; });
        _q.push(value);
        _cv_pop.notify_one();
    }

    // pop data (wait if the queue is empty)
    T pop() {
        std::unique_lock<std::mutex> lock(_mtx);
        _cv_pop.wait(lock, [this]() { return !_q.empty(); });
        T value = _q.front();
        _q.pop();
        _cv_push.notify_one();
        return value;
    }

    // try to pop data with timeout (returns false if timeout)
    bool try_pop(T& value, int timeout_ms) {
        std::unique_lock<std::mutex> lock(_mtx);
        if (!_cv_pop.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                             [this]() { return !_q.empty(); })) {
            return false; // timeout
        }
        value = _q.front();
        _q.pop();
        _cv_push.notify_one();
        return true;
    }

    // check if the queue is empty
    bool empty() const {
        std::unique_lock<std::mutex> lock(_mtx);
        return _q.empty();
    }

    // clear all items
    void clear() {
        std::unique_lock<std::mutex> lock(_mtx);
        while (!_q.empty()) {
            _q.pop();
        }
    }

    // current queue size
    size_t size() const {
        std::unique_lock<std::mutex> lock(_mtx);
        return _q.size();
    }

    // max queue size
    size_t maxSize() const {
        std::unique_lock<std::mutex> lock(_mtx);
        return _max_size;
    }

private:
    std::queue<T> _q; // stl queue
    mutable std::mutex _mtx;  // mutex lock
    std::condition_variable _cv_push; // push condition variable
    std::condition_variable _cv_pop;  // pop confition variable
    const size_t _max_size;  // max queue size
};
