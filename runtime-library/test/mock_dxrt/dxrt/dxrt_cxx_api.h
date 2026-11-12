#pragma once

// Minimal mock of the DX-RT API surface used by runtime_core.cpp, so the runtime
// can be unit-tested without DeepX hardware or the real libdxrt.
// Keep it in sync with the calls made in src/runtime_core.cpp.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Real DX-RT derives this from the device; a small fixed value keeps the
// backpressure test fast (pool capacity = this * device count).
inline int dxrt_get_task_max_load() { return 8; }

namespace dxrt {

enum DataType { FLOAT, UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64 };

// Test knob: make mock outputs rank-0 (scalar), shape = {}.
inline std::atomic<bool> &scalar_output() {
    static std::atomic<bool> v{false};
    return v;
}

// Test knob: RegisterCallback lets this many calls through, then throws
// std::bad_alloc once (negative = never).  Exercises load rollback.
inline std::atomic<int> &register_callback_throws_after() {
    static std::atomic<int> v{-1};
    return v;
}

// Test knob: this many of the next RunAsync calls throw std::runtime_error.
inline std::atomic<int> &run_async_failures() {
    static std::atomic<int> v{0};
    return v;
}

// Test knob: the first int of every RunAsync input, in call order.
struct RunAsyncLog {
    std::mutex mtx;
    std::vector<int> first_ints;
};
inline RunAsyncLog &run_async_log() {
    static RunAsyncLog log;
    return log;
}

class Tensor {
public:
    Tensor(size_t size, uint8_t fill) : data_buf_(malloc(size)), size_(size) {
        if (data_buf_) memset(data_buf_, fill, size);
    }
    ~Tensor() { free(data_buf_); }

    std::string name() const { return "output_0"; }
    std::vector<int64_t> shape() const {
        if (scalar_output().load()) return {};
        return {1, 3, 224, 224};
    }
    DataType type() const { return FLOAT; }
    void *data() const { return data_buf_; }

private:
    void *data_buf_;
    size_t size_;
};

using TensorPtrs = std::vector<std::shared_ptr<Tensor>>;

class InferenceEngine {
public:
    explicit InferenceEngine(const std::string &) {}

    // A valid in-memory model is "DXNN" followed by the byte outputs are filled
    // with.  Like DX-RT, everything needed is taken during construction; the
    // buffer is never read afterwards.
    InferenceEngine(const uint8_t *data, size_t size) {
        if (size < 5 || std::memcmp(data, "DXNN", 4) != 0) throw std::runtime_error("mock: invalid model buffer");
        fill_ = data[4];
    }

    // Like the real engine, destruction waits for every in-flight job (and its callback).
    ~InferenceEngine() {
        std::vector<std::thread> threads;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            threads.swap(threads_);
        }
        for (auto &t : threads) t.join();
    }

    std::vector<uint64_t> GetOutputTensorSizes() { return {1 * 3 * 224 * 224 * sizeof(float)}; }
    uint64_t GetOutputSize() { return 1 * 3 * 224 * 224 * sizeof(float); }
    uint64_t GetInputSize() { return 1 * 3 * 224 * 224 * sizeof(float); }

    void RegisterCallback(std::function<int(TensorPtrs &, void *)> cb) {
        const int left = register_callback_throws_after().load();
        if (left >= 0) {
            register_callback_throws_after().store(left - 1);
            if (left == 0) throw std::bad_alloc();
        }
        callback_ = std::move(cb);
    }

    // Completes asynchronously on a worker thread (5~15 ms simulated latency) and
    // invokes the registered callback, like DX-RT's completion threads do.
    int RunAsync(void *input, void *userArg, void *) {
        int failures = run_async_failures().load();
        while (failures > 0 && !run_async_failures().compare_exchange_weak(failures, failures - 1)) {
        }
        if (failures > 0) throw std::runtime_error("mock: RunAsync failed");
        {
            int first = 0;
            std::memcpy(&first, input, sizeof(first));
            std::lock_guard<std::mutex> lk(run_async_log().mtx);
            run_async_log().first_ints.push_back(first);
        }
        const int job_id = next_job_id_.fetch_add(1);
        std::lock_guard<std::mutex> lk(mtx_);
        threads_.emplace_back([this, job_id, userArg]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(5 + (job_id % 11)));
            TensorPtrs outputs{std::make_shared<Tensor>(GetOutputSize(), fill_)};
            if (callback_) callback_(outputs, userArg);
        });
        return job_id;
    }

private:
    uint8_t fill_{0x42};  // output fill byte: buf[4] for in-memory models
    std::function<int(TensorPtrs &, void *)> callback_;
    std::atomic<int> next_job_id_{0};
    std::mutex mtx_;
    std::vector<std::thread> threads_;
};

struct DeviceStatus {
    static int GetDeviceCount() { return 2; }
};

}  // namespace dxrt
