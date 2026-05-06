#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>
#include <atomic>

namespace dxrt {

enum DataType { FLOAT, UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64 };

class Tensor {
public:
    explicit Tensor(size_t size, int id = 0)
        : data_buf_(malloc(size)), size_(size) {
        if (data_buf_) memset(data_buf_, 0x42, size); // fill with pattern
    }
    ~Tensor() { free(data_buf_); }

    std::string name() const { return "output_0"; }
    std::vector<int64_t> shape() const { return {1, 3, 224, 224}; }
    DataType type() const { return FLOAT; }
    void *data() const { return data_buf_; }

private:
    void *data_buf_;
    size_t size_;
};

class InferenceEngine {
public:
    explicit InferenceEngine(const std::string &) {}
    InferenceEngine(const uint8_t *, size_t) {}

    std::vector<uint64_t> GetOutputTensorSizes() {
        return {1 * 3 * 224 * 224 * sizeof(float)};
    }

    uint64_t GetOutputSize() {
        return 1 * 3 * 224 * 224 * sizeof(float);
    }

    int RunAsync(uint8_t *, void *, void *outputs_ptr) {
        (void)outputs_ptr;
        return next_job_id_.fetch_add(1);
    }

    std::vector<std::shared_ptr<Tensor>> Wait(int job_id) {
        (void)job_id;
        // Simulate inference latency (5~15ms random)
        int delay = 5 + (job_id % 11);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        return {std::make_shared<Tensor>(GetOutputSize(), job_id)};
    }

private:
    std::atomic<int> next_job_id_{0};
};

struct DeviceStatus {
    static int GetDeviceCount() { return 2; }
};

} // namespace dxrt
