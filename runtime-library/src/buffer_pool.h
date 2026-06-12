// buffer_pool.h — bounded, lazily-grown buffer pool.
//
// Purpose: supply output buffers for DX-RT RunAsync calls from a reusable,
// capped set of allocations.  Capping the pool bounds memory use and provides
// backpressure (a full pool means too many requests are in flight), while
// growing lazily lets it scale automatically from a single channel up to many
// channels without manual tuning.  (Inputs are zero-copy and do not use a
// pool; this pool holds output buffers only.)
//
// Behavior:
//   - try_acquire() returns a pooled buffer immediately, or nullptr if the
//     pool is empty and already at its hard cap.  It never blocks: the runtime
//     turns a nullptr into RUNTIME_STATUS_TIMEOUT (backpressure) so callers can
//     drop the frame instead of stalling, per the OAAX non-blocking contract.
//   - release() returns the buffer to the pool for reuse; buffers are not
//     freed until the pool itself is destroyed.
//   - shutdown() makes subsequent try_acquire() calls fail during cleanup.
//
// Thread-safety: all methods are safe to call concurrently.

#pragma once

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <queue>

class BufferPool {
public:
    BufferPool(size_t buf_size, size_t hard_cap)
        : buf_size_(buf_size), hard_cap_(hard_cap == 0 ? 1 : hard_cap) {}

    ~BufferPool() {
        std::lock_guard<std::mutex> lk(mtx_);
        while (!free_buffers_.empty()) {
            std::free(free_buffers_.front());
            free_buffers_.pop();
        }
        allocated_ = 0;
    }

    BufferPool(const BufferPool &) = delete;
    BufferPool &operator=(const BufferPool &) = delete;

    // Returns a buffer of buf_size bytes immediately, or nullptr if the pool is
    // exhausted (at hard cap with none free) or shut down.  Never blocks.
    void *try_acquire() {
        std::lock_guard<std::mutex> lk(mtx_);
        if (shutting_down_) return nullptr;
        if (!free_buffers_.empty()) {
            void *p = free_buffers_.front();
            free_buffers_.pop();
            return p;
        }
        if (allocated_ >= hard_cap_) return nullptr;
        void *p = std::malloc(buf_size_);
        if (p) {
            std::memset(p, 0, buf_size_);
            ++allocated_;
        }
        return p;
    }

    void release(void *p) {
        if (!p) return;
        std::lock_guard<std::mutex> lk(mtx_);
        free_buffers_.push(p);
    }

    void shutdown() {
        std::lock_guard<std::mutex> lk(mtx_);
        shutting_down_ = true;
    }

    size_t allocated() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return allocated_;
    }
    size_t in_use() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return allocated_ - free_buffers_.size();
    }
    size_t capacity() const { return hard_cap_; }

private:
    mutable std::mutex mtx_;
    std::queue<void *> free_buffers_;
    size_t buf_size_;
    size_t hard_cap_;
    size_t allocated_{0};
    bool shutting_down_{false};
};
