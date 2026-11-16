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
//     pool is empty and already at its hard cap.  It never blocks, so the
//     runtime can queue the request or apply backpressure instead of stalling,
//     per the OAAX non-blocking contract.  It throws std::bad_alloc if a new
//     buffer cannot be allocated.
//   - release() returns the buffer to the pool for reuse; buffers are not
//     freed until the pool itself is destroyed.  It never throws: the free
//     list reserves room for every buffer up front.
//   - shutdown() makes subsequent try_acquire() calls fail during cleanup.
//
// Thread-safety: all methods are safe to call concurrently.

#pragma once

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

class BufferPool {
public:
    // Throws std::bad_alloc if the free list cannot be reserved.
    BufferPool(size_t buf_size, size_t hard_cap) : buf_size_(buf_size), hard_cap_(hard_cap == 0 ? 1 : hard_cap) {
        free_buffers_.reserve(hard_cap_);
    }

    ~BufferPool() {
        std::lock_guard<std::mutex> lk(mtx_);
        for (void *p : free_buffers_) std::free(p);
    }

    BufferPool(const BufferPool &) = delete;
    BufferPool &operator=(const BufferPool &) = delete;

    // Returns a buffer of buf_size bytes immediately, or nullptr if the pool is
    // exhausted (at hard cap with none free) or shut down.  Never blocks.
    // Throws std::bad_alloc if allocating a new buffer fails.
    void *try_acquire() {
        std::lock_guard<std::mutex> lk(mtx_);
        if (shutting_down_) return nullptr;
        if (!free_buffers_.empty()) {
            void *p = free_buffers_.back();
            free_buffers_.pop_back();
            return p;
        }
        if (allocated_ >= hard_cap_) return nullptr;
        void *p = std::malloc(buf_size_);
        if (!p) throw std::bad_alloc();
        std::memset(p, 0, buf_size_);
        ++allocated_;
        return p;
    }

    void release(void *p) noexcept {
        if (!p) return;
        std::lock_guard<std::mutex> lk(mtx_);
        free_buffers_.push_back(p);  // within the reserved capacity
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
    std::vector<void *> free_buffers_;  // capacity hard_cap_
    size_t buf_size_;
    size_t hard_cap_;
    size_t allocated_{0};
    bool shutting_down_{false};
};
