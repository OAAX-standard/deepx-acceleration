// runtime_core.cpp — OAAX 2.0 runtime for DeepX NPUs (callback-based).
//
// Design overview:
//   - DX-RT scheduling (device & core distribution) is delegated to libdxrt.
//     We pass the default InferenceOption so empty `devices` selects all
//     available NPUs and boundOption=NPU_ALL uses every core.
//   - Each model is encapsulated in a ModelContext (engine + output pool +
//     in-flight counter).  Single-model today, multi-model ready.
//   - Inference completion is delivered via dxrt::InferenceEngine::
//     RegisterCallback.  No background polling thread, no per-job Wait():
//     completions arrive on DX-RT's threads (in no guaranteed order) and are
//     pushed to a single lock-free output queue.  Callers match each result
//     to its request via the echoed Tensors::id, so ordering is irrelevant.
//     This removes the head-of-line block that capped throughput under
//     multi-channel load.
//   - Input is zero-copy: the caller's input_tensors pointer is handed
//     straight to RunAsync and kept alive (owned by the JobData) until the
//     completion callback fires, by which point DX-RT's asynchronous input
//     encoding has already consumed it.  No input buffer pool, no input memcpy.
//   - The output buffer pool grows lazily up to DXRT_TASK_MAX_LOAD_VALUE ×
//     NumDevice, matching DX-RT's own in-flight limit.  It caps memory and
//     applies backpressure without manual tuning: a 1-channel setup uses a
//     tiny pool, a 32-channel quad-H1 setup a large one.

#include "runtime_core.h"

#include <dxrt/dxrt_api.h>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <concurrentqueue.h>

#include "buffer_pool.h"

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

namespace {

struct JobData {
    int model_id;
    int request_id;
    Tensors *input_tensors;  // caller's input; owned here, freed in completion callback
    void *outputs_ptr;
    dxrt::TensorPtrs dxrt_outputs;
};

// ---------------------------------------------------------------------------
// OutputSemaphore — counts completed jobs available in the lock-free queue.
//
// The lock-free ConcurrentQueue carries the job pointers with no shared lock,
// but it has no built-in way to *block* a consumer until an item arrives.
// This tiny counting semaphore supplies that wait/notify signal: signal() runs
// once per enqueue, wait() consumes one token (honouring OAAX timeout
// semantics).  Its mutex only guards an integer increment/decrement — a
// nanosecond-scale critical section — so it does not reintroduce the
// contention of guarding the whole queue.  Because the count is persistent, no
// wakeup is ever lost (unlike a bare condition_variable paired with a
// lock-free queue).
// ---------------------------------------------------------------------------
class OutputSemaphore {
   public:
    void signal() {
        std::lock_guard<std::mutex> lk(m_);
        ++count_;
        cv_.notify_one();
    }

    // Consume one token. timeout_ms: 0 = poll, >0 = bounded wait, <0 = forever.
    // Returns false on timeout or when `abort` becomes true (shutdown).
    bool wait(int timeout_ms, const std::atomic<bool> &abort) {
        std::unique_lock<std::mutex> lk(m_);
        auto ready = [&] { return count_ > 0 || abort.load(); };
        if (timeout_ms == 0) {
            if (!ready()) return false;
        } else if (timeout_ms > 0) {
            if (!cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready)) return false;
        } else {
            cv_.wait(lk, ready);
        }
        if (count_ == 0) return false;  // woken by abort, no item
        --count_;
        return true;
    }

    void wake_all() {
        std::lock_guard<std::mutex> lk(m_);
        cv_.notify_all();
    }

   private:
    std::mutex m_;
    std::condition_variable cv_;
    long count_{0};
};

struct ModelContext {
    std::unique_ptr<dxrt::InferenceEngine> engine;
    std::vector<uint64_t> output_sizes;
    size_t input_size{0};
    std::unique_ptr<BufferPool> output_pool;

    std::atomic<int> in_flight{0};
    std::mutex in_flight_mtx;
    std::condition_variable in_flight_cv;
};

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

std::shared_ptr<spdlog::logger> g_logger;
std::atomic<bool> g_initialized{false};
std::atomic<bool> g_shutting_down{false};

std::vector<std::unique_ptr<ModelContext>> g_models;
std::mutex g_models_mtx;

// Lock-free MPMC queue of completed jobs.  DX-RT completion callbacks (many
// threads) enqueue; retrieve() consumers (many threads) dequeue — neither side
// serializes on a shared mutex, removing the single-lock contention that the
// std::queue + mutex design imposed under high channel counts.
moodycamel::ConcurrentQueue<JobData *> g_output_queue;

// Signals "a completed job is ready" so retrieve() can block with a timeout
// instead of busy-polling the lock-free queue.
OutputSemaphore g_output_sem;

std::string g_last_error;
std::string g_info_json;

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

void deep_free_tensors(Tensors *t) {
    if (!t) return;
    if (t->tensors) {
        for (int i = 0; i < t->num_tensors; ++i) {
            std::free(t->tensors[i].name);
            std::free(t->tensors[i].shape);
            std::free(t->tensors[i].data);
        }
        std::free(t->tensors);
    }
    std::free(t);
}

TensorElementType map_data_type(dxrt::DataType dtype) {
    switch (dtype) {
        case dxrt::UINT8:  return DATA_TYPE_UINT8;
        case dxrt::UINT16: return DATA_TYPE_UINT16;
        case dxrt::UINT32: return DATA_TYPE_UINT32;
        case dxrt::UINT64: return DATA_TYPE_UINT64;
        case dxrt::INT8:   return DATA_TYPE_INT8;
        case dxrt::INT16:  return DATA_TYPE_INT16;
        case dxrt::INT32:  return DATA_TYPE_INT32;
        case dxrt::INT64:  return DATA_TYPE_INT64;
        case dxrt::FLOAT:  return DATA_TYPE_FLOAT;
        default:           return DATA_TYPE_UNDEFINED;
    }
}

Tensors *build_output_tensors(const ModelContext &ctx, int request_id,
                              const dxrt::TensorPtrs &outputs) {
    const int num = static_cast<int>(outputs.size());
    if (num == 0 || static_cast<int>(ctx.output_sizes.size()) != num) {
        spdlog::error("Output tensor count mismatch: dxrt={}, expected={}", num,
                      ctx.output_sizes.size());
        return nullptr;
    }

    Tensors *t = static_cast<Tensors *>(std::malloc(sizeof(Tensors)));
    if (!t) return nullptr;
    t->id = request_id;
    t->num_tensors = num;
    t->tensors = static_cast<TensorDescriptor *>(std::calloc(num, sizeof(TensorDescriptor)));
    if (!t->tensors) {
        std::free(t);
        return nullptr;
    }

    for (int i = 0; i < num; ++i) {
        const auto &src = outputs[i];
        const auto name = src->name();
        const auto shape = src->shape();
        const auto dtype = src->type();
        const size_t size = ctx.output_sizes[i];

        t->tensors[i].name = strdup(name.c_str());
        t->tensors[i].data_type = map_data_type(dtype);
        t->tensors[i].rank = static_cast<int>(shape.size());
        t->tensors[i].shape = static_cast<int *>(std::malloc(shape.size() * sizeof(int)));
        t->tensors[i].data_size = size;
        t->tensors[i].data = std::malloc(size);

        if (!t->tensors[i].name || !t->tensors[i].shape || !t->tensors[i].data) {
            deep_free_tensors(t);
            return nullptr;
        }
        for (size_t j = 0; j < shape.size(); ++j) {
            t->tensors[i].shape[j] = static_cast<int>(shape[j]);
        }
        std::memcpy(t->tensors[i].data, src->data(), size);
    }
    return t;
}

// ---------------------------------------------------------------------------
// Output pool sizing — matches DX-RT's per-device in-flight limit, no manual
// tuning required.
// ---------------------------------------------------------------------------

size_t compute_pool_capacity() {
    int num_devices = 0;
    try {
        num_devices = dxrt::DeviceStatus::GetDeviceCount();
    } catch (...) {
        num_devices = 0;
    }
    if (num_devices < 1) num_devices = 1;
    const size_t per_device = static_cast<size_t>(DXRT_TASK_MAX_LOAD_VALUE);
    return per_device * static_cast<size_t>(num_devices);
}

}  // namespace

// ===========================================================================
// Public API
// ===========================================================================

RuntimeStatus runtime_init(Config config) {
    if (g_initialized.load()) {
        g_last_error = "Runtime already initialized";
        return RUNTIME_STATUS_ALREADY_INITIALIZED;
    }

    // Enable DX-RT dynamic CPU threading by default.  When a model has CPU
    // tasks (run via ONNX Runtime), a fixed thread pool can become the
    // bottleneck and leave the NPU under-utilized.  DX-RT reads this env var
    // (exact value "ON") while constructing the InferenceEngine, which happens
    // later in runtime_load_models — so it must be set here, before the engine
    // exists.  We only supply the default when nothing is set, so an explicit
    // operator setting (ON or OFF) is always respected; this spares service
    // deployments (e.g. systemd-launched servers) from having to inject the
    // variable into the unit environment.
    if (std::getenv("DXRT_DYNAMIC_CPU_THREAD") == nullptr) {
#ifdef _WIN32
        _putenv_s("DXRT_DYNAMIC_CPU_THREAD", "ON");
#else
        setenv("DXRT_DYNAMIC_CPU_THREAD", "ON", 0);
#endif
    }

    try {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("runtime.log", true);
        auto stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        g_logger = std::make_shared<spdlog::logger>(
            runtime_get_name(), spdlog::sinks_init_list{file_sink, stderr_sink});
        spdlog::set_default_logger(g_logger);
        spdlog::set_level(spdlog::level::info);
        spdlog::flush_on(spdlog::level::warn);
        spdlog::cfg::load_env_levels();  // honor SPDLOG_LEVEL env var
        spdlog::info("Runtime initialized");
    } catch (const spdlog::spdlog_ex &ex) {
        std::fprintf(stderr, "Warning: logger init failed: %s\n", ex.what());
    }

    for (int i = 0; i < config.length; ++i) {
        if (config.keys[i]) {
            spdlog::debug("Config[{}] = {}", config.keys[i],
                          config.values[i] ? config.values[i] : "(null)");
        }
    }

    g_shutting_down.store(false);
    g_initialized.store(true);
    return RUNTIME_STATUS_SUCCESS;
}

RuntimeStatus runtime_load_models(int num_models, const ModelConfig *model_configs) {
    if (!g_initialized.load()) {
        g_last_error = "Runtime not initialized";
        return RUNTIME_STATUS_NOT_INITIALIZED;
    }
    if (num_models <= 0 || !model_configs) {
        g_last_error = "Invalid argument";
        return RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    // Phase 1: single model only.  Future phases simply iterate.
    if (num_models != 1) {
        g_last_error = "Multiple models not yet supported";
        return RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    {
        std::lock_guard<std::mutex> lk(g_models_mtx);
        if (!g_models.empty()) {
            g_last_error = "Model already loaded";
            return RUNTIME_STATUS_INVALID_ARGUMENT;
        }
    }

    const ModelConfig &mc = model_configs[0];
    if (!mc.file_path) {
        g_last_error = "Model file path is null";
        return RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    {
        std::ifstream f(mc.file_path, std::ios::binary);
        if (!f) {
            g_last_error = std::string("Model file not found: ") + mc.file_path;
            spdlog::error("{}", g_last_error);
            return RUNTIME_STATUS_FILE_NOT_FOUND;
        }
    }

    spdlog::info("Loading model: {}", mc.file_path);

    auto ctx = std::unique_ptr<ModelContext>(new ModelContext());
    try {
        ctx->engine.reset(new dxrt::InferenceEngine(std::string(mc.file_path)));
    } catch (const std::exception &e) {
        g_last_error = std::string("Failed to load model: ") + e.what();
        spdlog::error("{}", g_last_error);
        return RUNTIME_STATUS_INVALID_MODEL;
    }

    ctx->output_sizes = ctx->engine->GetOutputTensorSizes();
    ctx->input_size = ctx->engine->GetInputSize();
    const uint64_t output_total = ctx->engine->GetOutputSize();

    const size_t cap = compute_pool_capacity();
    ctx->output_pool.reset(new BufferPool(output_total, cap));
    spdlog::info("Output pool capacity: {} (input_size={} bytes, output_size={} bytes)", cap,
                 ctx->input_size, output_total);

    // Register completion callback.  Captures the raw context pointer; the
    // engine is destroyed before the context, so this is safe.
    ModelContext *ctx_raw = ctx.get();
    ctx_raw->engine->RegisterCallback(
        [ctx_raw](dxrt::TensorPtrs &outputs, void *userArg) -> int {
            auto *jd = static_cast<JobData *>(userArg);
            if (jd) {
                jd->dxrt_outputs = outputs;
                // Inference is complete, so DX-RT's asynchronous input encoding
                // has already read the caller's input.  Release it now (zero-copy
                // path: nothing was copied at enqueue time).
                if (jd->input_tensors) {
                    deep_free_tensors(jd->input_tensors);
                    jd->input_tensors = nullptr;
                }
                // Lock-free push, then signal a waiting consumer.
                g_output_queue.enqueue(jd);
                g_output_sem.signal();
            }
            // Decrement in-flight count.  The atomic itself is the source of
            // truth; we only take the lock + notify on the 0-transition, which
            // is all cleanup waits for.  This keeps the steady-state completion
            // path lock-free (the common N->N-1 case touches only the atomic).
            if (ctx_raw->in_flight.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(ctx_raw->in_flight_mtx);
                ctx_raw->in_flight_cv.notify_all();
            }
            return 0;
        });

    {
        std::lock_guard<std::mutex> lk(g_models_mtx);
        g_models.push_back(std::move(ctx));
    }
    return RUNTIME_STATUS_SUCCESS;
}

RuntimeStatus runtime_enqueue_input(int model_id, Tensors *input_tensors) {
    if (!g_initialized.load()) {
        g_last_error = "Runtime not initialized";
        return RUNTIME_STATUS_NOT_INITIALIZED;
    }
    if (g_shutting_down.load()) {
        g_last_error = "Runtime is shutting down";
        return RUNTIME_STATUS_ERROR;
    }
    if (!input_tensors || input_tensors->num_tensors < 1 || !input_tensors->tensors) {
        g_last_error = "Invalid input tensors";
        return RUNTIME_STATUS_INVALID_TENSOR;
    }

    ModelContext *ctx = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_models_mtx);
        if (g_models.empty()) {
            g_last_error = "Model not loaded";
            return RUNTIME_STATUS_MODEL_NOT_LOADED;
        }
        if (model_id < 0 || static_cast<size_t>(model_id) >= g_models.size()) {
            g_last_error = "Invalid model_id";
            return RUNTIME_STATUS_INVALID_MODEL_ID;
        }
        ctx = g_models[model_id].get();
    }

    // Validate input size against the model's expected input before taking any
    // resources.  A mismatch in either direction means the caller's tensor does
    // not match this model: undersized would make DX-RT read past the buffer
    // (OOB), oversized feeds misaligned data and yields a silently wrong result.
    // Reject it; the caller keeps ownership of input_tensors.
    const size_t in_bytes = input_tensors->tensors[0].data_size;
    if (in_bytes != ctx->input_size) {
        g_last_error = "Input size " + std::to_string(in_bytes) +
                       " does not match model input size " + std::to_string(ctx->input_size);
        spdlog::error("{}", g_last_error);
        return RUNTIME_STATUS_TENSOR_SHAPE_MISMATCH;
    }

    // Non-blocking acquire of an output buffer.  On exhaustion we signal
    // backpressure with RUNTIME_STATUS_TIMEOUT — the only standard OAAX 2.0
    // status meaning "retry later" (the enum must not be extended) — so the
    // caller drops this frame instead of blocking.  Ownership of input_tensors
    // stays with the caller on any non-success return.
    void *outputs_ptr = ctx->output_pool->try_acquire();
    if (!outputs_ptr) {
        g_last_error = "Output pool exhausted (backpressure)";
        return RUNTIME_STATUS_TIMEOUT;
    }

    // Zero-copy: hand the caller's input buffer directly to RunAsync.  DX-RT
    // reads it asynchronously (input encoding runs on its worker threads after
    // RunAsync returns), so the buffer must stay alive until completion — we
    // transfer ownership to the JobData and free it in the callback.
    void *input_data = input_tensors->tensors[0].data;

    auto jd = std::unique_ptr<JobData>(new JobData());
    jd->model_id = model_id;
    jd->request_id = input_tensors->id;
    jd->input_tensors = input_tensors;
    jd->outputs_ptr = outputs_ptr;

    ctx->in_flight.fetch_add(1, std::memory_order_acq_rel);

    try {
        ctx->engine->RunAsync(input_data, jd.get(), outputs_ptr);
    } catch (const std::exception &e) {
        ctx->in_flight.fetch_sub(1, std::memory_order_acq_rel);
        {
            std::lock_guard<std::mutex> lk(ctx->in_flight_mtx);
            ctx->in_flight_cv.notify_all();
        }
        ctx->output_pool->release(outputs_ptr);
        g_last_error = std::string("RunAsync failed: ") + e.what();
        spdlog::error("{}", g_last_error);
        // Caller retains ownership of input_tensors on failure — do not free.
        return RUNTIME_STATUS_INFERENCE_ERROR;
    }

    // RunAsync took ownership of the JobData (via userArg).  The JobData now
    // owns input_tensors and frees them in the completion callback, once DX-RT
    // has finished reading the input.
    jd.release();
    return RUNTIME_STATUS_SUCCESS;
}

RuntimeStatus runtime_retrieve_output(int *model_id, Tensors **output_tensors, int timeout_ms) {
    if (!output_tensors) {
        g_last_error = "output_tensors pointer is null";
        return RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    *output_tensors = nullptr;
    if (!g_initialized.load()) {
        g_last_error = "Runtime not initialized";
        return RUNTIME_STATUS_NOT_INITIALIZED;
    }

    JobData *jd = nullptr;
    // Block (per OAAX timeout semantics) until a completed job is signalled.
    if (!g_output_sem.wait(timeout_ms, g_shutting_down)) {
        return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
    }
    // A token was acquired, so an item was enqueued with a happens-before
    // relationship established through the semaphore's mutex.  try_dequeue
    // normally succeeds on the first attempt; the yield-spin only covers the
    // rare window where another consumer momentarily holds the matching slot.
    while (!g_output_queue.try_dequeue(jd)) {
        if (g_shutting_down.load()) return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
        std::this_thread::yield();
    }

    ModelContext *ctx = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_models_mtx);
        if (jd->model_id < 0 || static_cast<size_t>(jd->model_id) >= g_models.size()) {
            // Should not happen unless cleanup races with retrieve.
            delete jd;
            g_last_error = "Stale model_id in output";
            return RUNTIME_STATUS_ERROR;
        }
        ctx = g_models[jd->model_id].get();
    }

    Tensors *result = build_output_tensors(*ctx, jd->request_id, jd->dxrt_outputs);

    // Return the output buffer to its pool regardless of build outcome.  The
    // input was already freed in the completion callback (zero-copy path).
    ctx->output_pool->release(jd->outputs_ptr);
    const int mid = jd->model_id;
    delete jd;

    if (!result) {
        g_last_error = "Failed to build output tensors";
        return RUNTIME_STATUS_OUT_OF_MEMORY;
    }

    *output_tensors = result;
    if (model_id) *model_id = mid;
    return RUNTIME_STATUS_SUCCESS;
}

RuntimeStatus runtime_cleanup(void) {
    if (!g_initialized.load()) return RUNTIME_STATUS_SUCCESS;

    spdlog::info("Runtime cleanup begin");
    g_shutting_down.store(true);
    g_output_sem.wake_all();

    // 1) Wait for every in-flight job to complete (callback fired).
    {
        std::lock_guard<std::mutex> lk(g_models_mtx);
        for (auto &ctx : g_models) {
            std::unique_lock<std::mutex> wlk(ctx->in_flight_mtx);
            ctx->in_flight_cv.wait(wlk, [&] { return ctx->in_flight.load() == 0; });
        }
    }

    // 2) Tear down engines (joins DX-RT internal threads — no more callbacks).
    {
        std::lock_guard<std::mutex> lk(g_models_mtx);
        for (auto &ctx : g_models) {
            ctx->engine.reset();
        }
    }

    // 3) Drain any outputs that arrived between in_flight==0 and engine reset.
    //    (Callbacks decrement in_flight last, so the queue push happened first.)
    {
        std::lock_guard<std::mutex> mlk(g_models_mtx);
        JobData *jd = nullptr;
        while (g_output_queue.try_dequeue(jd)) {
            if (jd->input_tensors) deep_free_tensors(jd->input_tensors);
            if (jd->model_id >= 0 && static_cast<size_t>(jd->model_id) < g_models.size()) {
                auto &ctx = g_models[jd->model_id];
                if (ctx->output_pool) ctx->output_pool->release(jd->outputs_ptr);
            }
            delete jd;
        }
    }

    // 4) Shutdown + free pools, then destroy contexts.
    {
        std::lock_guard<std::mutex> lk(g_models_mtx);
        for (auto &ctx : g_models) {
            if (ctx->output_pool) ctx->output_pool->shutdown();
        }
        g_models.clear();
    }

    g_initialized.store(false);
    g_shutting_down.store(false);
    g_last_error.clear();

    spdlog::info("Runtime cleanup done");
    if (g_logger) {
        g_logger->flush();
        g_logger.reset();
    }
    return RUNTIME_STATUS_SUCCESS;
}

const char *runtime_get_error(void) {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

const char *runtime_get_version(void) { return OAAX_RUNTIME_VERSION; }

const char *runtime_get_name(void) { return "DEEPX"; }

const char *runtime_get_info(void) {
    if (!g_initialized.load()) return nullptr;

    size_t loaded = 0;
    size_t in_flight = 0;
    size_t pool_in_use = 0;
    size_t pool_size = 0;
    size_t pool_capacity = 0;
    {
        std::lock_guard<std::mutex> lk(g_models_mtx);
        loaded = g_models.size();
        for (auto &ctx : g_models) {
            in_flight += ctx->in_flight.load();
            if (ctx->output_pool) {
                pool_in_use += ctx->output_pool->in_use();
                pool_size += ctx->output_pool->allocated();
                pool_capacity += ctx->output_pool->capacity();
            }
        }
    }
    // size_approx() is a lock-free estimate — exact counts aren't possible (or
    // needed) for a concurrently-mutated lock-free queue.
    size_t queued = g_output_queue.size_approx();

    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "{\"loaded_models\": %zu, \"requests_in_flight\": %zu, "
                  "\"output_queue\": %zu, \"pool_in_use\": %zu, "
                  "\"pool_size\": %zu, \"pool_capacity\": %zu}",
                  loaded, in_flight, queued, pool_in_use, pool_size, pool_capacity);
    g_info_json = buf;
    return g_info_json.c_str();
}
