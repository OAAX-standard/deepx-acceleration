#include "runtime_core.h"

#include <dxrt/dxrt_api.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <stdio.h>
#include <string.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

struct JobData {
    int job_id;
    int model_id;
    int request_id;
    void *outputs_ptr;
    void *input_buf;   // points into the input_buf_pool — returned to pool after Wait
    Tensors *input_tensors;
    std::vector<std::shared_ptr<dxrt::Tensor>> dxrt_outputs;
};

// ---------------------------------------------------------------------------
// Static state
// ---------------------------------------------------------------------------

static std::shared_ptr<spdlog::logger> logger;
static std::atomic<bool> g_initialized{false};

static dxrt::InferenceEngine *inference_engine = nullptr;
static std::vector<uint64_t> OutputTensorSizes;

// TODO(multi-model): When supporting multiple models concurrently, each model
// will have different input/output sizes.  The current global pools and
// INPUT_BUF_SIZE must be refactored into a per-model context structure, e.g.:
//   struct ModelContext {
//       InferenceEngine *engine;
//       std::queue<void*> input_buf_pool;
//       std::queue<void*> outputs_ptr_pool;
//       size_t input_buf_size;
//       std::vector<uint64_t> output_tensor_sizes;
//   };
// TODO(pool-config): The pool capacity multiplier (currently ×10) is derived
// from observation of DX-RT DMA registration limits.  Consider making it
// configurable via runtime Config or an environment variable to accommodate
// different hardware revisions or higher pipeline depths.
static size_t OUTPUTS_POOL_CAPACITY = 0;
static size_t NumDevice = 0;

static std::queue<void *> outputs_ptr_pool;
static std::mutex outputs_pool_mutex;
static std::condition_variable outputs_pool_cv;

// Pre-allocated input buffer pool — each RunAsync reuses a pooled buffer so
// the same physical addresses are registered with the DX-RT DMA engine across
// repeated inferences.  Submitting a freshly malloc-ed pointer on every call
// causes DX-RT to exhaust its internal DMA registration table (observed as
// errno=-70 WRITE_INPUT failures starting at reqId=10 for SqueezeNet).
static std::queue<void *> input_buf_pool;
static std::mutex input_buf_pool_mutex;
static std::condition_variable input_buf_pool_cv;
static size_t INPUT_BUF_SIZE = 0;

static std::queue<JobData> job_data_queue;
static std::mutex job_data_queue_mutex;
static std::condition_variable job_data_queue_cv;

static std::queue<JobData> output_queue;
static std::mutex output_queue_mutex;
static std::condition_variable output_queue_cv;

static std::atomic<bool> stop_wait_thread{false};
static std::atomic<bool> wait_thread_started{false};
static std::thread wait_thread;

static std::string g_last_error;
static std::string g_info_json;

// ---------------------------------------------------------------------------
// Helper: free Tensors (OAAX 2.0 structure)
// ---------------------------------------------------------------------------

static void deep_free_tensors(Tensors *t) {
    if (!t) return;
    if (t->tensors) {
        for (int i = 0; i < t->num_tensors; i++) {
            free(t->tensors[i].name);
            free(t->tensors[i].shape);
            free(t->tensors[i].data);
        }
        free(t->tensors);
    }
    free(t);
}

// ---------------------------------------------------------------------------
// Helper: map dxrt DataType to TensorElementType
// ---------------------------------------------------------------------------

static TensorElementType mapDataType(dxrt::DataType dtype) {
    switch (dtype) {
        case dxrt::UINT8:
            return DATA_TYPE_UINT8;
        case dxrt::UINT16:
            return DATA_TYPE_UINT16;
        case dxrt::UINT32:
            return DATA_TYPE_UINT32;
        case dxrt::UINT64:
            return DATA_TYPE_UINT64;
        case dxrt::INT8:
            return DATA_TYPE_INT8;
        case dxrt::INT16:
            return DATA_TYPE_INT16;
        case dxrt::INT32:
            return DATA_TYPE_INT32;
        case dxrt::INT64:
            return DATA_TYPE_INT64;
        case dxrt::FLOAT:
            return DATA_TYPE_FLOAT;
        default:
            return DATA_TYPE_UNDEFINED;
    }
}

// ---------------------------------------------------------------------------
// Helper: create output Tensors with pre-allocated data buffers
// ---------------------------------------------------------------------------

static Tensors *create_output_tensors(int request_id) {
    int num_tensors = (int)OutputTensorSizes.size();
    if (num_tensors == 0) return nullptr;

    Tensors *t = (Tensors *)malloc(sizeof(Tensors));
    if (!t) return nullptr;

    t->id = request_id;
    t->num_tensors = num_tensors;
    t->tensors = (TensorDescriptor *)calloc(num_tensors, sizeof(TensorDescriptor));
    if (!t->tensors) {
        free(t);
        return nullptr;
    }

    for (int i = 0; i < num_tensors; i++) {
        t->tensors[i].data_size = OutputTensorSizes[i];
        t->tensors[i].data = malloc(OutputTensorSizes[i]);
        if (!t->tensors[i].data) {
            deep_free_tensors(t);
            return nullptr;
        }
    }
    return t;
}

// ---------------------------------------------------------------------------
// Helper: copy dxrt outputs into the Tensors structure
// ---------------------------------------------------------------------------

static Tensors *copy_dxrt_outputs(const std::vector<std::shared_ptr<dxrt::Tensor>> &outputs, Tensors *out_tensors) {
    if (!out_tensors) return nullptr;

    int num = (int)outputs.size();
    if (num == 0 || out_tensors->num_tensors != num) {
        spdlog::error("Output tensor count mismatch: dxrt={}, expected={}", num, out_tensors->num_tensors);
        return nullptr;
    }

    for (int i = 0; i < num; i++) {
        const auto &output = outputs[i];

        auto name = output->name();
        auto shape = output->shape();
        auto dtype = output->type();
        auto data = output->data();

        out_tensors->tensors[i].name = strdup(name.c_str());
        if (!out_tensors->tensors[i].name) {
            spdlog::error("Failed to allocate name for tensor {}", i);
            deep_free_tensors(out_tensors);
            return nullptr;
        }

        out_tensors->tensors[i].data_type = mapDataType(dtype);
        out_tensors->tensors[i].rank = (int)shape.size();
        out_tensors->tensors[i].shape = (int *)malloc(shape.size() * sizeof(int));
        if (!out_tensors->tensors[i].shape) {
            spdlog::error("Failed to allocate shape for tensor {}", i);
            deep_free_tensors(out_tensors);
            return nullptr;
        }

        for (size_t j = 0; j < shape.size(); j++) {
            out_tensors->tensors[i].shape[j] = (int)shape[j];
        }

        memcpy(out_tensors->tensors[i].data, data, OutputTensorSizes[i]);
    }
    return out_tensors;
}

// ---------------------------------------------------------------------------
// wait_loop: background thread that waits for inference completion
// ---------------------------------------------------------------------------

static void wait_loop() {
    while (true) {
        JobData job_data{};
        {
            std::unique_lock<std::mutex> lock(job_data_queue_mutex);
            job_data_queue_cv.wait(lock, []() { return stop_wait_thread.load() || !job_data_queue.empty(); });
            if (stop_wait_thread.load() && job_data_queue.empty()) {
                break;
            }
            job_data = job_data_queue.front();
            job_data_queue.pop();
        }

        spdlog::trace("[wait_loop] calling Wait for job_id={} request_id={}", job_data.job_id,
                      job_data.request_id);
        try {
            job_data.dxrt_outputs = inference_engine->Wait(job_data.job_id);
            spdlog::trace("[wait_loop] Wait completed job_id={} request_id={} num_outputs={}", job_data.job_id,
                          job_data.request_id, job_data.dxrt_outputs.size());
        } catch (const std::exception &e) {
            spdlog::error("[wait_loop] Wait failed job_id={} request_id={}: {}", job_data.job_id,
                          job_data.request_id, e.what());
            if (job_data.input_tensors) deep_free_tensors(job_data.input_tensors);
            if (job_data.input_buf) {
                std::lock_guard<std::mutex> lock(input_buf_pool_mutex);
                input_buf_pool.push(job_data.input_buf);
                input_buf_pool_cv.notify_one();
            }
            if (job_data.outputs_ptr) {
                std::lock_guard<std::mutex> lock(outputs_pool_mutex);
                outputs_ptr_pool.push(job_data.outputs_ptr);
                outputs_pool_cv.notify_one();
            }
            continue;
        } catch (...) {
            spdlog::error("[wait_loop] Wait failed job_id={} request_id={}: unknown exception", job_data.job_id,
                          job_data.request_id);
            if (job_data.input_tensors) deep_free_tensors(job_data.input_tensors);
            if (job_data.input_buf) {
                std::lock_guard<std::mutex> lock(input_buf_pool_mutex);
                input_buf_pool.push(job_data.input_buf);
                input_buf_pool_cv.notify_one();
            }
            if (job_data.outputs_ptr) {
                std::lock_guard<std::mutex> lock(outputs_pool_mutex);
                outputs_ptr_pool.push(job_data.outputs_ptr);
                outputs_pool_cv.notify_one();
            }
            continue;
        }

        if (job_data.input_tensors) {
            deep_free_tensors(job_data.input_tensors);
            job_data.input_tensors = nullptr;
        }

        // Return input buffer to pool so the same physical address is reused
        if (job_data.input_buf) {
            std::lock_guard<std::mutex> lock(input_buf_pool_mutex);
            input_buf_pool.push(job_data.input_buf);
            input_buf_pool_cv.notify_one();
        }

        {
            std::lock_guard<std::mutex> lock(output_queue_mutex);
            output_queue.push(job_data);
            output_queue_cv.notify_one();
            spdlog::trace("[wait_loop] pushed job_id={} request_id={} to output_queue (queue_size={})",
                          job_data.job_id, job_data.request_id, output_queue.size());
        }
    }
}

// ===========================================================================
// Public API implementation (OAAX 2.0)
// ===========================================================================

RuntimeStatus runtime_init(Config config) {
    if (g_initialized.load()) {
        g_last_error = "Runtime already initialized";
        return RUNTIME_STATUS_ALREADY_INITIALIZED;
    }

    try {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("runtime.log", true);
        auto stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        logger = std::make_shared<spdlog::logger>(runtime_get_name(),
                                                  spdlog::sinks_init_list{file_sink, stderr_sink});
        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::trace);
        spdlog::flush_on(spdlog::level::trace);
        spdlog::info("Initializing the runtime environment");
    } catch (const spdlog::spdlog_ex &ex) {
        printf("Warning: Failed to create logger: %s, continuing without file logging\n", ex.what());
    }

    for (int i = 0; i < config.length; i++) {
        if (config.keys[i]) {
            spdlog::debug("Config key: {} = {}", config.keys[i], config.values[i] ? config.values[i] : "(null)");
        }
    }

    g_initialized.store(true);
    return RUNTIME_STATUS_SUCCESS;
}

RuntimeStatus runtime_load_models(int num_models, const ModelConfig *model_configs) {
    if (!g_initialized.load()) {
        g_last_error = "Runtime not initialized";
        return RUNTIME_STATUS_NOT_INITIALIZED;
    }

    if (num_models <= 0 || !model_configs) {
        g_last_error = "Invalid argument: num_models or model_configs";
        return RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    // Phase 1: single model support only
    if (num_models != 1) {
        g_last_error = "Multiple models not yet supported (Phase 1)";
        return RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    const ModelConfig &mc = model_configs[0];
    if (!mc.file_path) {
        g_last_error = "Model file path is null";
        return RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    {
        std::ifstream f(mc.file_path, std::ios::binary);
        if (!f) {
            spdlog::error("Model file does not exist: {}", mc.file_path);
            g_last_error = std::string("Model file not found: ") + mc.file_path;
            return RUNTIME_STATUS_FILE_NOT_FOUND;
        }
    }

    spdlog::info("Loading model from: {}", mc.file_path);

    try {
        inference_engine = new dxrt::InferenceEngine(std::string(mc.file_path));
        if (inference_engine == nullptr) {
            g_last_error = "Failed to create inference engine";
            spdlog::error("{}", g_last_error);
            return RUNTIME_STATUS_ERROR;
        }

        NumDevice = dxrt::DeviceStatus::GetDeviceCount();
        OUTPUTS_POOL_CAPACITY = NumDevice * 10;

        OutputTensorSizes = inference_engine->GetOutputTensorSizes();
        uint64_t OutputSize = inference_engine->GetOutputSize();
        {
            std::lock_guard<std::mutex> lock(outputs_pool_mutex);
            while (!outputs_ptr_pool.empty()) outputs_ptr_pool.pop();
            for (size_t i = 0; i < OUTPUTS_POOL_CAPACITY; ++i) {
                void *buf = malloc(OutputSize);
                if (!buf) {
                    spdlog::error("Failed to allocate output buffer {}", i);
                    continue;
                }
                outputs_ptr_pool.push(buf);
            }
            spdlog::info("Initialized outputs_ptr_pool with {} buffers for {} devices", outputs_ptr_pool.size(),
                         NumDevice);
        }
        outputs_pool_cv.notify_one();

        // Initialize the input buffer pool with the same capacity as the output
        // pool.  Reusing these fixed allocations across RunAsync calls keeps the
        // same physical addresses registered with the DX-RT DMA engine,
        // preventing the errno=-70 WRITE_INPUT failures that occur when fresh
        // malloc pointers are submitted on every call.
        INPUT_BUF_SIZE = inference_engine->GetInputSize();
        {
            std::lock_guard<std::mutex> lock(input_buf_pool_mutex);
            while (!input_buf_pool.empty()) input_buf_pool.pop();
            for (size_t i = 0; i < OUTPUTS_POOL_CAPACITY; ++i) {
                void *buf = malloc(INPUT_BUF_SIZE);
                if (!buf) {
                    spdlog::error("Failed to allocate input buffer {}", i);
                    continue;
                }
                memset(buf, 0, INPUT_BUF_SIZE);
                input_buf_pool.push(buf);
            }
            spdlog::info("Initialized input_buf_pool with {} buffers (each {} bytes)", input_buf_pool.size(),
                         INPUT_BUF_SIZE);
        }
        input_buf_pool_cv.notify_one();

        stop_wait_thread.store(false);
        try {
            wait_thread = std::thread(wait_loop);
            wait_thread_started.store(true);
        } catch (...) {
            spdlog::error("Failed to create wait thread");
            {
                std::lock_guard<std::mutex> lock(outputs_pool_mutex);
                while (!outputs_ptr_pool.empty()) {
                    free(outputs_ptr_pool.front());
                    outputs_ptr_pool.pop();
                }
            }
            {
                std::lock_guard<std::mutex> lock(input_buf_pool_mutex);
                while (!input_buf_pool.empty()) {
                    free(input_buf_pool.front());
                    input_buf_pool.pop();
                }
            }
            delete inference_engine;
            inference_engine = nullptr;
            g_last_error = "Failed to create wait thread";
            return RUNTIME_STATUS_ERROR;
        }

        return RUNTIME_STATUS_SUCCESS;
    } catch (const std::exception &e) {
        spdlog::error("Failed to load model: {}", e.what());
        g_last_error = std::string("Failed to load model: ") + e.what();
        return RUNTIME_STATUS_INVALID_MODEL;
    }
}

RuntimeStatus runtime_enqueue_input(int model_id, Tensors *input_tensors) {
    if (!g_initialized.load()) {
        g_last_error = "Runtime not initialized";
        return RUNTIME_STATUS_NOT_INITIALIZED;
    }

    if (!inference_engine) {
        g_last_error = "Model not loaded";
        return RUNTIME_STATUS_MODEL_NOT_LOADED;
    }

    // Phase 1: single model, model_id must be 0
    if (model_id != 0) {
        g_last_error = "Invalid model_id";
        return RUNTIME_STATUS_INVALID_MODEL_ID;
    }

    if (!input_tensors || input_tensors->num_tensors < 1 || !input_tensors->tensors) {
        g_last_error = "Invalid input tensors";
        return RUNTIME_STATUS_INVALID_TENSOR;
    }

    spdlog::trace("[enqueue] request_id={} model_id={} input_data_size={}", input_tensors->id, model_id,
                  input_tensors->tensors[0].data_size);

    void *outputs_ptr = nullptr;
    {
        std::unique_lock<std::mutex> lock(outputs_pool_mutex);
        spdlog::trace("[enqueue] waiting for output buffer (pool_size={} pool_capacity={})",
                      outputs_ptr_pool.size(), OUTPUTS_POOL_CAPACITY);
        outputs_pool_cv.wait(lock, []() { return !outputs_ptr_pool.empty() || stop_wait_thread.load(); });
        if (stop_wait_thread.load() && outputs_ptr_pool.empty()) {
            g_last_error = "Runtime is shutting down";
            return RUNTIME_STATUS_ERROR;
        }
        outputs_ptr = outputs_ptr_pool.front();
        outputs_ptr_pool.pop();
        spdlog::trace("[enqueue] acquired output buffer (pool_size_remaining={})", outputs_ptr_pool.size());
    }

    // Acquire a pooled input buffer and copy the caller's data into it.
    // Using a reused allocation keeps the same physical addresses registered
    // with the DX-RT DMA engine, avoiding errno=-70 failures that occur when
    // a fresh malloc pointer is submitted on every RunAsync call.
    void *input_buf = nullptr;
    {
        std::unique_lock<std::mutex> lock(input_buf_pool_mutex);
        input_buf_pool_cv.wait(lock, []() { return !input_buf_pool.empty() || stop_wait_thread.load(); });
        if (stop_wait_thread.load() && input_buf_pool.empty()) {
            std::lock_guard<std::mutex> out_lock(outputs_pool_mutex);
            outputs_ptr_pool.push(outputs_ptr);
            outputs_pool_cv.notify_one();
            g_last_error = "Runtime is shutting down";
            return RUNTIME_STATUS_ERROR;
        }
        input_buf = input_buf_pool.front();
        input_buf_pool.pop();
    }

    size_t copy_size = input_tensors->tensors[0].data_size;
    if (copy_size > INPUT_BUF_SIZE) {
        spdlog::warn("[enqueue] input data_size ({}) > pool buffer size ({}); clamping", copy_size, INPUT_BUF_SIZE);
        copy_size = INPUT_BUF_SIZE;
    }
    if (input_tensors->tensors[0].data && copy_size > 0)
        memcpy(input_buf, input_tensors->tensors[0].data, copy_size);

    // input_tensors data has been copied into the pooled buffer — free it now
    // to reduce memory pressure during pipelined inference.
    int request_id = input_tensors->id;
    deep_free_tensors(input_tensors);
    input_tensors = nullptr;

    int job_id = -1;
    try {
        spdlog::trace("[enqueue] calling RunAsync for request_id={}", request_id);
        job_id = inference_engine->RunAsync(static_cast<uint8_t *>(input_buf), nullptr, outputs_ptr);
        spdlog::trace("[enqueue] RunAsync returned job_id={} for request_id={}", job_id, request_id);
    } catch (const std::exception &e) {
        spdlog::error("[enqueue] RunAsync failed for request_id={}: {}", request_id, e.what());
        g_last_error = std::string("Inference failed: ") + e.what();
        {
            std::lock_guard<std::mutex> lock(input_buf_pool_mutex);
            input_buf_pool.push(input_buf);
            input_buf_pool_cv.notify_one();
        }
        {
            std::lock_guard<std::mutex> lock(outputs_pool_mutex);
            outputs_ptr_pool.push(outputs_ptr);
            outputs_pool_cv.notify_one();
        }
        return RUNTIME_STATUS_INFERENCE_ERROR;
    }

    JobData job_data{};
    job_data.job_id = job_id;
    job_data.model_id = model_id;
    job_data.request_id = request_id;
    job_data.input_tensors = nullptr;
    job_data.input_buf = input_buf;
    job_data.outputs_ptr = outputs_ptr;

    {
        std::unique_lock<std::mutex> lock(job_data_queue_mutex);
        job_data_queue.push(job_data);
        job_data_queue_cv.notify_one();
        spdlog::trace("[enqueue] pushed job_id={} request_id={} to job_data_queue (queue_size={})",
                      job_id, job_data.request_id, job_data_queue.size());
    }

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

    spdlog::trace("[retrieve] called timeout_ms={}", timeout_ms);

    JobData job_data{};
    {
        std::unique_lock<std::mutex> lock(output_queue_mutex);

        auto predicate = []() { return stop_wait_thread.load() || !output_queue.empty(); };

        if (timeout_ms == 0) {
            // Non-blocking poll
            if (!predicate()) {
                return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
            }
        } else if (timeout_ms > 0) {
            // Timed wait
            bool got = output_queue_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), predicate);
            if (!got) {
                spdlog::trace("[retrieve] timed out after {}ms", timeout_ms);
                return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
            }
        } else {
            // Infinite wait (negative timeout)
            output_queue_cv.wait(lock, predicate);
        }

        if (stop_wait_thread.load() && output_queue.empty()) {
            return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
        }

        job_data = output_queue.front();
        output_queue.pop();
        spdlog::trace("[retrieve] dequeued job_id={} request_id={} (output_queue_size={})",
                      job_data.job_id, job_data.request_id, output_queue.size());
    }

    Tensors *result = create_output_tensors(job_data.request_id);
    if (!result) {
        spdlog::error("[retrieve_output] Failed to allocate output tensors");
        g_last_error = "Failed to allocate output tensors";
        if (job_data.outputs_ptr) {
            std::lock_guard<std::mutex> lock(outputs_pool_mutex);
            outputs_ptr_pool.push(job_data.outputs_ptr);
            outputs_pool_cv.notify_one();
        }
        return RUNTIME_STATUS_OUT_OF_MEMORY;
    }

    Tensors *copied = copy_dxrt_outputs(job_data.dxrt_outputs, result);
    if (!copied) {
        spdlog::error("[retrieve_output] Failed to copy dxrt outputs");
        g_last_error = "Failed to copy output tensors";
        if (job_data.outputs_ptr) {
            std::lock_guard<std::mutex> lock(outputs_pool_mutex);
            outputs_ptr_pool.push(job_data.outputs_ptr);
            outputs_pool_cv.notify_one();
        }
        return RUNTIME_STATUS_ERROR;
    }

    // Return outputs_ptr to pool
    if (job_data.outputs_ptr) {
        std::lock_guard<std::mutex> lock(outputs_pool_mutex);
        outputs_ptr_pool.push(job_data.outputs_ptr);
        outputs_pool_cv.notify_one();
    }

    *output_tensors = copied;
    if (model_id) {
        *model_id = job_data.model_id;
    }
    spdlog::trace("[retrieve] returning output for request_id={} model_id={}", job_data.request_id,
                  job_data.model_id);
    return RUNTIME_STATUS_SUCCESS;
}

RuntimeStatus runtime_cleanup(void) {
    if (!g_initialized.load()) {
        // Idempotent: already cleaned up
        return RUNTIME_STATUS_SUCCESS;
    }

    spdlog::info("Cleaning up the runtime environment");

    stop_wait_thread.store(true);
    job_data_queue_cv.notify_all();
    output_queue_cv.notify_all();
    outputs_pool_cv.notify_all();
    input_buf_pool_cv.notify_all();

    // Drain output queue
    {
        std::lock_guard<std::mutex> lock(output_queue_mutex);
        while (!output_queue.empty()) {
            JobData r = std::move(output_queue.front());
            output_queue.pop();
            if (r.input_tensors) deep_free_tensors(r.input_tensors);
            if (r.input_buf) {
                std::lock_guard<std::mutex> ibp_lock(input_buf_pool_mutex);
                input_buf_pool.push(r.input_buf);
            }
            if (r.outputs_ptr) free(r.outputs_ptr);
        }
    }

    // Drain job queue
    {
        std::lock_guard<std::mutex> lock(job_data_queue_mutex);
        while (!job_data_queue.empty()) {
            JobData j = job_data_queue.front();
            job_data_queue.pop();
            if (j.input_tensors) deep_free_tensors(j.input_tensors);
            if (j.input_buf) {
                std::lock_guard<std::mutex> ibp_lock(input_buf_pool_mutex);
                input_buf_pool.push(j.input_buf);
            }
            if (j.outputs_ptr) free(j.outputs_ptr);
        }
    }

    // Free output pool
    {
        std::lock_guard<std::mutex> lock(outputs_pool_mutex);
        while (!outputs_ptr_pool.empty()) {
            free(outputs_ptr_pool.front());
            outputs_ptr_pool.pop();
        }
    }

    // Free input buffer pool
    {
        std::lock_guard<std::mutex> lock(input_buf_pool_mutex);
        while (!input_buf_pool.empty()) {
            free(input_buf_pool.front());
            input_buf_pool.pop();
        }
        INPUT_BUF_SIZE = 0;
    }

    // Join wait thread
    if (wait_thread_started.load()) {
        if (wait_thread.joinable()) {
            wait_thread.join();
        }
        wait_thread_started.store(false);
    }

    // Destroy inference engine
    if (inference_engine != nullptr) {
        delete inference_engine;
        inference_engine = nullptr;
        spdlog::info("Inference engine destroyed");
    }

    OutputTensorSizes.clear();
    OUTPUTS_POOL_CAPACITY = 0;
    NumDevice = 0;
    INPUT_BUF_SIZE = 0;

    spdlog::info("Runtime cleanup completed");
    if (logger) {
        logger->flush();
        logger.reset();
    }

    g_initialized.store(false);
    g_last_error.clear();

    return RUNTIME_STATUS_SUCCESS;
}

const char *runtime_get_error(void) {
    if (g_last_error.empty()) return nullptr;
    return g_last_error.c_str();
}

const char *runtime_get_version(void) { return OAAX_RUNTIME_VERSION; }

const char *runtime_get_name(void) { return "DEEPX"; }

const char *runtime_get_info(void) {
    if (!g_initialized.load()) return nullptr;

    size_t pool_size = 0;
    size_t in_flight = 0;
    {
        std::lock_guard<std::mutex> lock(outputs_pool_mutex);
        pool_size = outputs_ptr_pool.size();
    }
    {
        std::lock_guard<std::mutex> lock(job_data_queue_mutex);
        in_flight = job_data_queue.size();
    }
    {
        std::lock_guard<std::mutex> lock(output_queue_mutex);
        in_flight += output_queue.size();
    }

    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"loaded_models\": %d, \"requests_in_flight\": %zu, "
             "\"pool_size\": %zu, \"pool_capacity\": %zu, \"num_devices\": %zu}",
             inference_engine ? 1 : 0, in_flight, pool_size, OUTPUTS_POOL_CAPACITY, NumDevice);

    g_info_json = buf;
    return g_info_json.c_str();
}
