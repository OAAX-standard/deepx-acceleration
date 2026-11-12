#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "dxrt/dxrt_cxx_api.h"  // mock: scalar_output() test knob
#include "runtime_core.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const size_t kModelInputBytes = 1 * 3 * 224 * 224 * sizeof(float);

static Tensors *make_input(int id, size_t data_size = kModelInputBytes) {
    Tensors *t = (Tensors *)malloc(sizeof(Tensors));
    assert(t);
    t->id = id;
    t->num_tensors = 1;
    t->tensors = (TensorDescriptor *)calloc(1, sizeof(TensorDescriptor));
    assert(t->tensors);

    t->tensors[0].name = strdup("input_0");
    t->tensors[0].data_type = DATA_TYPE_FLOAT;
    t->tensors[0].rank = 4;
    t->tensors[0].shape = (int *)malloc(4 * sizeof(int));
    assert(t->tensors[0].shape);
    int shape[] = {1, 3, 224, 224};
    memcpy(t->tensors[0].shape, shape, sizeof(shape));

    t->tensors[0].data_size = data_size;
    t->tensors[0].data = malloc(data_size);
    assert(t->tensors[0].data);
    memset(t->tensors[0].data, 0xAA, data_size);

    return t;
}

// Smallest buffer the mock engine accepts: "DXNN" magic, then the byte it fills outputs with.
static std::vector<unsigned char> make_model_blob(unsigned char fill) { return {'D', 'X', 'N', 'N', fill, 0, 0, 0}; }

static void free_output(Tensors *t) {
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

static ModelConfig file_model(const char *path) {
    ModelConfig mc{};
    mc.file_path = path;
    return mc;
}

static ModelConfig memory_model(const std::vector<unsigned char> &blob) {
    ModelConfig mc{};
    mc.model_data = blob.data();
    mc.model_size = blob.size();
    return mc;
}

// Sends one request to each of model ids 0..n-1 and returns the first output byte per model id.
static std::vector<unsigned char> first_output_byte_per_model(int n) {
    for (int i = 0; i < n; i++) assert(runtime_enqueue_input(i, make_input(100 + i)) == RUNTIME_STATUS_SUCCESS);
    std::vector<unsigned char> bytes(n);
    std::vector<bool> seen(n, false);
    for (int i = 0; i < n; i++) {
        int mid = -1;
        Tensors *output = nullptr;
        assert(runtime_retrieve_output(&mid, &output, 2000) == RUNTIME_STATUS_SUCCESS);
        assert(mid >= 0 && mid < n && !seen[mid]);
        assert(output->id == 100 + mid);
        seen[mid] = true;
        bytes[mid] = ((unsigned char *)output->tensors[0].data)[0];
        free_output(output);
    }
    return bytes;
}

// Initializes the runtime with the config key max_pending_requests set to `value`.
static RuntimeStatus init_with_max_pending(const char *value) {
    const char *keys[] = {"max_pending_requests"};
    const char *values[] = {value};
    Config cfg = {1, keys, values};
    return runtime_init(cfg);
}

// Enqueues request ids first..first+n-1 on model 0; each must be accepted.
static void enqueue_accepted(int first, int n) {
    for (int i = first; i < first + n; i++) assert(runtime_enqueue_input(0, make_input(i)) == RUNTIME_STATUS_SUCCESS);
}

// Retrieves n outputs, which must answer request ids 0..n-1 of model 0, each exactly once.
static void expect_answers(int n) {
    std::vector<int> times_seen(n, 0);
    for (int i = 0; i < n; i++) {
        int mid = -1;
        Tensors *output = nullptr;
        assert(runtime_retrieve_output(&mid, &output, 2000) == RUNTIME_STATUS_SUCCESS);
        assert(mid == 0 && output->id >= 0 && output->id < n);
        assert(times_seen[output->id]++ == 0);
        free_output(output);
    }
}

// Returns f's result, or fails the test (instead of hanging it) if f is not ready within timeout_ms.
template <typename T>
static T get_within(std::future<T> &f, int timeout_ms, const char *what) {
    if (f.wait_for(std::chrono::milliseconds(timeout_ms)) != std::future_status::ready) {
        fprintf(stderr, "  FAILED: %s did not finish within %d ms\n", what, timeout_ms);
        abort();
    }
    return f.get();
}

// ---------------------------------------------------------------------------
// Test 1: Basic lifecycle
// ---------------------------------------------------------------------------

static void test_basic_lifecycle() {
    printf("=== test_basic_lifecycle ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);

    // Double init should fail
    assert(runtime_init(cfg) == RUNTIME_STATUS_ALREADY_INITIALIZED);

    // Version and name should work
    assert(runtime_get_version() != nullptr);
    assert(strcmp(runtime_get_name(), "DEEPX") == 0);

    // Cleanup
    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);

    // Double cleanup should be idempotent
    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);

    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 2: Model load without init
// ---------------------------------------------------------------------------

static void test_load_without_init() {
    printf("=== test_load_without_init ===\n");

    ModelConfig mc{};
    mc.file_path = "dummy.dxnn";
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_NOT_INITIALIZED);

    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 3: Single synchronous request (enqueue + retrieve)
// ---------------------------------------------------------------------------

static void test_single_request() {
    printf("=== test_single_request ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);

    ModelConfig mc{};
    mc.file_path = "dummy.dxnn";
    mc.config = cfg;
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    // Enqueue input
    Tensors *input = make_input(42);
    assert(runtime_enqueue_input(0, input) == RUNTIME_STATUS_SUCCESS);

    // Retrieve output (infinite wait)
    int out_model_id = -1;
    Tensors *output = nullptr;
    assert(runtime_retrieve_output(&out_model_id, &output, -1) == RUNTIME_STATUS_SUCCESS);
    assert(output != nullptr);
    assert(output->id == 42);   // id propagation
    assert(out_model_id == 0);  // model_id propagation
    assert(output->num_tensors == 1);
    assert(output->tensors[0].name != nullptr);
    assert(output->tensors[0].data_size > 0);
    assert(output->tensors[0].rank == 4);
    assert(output->tensors[0].shape != nullptr);
    assert(output->tensors[0].data != nullptr);

    printf("  output id=%d, model_id=%d, name=%s, rank=%d, data_size=%zu\n", output->id, out_model_id,
           output->tensors[0].name, output->tensors[0].rank, output->tensors[0].data_size);

    free_output(output);
    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);

    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 4: Timeout behavior
// ---------------------------------------------------------------------------

static void test_timeout() {
    printf("=== test_timeout ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);

    ModelConfig mc{};
    mc.file_path = "dummy.dxnn";
    mc.config = cfg;
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    // No input enqueued, timeout=0 should return immediately
    int out_model_id = -1;
    Tensors *output = nullptr;
    RuntimeStatus s = runtime_retrieve_output(&out_model_id, &output, 0);
    assert(s == RUNTIME_STATUS_NO_OUTPUT_AVAILABLE);
    assert(output == nullptr);

    // timeout=50ms should also return quickly
    s = runtime_retrieve_output(&out_model_id, &output, 50);
    assert(s == RUNTIME_STATUS_NO_OUTPUT_AVAILABLE);
    assert(output == nullptr);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 5: Invalid model_id
// ---------------------------------------------------------------------------

static void test_invalid_model_id() {
    printf("=== test_invalid_model_id ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);

    ModelConfig mc{};
    mc.file_path = "dummy.dxnn";
    mc.config = cfg;
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    Tensors *input = make_input(0);
    RuntimeStatus s = runtime_enqueue_input(99, input);
    assert(s == RUNTIME_STATUS_INVALID_MODEL_ID);
    // On failure, caller must free input
    free_output(input);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 6: Async pipeline (multiple producer/consumer)
// ---------------------------------------------------------------------------

static void test_async_pipeline() {
    printf("=== test_async_pipeline ===\n");

    const int NUM_REQUESTS = 30;

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);

    ModelConfig mc{};
    mc.file_path = "dummy.dxnn";
    mc.config = cfg;
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    std::unordered_set<int> received_ids;
    std::mutex received_mutex;

    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < NUM_REQUESTS; i++) {
            Tensors *input = make_input(i);
            // TIMEOUT = output pool exhausted (backpressure); the caller keeps the input and retries.
            RuntimeStatus s;
            while ((s = runtime_enqueue_input(0, input)) == RUNTIME_STATUS_TIMEOUT) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            assert(s == RUNTIME_STATUS_SUCCESS);
            // Vary timing
            if (i % 7 == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
        }
    });

    // Consumer thread
    std::thread consumer([&]() {
        int count = 0;
        while (count < NUM_REQUESTS) {
            int out_model_id = -1;
            Tensors *output = nullptr;
            RuntimeStatus s = runtime_retrieve_output(&out_model_id, &output, 500);

            if (s == RUNTIME_STATUS_NO_OUTPUT_AVAILABLE) {
                continue;
            }

            assert(s == RUNTIME_STATUS_SUCCESS);
            assert(output != nullptr);
            assert(out_model_id == 0);
            assert(output->num_tensors == 1);
            assert(output->tensors[0].data != nullptr);
            assert(output->tensors[0].data_size > 0);

            {
                std::lock_guard<std::mutex> lock(received_mutex);
                // No duplicates
                assert(received_ids.find(output->id) == received_ids.end());
                received_ids.insert(output->id);
            }

            free_output(output);
            count++;
        }
    });

    producer.join();
    consumer.join();

    // All request IDs received
    assert((int)received_ids.size() == NUM_REQUESTS);
    for (int i = 0; i < NUM_REQUESTS; i++) {
        assert(received_ids.count(i) == 1);
    }

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED (%d async requests)\n", NUM_REQUESTS);
}

// ---------------------------------------------------------------------------
// Test 7: runtime_get_info
// ---------------------------------------------------------------------------

static void test_get_info() {
    printf("=== test_get_info ===\n");

    // Before init, should return null
    assert(runtime_get_info() == nullptr);

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);

    ModelConfig mc{};
    mc.file_path = "dummy.dxnn";
    mc.config = cfg;
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    const char *info = runtime_get_info();
    assert(info != nullptr);
    printf("  info: %s\n", info);

    // Should contain expected fields
    assert(strstr(info, "loaded_models") != nullptr);
    assert(strstr(info, "pool_size") != nullptr);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 8: Input size validation
// ---------------------------------------------------------------------------

static void test_input_size_mismatch() {
    printf("=== test_input_size_mismatch ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mc{};
    mc.file_path = "dummy.dxnn";
    mc.config = cfg;
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    // Both undersized and oversized inputs are rejected; the caller keeps ownership.
    Tensors *small = make_input(1, kModelInputBytes - 4);
    assert(runtime_enqueue_input(0, small) == RUNTIME_STATUS_TENSOR_SHAPE_MISMATCH);
    free_output(small);
    Tensors *big = make_input(2, kModelInputBytes + 4);
    assert(runtime_enqueue_input(0, big) == RUNTIME_STATUS_TENSOR_SHAPE_MISMATCH);
    free_output(big);

    assert(runtime_get_error() != nullptr);
    assert(strstr(runtime_get_error(), "does not match") != nullptr);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 9: Backpressure — a full output pool and pending queue return TIMEOUT, then recover
// ---------------------------------------------------------------------------

static void test_backpressure() {
    printf("=== test_backpressure ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mc{};
    mc.file_path = "dummy.dxnn";
    mc.config = cfg;
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    // Pool capacity = dxrt_get_task_max_load() x mock device count (2), plus 256 pending (the default).
    const int capacity = dxrt_get_task_max_load() * 2 + 256;
    enqueue_accepted(0, capacity);

    // Outputs are only released on retrieve, so the next enqueue must fail fast.
    Tensors *rejected = make_input(capacity);
    assert(runtime_enqueue_input(0, rejected) == RUNTIME_STATUS_TIMEOUT);
    free_output(rejected);  // caller keeps ownership on failure

    for (int i = 0; i < capacity; i++) {
        Tensors *output = nullptr;
        assert(runtime_retrieve_output(nullptr, &output, 1000) == RUNTIME_STATUS_SUCCESS);
        free_output(output);
    }

    // Buffers are back in the pool: enqueue works again.
    assert(runtime_enqueue_input(0, make_input(1000)) == RUNTIME_STATUS_SUCCESS);
    Tensors *output = nullptr;
    assert(runtime_retrieve_output(nullptr, &output, 1000) == RUNTIME_STATUS_SUCCESS);
    free_output(output);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 10: Rank-0 (scalar) output tensors are returned, not treated as OOM
// ---------------------------------------------------------------------------

static void test_scalar_output() {
    printf("=== test_scalar_output ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mc{};
    mc.file_path = "dummy.dxnn";
    mc.config = cfg;
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    dxrt::scalar_output().store(true);
    assert(runtime_enqueue_input(0, make_input(7)) == RUNTIME_STATUS_SUCCESS);
    Tensors *output = nullptr;
    RuntimeStatus s = runtime_retrieve_output(nullptr, &output, 1000);
    dxrt::scalar_output().store(false);

    assert(s == RUNTIME_STATUS_SUCCESS);
    assert(output != nullptr && output->id == 7);
    assert(output->tensors[0].rank == 0);
    assert(output->tensors[0].shape != nullptr);
    free_output(output);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 11: runtime_get_error is null until an error happens, valid across threads
// ---------------------------------------------------------------------------

static void test_get_error_thread_safe() {
    printf("=== test_get_error_thread_safe ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    assert(runtime_get_error() == nullptr);

    // Hammer error-setting and error-reading from several threads (run under TSan to check races).
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; t++) {
        threads.emplace_back([]() {
            for (int i = 0; i < 200; i++) {
                Tensors *in = make_input(i);
                assert(runtime_enqueue_input(0, in) == RUNTIME_STATUS_MODEL_NOT_LOADED);
                free_output(in);
                const char *err = runtime_get_error();
                assert(err != nullptr && strcmp(err, "Model not loaded") == 0);
            }
        });
    }
    for (auto &t : threads) t.join();

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 12: Missing model file
// ---------------------------------------------------------------------------

static void test_model_file_not_found() {
    printf("=== test_model_file_not_found ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mc{};
    mc.file_path = "does_not_exist.dxnn";
    mc.config = cfg;
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_FILE_NOT_FOUND);
    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 13: In-memory model loading (ModelConfig.model_data)
// ---------------------------------------------------------------------------

static void test_load_from_memory() {
    printf("=== test_load_from_memory ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    std::vector<unsigned char> blob = make_model_blob(0x07);
    ModelConfig mc{};
    mc.file_path = nullptr;
    mc.model_data = blob.data();
    mc.model_size = blob.size();
    mc.config = cfg;
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    assert(runtime_enqueue_input(0, make_input(11)) == RUNTIME_STATUS_SUCCESS);
    int out_model_id = -1;
    Tensors *output = nullptr;
    assert(runtime_retrieve_output(&out_model_id, &output, 2000) == RUNTIME_STATUS_SUCCESS);
    assert(out_model_id == 0);
    assert(output->id == 11);
    assert(((unsigned char *)output->tensors[0].data)[0] == 0x07);
    free_output(output);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// The caller may reuse or free model_data as soon as runtime_load_models returns.
static void test_memory_buffer_reusable_after_load() {
    printf("=== test_memory_buffer_reusable_after_load ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    std::vector<unsigned char> blob = make_model_blob(0x07);
    ModelConfig mc{};
    mc.model_data = blob.data();
    mc.model_size = blob.size();
    mc.config = cfg;
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);
    memset(blob.data(), 0, blob.size());

    assert(runtime_enqueue_input(0, make_input(12)) == RUNTIME_STATUS_SUCCESS);
    Tensors *output = nullptr;
    assert(runtime_retrieve_output(nullptr, &output, 2000) == RUNTIME_STATUS_SUCCESS);
    assert(((unsigned char *)output->tensors[0].data)[0] == 0x07);
    free_output(output);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

static void test_load_requires_source() {
    printf("=== test_load_requires_source ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mc{};
    mc.config = cfg;
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_INVALID_ARGUMENT);

    std::vector<unsigned char> blob = make_model_blob(0x07);
    mc.model_data = blob.data();
    mc.model_size = 0;
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_INVALID_ARGUMENT);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

static void test_invalid_model_buffer() {
    printf("=== test_invalid_model_buffer ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    const unsigned char junk[] = {'X', 'X', 'X', 'X', 0x07, 0, 0, 0};
    ModelConfig mc{};
    mc.model_data = junk;
    mc.model_size = sizeof(junk);
    mc.config = cfg;
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_INVALID_MODEL);
    assert(runtime_get_error() != nullptr);
    assert(strstr(runtime_get_error(), "model[0]") != nullptr);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// When both sources are set, model_data wins and file_path only names the model.
static void test_memory_takes_precedence_over_file() {
    printf("=== test_memory_takes_precedence_over_file ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    std::vector<unsigned char> blob = make_model_blob(0x07);
    ModelConfig mc = memory_model(blob);
    mc.file_path = "does_not_exist.dxnn";
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);
    assert(first_output_byte_per_model(1)[0] == 0x07);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 14: Several models in one runtime_load_models call (model id = index)
// ---------------------------------------------------------------------------

static void test_load_multiple_models() {
    printf("=== test_load_multiple_models ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    std::vector<unsigned char> blob1 = make_model_blob(0x01);
    std::vector<unsigned char> blob2 = make_model_blob(0x02);
    ModelConfig mcs[] = {memory_model(blob1), memory_model(blob2)};
    assert(runtime_load_models(2, mcs) == RUNTIME_STATUS_SUCCESS);

    std::vector<unsigned char> bytes = first_output_byte_per_model(2);
    assert(bytes[0] == 0x01 && bytes[1] == 0x02);
    assert(strstr(runtime_get_info(), "\"loaded_models\": 2") != nullptr);

    Tensors *input = make_input(1);
    assert(runtime_enqueue_input(2, input) == RUNTIME_STATUS_INVALID_MODEL_ID);
    free_output(input);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

static void test_load_same_file_twice() {
    printf("=== test_load_same_file_twice ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mcs[] = {file_model("dummy.dxnn"), file_model("dummy.dxnn")};
    assert(runtime_load_models(2, mcs) == RUNTIME_STATUS_SUCCESS);
    first_output_byte_per_model(2);  // asserts each id answers on its own model id

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

static void test_mixed_sources() {
    printf("=== test_mixed_sources ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    std::vector<unsigned char> blob = make_model_blob(0x07);
    ModelConfig mcs[] = {file_model("dummy.dxnn"), memory_model(blob)};
    assert(runtime_load_models(2, mcs) == RUNTIME_STATUS_SUCCESS);
    std::vector<unsigned char> bytes = first_output_byte_per_model(2);
    assert(bytes[0] == 0x42 && bytes[1] == 0x07);  // 0x42: mock fill for file-loaded models

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

static void test_second_load_rejected() {
    printf("=== test_second_load_rejected ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mc = file_model("dummy.dxnn");
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_ALREADY_INITIALIZED);
    first_output_byte_per_model(1);  // the loaded model still answers

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// A failure on any model unloads the ones built before it; the next call starts afresh.
static void test_failed_load_rolls_back() {
    printf("=== test_failed_load_rolls_back ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    std::vector<unsigned char> blob = make_model_blob(0x01);
    ModelConfig mcs[] = {memory_model(blob), file_model("does_not_exist.dxnn")};
    assert(runtime_load_models(2, mcs) == RUNTIME_STATUS_FILE_NOT_FOUND);
    assert(strstr(runtime_get_error(), "model[1]") != nullptr);
    assert(strstr(runtime_get_info(), "\"loaded_models\": 0") != nullptr);

    assert(runtime_load_models(1, mcs) == RUNTIME_STATUS_SUCCESS);
    assert(first_output_byte_per_model(1)[0] == 0x01);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// An exception thrown while building a model (here std::bad_alloc) is reported, not propagated.
static void test_load_exception_rolls_back() {
    printf("=== test_load_exception_rolls_back ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mcs[] = {file_model("dummy.dxnn"), file_model("dummy.dxnn")};
    dxrt::register_callback_throws_after().store(1);
    assert(runtime_load_models(2, mcs) == RUNTIME_STATUS_OUT_OF_MEMORY);
    assert(strstr(runtime_get_error(), "model[1]") != nullptr);
    assert(strstr(runtime_get_info(), "\"loaded_models\": 0") != nullptr);

    assert(runtime_load_models(2, mcs) == RUNTIME_STATUS_SUCCESS);
    first_output_byte_per_model(2);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 15: Concurrent inference on several models
// ---------------------------------------------------------------------------

// Producers on several threads share two models and never retry (OAAX conformance RT-M-29): every
// request is accepted, and answered exactly once by its own model.
static void test_concurrent_producers_no_retry() {
    printf("=== test_concurrent_producers_no_retry ===\n");

    const int kProducers = 4;
    const int kPerProducer = 25;
    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mcs[] = {file_model("dummy.dxnn"), file_model("dummy.dxnn")};
    assert(runtime_load_models(2, mcs) == RUNTIME_STATUS_SUCCESS);

    // Request id t*100+k goes to model k%2.
    std::vector<std::thread> producers;
    for (int t = 0; t < kProducers; t++) {
        producers.emplace_back([=]() {
            for (int k = 0; k < kPerProducer; k++) {
                assert(runtime_enqueue_input(k % 2, make_input(t * 100 + k)) == RUNTIME_STATUS_SUCCESS);
            }
        });
    }

    std::vector<int> times_seen(kProducers * 100, 0);
    std::thread consumer([&]() {
        for (int i = 0; i < kProducers * kPerProducer; i++) {
            int mid = -1;
            Tensors *output = nullptr;
            assert(runtime_retrieve_output(&mid, &output, 2000) == RUNTIME_STATUS_SUCCESS);
            const int id = output->id;
            assert(id >= 0 && id / 100 < kProducers && id % 100 < kPerProducer);
            assert(times_seen[id]++ == 0);
            assert(mid == (id % 100) % 2);
            free_output(output);
        }
    });

    for (auto &p : producers) p.join();
    consumer.join();

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// A full output pool and pending queue on one model do not block another model.
static void test_backpressure_isolated() {
    printf("=== test_backpressure_isolated ===\n");

    assert(init_with_max_pending("4") == RUNTIME_STATUS_SUCCESS);
    ModelConfig mcs[] = {file_model("dummy.dxnn"), file_model("dummy.dxnn")};
    assert(runtime_load_models(2, mcs) == RUNTIME_STATUS_SUCCESS);

    // Fill model 0: pool capacity = dxrt_get_task_max_load() x mock device count (2), plus 4 pending.
    const int capacity = dxrt_get_task_max_load() * 2 + 4;
    int accepted = 0;
    for (int i = 0; i <= capacity; i++) {
        Tensors *input = make_input(i);
        const RuntimeStatus s = runtime_enqueue_input(0, input);
        if (s == RUNTIME_STATUS_TIMEOUT) {
            free_output(input);  // caller keeps ownership on failure
            break;
        }
        assert(s == RUNTIME_STATUS_SUCCESS);
        accepted++;
    }
    assert(accepted == capacity);

    assert(runtime_enqueue_input(1, make_input(1000)) == RUNTIME_STATUS_SUCCESS);

    for (int i = 0; i < accepted + 1; i++) {
        Tensors *output = nullptr;
        assert(runtime_retrieve_output(nullptr, &output, 2000) == RUNTIME_STATUS_SUCCESS);
        free_output(output);
    }
    assert(runtime_enqueue_input(0, make_input(1001)) == RUNTIME_STATUS_SUCCESS);
    Tensors *output = nullptr;
    assert(runtime_retrieve_output(nullptr, &output, 2000) == RUNTIME_STATUS_SUCCESS);
    free_output(output);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// requests_in_flight counts requests accepted but not yet retrieved, including completed ones.
static void test_info_counts_unretrieved() {
    printf("=== test_info_counts_unretrieved ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mc = file_model("dummy.dxnn");
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    for (int i = 0; i < 3; i++) assert(runtime_enqueue_input(0, make_input(i)) == RUNTIME_STATUS_SUCCESS);
    std::this_thread::sleep_for(std::chrono::milliseconds(150));  // mock completes within 15 ms
    assert(strstr(runtime_get_info(), "\"requests_in_flight\": 3") != nullptr);

    for (int i = 0; i < 3; i++) {
        Tensors *output = nullptr;
        assert(runtime_retrieve_output(nullptr, &output, 2000) == RUNTIME_STATUS_SUCCESS);
        free_output(output);
    }
    assert(strstr(runtime_get_info(), "\"requests_in_flight\": 0") != nullptr);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// Cleanup with requests pending on several models releases a thread blocked in retrieve_output(-1).
static void test_cleanup_multi_model_wakes_retriever() {
    printf("=== test_cleanup_multi_model_wakes_retriever ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mcs[] = {file_model("dummy.dxnn"), file_model("dummy.dxnn")};
    assert(runtime_load_models(2, mcs) == RUNTIME_STATUS_SUCCESS);

    for (int i = 0; i < 10; i++) {
        for (int m = 0; m < 2; m++) assert(runtime_enqueue_input(m, make_input(m * 10 + i)) == RUNTIME_STATUS_SUCCESS);
    }
    // Takes outputs until the runtime stops handing them out; cleanup frees the rest.
    std::future<RuntimeStatus> retriever = std::async(std::launch::async, []() {
        int mid = -1;
        Tensors *output = nullptr;
        RuntimeStatus s;
        while ((s = runtime_retrieve_output(&mid, &output, -1)) == RUNTIME_STATUS_SUCCESS) free_output(output);
        return s;
    });

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    const RuntimeStatus s = get_within(retriever, 5000, "retriever after cleanup");
    assert(s == RUNTIME_STATUS_NO_OUTPUT_AVAILABLE || s == RUNTIME_STATUS_NOT_INITIALIZED);

    // The next session starts empty: nothing left over from the outputs cleanup freed.
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    assert(runtime_load_models(2, mcs) == RUNTIME_STATUS_SUCCESS);
    std::future<RuntimeStatus> poll = std::async(std::launch::async, []() {
        Tensors *output = nullptr;
        return runtime_retrieve_output(nullptr, &output, 0);
    });
    assert(get_within(poll, 1000, "retrieve_output(timeout 0) after re-init") == RUNTIME_STATUS_NO_OUTPUT_AVAILABLE);
    first_output_byte_per_model(2);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 16: Requests beyond the output pool wait in a bounded per-model queue
// ---------------------------------------------------------------------------

static void test_enqueue_beyond_pool_is_queued() {
    printf("=== test_enqueue_beyond_pool_is_queued ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mc = file_model("dummy.dxnn");
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    enqueue_accepted(0, 40);  // pool capacity is 16
    expect_answers(40);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

static void test_pending_queue_full_returns_timeout() {
    printf("=== test_pending_queue_full_returns_timeout ===\n");

    assert(init_with_max_pending("4") == RUNTIME_STATUS_SUCCESS);
    ModelConfig mc = file_model("dummy.dxnn");
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    enqueue_accepted(0, 16 + 4);
    Tensors *rejected = make_input(20);
    assert(runtime_enqueue_input(0, rejected) == RUNTIME_STATUS_TIMEOUT);

    // Retrieving one request makes room again; the caller still owns the rejected input.
    Tensors *output = nullptr;
    assert(runtime_retrieve_output(nullptr, &output, 2000) == RUNTIME_STATUS_SUCCESS);
    const int retrieved_id = output->id;
    free_output(output);
    assert(runtime_enqueue_input(0, rejected) == RUNTIME_STATUS_SUCCESS);

    std::vector<int> times_seen(21, 0);
    times_seen[retrieved_id] = 1;
    for (int i = 0; i < 20; i++) {
        assert(runtime_retrieve_output(nullptr, &output, 2000) == RUNTIME_STATUS_SUCCESS);
        assert(output->id >= 0 && output->id <= 20 && times_seen[output->id]++ == 0);
        free_output(output);
    }

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

static void test_max_pending_zero_keeps_old_behaviour() {
    printf("=== test_max_pending_zero_keeps_old_behaviour ===\n");

    assert(init_with_max_pending("0") == RUNTIME_STATUS_SUCCESS);
    ModelConfig mc = file_model("dummy.dxnn");
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    enqueue_accepted(0, 16);
    Tensors *rejected = make_input(16);
    assert(runtime_enqueue_input(0, rejected) == RUNTIME_STATUS_TIMEOUT);
    free_output(rejected);
    expect_answers(16);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

static void test_invalid_max_pending_rejected() {
    printf("=== test_invalid_max_pending_rejected ===\n");

    const char *invalid[] = {"-1", "abc", "", "12x", "99999999999999999999999"};
    for (const char *value : invalid) {
        assert(init_with_max_pending(value) == RUNTIME_STATUS_INVALID_ARGUMENT);
        assert(strstr(runtime_get_error(), "max_pending_requests") != nullptr);
        assert(runtime_get_info() == nullptr);  // left uninitialized
    }
    assert(init_with_max_pending("8") == RUNTIME_STATUS_SUCCESS);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

static void test_cleanup_frees_pending() {
    printf("=== test_cleanup_frees_pending ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mc = file_model("dummy.dxnn");
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);
    enqueue_accepted(0, 40);
    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);  // ASan reports any input it did not free

    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);
    first_output_byte_per_model(1);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

static void test_info_counts_pending() {
    printf("=== test_info_counts_pending ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mc = file_model("dummy.dxnn");
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    enqueue_accepted(0, 40);
    const char *info = runtime_get_info();
    assert(strstr(info, "\"requests_in_flight\": 40") != nullptr);
    assert(strstr(info, "\"pending_requests\": 24") != nullptr);
    expect_answers(40);
    assert(strstr(runtime_get_info(), "\"requests_in_flight\": 0") != nullptr);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// DX-RT rejecting a queued request answers it with an error; rejecting a new one returns the error.
static void test_rejected_submission_answered() {
    printf("=== test_rejected_submission_answered ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mc = file_model("dummy.dxnn");
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);

    dxrt::run_async_failures().store(1);
    Tensors *input = make_input(0);
    assert(runtime_enqueue_input(0, input) == RUNTIME_STATUS_INFERENCE_ERROR);
    free_output(input);  // caller keeps ownership on failure

    enqueue_accepted(0, 17);  // request 16 waits for a buffer
    dxrt::run_async_failures().store(1);
    std::vector<int> times_seen(17, 0);
    for (int i = 0; i < 17; i++) {
        int mid = -1;
        Tensors *output = nullptr;
        const RuntimeStatus s = runtime_retrieve_output(&mid, &output, 2000);
        assert(mid == 0);
        if (s == RUNTIME_STATUS_INFERENCE_ERROR) {
            assert(output == nullptr && strstr(runtime_get_error(), "request 16") != nullptr);
            times_seen[16]++;
            continue;
        }
        assert(s == RUNTIME_STATUS_SUCCESS && output->id >= 0 && output->id < 16);
        times_seen[output->id]++;
        free_output(output);
    }
    for (int n : times_seen) assert(n == 1);
    assert(strstr(runtime_get_info(), "\"requests_in_flight\": 0") != nullptr);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// A thread's last error is its own: another thread's failure does not replace the request id of a
// rejection, and a successful retrieve that submits a pending request leaves no error.
static void test_last_error_is_per_thread() {
    printf("=== test_last_error_is_per_thread ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mc = file_model("dummy.dxnn");
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS);
    enqueue_accepted(0, 17);  // request 16 waits for a buffer
    dxrt::run_async_failures().store(1);

    std::promise<void> rejected, other_failed;
    std::future<void> other_failed_done = other_failed.get_future();
    std::thread a([&]() {
        bool seen_rejection = false;
        for (int i = 0; i < 17; i++) {
            Tensors *output = nullptr;
            const RuntimeStatus s = runtime_retrieve_output(nullptr, &output, 2000);
            if (s == RUNTIME_STATUS_SUCCESS) {
                assert(seen_rejection || runtime_get_error() == nullptr);
                free_output(output);
                continue;
            }
            assert(s == RUNTIME_STATUS_INFERENCE_ERROR && !seen_rejection);
            seen_rejection = true;
            rejected.set_value();
            other_failed_done.wait();
            assert(strstr(runtime_get_error(), "request 16") != nullptr);
        }
        assert(seen_rejection);
    });
    std::thread b([&]() {
        rejected.get_future().wait();
        Tensors *input = make_input(99);
        assert(runtime_enqueue_input(5, input) == RUNTIME_STATUS_INVALID_MODEL_ID);
        free_output(input);
        assert(strcmp(runtime_get_error(), "Invalid model_id") == 0);
        other_failed.set_value();
    });
    a.join();
    b.join();

    // runtime_cleanup and a successful runtime_init clear the thread's last error (conformance RT-L-15).
    Tensors *input = make_input(0);
    assert(runtime_enqueue_input(5, input) == RUNTIME_STATUS_INVALID_MODEL_ID);
    free_output(input);
    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    assert(runtime_get_error() == nullptr);
    assert(runtime_load_models(1, &mc) == RUNTIME_STATUS_NOT_INITIALIZED);
    assert(runtime_get_error() != nullptr);
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    assert(runtime_get_error() == nullptr);

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// Each model's pending requests get output buffers in the order they were enqueued.
static void test_pending_fifo_order() {
    printf("=== test_pending_fifo_order ===\n");

    Config cfg = {0, nullptr, nullptr};
    assert(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS);
    ModelConfig mcs[] = {file_model("dummy.dxnn"), file_model("dummy.dxnn")};
    assert(runtime_load_models(2, mcs) == RUNTIME_STATUS_SUCCESS);
    {
        std::lock_guard<std::mutex> lk(dxrt::run_async_log().mtx);
        dxrt::run_async_log().first_ints.clear();
    }

    // Request i goes to model i % 2 and carries i in its first input bytes; 36 per model, 16 fit the pool.
    const int kRequests = 72;
    for (int i = 0; i < kRequests; i++) {
        Tensors *input = make_input(i);
        memcpy(input->tensors[0].data, &i, sizeof(i));
        assert(runtime_enqueue_input(i % 2, input) == RUNTIME_STATUS_SUCCESS);
    }
    for (int i = 0; i < kRequests; i++) {
        Tensors *output = nullptr;
        assert(runtime_retrieve_output(nullptr, &output, 2000) == RUNTIME_STATUS_SUCCESS);
        free_output(output);
    }

    std::vector<int> submitted;
    {
        std::lock_guard<std::mutex> lk(dxrt::run_async_log().mtx);
        submitted = dxrt::run_async_log().first_ints;
    }
    assert((int)submitted.size() == kRequests);
    int last[2] = {-1, -1};
    for (int id : submitted) {
        assert(id > last[id % 2]);
        last[id % 2] = id;
    }

    assert(runtime_cleanup() == RUNTIME_STATUS_SUCCESS);
    printf("  PASSED\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    printf("\n====== OAAX 2.0 Runtime Tests ======\n\n");

    // The runtime checks the model file exists before handing it to DX-RT;
    // the mock engine ignores its contents, so an empty placeholder suffices.
    FILE *f = fopen("dummy.dxnn", "wb");
    assert(f);
    fclose(f);

    test_basic_lifecycle();
    test_load_without_init();
    test_single_request();
    test_timeout();
    test_invalid_model_id();
    test_async_pipeline();
    test_get_info();
    test_input_size_mismatch();
    test_backpressure();
    test_scalar_output();
    test_get_error_thread_safe();
    test_model_file_not_found();
    test_load_from_memory();
    test_memory_buffer_reusable_after_load();
    test_load_requires_source();
    test_invalid_model_buffer();
    test_memory_takes_precedence_over_file();
    test_load_multiple_models();
    test_load_same_file_twice();
    test_mixed_sources();
    test_second_load_rejected();
    test_failed_load_rolls_back();
    test_load_exception_rolls_back();
    test_concurrent_producers_no_retry();
    test_backpressure_isolated();
    test_info_counts_unretrieved();
    test_cleanup_multi_model_wakes_retriever();
    test_enqueue_beyond_pool_is_queued();
    test_pending_queue_full_returns_timeout();
    test_max_pending_zero_keeps_old_behaviour();
    test_invalid_max_pending_rejected();
    test_cleanup_frees_pending();
    test_info_counts_pending();
    test_rejected_submission_answered();
    test_last_error_is_per_thread();
    test_pending_fifo_order();

    remove("dummy.dxnn");
    printf("\n====== ALL TESTS PASSED ======\n\n");
    return 0;
}
