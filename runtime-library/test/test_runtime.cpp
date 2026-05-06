#include "runtime_core.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <atomic>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static Tensors *make_input(int id) {
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

    size_t data_size = 1 * 3 * 224 * 224 * sizeof(float);
    t->tensors[0].data_size = data_size;
    t->tensors[0].data = malloc(data_size);
    assert(t->tensors[0].data);
    memset(t->tensors[0].data, 0xAA, data_size);

    return t;
}

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
    assert(output->id == 42);        // id propagation
    assert(out_model_id == 0);       // model_id propagation
    assert(output->num_tensors == 1);
    assert(output->tensors[0].name != nullptr);
    assert(output->tensors[0].data_size > 0);
    assert(output->tensors[0].rank == 4);
    assert(output->tensors[0].shape != nullptr);
    assert(output->tensors[0].data != nullptr);

    printf("  output id=%d, model_id=%d, name=%s, rank=%d, data_size=%zu\n",
           output->id, out_model_id, output->tensors[0].name,
           output->tensors[0].rank, output->tensors[0].data_size);

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
            RuntimeStatus s = runtime_enqueue_input(0, input);
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
// Main
// ---------------------------------------------------------------------------

int main() {
    printf("\n====== OAAX 2.0 Runtime Tests ======\n\n");

    test_basic_lifecycle();
    test_load_without_init();
    test_single_request();
    test_timeout();
    test_invalid_model_id();
    test_async_pipeline();
    test_get_info();

    printf("\n====== ALL TESTS PASSED ======\n\n");
    return 0;
}
