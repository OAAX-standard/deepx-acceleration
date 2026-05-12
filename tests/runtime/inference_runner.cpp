#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "runtime_core.h"

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --model <path> --input-size <bytes> [options]\n"
            "\n"
            "Required:\n"
            "  --model <path>         Path to the .dxnn model file\n"
            "  --input-size <bytes>   Size of the input tensor data in bytes\n"
            "\n"
            "Optional:\n"
            "  --input-shape N,H,W,C  Input tensor shape (default: unset)\n"
            "  --runs <N>             Number of timed inference runs (default: 100)\n"
            "  --warmup <N>           Number of warmup runs, sequential (default: 10)\n"
            "  --pipeline-depth <N>   Requests kept in-flight simultaneously (default: 4)\n"
            "  --csv                  Print results in CSV format\n"
            "\n"
            "CSV header: model,runs,avg_ms,p50_ms,p95_ms,min_ms,max_ms,throughput_fps\n",
            prog);
}

// ---------------------------------------------------------------------------
// Simple semaphore — limits in-flight requests to pipeline_depth
// ---------------------------------------------------------------------------

struct Semaphore {
    explicit Semaphore(int n) : count_(n) {}

    void acquire() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return count_ > 0; });
        --count_;
    }

    void release() {
        std::lock_guard<std::mutex> lock(mtx_);
        ++count_;
        cv_.notify_one();
    }

private:
    int count_;
    std::mutex mtx_;
    std::condition_variable cv_;
};

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

static int g_request_id = 0;

static Tensors *make_input(size_t data_size, const std::vector<int> &shape) {
    Tensors *t = (Tensors *)malloc(sizeof(Tensors));
    if (!t) return nullptr;

    t->id = g_request_id++;
    t->num_tensors = 1;
    t->tensors = (TensorDescriptor *)calloc(1, sizeof(TensorDescriptor));
    if (!t->tensors) {
        free(t);
        return nullptr;
    }

    t->tensors[0].name = strdup("input");
    t->tensors[0].data_type = DATA_TYPE_UINT8;
    t->tensors[0].rank = (int)shape.size();
    t->tensors[0].shape = nullptr;
    if (!shape.empty()) {
        t->tensors[0].shape = (int *)malloc(shape.size() * sizeof(int));
        if (t->tensors[0].shape)
            memcpy(t->tensors[0].shape, shape.data(), shape.size() * sizeof(int));
    }
    t->tensors[0].data_size = data_size;
    t->tensors[0].data = calloc(1, data_size);
    if (!t->tensors[0].data || !t->tensors[0].name) {
        free(t->tensors[0].name);
        free(t->tensors[0].shape);
        free(t->tensors[0].data);
        free(t->tensors);
        free(t);
        return nullptr;
    }
    return t;
}

static void free_tensors(Tensors *t) {
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

static double percentile(std::vector<double> &v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)(p / 100.0 * (v.size() - 1) + 0.5);
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

static std::vector<int> parse_shape(const char *s) {
    std::vector<int> shape;
    std::string str(s);
    size_t pos = 0;
    while (pos < str.size()) {
        size_t end = str.find(',', pos);
        if (end == std::string::npos) end = str.size();
        shape.push_back(std::stoi(str.substr(pos, end - pos)));
        pos = end + 1;
    }
    return shape;
}

// ---------------------------------------------------------------------------
// Shared state between producer and consumer threads
// ---------------------------------------------------------------------------

struct SharedState {
    // Enqueue timestamps, pushed by producer, popped by consumer (FIFO)
    std::queue<std::chrono::high_resolution_clock::time_point> enqueue_times;
    std::mutex times_mutex;

    std::atomic<bool> error{false};
};

// ---------------------------------------------------------------------------
// Producer: enqueues num_runs requests, bounded by semaphore
// ---------------------------------------------------------------------------

static void producer_thread(int num_runs, size_t input_size, const std::vector<int> &shape,
                             Semaphore &sem, SharedState &state) {
    for (int i = 0; i < num_runs && !state.error.load(); i++) {
        sem.acquire();

        Tensors *inp = make_input(input_size, shape);
        if (!inp) {
            fprintf(stderr, "[producer] OOM at run %d/%d\n", i + 1, num_runs);
            state.error.store(true);
            sem.release();
            return;
        }

        auto t = std::chrono::high_resolution_clock::now();
        {
            std::lock_guard<std::mutex> lock(state.times_mutex);
            state.enqueue_times.push(t);
        }

        RuntimeStatus s = runtime_enqueue_input(0, inp);
        if (s != RUNTIME_STATUS_SUCCESS) {
            fprintf(stderr, "[producer] enqueue failed at run %d/%d: %s\n", i + 1, num_runs, runtime_get_error());
            free_tensors(inp);
            state.error.store(true);
            sem.release();
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Consumer: retrieves num_runs outputs, records per-request latency
// ---------------------------------------------------------------------------

static void consumer_thread(int num_runs, Semaphore &sem, SharedState &state,
                             std::vector<double> &latencies_ms) {
    latencies_ms.reserve(num_runs);

    for (int i = 0; i < num_runs && !state.error.load(); i++) {
        int out_model = -1;
        Tensors *out = nullptr;
        RuntimeStatus s = runtime_retrieve_output(&out_model, &out, -1);
        auto t_out = std::chrono::high_resolution_clock::now();

        if (s != RUNTIME_STATUS_SUCCESS) {
            fprintf(stderr, "[consumer] retrieve failed at run %d/%d: %s\n", i + 1, num_runs, runtime_get_error());
            free_tensors(out);
            sem.release();
            state.error.store(true);
            return;
        }
        if (out_model != 0)
            fprintf(stderr, "[consumer] unexpected out_model=%d at run %d/%d\n", out_model, i + 1, num_runs);

        free_tensors(out);
        sem.release();

        std::chrono::high_resolution_clock::time_point t_in;
        {
            std::lock_guard<std::mutex> lock(state.times_mutex);
            t_in = state.enqueue_times.front();
            state.enqueue_times.pop();
        }

        latencies_ms.push_back(std::chrono::duration<double, std::milli>(t_out - t_in).count());
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    const char *model_path = nullptr;
    size_t input_size = 0;
    std::vector<int> input_shape;
    int num_runs = 100;
    int warmup = 10;
    int pipeline_depth = 4;
    bool csv_mode = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--input-size") == 0 && i + 1 < argc) {
            input_size = (size_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--input-shape") == 0 && i + 1 < argc) {
            input_shape = parse_shape(argv[++i]);
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            num_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pipeline-depth") == 0 && i + 1 < argc) {
            pipeline_depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0) {
            csv_mode = true;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path || input_size == 0) {
        print_usage(argv[0]);
        return 1;
    }

    // Startup config summary
    fprintf(stderr, "[config] model=%s input_size=%zu runs=%d warmup=%d pipeline_depth=%d\n",
            model_path, input_size, num_runs, warmup, pipeline_depth);
    if (!input_shape.empty()) {
        std::string shape_str = "[";
        for (size_t i = 0; i < input_shape.size(); i++)
            shape_str += std::to_string(input_shape[i]) + (i + 1 < input_shape.size() ? "," : "");
        shape_str += "]";
        fprintf(stderr, "[config] input_shape=%s\n", shape_str.c_str());
    }

    Config cfg = {0, nullptr, nullptr};
    RuntimeStatus s = runtime_init(cfg);
    if (s != RUNTIME_STATUS_SUCCESS) {
        fprintf(stderr, "runtime_init failed: %s\n", runtime_get_error());
        return 1;
    }
    fprintf(stderr, "[init] runtime initialized\n");

    ModelConfig mc{};
    mc.file_path = model_path;
    mc.config = cfg;
    s = runtime_load_models(1, &mc);
    if (s != RUNTIME_STATUS_SUCCESS) {
        fprintf(stderr, "runtime_load_models failed: %s\n", runtime_get_error());
        runtime_cleanup();
        return 1;
    }
    fprintf(stderr, "[init] model loaded: %s\n", model_path);

    // Print input tensor metadata once
    {
        Tensors *inp = make_input(input_size, input_shape);
        if (inp) {
            fprintf(stderr, "[input tensor] num_tensors=%d\n", inp->num_tensors);
            for (int i = 0; i < inp->num_tensors; i++) {
                const TensorDescriptor &td = inp->tensors[i];
                std::string shape_str = "[";
                for (int j = 0; j < td.rank; j++)
                    shape_str += std::to_string(td.shape[j]) + (j + 1 < td.rank ? "," : "");
                shape_str += "]";
                fprintf(stderr, "[input tensor] [%d] name=%s data_type=%d rank=%d shape=%s data_size=%zu\n",
                        i, td.name ? td.name : "(null)", (int)td.data_type, td.rank, shape_str.c_str(), td.data_size);
            }
            free_tensors(inp);
            g_request_id = 0;
        }
    }

    // Warmup: sequential so hardware is in a stable state before pipelined timing
    fprintf(stderr, "[warmup] starting %d warmup run(s)\n", warmup);
    for (int i = 0; i < warmup; i++) {
        Tensors *inp = make_input(input_size, input_shape);
        if (!inp) {
            fprintf(stderr, "[warmup] OOM at run %d/%d\n", i + 1, warmup);
            runtime_cleanup();
            return 1;
        }
        if (runtime_enqueue_input(0, inp) != RUNTIME_STATUS_SUCCESS) {
            fprintf(stderr, "[warmup] enqueue failed at run %d/%d: %s\n", i + 1, warmup, runtime_get_error());
            free_tensors(inp);
            runtime_cleanup();
            return 1;
        }
        int out_model = -1;
        Tensors *out = nullptr;
        if (runtime_retrieve_output(&out_model, &out, -1) != RUNTIME_STATUS_SUCCESS) {
            fprintf(stderr, "[warmup] retrieve failed at run %d/%d: %s\n", i + 1, warmup, runtime_get_error());
            free_tensors(out);
            runtime_cleanup();
            return 1;
        }
        if (out_model != 0)
            fprintf(stderr, "[warmup] unexpected out_model=%d at run %d/%d\n", out_model, i + 1, warmup);
        // Log output tensor metadata on the first warmup output
        if (i == 0 && out) {
            fprintf(stderr, "[output tensor] num_tensors=%d\n", out->num_tensors);
            for (int j = 0; j < out->num_tensors; j++) {
                const TensorDescriptor &td = out->tensors[j];
                std::string shape_str = "[";
                for (int k = 0; k < td.rank; k++)
                    shape_str += std::to_string(td.shape[k]) + (k + 1 < td.rank ? "," : "");
                shape_str += "]";
                fprintf(stderr, "[output tensor] [%d] name=%s data_type=%d rank=%d shape=%s data_size=%zu\n",
                        j, td.name ? td.name : "(null)", (int)td.data_type, td.rank, shape_str.c_str(), td.data_size);
            }
        }
        free_tensors(out);
    }
    fprintf(stderr, "[warmup] done\n");
    g_request_id = 0;

    // Pipelined timed runs: producer and consumer run concurrently.
    // The semaphore limits in-flight requests to pipeline_depth.
    fprintf(stderr, "[bench] starting %d timed run(s) with pipeline_depth=%d\n", num_runs, pipeline_depth);
    Semaphore sem(pipeline_depth);
    SharedState state;
    std::vector<double> latencies_ms;

    auto wall_start = std::chrono::high_resolution_clock::now();

    std::thread producer(producer_thread, num_runs, input_size, std::cref(input_shape),
                         std::ref(sem), std::ref(state));
    std::thread consumer(consumer_thread, num_runs, std::ref(sem), std::ref(state),
                         std::ref(latencies_ms));

    producer.join();
    consumer.join();

    auto wall_end = std::chrono::high_resolution_clock::now();

    runtime_cleanup();

    if (state.error.load()) {
        fprintf(stderr, "Inference run failed.\n");
        return 1;
    }

    double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    double throughput_fps = (wall_ms > 0) ? (num_runs * 1000.0 / wall_ms) : 0.0;

    double avg = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / latencies_ms.size();
    double p50 = percentile(latencies_ms, 50.0);
    double p95 = percentile(latencies_ms, 95.0);
    double min_ms = *std::min_element(latencies_ms.begin(), latencies_ms.end());
    double max_ms = *std::max_element(latencies_ms.begin(), latencies_ms.end());

    std::string path_str(model_path);
    size_t slash = path_str.rfind('/');
    std::string model_name = (slash == std::string::npos) ? path_str : path_str.substr(slash + 1);
    size_t dot = model_name.rfind('.');
    if (dot != std::string::npos) model_name = model_name.substr(0, dot);

    if (csv_mode) {
        printf("%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f\n",
               model_name.c_str(), num_runs, avg, p50, p95, min_ms, max_ms, throughput_fps);
    } else {
        printf("model=%s runs=%d avg_ms=%.3f p50_ms=%.3f p95_ms=%.3f "
               "min_ms=%.3f max_ms=%.3f throughput_fps=%.1f\n",
               model_name.c_str(), num_runs, avg, p50, p95, min_ms, max_ms, throughput_fps);
    }

    return 0;
}
