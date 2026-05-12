#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <numeric>
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
            "  --runs <N>             Number of inference runs (default: 100)\n"
            "  --warmup <N>           Number of warmup runs (default: 10)\n"
            "  --csv                  Print results in CSV format\n"
            "\n"
            "CSV header: model,runs,avg_ms,p50_ms,p95_ms,min_ms,max_ms,throughput_fps\n",
            prog);
}

static Tensors *make_input(int id, size_t data_size) {
    Tensors *t = (Tensors *)malloc(sizeof(Tensors));
    if (!t) return nullptr;

    t->id = id;
    t->num_tensors = 1;
    t->tensors = (TensorDescriptor *)calloc(1, sizeof(TensorDescriptor));
    if (!t->tensors) {
        free(t);
        return nullptr;
    }

    t->tensors[0].name = strdup("input");
    t->tensors[0].data_type = DATA_TYPE_UINT8;
    t->tensors[0].rank = 0;
    t->tensors[0].shape = nullptr;
    t->tensors[0].data_size = data_size;
    t->tensors[0].data = calloc(1, data_size);
    if (!t->tensors[0].data || !t->tensors[0].name) {
        free(t->tensors[0].name);
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

int main(int argc, char **argv) {
    const char *model_path = nullptr;
    size_t input_size = 0;
    int num_runs = 100;
    int warmup = 10;
    bool csv_mode = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--input-size") == 0 && i + 1 < argc) {
            input_size = (size_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            num_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
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

    Config cfg = {0, nullptr, nullptr};
    RuntimeStatus s = runtime_init(cfg);
    if (s != RUNTIME_STATUS_SUCCESS) {
        fprintf(stderr, "runtime_init failed: %s\n", runtime_get_error());
        return 1;
    }

    ModelConfig mc{};
    mc.file_path = model_path;
    mc.config = cfg;
    s = runtime_load_models(1, &mc);
    if (s != RUNTIME_STATUS_SUCCESS) {
        fprintf(stderr, "runtime_load_models failed: %s\n", runtime_get_error());
        runtime_cleanup();
        return 1;
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        Tensors *inp = make_input(i, input_size);
        if (!inp) {
            fprintf(stderr, "OOM\n");
            runtime_cleanup();
            return 1;
        }
        s = runtime_enqueue_input(0, inp);
        if (s != RUNTIME_STATUS_SUCCESS) {
            fprintf(stderr, "Enqueue failed during warmup: %s\n", runtime_get_error());
            free_tensors(inp);
            runtime_cleanup();
            return 1;
        }
        int out_model = -1;
        Tensors *out = nullptr;
        runtime_retrieve_output(&out_model, &out, -1);
        free_tensors(out);
    }

    // Timed runs
    std::vector<double> latencies;
    latencies.reserve(num_runs);

    for (int i = 0; i < num_runs; i++) {
        Tensors *inp = make_input(i, input_size);
        if (!inp) {
            fprintf(stderr, "OOM\n");
            runtime_cleanup();
            return 1;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        s = runtime_enqueue_input(0, inp);
        if (s != RUNTIME_STATUS_SUCCESS) {
            fprintf(stderr, "Enqueue failed at run %d: %s\n", i, runtime_get_error());
            free_tensors(inp);
            runtime_cleanup();
            return 1;
        }

        int out_model = -1;
        Tensors *out = nullptr;
        s = runtime_retrieve_output(&out_model, &out, -1);
        auto t1 = std::chrono::high_resolution_clock::now();

        if (s != RUNTIME_STATUS_SUCCESS) {
            fprintf(stderr, "Retrieve failed at run %d: %s\n", i, runtime_get_error());
            runtime_cleanup();
            return 1;
        }
        free_tensors(out);

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies.push_back(ms);
    }

    runtime_cleanup();

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    double p50 = percentile(latencies, 50.0);
    double p95 = percentile(latencies, 95.0);
    double min_ms = *std::min_element(latencies.begin(), latencies.end());
    double max_ms = *std::max_element(latencies.begin(), latencies.end());
    double fps = 1000.0 / avg;

    // Extract model name from path
    std::string path_str(model_path);
    size_t slash = path_str.rfind('/');
    std::string model_name = (slash == std::string::npos) ? path_str : path_str.substr(slash + 1);
    size_t dot = model_name.rfind('.');
    if (dot != std::string::npos) model_name = model_name.substr(0, dot);

    if (csv_mode) {
        printf("%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f\n", model_name.c_str(), num_runs, avg, p50, p95, min_ms, max_ms,
               fps);
    } else {
        printf(
            "model=%s runs=%d avg_ms=%.3f p50_ms=%.3f p95_ms=%.3f "
            "min_ms=%.3f max_ms=%.3f throughput_fps=%.1f\n",
            model_name.c_str(), num_runs, avg, p50, p95, min_ms, max_ms, fps);
    }

    return 0;
}
