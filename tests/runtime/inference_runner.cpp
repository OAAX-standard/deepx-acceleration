#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "runtime_core.h"
#include "yolo_postprocess.hpp"

typedef std::chrono::high_resolution_clock Clock;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --model <path> --input-size <bytes> [--model <path> --input-size <bytes> ...] [options]\n"
            "\n"
            "Per model (repeat for each model; the Nth of each option belongs to the Nth --model):\n"
            "  --model <path>         Path to the .dxnn model file\n"
            "  --input-size <bytes>   Size of the input tensor data in bytes (required)\n"
            "  --input-shape N,H,W,C  Input tensor shape (default: unset; give for all models or none)\n"
            "  --image <path>         Raw uint8 HWC input file (default: zeros; give for all models or none)\n"
            "\n"
            "Optional:\n"
            "  --from-memory          Load models from memory buffers (freed right after loading)\n"
            "  --runs <N>             Number of timed inference runs per model (default: 100)\n"
            "  --warmup <N>           Number of warmup runs per model, sequential (default: 10)\n"
            "  --pipeline-depth <N>   Requests kept in flight per model (default: 4)\n"
            "  --csv                  Print results in CSV format, one row per model\n"
            "\n"
            "Each model has its own producer thread, so all models run concurrently. throughput_fps is\n"
            "per model: its runs over its own window (first enqueue to last output retrieved).\n"
            "\n"
            "CSV header: model,runs,avg_ms,p50_ms,p95_ms,min_ms,max_ms,throughput_fps\n",
            prog);
}

// ---------------------------------------------------------------------------
// Model and tensor helpers
// ---------------------------------------------------------------------------

struct Model {
    const char *path;
    std::string name;  // file name without directory and extension
    size_t input_size;
    std::vector<int> shape;
    std::vector<uint8_t> image;  // input data; empty means zeros
    yolo::Family family;
};

static std::string file_stem(const std::string &path) {
#ifdef _WIN32
    size_t slash = path.find_last_of("/\\");
#else
    size_t slash = path.rfind('/');
#endif
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.rfind('.');
    return (dot == std::string::npos) ? name : name.substr(0, dot);
}

static std::string shape_str(const int *dims, int rank) {
    std::string s = "[";
    for (int i = 0; i < rank; i++) s += std::to_string(dims[i]) + (i + 1 < rank ? "," : "");
    return s + "]";
}

static bool read_file(const char *path, std::vector<uint8_t> &data) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    long size = (fseek(f, 0, SEEK_END) == 0) ? ftell(f) : -1;
    bool ok = size >= 0 && fseek(f, 0, SEEK_SET) == 0;
    if (ok) {
        data.resize((size_t)size);
        ok = fread(data.data(), 1, data.size(), f) == data.size();
    }
    fclose(f);
    return ok;
}

static Tensors *make_input(int id, const Model &model) {
    Tensors *t = (Tensors *)malloc(sizeof(Tensors));
    if (!t) return nullptr;

    t->id = id;
    t->num_tensors = 1;
    t->tensors = (TensorDescriptor *)calloc(1, sizeof(TensorDescriptor));
    if (!t->tensors) {
        free(t);
        return nullptr;
    }

    TensorDescriptor &td = t->tensors[0];
    td.name = strdup("input");
    td.data_type = DATA_TYPE_UINT8;
    td.rank = (int)model.shape.size();
    td.shape = nullptr;
    if (!model.shape.empty()) {
        td.shape = (int *)malloc(model.shape.size() * sizeof(int));
        if (td.shape) memcpy(td.shape, model.shape.data(), model.shape.size() * sizeof(int));
    }
    td.data_size = model.input_size;
    td.data = malloc(model.input_size);
    if (!td.data || !td.name || (td.rank > 0 && !td.shape)) {
        free(td.name);
        free(td.shape);
        free(td.data);
        free(t->tensors);
        free(t);
        return nullptr;
    }
    if (model.image.empty())
        memset(td.data, 0, model.input_size);
    else
        memcpy(td.data, model.image.data(), model.input_size);
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

// Parses s as a whole decimal integer in [min, max] into out; out is untouched on failure.
template <typename T>
static bool parse_number(const char *s, long long min, long long max, T &out) {
    char *end = nullptr;
    errno = 0;
    const long long v = strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < min || v > max) return false;
    out = (T)v;
    return true;
}

// Parses a comma-separated list of positive dimensions, e.g. "1,224,224,3".
static bool parse_shape(const char *s, std::vector<int> &shape) {
    std::string str(s);
    size_t pos = 0;
    do {
        size_t end = str.find(',', pos);
        if (end == std::string::npos) end = str.size();
        int dim = 0;
        if (!parse_number(str.substr(pos, end - pos).c_str(), 1, INT_MAX, dim)) return false;
        shape.push_back(dim);
        pos = end + 1;
    } while (pos <= str.size());
    return true;
}

// Loads all models in one runtime_load_models call; model i gets model id i.
// With from_memory, the file contents are passed as model_data and freed as soon
// as the call returns: the runtime keeps no reference to them.
static bool load_models(const std::vector<Model> &models, bool from_memory) {
    std::vector<ModelConfig> configs(models.size());  // value-initialized: all fields null/zero
    std::vector<std::vector<uint8_t>> blobs(from_memory ? models.size() : 0);
    for (size_t i = 0; i < models.size(); i++) {
        if (!from_memory) {
            configs[i].file_path = models[i].path;
            continue;
        }
        if (!read_file(models[i].path, blobs[i])) {
            fprintf(stderr, "Error: cannot read model file: %s\n", models[i].path);
            return false;
        }
        configs[i].model_data = blobs[i].data();
        configs[i].model_size = blobs[i].size();
    }
    if (runtime_load_models((int)models.size(), configs.data()) != RUNTIME_STATUS_SUCCESS) {
        fprintf(stderr, "runtime_load_models failed: %s\n", runtime_get_error());
        return false;
    }
    return true;
}

// Sequential runs so the hardware is in a stable state before pipelined timing.
// Logs the output tensors (and YOLO detections) of each model's first run.
static bool warmup(const std::vector<Model> &models, int runs) {
    for (int i = 0; i < runs; i++) {
        for (size_t m = 0; m < models.size(); m++) {
            const Model &model = models[m];
            Tensors *inp = make_input(i, model);
            if (!inp) {
                fprintf(stderr, "[warmup] OOM at run %d/%d\n", i + 1, runs);
                return false;
            }
            if (runtime_enqueue_input((int)m, inp) != RUNTIME_STATUS_SUCCESS) {
                fprintf(stderr, "[warmup] %s: enqueue failed at run %d/%d: %s\n", model.name.c_str(), i + 1, runs,
                        runtime_get_error());
                free_tensors(inp);
                return false;
            }
            int out_model = -1;
            Tensors *out = nullptr;
            if (runtime_retrieve_output(&out_model, &out, -1) != RUNTIME_STATUS_SUCCESS) {
                fprintf(stderr, "[warmup] %s: retrieve failed at run %d/%d: %s\n", model.name.c_str(), i + 1, runs,
                        runtime_get_error());
                free_tensors(out);
                return false;
            }
            if (out_model != (int)m) {
                fprintf(stderr, "[warmup] %s: output returned for model %d, expected %zu\n", model.name.c_str(),
                        out_model, m);
                free_tensors(out);
                return false;
            }
            if (i == 0) {
                fprintf(stderr, "[output tensor] %s: num_tensors=%d\n", model.name.c_str(), out->num_tensors);
                for (int j = 0; j < out->num_tensors; j++) {
                    const TensorDescriptor &td = out->tensors[j];
                    fprintf(stderr, "[output tensor] [%d] name=%s data_type=%d rank=%d shape=%s data_size=%zu\n", j,
                            td.name ? td.name : "(null)", (int)td.data_type, td.rank,
                            shape_str(td.shape, td.rank).c_str(), td.data_size);
                }
                if (model.family != yolo::Family::UNKNOWN)
                    yolo::log_detections(yolo::postprocess(out, model.family), model.name.c_str());
            }
            free_tensors(out);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Shared state between the producer threads and the consumer thread
// ---------------------------------------------------------------------------

struct ModelStats {
    std::vector<double> latencies_ms;
    int in_flight = 0;           // accepted, not yet retrieved
    Clock::time_point last_out;  // when the last output of this model was retrieved
};

struct SharedState {
    SharedState(size_t num_models, int runs)
        : runs(runs), enqueue_times(num_models * runs), models(num_models), active_producers((int)num_models) {}

    const int runs;  // per model; model m owns request ids [m * runs, (m + 1) * runs)

    // Guards everything below; cv signals a change of submitted/retired/in_flight/
    // active_producers/error.
    std::mutex mutex;
    std::condition_variable cv;

    std::vector<Clock::time_point> enqueue_times;  // indexed by request id
    std::vector<ModelStats> models;
    int submitted = 0;  // requests accepted by the runtime
    int retired = 0;    // outputs retrieved
    int active_producers;
    bool error = false;
};

static void fail(SharedState &state) {
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.error = true;
    }
    state.cv.notify_all();
}

// ---------------------------------------------------------------------------
// Producer (one per model): enqueues the model's runs requests, at most
// pipeline_depth in flight
// ---------------------------------------------------------------------------

static void producer_thread(const Model &model, int m, int pipeline_depth, SharedState &state) {
    ModelStats &stats = state.models[m];
    for (int i = 0; i < state.runs; i++) {
        {
            std::unique_lock<std::mutex> lock(state.mutex);
            state.cv.wait(lock, [&] { return stats.in_flight < pipeline_depth || state.error; });
            if (state.error) break;
        }

        const int id = m * state.runs + i;
        Tensors *inp = make_input(id, model);
        if (!inp) {
            fprintf(stderr, "[producer] %s: OOM at run %d/%d\n", model.name.c_str(), i + 1, state.runs);
            fail(state);
            break;
        }

        RuntimeStatus s;
        {
            std::unique_lock<std::mutex> lock(state.mutex);
            state.enqueue_times[id] = Clock::now();
            // A full output pool (TIMEOUT) is backpressure: the input stays ours.
            // The consumer is the only one retrieving, so wait for it to retire
            // an output (freeing a pool slot), then retry.
            for (;;) {
                const int seen = state.retired;
                lock.unlock();
                s = runtime_enqueue_input(m, inp);
                lock.lock();
                if (s != RUNTIME_STATUS_TIMEOUT) break;
                state.cv.wait(lock, [&] { return state.retired != seen || state.error; });
                if (state.error) break;
            }
            if (s == RUNTIME_STATUS_SUCCESS) {
                stats.in_flight++;
                state.submitted++;
            }
        }
        if (s != RUNTIME_STATUS_SUCCESS) {
            if (s != RUNTIME_STATUS_TIMEOUT)
                fprintf(stderr, "[producer] %s: enqueue failed at run %d/%d: %s\n", model.name.c_str(), i + 1,
                        state.runs, runtime_get_error());
            free_tensors(inp);
            fail(state);
            break;
        }
        state.cv.notify_all();
    }

    // Lets the consumer stop waiting for outputs that will never come.
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.active_producers--;
    }
    state.cv.notify_all();
}

// ---------------------------------------------------------------------------
// Consumer: the only thread retrieving outputs; records per-request latency
// under the model id the runtime returns with each output
// ---------------------------------------------------------------------------

static void consumer_thread(SharedState &state) {
    for (;;) {
        {
            // Retrieve only while an output is guaranteed to arrive, so the
            // blocking retrieve below can never outlive the producers.
            std::unique_lock<std::mutex> lock(state.mutex);
            state.cv.wait(
                lock, [&] { return state.retired < state.submitted || state.active_producers == 0 || state.error; });
            if (state.error || state.retired == state.submitted) return;
        }

        int m = -1;
        Tensors *out = nullptr;
        RuntimeStatus s = runtime_retrieve_output(&m, &out, -1);
        const Clock::time_point t_out = Clock::now();
        const int id = out ? out->id : -1;
        free_tensors(out);
        if (s != RUNTIME_STATUS_SUCCESS) {
            fprintf(stderr, "[consumer] retrieve failed: %s\n", runtime_get_error());
            fail(state);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(state.mutex);
            if (m < 0 || (size_t)m >= state.models.size() || id < 0 || id / state.runs != m) {
                fprintf(stderr, "[consumer] unexpected output: model %d, request id %d\n", m, id);
                state.error = true;
            } else {
                ModelStats &stats = state.models[m];
                stats.latencies_ms.push_back(
                    std::chrono::duration<double, std::milli>(t_out - state.enqueue_times[id]).count());
                stats.last_out = t_out;
                stats.in_flight--;
                state.retired++;
            }
        }
        state.cv.notify_all();
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    std::vector<const char *> model_paths, image_paths;
    std::vector<size_t> input_sizes;
    std::vector<std::vector<int>> input_shapes;
    int num_runs = 100;
    int warmup_runs = 10;
    int pipeline_depth = 4;
    bool from_memory = false;
    bool csv_mode = false;

    for (int i = 1; i < argc; i++) {
        const char *opt = argv[i];
        if (strcmp(opt, "--from-memory") == 0) {
            from_memory = true;
            continue;
        }
        if (strcmp(opt, "--csv") == 0) {
            csv_mode = true;
            continue;
        }

        // Every other option takes a value.
        const char *val = (i + 1 < argc) ? argv[++i] : nullptr;
        bool ok = val != nullptr;
        if (strcmp(opt, "--model") == 0) {
            if (ok) model_paths.push_back(val);
        } else if (strcmp(opt, "--image") == 0) {
            if (ok) image_paths.push_back(val);
        } else if (strcmp(opt, "--input-size") == 0) {
            size_t size = 0;
            ok = ok && parse_number(val, 1, LLONG_MAX, size);
            if (ok) input_sizes.push_back(size);
        } else if (strcmp(opt, "--input-shape") == 0) {
            std::vector<int> shape;
            ok = ok && parse_shape(val, shape);
            if (ok) input_shapes.push_back(shape);
        } else if (strcmp(opt, "--runs") == 0) {
            ok = ok && parse_number(val, 1, INT_MAX, num_runs);
        } else if (strcmp(opt, "--warmup") == 0) {
            ok = ok && parse_number(val, 0, INT_MAX, warmup_runs);
        } else if (strcmp(opt, "--pipeline-depth") == 0) {
            ok = ok && parse_number(val, 1, INT_MAX, pipeline_depth);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", opt);
            print_usage(argv[0]);
            return 2;
        }
        if (!ok) {
            fprintf(stderr, "Error: missing or invalid value for %s\n", opt);
            print_usage(argv[0]);
            return 2;
        }
    }

    const size_t num_models = model_paths.size();
    if (num_models == 0 || input_sizes.size() != num_models ||
        (!input_shapes.empty() && input_shapes.size() != num_models) ||
        (!image_paths.empty() && image_paths.size() != num_models)) {
        fprintf(stderr, "Error: each --model needs one --input-size (and one --input-shape/--image if any is given)\n");
        print_usage(argv[0]);
        return 2;
    }
    if ((long long)num_runs * num_models > INT_MAX) {  // request ids are ints
        fprintf(stderr, "Error: --runs times the number of models exceeds %d\n", INT_MAX);
        return 2;
    }

    fprintf(stderr, "[config] models=%zu runs=%d warmup=%d pipeline_depth=%d from_memory=%d\n", num_models, num_runs,
            warmup_runs, pipeline_depth, (int)from_memory);
    std::vector<Model> models(num_models);
    for (size_t m = 0; m < num_models; m++) {
        Model &model = models[m];
        model.path = model_paths[m];
        model.name = file_stem(model.path);
        model.input_size = input_sizes[m];
        if (!input_shapes.empty()) model.shape = input_shapes[m];
        if (!image_paths.empty()) {
            if (!read_file(image_paths[m], model.image) || model.image.size() != model.input_size) {
                fprintf(stderr, "Error: cannot read %zu bytes of input from %s\n", model.input_size, image_paths[m]);
                return 1;
            }
        }
        model.family = yolo::detect_family(model.name);
        fprintf(stderr, "[config] model[%zu]=%s input_size=%zu input_shape=%s image=%s%s\n", m, model.path,
                model.input_size, shape_str(model.shape.data(), (int)model.shape.size()).c_str(),
                image_paths.empty() ? "(zeros)" : image_paths[m],
                model.family != yolo::Family::UNKNOWN ? " (YOLO post-processing)" : "");
    }

    Config cfg = {0, nullptr, nullptr};
    if (runtime_init(cfg) != RUNTIME_STATUS_SUCCESS) {
        fprintf(stderr, "runtime_init failed: %s\n", runtime_get_error());
        return 1;
    }
    if (!load_models(models, from_memory)) {
        runtime_cleanup();
        return 1;
    }
    fprintf(stderr, "[init] %zu model(s) loaded\n", num_models);

    fprintf(stderr, "[warmup] starting %d warmup run(s) per model\n", warmup_runs);
    if (!warmup(models, warmup_runs)) {
        runtime_cleanup();
        return 1;
    }
    fprintf(stderr, "[warmup] done\n");

    // Pipelined timed runs: one producer per model, one consumer for all of them.
    fprintf(stderr, "[bench] starting %d timed run(s) per model with pipeline_depth=%d\n", num_runs, pipeline_depth);
    SharedState state(num_models, num_runs);
    std::vector<std::thread> producers;
    for (size_t m = 0; m < num_models; m++)
        producers.push_back(
            std::thread(producer_thread, std::cref(models[m]), (int)m, pipeline_depth, std::ref(state)));
    std::thread consumer(consumer_thread, std::ref(state));
    for (size_t m = 0; m < num_models; m++) producers[m].join();
    consumer.join();

    runtime_cleanup();

    if (state.error) {
        fprintf(stderr, "Inference run failed.\n");
        return 1;
    }

    for (size_t m = 0; m < num_models; m++) {
        ModelStats &stats = state.models[m];
        std::vector<double> &lat = stats.latencies_ms;
        // The model's own window, so a model starved by the others shows a lower rate.
        const double window_ms =
            std::chrono::duration<double, std::milli>(stats.last_out - state.enqueue_times[m * num_runs]).count();
        double throughput_fps = (window_ms > 0) ? (num_runs * 1000.0 / window_ms) : 0.0;
        double avg = std::accumulate(lat.begin(), lat.end(), 0.0) / lat.size();
        double p50 = percentile(lat, 50.0);  // sorts lat, so min/max are its ends
        double p95 = percentile(lat, 95.0);
        const char *name = models[m].name.c_str();
        if (csv_mode) {
            printf("%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f\n", name, num_runs, avg, p50, p95, lat.front(), lat.back(),
                   throughput_fps);
        } else {
            printf(
                "model=%s runs=%d avg_ms=%.3f p50_ms=%.3f p95_ms=%.3f "
                "min_ms=%.3f max_ms=%.3f throughput_fps=%.1f\n",
                name, num_runs, avg, p50, p95, lat.front(), lat.back(), throughput_fps);
        }
    }
    return 0;
}
