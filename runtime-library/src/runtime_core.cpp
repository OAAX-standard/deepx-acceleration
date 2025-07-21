#include "runtime_core.h"
#include "concurrent_queue.h"

extern "C" {
#include "queue.h"
#include "logger.h"
}

#include <atomic>
#include <iostream>
#include <memory>

#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

#include <dxrt/dxrt_api.h>

#define QUEUE_CAPACITY 100

// Structure to track job_id with its associated aligned_buffer
struct JobData {
    int job_id;
    uint8_t *aligned_buffer;
};

// Global variables
static dxrt::InferenceEngine *inference_engine = nullptr;
static char *_error_message = NULL;
Logger *logger = NULL;  // Make it accessible from C code
static LogLevel log_level = LOG_INFO;

// Queues
static Queue *input_queue = NULL, *output_queue = NULL;
static ConcurrentQueue<JobData> job_data_queue(QUEUE_CAPACITY);

// Running threads
static pthread_t inference_thread, wait_thread;
static std::atomic<int> stop_thread{0};

// Function prototypes
static void *inference_loop(void *arg);
static void *wait_loop(void *arg);
static int runtime_inference_execution(tensors_struct *input_tensors);
static tensors_struct* create_tensors_struct(size_t num_tensors);
static tensors_struct* create_output_tensors_from_dxrt(const std::vector<std::shared_ptr<dxrt::Tensor>>& outputs);
static tensor_data_type mapDataTypeToTensorDataType(dxrt::DataType dtype);

// Helper function to initialize tensors_struct
static tensors_struct* create_tensors_struct(size_t num_tensors) {
    if (num_tensors == 0) {
        return NULL;
    }
    
    // Allocate memory for the tensors_struct
    tensors_struct* tensors = (tensors_struct*)malloc(sizeof(tensors_struct));
    if (tensors == NULL) {
        return NULL;
    }
    
    tensors->num_tensors = num_tensors;
    
    // Allocate memory for names
    tensors->names = (char**)malloc(num_tensors * sizeof(char*));
    if (tensors->names == NULL) {
        free(tensors);
        return NULL;
    }
    
    // Allocate memory for data_types
    tensors->data_types = (tensor_data_type*)malloc(num_tensors * sizeof(tensor_data_type));
    if (tensors->data_types == NULL) {
        free(tensors->names);
        free(tensors);
        return NULL;
    }
    
    // Allocate memory for ranks
    tensors->ranks = (size_t*)malloc(num_tensors * sizeof(size_t));
    if (tensors->ranks == NULL) {
        free(tensors->data_types);
        free(tensors->names);
        free(tensors);
        return NULL;
    }
    
    // Allocate memory for shapes
    tensors->shapes = (size_t**)malloc(num_tensors * sizeof(size_t*));
    if (tensors->shapes == NULL) {
        free(tensors->ranks);
        free(tensors->data_types);
        free(tensors->names);
        free(tensors);
        return NULL;
    }
    
    // Allocate memory for data
    tensors->data = (void**)malloc(num_tensors * sizeof(void*));
    if (tensors->data == NULL) {
        free(tensors->shapes);
        free(tensors->ranks);
        free(tensors->data_types);
        free(tensors->names);
        free(tensors);
        return NULL;
    }
    
    // Initialize arrays to NULL
    for (size_t i = 0; i < num_tensors; i++) {
        tensors->names[i] = NULL;
        tensors->data_types[i] = DATA_TYPE_UNDEFINED;
        tensors->ranks[i] = 0;
        tensors->shapes[i] = NULL;
        tensors->data[i] = NULL;
    }
    
    return tensors;
}

// Helper function to create output tensors from dxrt::Tensor objects
static tensors_struct* create_output_tensors_from_dxrt(const std::vector<std::shared_ptr<dxrt::Tensor>>& outputs) {
    size_t num_output_tensors = outputs.size();
    tensors_struct* output_tensors = create_tensors_struct(num_output_tensors);
    if (output_tensors == NULL) {
        return NULL;
    }
    
    // Process each output tensor
    for (size_t i = 0; i < num_output_tensors; i++) {
        const auto& output = outputs[i];
        
        // Get tensor info
        auto shape = output->shape();
        auto dtype = output->type();
        auto data = output->data();
        
        // Set tensor name (use index if no name)
        output_tensors->names[i] = strdup(("output_" + std::to_string(i)).c_str());
        if (!output_tensors->names[i]) {
            log_error(logger, "Failed to allocate name for tensor %zu", i);
            deep_free_tensors_struct(output_tensors);
            return NULL;
        }
        
        // Set data type
        output_tensors->data_types[i] = mapDataTypeToTensorDataType(dtype);
        
        // Set rank and shape
        output_tensors->ranks[i] = shape.size();
        output_tensors->shapes[i] = (size_t*)malloc(shape.size() * sizeof(size_t));
        if (!output_tensors->shapes[i]) {
            log_error(logger, "Failed to allocate shape for tensor %zu", i);
            deep_free_tensors_struct(output_tensors);
            return NULL;
        }
        
        for (size_t j = 0; j < shape.size(); j++) {
            output_tensors->shapes[i][j] = shape[j];
        }
        
        // Calculate data size and copy data
        size_t data_size = output->elem_size();
        for (size_t d = 0; d < output_tensors->ranks[i]; d++) {
            data_size *= output_tensors->shapes[i][d];
        }
        
        output_tensors->data[i] = malloc(data_size);
        if (!output_tensors->data[i]) {
            log_error(logger, "Failed to allocate data for tensor %zu", i);
            deep_free_tensors_struct(output_tensors);
            return NULL;
        }
        
        memcpy(output_tensors->data[i], data, data_size);
    }
    
    return output_tensors;
}

static tensor_data_type mapDataTypeToTensorDataType(dxrt::DataType dtype) {
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
        return DATA_TYPE_FLOAT;  // Default case
    }
}

static int runtime_inference_execution(tensors_struct *input_tensors) {
    
    uint8_t *aligned_buffer = nullptr;
    
    try {
        if (input_tensors->num_tensors != 1) {
            throw std::invalid_argument(
                "Input tensors size must be 1 (not support multi-inputs).");
        }
        
        if (input_tensors->data_types[0] != DATA_TYPE_UINT8) {
            throw std::invalid_argument("Input tensors must be uint8 format.");
        }

        int batch_size = input_tensors->shapes[0][0];
        int align_factor = (64 - ((input_tensors->shapes[0][2] * input_tensors->shapes[0][3]) % 64)) % 64;
        size_t aligned_element_size = input_tensors->shapes[0][1] * (input_tensors->shapes[0][2] * input_tensors->shapes[0][3] + align_factor);

        if (batch_size > 1) {
            log_warning(logger, "Batch size > 1 detected (%d), processing only first batch", batch_size);
        }

        aligned_buffer = (uint8_t *)malloc(aligned_element_size);
        if (!aligned_buffer) {
            throw std::runtime_error("Failed to allocate aligned buffer");
        }
        
        uint8_t *input = static_cast<uint8_t *>(input_tensors->data[0]);
        
        // Align and copy first batch data
        for (size_t y = 0; y < input_tensors->shapes[0][1]; y++) {
            memcpy(aligned_buffer + y * (input_tensors->shapes[0][2] * input_tensors->shapes[0][3] + align_factor),
                   input + y * input_tensors->shapes[0][2] * input_tensors->shapes[0][3],
                   input_tensors->shapes[0][2] * input_tensors->shapes[0][3]);
        }

        int job_id = inference_engine->RunAsync(aligned_buffer);
        
        // Store job_id and aligned_buffer in concurrent queue
        JobData job_data;
        job_data.job_id = job_id;
        job_data.aligned_buffer = aligned_buffer;
        job_data_queue.push(job_data);
        
        log_debug(logger, "Pushed job_id: %d", job_id);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in runtime_inference_execution: " << e.what() << std::endl;
        log_error(logger, "Inference execution failed: %s", e.what());
        
        if (aligned_buffer) {
            free(aligned_buffer);
        }
        
        return 1;
    }
}

static void *inference_loop(void *arg) {
    (void)arg; // Suppress unused parameter warning
    
    log_info(logger, "Starting the inference thread");
    int exit_code = 0;
    
    while (1) {
        tensors_struct *input_tensors = dequeue(input_queue, 200);
        if (input_tensors != NULL) {
            // Check for sentinel (num_tensors = 0 indicates sentinel)
            if (input_tensors->num_tensors == 0) {
                log_info(logger, "Received sentinel, stopping inference thread");
                deep_free_tensors_struct(input_tensors);
                break;
            }
            
            log_debug(logger, "Running inference");
            exit_code = runtime_inference_execution(input_tensors);
            
            deep_free_tensors_struct(input_tensors);
            
            if (exit_code != 0) {
                log_error(logger, "Inference execution failed");
            }
        } else {
            log_debug(logger, "input_tensors is NULL - waiting for input");
        }

        if (stop_thread.load()) {
            log_info(logger, "Stopping inference thread");
            break;
        }
    }
    return NULL;
}

static void *wait_loop(void *arg) {
    (void)arg; // Suppress unused parameter warning
    
    log_info(logger, "Starting the wait thread");
    
    while (1) {
        // Pop item from queue (will wait if empty)
        JobData job_data = job_data_queue.pop();
        
        // Check for sentinel (job_id = -1 indicates sentinel)
        if (stop_thread.load() || job_data.job_id == -1) {
            log_info(logger, "Stopping wait thread");
            break;
        }
        
        log_debug(logger, "Waiting for job_id: %d", job_data.job_id);
        
        try {
            // Wait for the inference to complete
            std::vector<std::shared_ptr<dxrt::Tensor>> outputs = inference_engine->Wait(job_data.job_id);

            std::cout << "[Inference] Received output tensors for job_id : " << job_data.job_id << std::endl;
            
            // Free aligned_buffer after Wait() completes
            if (job_data.aligned_buffer) {
                free(job_data.aligned_buffer);
                job_data.aligned_buffer = nullptr;
            }
            
            // Create output tensors using helper function
            tensors_struct *output_tensors = create_output_tensors_from_dxrt(outputs);
            if (output_tensors == NULL) {
                log_error(logger, "Failed to create output tensors");
                continue;
            }
            
            // Enqueue output tensors
            if (enqueue(output_queue, output_tensors) != 0) {
                log_error(logger, "Failed to enqueue output tensors");
                deep_free_tensors_struct(output_tensors);
            }
            
        } catch (const std::exception& e) {
            log_error(logger, "Error in wait loop: %s", e.what());
        }
    }
    return NULL;
}

int runtime_initialization() {
    srand(time(NULL));
    
    // Initialize logger with better error handling
    logger = create_logger(runtime_name(), "runtime.log", log_level, LOG_INFO);
    if (logger == NULL) {
        printf("Warning: Failed to create logger, continuing without file logging\n");
        // Continue without logger - many functions now handle NULL logger safely
    } else {
        log_info(logger, "Initializing the runtime environment");
    }

    // Initialize queues
    input_queue = new_queue(QUEUE_CAPACITY);
    output_queue = new_queue(QUEUE_CAPACITY);

    if (!input_queue || !output_queue) {
        if (logger != NULL) {
            log_error(logger, "Failed to create queues");
        } else {
            printf("Error: Failed to create queues\n");
        }
        return 1;
    }

    return 0;
}

int runtime_initialization_with_args(int length, const char **keys, const void **values) {
    (void)values; // Suppress unused parameter warning
    
    int ret = runtime_initialization();
    if (ret != 0) {
        return ret;
    }

    log_info(logger, "Runtime initialized with arguments");
    for (int i = 0; i < length; i++) {
        log_debug(logger, "Using Key: %s", keys[i]);
    }

    return 0;
}

int runtime_model_loading(const char *file_path) {
    if (access(file_path, F_OK) != 0) {
        log_error(logger, "Model file does not exist: %s", file_path);
        return 2;
    }

    log_info(logger, "Loading model from: %s", file_path);

    try {
        inference_engine = new dxrt::InferenceEngine(std::string(file_path));
        // No callback registration needed - we'll use Wait() instead
        
        stop_thread.store(0);

        if (pthread_create(&inference_thread, NULL, inference_loop, NULL) != 0) {
            log_error(logger, "Failed to create inference thread");
            return 1;
        }

        if (pthread_create(&wait_thread, NULL, wait_loop, NULL) != 0) {
            log_error(logger, "Failed to create wait thread");
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        log_error(logger, "Failed to load model: %s", e.what());
        return 1;
    }
}

int send_input(tensors_struct *input_tensors) {
    log_debug(logger, "Sending input tensors to the queue");
    if (enqueue(input_queue, input_tensors) != 0) {
        log_error(logger, "Failed to enqueue input tensors");
        return 1;
    }
    return 0;
}

int receive_output(tensors_struct **output_tensors) {
    log_debug(logger, "Receiving output tensors from the queue");
    tensors_struct *output = dequeue(output_queue, 200);
    if (output == NULL) {
        log_debug(logger, "No output tensors available");
        *output_tensors = NULL;
        return 1;
    }

    *output_tensors = output;
    return 0;
}

int runtime_destruction() {
    log_info(logger, "Destroying the runtime environment");

    // Set stop flag
    stop_thread.store(1);

    // Push dummy job to unblock wait_loop
    JobData sentinel = {-1, nullptr};
    job_data_queue.push(sentinel);

    // Push dummy input tensors to unblock inference_loop
    tensors_struct *dummy_input = (tensors_struct *)calloc(1, sizeof(tensors_struct));
    if (dummy_input) {
        dummy_input->num_tensors = 0; // Mark as sentinel
        if (enqueue(input_queue, dummy_input) != 0) {
            // If enqueue fails, free the dummy input immediately
            free(dummy_input);
            log_warning(logger, "Failed to enqueue dummy input, freed immediately");
        }
    }

    // Give threads time to process the sentinel and stop naturally
    sleep(1);

    // Join inference thread (should terminate naturally with sentinel)
    int join_result = pthread_join(inference_thread, NULL);
    if (join_result != 0) {
        log_error(logger, "Failed to join inference thread: %d", join_result);
    } else {
        log_info(logger, "Inference thread terminated successfully");
    }
    
    // Join wait thread (should terminate naturally with sentinel)
    join_result = pthread_join(wait_thread, NULL);
    if (join_result != 0) {
        log_error(logger, "Failed to join wait thread: %d", join_result);
    } else {
        log_info(logger, "Wait thread terminated successfully");
    }

    // Clean up inference engine
    if (inference_engine != nullptr) {
        delete inference_engine;
        inference_engine = nullptr;
        log_info(logger, "Inference engine destroyed");
    }

    // Clean up queues
    if (input_queue != NULL) {
        shutdown_queue(input_queue);
        free_queue(input_queue);
        input_queue = NULL;
        log_info(logger, "Input queue destroyed");
    }
    
    if (output_queue != NULL) {
        shutdown_queue(output_queue);
        free_queue(output_queue);
        output_queue = NULL;
        log_info(logger, "Output queue destroyed");
    }

    // Clean up error message
    if (_error_message != NULL) {
        free(_error_message);
        _error_message = NULL;
    }

    // Log final message before closing logger
    log_info(logger, "Runtime destruction completed");

    // Clean up logger (moved to the end)
    if (logger != NULL) {
        close_logger(logger);
        logger = NULL;
    }

    return 0;
}

const char *runtime_error_message() { 
    return _error_message ? _error_message : ""; 
}

const char *runtime_version() { 
    return "1.0.0"; 
}

const char *runtime_name() { 
    return "DEEPX"; 
}
