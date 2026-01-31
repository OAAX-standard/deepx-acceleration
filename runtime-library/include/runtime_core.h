#ifndef RUNTIME_CORE_H
#define RUNTIME_CORE_H

#if defined(_WIN32) || defined(_WIN64)
#  ifdef RUNTIME_LIBRARY_EXPORTS
#    define RUNTIME_API __declspec(dllexport)
#  else
#    define RUNTIME_API __declspec(dllimport)
#  endif
#else
#  define RUNTIME_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

extern "C" {
#include "tensors_struct.h"
}

/**
 * @brief This function is called only once to initialize the ru    ntime environment.
 *
 * @return 0 if the initialization is successful, and non-zero otherwise.
 */
RUNTIME_API int runtime_initialization();

/**
 * @brief This function is called to initialize the runtime environment with arguments.
 * 
 * @note If this function is used to initialize the runtime, the runtime_initialization() function should not be called.
 * 
 * @note If an unknown key is passed, the runtime should ignore it.
 *
 * @param length The number of arguments.
 * @param keys The keys of the arguments.
 * @param values The values of the arguments.
 * @return 0 if the initialization is successful, and non-zero otherwise.
 */
RUNTIME_API int runtime_initialization_with_args(int length, const char **keys, const void **values);

/**
 * @brief This function is called to load the model from the file path.
 *
 * @param file_path The path to the model file.
 * @return 0 if the model is loaded successfully, and non-zero otherwise.
 */
RUNTIME_API int runtime_model_loading(const char *file_path);

/**
 * @brief This function is called to store the input tensors to be processed by the runtime when it's ready.
 * 
 * @note This function copies the reference of the input tensors, not the tensors themselves. 
 * The runtime will free the memory of the input tensors after its processed.
 * 
 * @warning If this function returns a non-zero value, the caller is expected to free the memory of the input tensors.
 *
 * @param tensors The input tensors for the inference processing. 
 * 
 * @return 0 if the input tensors are stored successfully, and non-zero otherwise.
 */
RUNTIME_API int send_input(tensors_struct *input_tensors);

/**
 * @brief This function is called to retrieve any available output tensors after the inference process is done.
 *
 * @note The caller is responsible for managing the memory of the output tensors.
 * 
 * @param output_tensors The output tensors of the inference process.
 * 
 * @return 0 if an output is available and returned, and non-zero otherwise.
 */
RUNTIME_API int receive_output(tensors_struct **output_tensors);

/**
 * @brief This function is called to destroy the runtime environment after the inference process is stopped.

 * @return 0 if the finalization is successful, and non-zero otherwise.
 */
RUNTIME_API int runtime_destruction();

/**
 * @brief This function is called to get the error message in case of a runtime error.

 * @return The error message in a human-readable format. This should be allocated by the shared library, and proper deallocation should be handled by the library.
 */
RUNTIME_API const char *runtime_error_message();

/**
 * @brief This function is called to get the version of the shared library.

 * @return The version of the shared library. This should be allocated by the shared library, and proper deallocation should be handled by the library.
 */
RUNTIME_API const char *runtime_version();

/**
 * @brief This function is called to get the name of the shared library.

 * @return The name of the shared library. This should be allocated by the shared library, and proper deallocation should be handled by the library.
 */
RUNTIME_API const char *runtime_name();

#ifdef __cplusplus
}
#endif

#endif // RUNTIME_CORE_H  