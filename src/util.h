#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <optix.h>

#define CUDA_CHECK( call )                                                     \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if(error != cudaSuccess)                                               \
        {                                                                      \
            throw std::runtime_error("A CUDA runtime error occurred:\n'" + std::string(cudaGetErrorString(error)) + "'\n at (" + __FILE__ + ":" + std::to_string(__LINE__) + ")\n"); \
        } \
    } while( 0 )

#define OPTIX_CHECK( call )                                                    \
    do                                                                         \
    {                                                                          \
        OptixResult error = call;                                              \
        if(error != OPTIX_SUCCESS)                                               \
        {                                                                      \
            throw std::runtime_error("An Optix runtime error occurred:\n'" + std::string(optixGetErrorString(error)) + "'\n at (" + __FILE__ + ":" + std::to_string(__LINE__) + ")\n"); \
        } \
    } while( 0 )

#define ARRAY_LENGTH(arr) (sizeof(arr) / sizeof(arr[0]))

namespace mcx {
}