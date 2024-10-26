#include "mcx_utils.h"
#include "mcx_core.h"
#include "mcx_const.h"
#include <string.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define CUDA_ASSERT(a)      mcx_cu_assess((a),__FILE__,__LINE__) //< macro to report CUDA errors

/**
   assert cuda memory allocation result
*/
void mcx_cu_assess(cudaError_t cuerr, const char* file, const int linenum) {
    if (cuerr != cudaSuccess) {
        CUDA_ASSERT(cudaDeviceReset());
        mcx_error(-(int)cuerr, (char*)cudaGetErrorString(cuerr), file, linenum);
    }
}

/**
 * @brief Utility function to calculate the GPU stream processors (cores) per SM
 *
 * Obtain GPU core number per MP, this replaces
 * ConvertSMVer2Cores() in libcudautils to avoid
 * extra dependency.
 *
 * @param[in] v1: the major version of an NVIDIA GPU
 * @param[in] v2: the minor version of an NVIDIA GPU
 */

int mcx_corecount(int v1, int v2) {
    int v = v1 * 10 + v2;

    if (v < 20) {
        return 8;
    } else if (v < 21) {
        return 32;
    } else if (v < 30) {
        return 48;
    } else if (v < 50) {
        return 192;
    } else if (v < 60 || v == 61) {
        return 128;
    } else {
        return 64;
    }
}


/**
 * @brief Utility function to calculate the maximum blocks per SM
 *
 *
 * @param[in] v1: the major version of an NVIDIA GPU
 * @param[in] v2: the minor version of an NVIDIA GPU
 */

int mcx_smxblock(int v1, int v2) {
    int v = v1 * 10 + v2;

    if (v < 30) {
        return 8;
    } else if (v < 50) {
        return 16;
    } else {
        return 32;
    }
}

/**
 * @brief Utility function to calculate the maximum blocks per SM
 *
 *
 * @param[in] v1: the major version of an NVIDIA GPU
 * @param[in] v2: the minor version of an NVIDIA GPU
 */

int mcx_threadmultiplier(int v1, int v2) {
    int v = v1 * 10 + v2;

    if (v <= 75) {
        return 1;
    } else {
        return 2;
    }
}

/**
 * @brief Utility function to query GPU info and set active GPU
 *
 * This function query and list all available GPUs on the system and print
 * their parameters. This is used when -L or -I is used.
 *
 * @param[in,out] cfg: the simulation configuration structure
 * @param[out] info: the GPU information structure
 */

int mcx_list_gpu(Config* cfg, GPUInfo** info) {

#if __DEVICE_EMULATION__
    return 1;
#else
    int dev;
    int deviceCount, activedev = 0;

    cudaError_t cuerr = cudaGetDeviceCount(&deviceCount);

    if (cuerr != cudaSuccess) {
        if (cuerr == (cudaError_t)30) {
            mcx_error(-(int)cuerr, "A CUDA-capable GPU is not found or configured", __FILE__, __LINE__);
        }

        CUDA_ASSERT(cuerr);
    }

    if (deviceCount == 0) {
        MCX_FPRINTF(stderr, S_RED "ERROR: No CUDA-capable GPU device found\n" S_RESET);
        return 0;
    }

    *info = (GPUInfo*)calloc(deviceCount, sizeof(GPUInfo));

    if (cfg->gpuid && cfg->gpuid > deviceCount) {
        MCX_FPRINTF(stderr, S_RED "ERROR: Specified GPU ID is out of range\n" S_RESET);
        return 0;
    }

    // scan from the first device
    for (dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp dp;
        CUDA_ASSERT(cudaGetDeviceProperties(&dp, dev));

        if (cfg->isgpuinfo == 3) {
            activedev++;
        } else if (cfg->deviceid[dev] == '1') {
            cfg->deviceid[dev] = '\0';
            cfg->deviceid[activedev] = dev + 1;
            activedev++;
        }

        strncpy((*info)[dev].name, dp.name, MAX_SESSION_LENGTH);
        (*info)[dev].id = dev + 1;
        (*info)[dev].devcount = deviceCount;
        (*info)[dev].major = dp.major;
        (*info)[dev].minor = dp.minor;
        (*info)[dev].globalmem = dp.totalGlobalMem;
        (*info)[dev].constmem = dp.totalConstMem;
        (*info)[dev].sharedmem = dp.sharedMemPerBlock;
        (*info)[dev].regcount = dp.regsPerBlock;
        (*info)[dev].clock = dp.clockRate;
        (*info)[dev].sm = dp.multiProcessorCount;
        (*info)[dev].core = dp.multiProcessorCount * mcx_corecount(dp.major, dp.minor);
        (*info)[dev].maxmpthread = dp.maxThreadsPerMultiProcessor;
        (*info)[dev].maxgate = cfg->maxgate;
        (*info)[dev].autoblock = MAX((*info)[dev].maxmpthread / mcx_smxblock(dp.major, dp.minor), 64);

        if ((*info)[dev].autoblock == 0) {
            MCX_FPRINTF(stderr, S_RED "WARNING: maxThreadsPerMultiProcessor can not be detected\n" S_RESET);
            (*info)[dev].autoblock = 64;
        }

        (*info)[dev].autothread = (*info)[dev].autoblock * mcx_smxblock(dp.major, dp.minor) * (*info)[dev].sm * mcx_threadmultiplier(dp.major, dp.minor);

        if (strncmp(dp.name, "Device Emulation", 16)) {
            if (cfg->isgpuinfo) {
                MCX_FPRINTF(stdout, S_BLUE"=============================   GPU Information  ================================\n" S_RESET);
                MCX_FPRINTF(stdout, "Device %d of %d:\t\t%s\n", (*info)[dev].id, (*info)[dev].devcount, (*info)[dev].name);
                MCX_FPRINTF(stdout, "Compute Capability:\t%u.%u\n", (*info)[dev].major, (*info)[dev].minor);
                MCX_FPRINTF(stdout, "Global Memory:\t\t%.0f B\nConstant Memory:\t%.0f B\n"
                            "Shared Memory:\t\t%.0f B\nRegisters:\t\t%u\nClock Speed:\t\t%.2f GHz\n",
                            (double)(*info)[dev].globalmem, (double)(*info)[dev].constmem,
                            (double)(*info)[dev].sharedmem, (unsigned int)(*info)[dev].regcount, (*info)[dev].clock * 1e-6f);
#if CUDART_VERSION >= 2000
                MCX_FPRINTF(stdout, "Number of SMs:\t\t%u\nNumber of Cores:\t%u\n",
                            (*info)[dev].sm, (*info)[dev].core);
#endif
                MCX_FPRINTF(stdout, "Auto-thread:\t\t%d\n", (*info)[dev].autothread);
                MCX_FPRINTF(stdout, "Auto-block:\t\t%d\n", (*info)[dev].autoblock);
            }
        }
    }

    if (cfg->isgpuinfo == 2 && cfg->parentid == mpStandalone) { //list GPU info only
        exit(0);
    }

    if (activedev < MAX_DEVICE) {
        cfg->deviceid[activedev] = '\0';
    }

    return activedev;
#endif
}
