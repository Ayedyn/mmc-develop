#include <cstdlib>
#ifdef _OPENMP
    #include <omp.h>
#endif

#include "mmc_optix_host.h"
#include "mmc_optix_utils.h"
#include "mmc_cuda_query_gpu.h"
#include "mcx_context.h"

/************************************************************************** In
this unit, we first launch a master thread and initialize the necessary data
structures.This include the command line options(cfg), tetrahedral mesh(mesh)
and the ray tracer precomputed data (tracer).
******************************************************************************/
void mmc_run_optix(mcconfig* cfg, tetmesh* mesh, raytracer* tracer, 
    void (*progressfun)(float, void*), void* handle) {
    GPUInfo* gpuinfo = NULL;      /** gpuinfo: structure to store GPU information */
    unsigned int activedev = 0;   /** activedev: count of total active GPUs to be used */

    if (!(activedev = mcx_list_cu_gpu(cfg, &gpuinfo))) {
        mcx_error(-1, "No GPU device found\n", __FILE__, __LINE__);
    }

#ifdef _OPENMP
    /**
        Now we are ready to launch one thread for each involked GPU to run the simulation
     */
    omp_set_num_threads(activedev);
    #pragma omp parallel
    {
#endif

#ifdef OPTIX_IMMC
        size_t num_spheres = sizeof(cfg->spheres) / sizeof(cfg->spheres[0]);
        size_t num_capsule_centers = sizeof(cfg->capsulecenters) / sizeof(cfg->capsulecenters[0]);
        const bool has_implicits = num_spheres>0 || num_capsule_centers>1; 
        if(has_implicits){ // at least one implicit sphere or capsule is simulated
            mcx::McxContext ctx = mcx::McxContext();
            ctx.simulate(mesh, cfg); 
        }
        else{
#endif 
        /**
            This line runs the main MCX simulation for each GPU inside each thread
         */
        optix_run_simulation(cfg, mesh, tracer, gpuinfo, progressfun, handle);
#ifdef OPTIX_IMMC
        }
#endif

#ifdef _OPENMP
    }
#endif
}
