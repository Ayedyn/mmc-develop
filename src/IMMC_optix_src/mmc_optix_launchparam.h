#ifndef _MMC_OPTIX_LAUNCHPARAM_H
#define _MMC_OPTIX_LAUNCHPARAM_H

#include <vector_types.h>

#define MAX_PROP_OPTIX 4000              /*maximum property number*/

/**
 * @brief struct for medium optical properties
 */
typedef struct __attribute__((aligned(16))) MCX_medium {
    float mua;                     /**<absorption coeff in 1/mm unit*/
    float mus;                     /**<scattering coeff in 1/mm unit*/
    float g;                       /**<anisotropy*/
    float n;                       /**<refractive index*/
} Medium;

/**
 * @brief struct for simulation configuration paramaters
 */
typedef struct __attribute__((aligned(16))) MMC_Parameter {
    OptixTraversableHandle gashandle0;

    CUdeviceptr seedbuffer;             /**< rng seed for each thread */
    CUdeviceptr outputbuffer;

    float3 srcpos;
    float3 srcdir;
    float3 nmin; // minimum corner coordinates of dual grid
    float3 nmax; // maximum corner coordinates of dual grid
    uint4 crop0;
    float dstep;
    float tstart, tend;
    float Rtstep;
    int maxgate;
    uint32_t mediumid0;             /**< initial medium type */

    uint isreflect;
    int outputtype;

    int threadphoton;
    int oddphoton;

    Medium medium[MAX_PROP_OPTIX];

    // Fields for IMMC optix (Consoliate in the future):
    uint3 dataSize;
    CUdeviceptr surfaceBoundaries; // device pointer to a vector of PrimitiveSurfaceData 
    // for IMMC primitives 
    CUdeviceptr curveData; // device pointer to a vector of curve geometry
    float duration;
    int timeSteps;
    uint32_t num_inside_prims;
    float WIDTH_ADJ;
    // Fields for IMMC optix ^^^^^^^^^^^^^^^^^^^^^^    

} MMCParam;

struct __attribute__((aligned(16))) TriangleMeshSBTData {
    float4 *fnorm; /**< x,y,z: face normal; w: neighboring medium type */
    OptixTraversableHandle *nbgashandle;
};

// Struct for IMMC with single fnorm and handle instead of pointers
// with triangles, capsules, and spheres:
// for a sphere's normal,
// record the center position, for a capsule use 0,0,0 to denote capsule
struct __attribute__((aligned(16))) PrimitiveSurfaceData {
    float4 fnorm;
    OptixTraversableHandle nbgashandle;    
};

#endif
