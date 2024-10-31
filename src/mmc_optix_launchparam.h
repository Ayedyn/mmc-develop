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
    unsigned int mediumid0;             /**< initial medium type */

    uint isreflect;
    int outputtype;

    int threadphoton;
    int oddphoton;

    Medium medium[MAX_PROP_OPTIX];

    // Fields for IMMC optix (Consoliate in the future):
    uint3 dataSize;
    CUdeviceptr surfaceBoundaries; // device pointer to a vector of surface boundaries 
    // for IMMC primitives 
    CUdeviceptr curveData; // device pointer to a vector of curve geometry
    float duration;
    int timeSteps;
    unsigned int num_inside_prims;
    float WIDTH_ADJ;

} MMCParam;

struct __attribute__((aligned(16))) TriangleMeshSBTData {
    float4 *fnorm; /**< x,y,z: face normal; w: neighboring medium type */
    OptixTraversableHandle *nbgashandle;
};

using PrimitiveSurfaceData = TriangleMeshSBTData;

//TODO: use proper header guards instead of inlining this
inline int print_MMCParam(MMCParam param){
    printf("\nStarting OptixTraversableHandle is: %llx", param.gashandle0);
    printf("\nSeedbuffer device pointer is: %llx", param.seedbuffer);
    printf("\nSrcpos is: %f, %f, %f", param.srcpos.x, param.srcpos.y, param.srcpos.z);
    printf("\nSrcdir is: %f, %f, %f", param.srcdir.x, param.srcdir.y, param.srcdir.z);
    printf("\nnmin is: %f, %f, %f", param.nmin.x, param.nmin.y, param.nmin.z);
    printf("\nnmax is: %f, %f, %f", param.nmax.x, param.nmax.y, param.nmax.z); 
    printf("\ncrop0 is: %d, %d, %d, %d", param.crop0.x, param.crop0.y, param.crop0.z, param.crop0.w);
    printf("\ndstep is: %f", param.dstep);
    printf("\nmaxgate: %d", param.maxgate);
    printf("\nisreflect: %d", param.isreflect);
    printf("\noutputtype: %d", param.outputtype);
    printf("\nthreadphoton is: %d", param.threadphoton);
    printf("\noddphoton is: %d\n", param.oddphoton);
/*
    int numMedia = sizeof(param.medium) / sizeof(param.medium[0]);
    for(size_t i=0; i<numMedia; ++i){
        printf("\nmedium #%d, mua=%f, musp=%f, g=%f, n=%f", i, param.medium[i].mua, param.medium[i].mus, param.medium[i].g, param.medium[i].n);
    }*/
}

#endif
