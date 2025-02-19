#include "mcx_context.h"
#include <cuda_runtime.h>
#include "CUDABuffer.h"
#include <optix.h>
//    #include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <stdio.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
// added this for memcpy
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
//#include <vector_functions.h>
//#include <vector_types.h>   

#include "implicit_geometries.h"
#include "incbin.h"
#include "shader_pipeline.h"
#include "util.h"
#include "device_buffer.h"
#include "mmc_optix_launchparam.h"
#include "mmc_utils.h"
#include "mmc_mesh.h"

// this includes lots of optix features
#ifndef NDEBUG
INCTXT(mmcShaderPtx, mmcShaderPtxSize, "/built/mmc_optix_core.ptx")
#else
INCTXT(mmcShaderPtx, mmcShaderPtxSize, "/built/mmc_optix_core.ptx");
#endif

#define SPHERE_MATERIAL 2

namespace mcx {

// struct for surface mesh of each medium
typedef struct surfaceMesh {
    std::vector<uint3> face;
    std::vector<float3> norm;
    std::vector<unsigned int> nbtype;
    std::vector<ImplicitCurve> curves;
    std::vector<ImplicitSphere> spheres;
} surfmesh;

std::vector<float3> convert_mmc_mesh_nodevector(tetmesh* mesh){
   std::vector<float3> result; 
    
   for(int i=0; i<mesh->nn; ++i){
       result.push_back(make_float3(mesh->node[i].x, mesh->node[i].y, mesh->node[i].z));
   }
  return result; 
}

// converts from c-style arrays to capsules
std::vector<mcx::ImplicitCurve> mcconfig_to_capsules(const mcconfig* cfg){
    float3* capsuleArray = cfg->capsulecenters;
    float* capsuleWidths = cfg->capsulewidths;
    
    size_t arraySize = sizeof(capsuleArray) / sizeof(capsuleArray[0]);
    if(arraySize%2 != 0){
        throw("Error: Capsule vertices count is uneven, two vertices needed per capsule.\n");
    }

    std::vector<mcx::ImplicitCurve> curveVector;
    curveVector.reserve(arraySize); 
    
    for (size_t i = 0; i < arraySize; i=i+2) {
        const float3& f1 = capsuleArray[i];
        const float3& f2 = capsuleArray[i+1];
        const float& w = capsuleWidths[i/2];

        mcx::ImplicitCurve curve;
        curve.vertex1 = f1;
        curve.vertex2 = f2; 
        curve.width = w;
        curveVector.push_back(curve);
    }

    return curveVector;
}

// converts from c-style arrays to spheres
std::vector<mcx::ImplicitSphere> mcconfig_to_spheres(const mcconfig* cfg){
    float4* sphereArray = cfg->spheres;  
    size_t arraySize = sizeof(sphereArray) / sizeof(sphereArray[0]);
    
    std::vector<mcx::ImplicitSphere> sphereVector;
    sphereVector.reserve(arraySize);

    for (size_t i = 0; i < arraySize; ++i) {
        const float4& f = sphereArray[i];
        ImplicitSphere sphere;
        sphere.position = make_float3(f.x,f.y,f.z);
        sphere.radius = f.w;
        sphereVector.push_back(sphere);
    }

    return sphereVector;
}

/**
 * @brief extract surface mesh for each medium
 */
void prepareSurfMeshArray(tetmesh *tmesh, surfmesh *smesh, const mcconfig *cfg) {

    const int ifaceorder[] = {3, 0, 2, 1};
    const int out[4][3] = {{0, 3, 1}, {3, 2, 1}, {0, 2, 3}, {0, 1, 2}};
    int *fnb = (int*)calloc(tmesh->ne * tmesh->elemlen, sizeof(int));
    memcpy(fnb, tmesh->facenb, (tmesh->ne * tmesh->elemlen) * sizeof(int));

    float3 v0, v1, v2, vec01, vec02, vnorm; 
    for (int i = 0; i < tmesh->ne; ++i) {
        // iterate over each tetrahedra
        unsigned int currmedid = tmesh->type[i];
        for(int j = 0; j < tmesh->elemlen; ++j){
            // iterate over each triangle
            int nexteid = fnb[(i * tmesh->elemlen) + j];
            if (nexteid == INT_MIN) continue;
            unsigned int nextmedid = ((nexteid < 0) ? 0 : tmesh->type[nexteid - 1]);
            if(currmedid != nextmedid) {
                // face nodes
                unsigned int n0 = tmesh->elem[(i * tmesh->elemlen) + out[ifaceorder[j]][0]] - 1;
                unsigned int n1 = tmesh->elem[(i * tmesh->elemlen) + out[ifaceorder[j]][1]] - 1;
                unsigned int n2 = tmesh->elem[(i * tmesh->elemlen) + out[ifaceorder[j]][2]] - 1;

                // face vertex indices
                smesh[currmedid].face.push_back(make_uint3(n0, n1, n2));
                smesh[nextmedid].face.push_back(make_uint3(n0, n2, n1));

                // outward-pointing face norm
                v0 = *(float3*)&tmesh->fnode[n0];
                v1 = *(float3*)&tmesh->fnode[n1];
                v2 = *(float3*)&tmesh->fnode[n2];
                vec_diff((MMCfloat3*)&v0, (MMCfloat3*)&v1, (MMCfloat3*)&vec01);
                vec_diff((MMCfloat3*)&v0, (MMCfloat3*)&v2, (MMCfloat3*)&vec02);
                vec_cross((MMCfloat3*)&vec01, (MMCfloat3*)&vec02, (MMCfloat3*)&vnorm);
                float mag = 1.0f / sqrtf(vec_dot((MMCfloat3*)&vnorm, (MMCfloat3*)&vnorm));
                vec_mult((MMCfloat3*)&vnorm, mag, (MMCfloat3*)&vnorm);
                smesh[currmedid].norm.push_back(vnorm);
                smesh[nextmedid].norm.push_back(-vnorm);

                // neighbour medium types
                smesh[currmedid].nbtype.push_back(nextmedid);
                smesh[nextmedid].nbtype.push_back(currmedid);

                fnb[(i * tmesh->elemlen) + j] = INT_MIN;
                if(nexteid > 0){
                    for(int k = 0; k < tmesh->elemlen; ++k){
                        if(fnb[((nexteid - 1) * tmesh->elemlen) + k] == i + 1) {
                            fnb[((nexteid - 1) * tmesh->elemlen) + k] = INT_MIN;
                            break;
                        }
                    }
                }
            }                                                                                                                                                                                                    
        }
    }

    // iterate over each surface mesh
    for(int j=0; j<=tmesh->prop; ++j){ 
        // Add spheres and capsules to smesh from cfg
        for (int i = 0; i<cfg->nspheres; i++){
            ImplicitSphere sphere = {make_float3(cfg->spheres[i].x, cfg->spheres[i].y, cfg->spheres[i].z), cfg->spheres[i].w};       
            smesh[j].spheres.push_back(sphere); 
        }

        // Add capsules to smesh from cfg
        for (int i = 0; i<cfg->ncapsules; i=i+2){
            ImplicitCurve capsule = {make_float3(cfg->capsulecenters[i].x, cfg->capsulecenters[i].y, cfg->capsulecenters[i].z), 
                                       make_float3(cfg->capsulecenters[i+1].x, cfg->capsulecenters[i+1].y, cfg->capsulecenters[i+1].z), 
                                       cfg->capsulewidths[i/2]};
            smesh[j].curves.push_back(capsule);
        }
    }
}

bool insideCurve(float3 position, std::vector<ImplicitCurve> curves) {
    for (ImplicitCurve curve : curves) {
        // vector along the line segment of the curve
        float3 vector_alongcurve =
            make_float3(curve.vertex2.x - curve.vertex1.x,
                curve.vertex2.y - curve.vertex1.y,                                                  
                curve.vertex2.z - curve.vertex1.z);
        // vector from the first vertex to the position
        float3 vector_toposition = make_float3(
            position.x - curve.vertex1.x, position.y - curve.vertex1.y,
            position.z - curve.vertex1.z);
        // computes the scalar projection of vector to position onto the
        // line segment
        float scalarproj = dot(vector_alongcurve, vector_toposition) /
                   dot(vector_alongcurve, vector_alongcurve);
        // computes the projection of the vector to position onto the
        // line segment
        float3 projection =
            make_float3(vector_alongcurve.x * scalarproj,
                vector_alongcurve.y * scalarproj,
                vector_alongcurve.z * scalarproj);
        // gets the vector for the shortest distance from line segment
        float3 distance_vector =
            make_float3(vector_toposition.x - projection.x,
                vector_toposition.y - projection.y,
                vector_toposition.z - projection.z);
        // get the magnitude of that vector
        float distance = sqrt(distance_vector.x * distance_vector.x +
                      distance_vector.y * distance_vector.y +
                      distance_vector.z * distance_vector.z);
        if (distance < curve.width) {
            return true;
        }
    }
    return false;
}

// checks if a given point is inside any spheres in the tetrahedral mesh
bool insideSphere(float3 position, std::vector<ImplicitSphere> spheres) {
    for (ImplicitSphere sphere : spheres) {
        float3 displacement = position - sphere.position;
        if (dot(displacement, displacement) <= sphere.radius) {
            return true;
        }
    }

    return false;
}

struct MCX_clock {
    std::chrono::system_clock::time_point starttime;
    MCX_clock() : starttime(std::chrono::system_clock::now()) {}
    double elapse() {
        std::chrono::duration<double> elapsetime = (std::chrono::system_clock::now() - starttime);
        return elapsetime.count() * 1000.;
    }
};

// store bits of uint into float for later retrieval from device-code
float storeuintAsFloat(unsigned int myUint) {
    float storedFloat;
    std::memcpy(&storedFloat, &myUint, sizeof(float));  // Bitwise copy uint to float
    return storedFloat;  // Return the stored float
}

// store bits of uint into float for later retrieval from device-code
unsigned int storeFloatAsuint(float myFloat) {
    unsigned int storedUint;
    std::memcpy(&storedUint, &myFloat, sizeof(unsigned int));  // Bitwise copy uint to float
    return storedUint;  // Return the stored float
}

static DeviceByteBuffer createAccelerationStructure(
    OptixDeviceContext ctx, OptixTraversableHandle& handle, int primitiveOffset,
    DeviceBuffer<float3>& vertexBuffer, DeviceBuffer<uint3>& indexBuffer,
    DeviceBuffer<uint32_t>& sbtIndexOffsets) {
	uint32_t triangleInputFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
	OptixBuildInput buildInput = {};
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
	buildInput.triangleArray.numVertices = vertexBuffer.count();
	buildInput.triangleArray.vertexBuffers = &vertexBuffer.handle();
	buildInput.triangleArray.indexFormat =
	    OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	buildInput.triangleArray.indexStrideInBytes = 0;
	buildInput.triangleArray.numIndexTriplets = indexBuffer.count();
	buildInput.triangleArray.indexBuffer = indexBuffer.handle();
	buildInput.triangleArray.flags = &triangleInputFlags;
	buildInput.triangleArray.numSbtRecords = 1;
	buildInput.triangleArray.primitiveIndexOffset = primitiveOffset;

	OptixAccelBuildOptions accelerationOptions = {};
	accelerationOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelerationOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes sizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &accelerationOptions,
						 &buildInput, 1, &sizes));

	DeviceByteBuffer tempBuffer = DeviceByteBuffer(sizes.tempSizeInBytes);
	DeviceByteBuffer outputBuffer =
	    DeviceByteBuffer(sizes.outputSizeInBytes);
	DeviceBuffer<size_t> compactedSize = DeviceBuffer<size_t>(1);

	OptixAccelEmitDesc emitProperty = {};
	emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitProperty.result = compactedSize.handle();

	OPTIX_CHECK(optixAccelBuild(
	    ctx, nullptr, &accelerationOptions, &buildInput, 1,
	    tempBuffer.handle(), sizes.tempSizeInBytes, outputBuffer.handle(),
	    sizes.outputSizeInBytes, &handle, &emitProperty, 1));

#ifndef NDEBUG
	printf("\nBuilt an optix acceleration structure of type: triangles");
#endif

	size_t compactSize;
	compactedSize.read(&compactSize);

	if (compactSize < sizes.outputSizeInBytes) {
		DeviceByteBuffer buffer = DeviceByteBuffer(compactSize);
		OPTIX_CHECK(optixAccelCompact(ctx, nullptr, handle,
					      buffer.handle(), compactSize,
					      &handle));
		return buffer;
	} else {
		return outputBuffer;
	}
}

// Builds a an acceleration structure of linear (cylindrical with spherical
// end-caps) custom curves. Had to make it custom because OptiX does not support
// in-to-out ray tracing vertexBuffer represents a list of the endpoints of each
// curve segment widthBuffer represents a list of the swept radius between each
// vertex Takes returns the AS as a devicebytebuffer and adds the AS to the
// traversable handle passed by reference
static DeviceByteBuffer createCurveAccelStructure(
    OptixDeviceContext ctx, OptixTraversableHandle& handle, int primitiveOffset,
    std::vector<float3>& vertexVector, std::vector<float>& widthVector,
    DeviceBuffer<unsigned int>& indexBuffer,
    DeviceBuffer<uint32_t>& sbtIndexOffsets) {
	OptixBuildInput buildInput = {};
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

	// tells the number of curve segments (pills) to create
	// there are two vertices per curve segment
	if (vertexVector.size() % 2 == 0) {
		buildInput.customPrimitiveArray.numPrimitives =
		    vertexVector.size() / 2;
	} else {
		throw std::runtime_error(
		    "Number of curve vertices is not an even number");
	}

	std::vector<OptixAabb> aabb = std::vector<OptixAabb>();
	// prepare axis aligned bounding boxes
	for (unsigned int i = 0; i < vertexVector.size(); i += 2) {
		// create one per curve primitive,
		//  figure out the maximum/min vertex in each direction
		//  and then calculate the maximum height/width/depth
		//  via adding or subtracting radii
		OptixAabb temp_aabb;
		float width = widthVector[i / 2];
		float3 vertex_one = vertexVector[i];
		float3 vertex_two = vertexVector[i + 1];
		//float epsilon = 1.0/1024;

		temp_aabb.minX = std::min(vertex_one.x, vertex_two.x) - width;//-epsilon;
		temp_aabb.minY = std::min(vertex_one.y, vertex_two.y) - width;//-epsilon;
		temp_aabb.minZ = std::min(vertex_one.z, vertex_two.z) - width;//-epsilon;

		temp_aabb.maxX = std::max(vertex_one.x, vertex_two.x) + width;//+epsilon;
		temp_aabb.maxY = std::max(vertex_one.y, vertex_two.y) + width;//+epsilon;
		temp_aabb.maxZ = std::max(vertex_one.z, vertex_two.z) + width;//+epsilon;
		aabb.push_back(temp_aabb);
	}

	DeviceBuffer<OptixAabb> aabbBuffer =
	    DeviceBuffer<OptixAabb>(aabb.data(), aabb.size());

	buildInput.customPrimitiveArray.aabbBuffers = &aabbBuffer.handle();

	uint32_t aabbInputFlags = OPTIX_GEOMETRY_FLAG_NONE;
	buildInput.customPrimitiveArray.flags = &aabbInputFlags;
	buildInput.customPrimitiveArray.numSbtRecords = 1;
	//buildInput.customPrimitiveArray.sbtIndexOffsetBuffer =
	//    sbtIndexOffsets.handle();
	//buildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes =
	//    sizeof(int32_t);
	buildInput.customPrimitiveArray.primitiveIndexOffset = primitiveOffset;

	// compact and build the acceleration structure

	OptixAccelBuildOptions accelerationOptions = {};
	accelerationOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelerationOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes sizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &accelerationOptions,
						 &buildInput, 1, &sizes));

	DeviceByteBuffer tempBuffer = DeviceByteBuffer(sizes.tempSizeInBytes);
	DeviceByteBuffer outputBuffer =
	    DeviceByteBuffer(sizes.outputSizeInBytes);
	DeviceBuffer<size_t> compactedSize = DeviceBuffer<size_t>(1);

	OptixAccelEmitDesc emitProperty = {};
	emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitProperty.result = compactedSize.handle();

	OPTIX_CHECK(optixAccelBuild(
	    ctx, nullptr, &accelerationOptions, &buildInput, 1,
	    tempBuffer.handle(), sizes.tempSizeInBytes, outputBuffer.handle(),
	    sizes.outputSizeInBytes, &handle, &emitProperty, 1));


#ifndef NDEBUG
	printf("\nBuilt an optix acceleration structure of type: custom capsule");
#endif


	size_t compactSize;
	compactedSize.read(&compactSize);

	if (compactSize < sizes.outputSizeInBytes) {
		DeviceByteBuffer buffer = DeviceByteBuffer(compactSize);
		OPTIX_CHECK(optixAccelCompact(ctx, nullptr, handle,
					      buffer.handle(), compactSize,
					      &handle));
		return buffer;
	} else {
		return outputBuffer;
	}
}

// Creates a sphere acceleration structure as a DeviceByteBuffer, adds it to the
// OptixTraversableHandle which is passed by reference this AS contains a series
// of spheres with the same material and varying radii & center points
static DeviceByteBuffer createSphereAccelerationStructure(
    OptixDeviceContext ctx, OptixTraversableHandle& handle, int primitiveOffset,
    DeviceBuffer<float3>& centerBuffer, DeviceBuffer<float>& radiusBuffer,
    DeviceBuffer<uint32_t>& sbtIndexOffsets) {
	uint32_t sphere_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

	OptixBuildInput buildInput = {};
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
	buildInput.sphereArray.vertexStrideInBytes = sizeof(float3);
	buildInput.sphereArray.vertexBuffers = &centerBuffer.handle();
	buildInput.sphereArray.numVertices = centerBuffer.count();
	buildInput.sphereArray.radiusBuffers = &radiusBuffer.handle();
	buildInput.sphereArray.radiusStrideInBytes = 0;
	buildInput.sphereArray.flags = sphere_input_flags;
	buildInput.sphereArray.numSbtRecords = 1;
	buildInput.sphereArray.primitiveIndexOffset = primitiveOffset;
	buildInput.sphereArray.singleRadius = false;

	OptixAccelBuildOptions accelerationOptions = {};
	accelerationOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelerationOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes sizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &accelerationOptions,
						 &buildInput, 1, &sizes));

	DeviceByteBuffer tempBuffer = DeviceByteBuffer(sizes.tempSizeInBytes);
	DeviceByteBuffer outputBuffer =
	    DeviceByteBuffer(sizes.outputSizeInBytes);
	DeviceBuffer<size_t> compactedSize = DeviceBuffer<size_t>(1);

	OptixAccelEmitDesc emitProperty = {};
	emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitProperty.result = compactedSize.handle();

	OPTIX_CHECK(optixAccelBuild(
	    ctx, nullptr, &accelerationOptions, &buildInput, 1,
	    tempBuffer.handle(), sizes.tempSizeInBytes, outputBuffer.handle(),
	    sizes.outputSizeInBytes, &handle, &emitProperty, 1));

#ifndef NDEBUG
	printf("\nBuilt an optix acceleration structure of type: sphere");
#endif

	size_t compactSize;
	compactedSize.read(&compactSize);

	if (compactSize < sizes.outputSizeInBytes) {
		DeviceByteBuffer buffer = DeviceByteBuffer(compactSize);
		OPTIX_CHECK(optixAccelCompact(ctx, nullptr, handle,
					      buffer.handle(), compactSize,
					      &handle));
		return buffer;
	}

    return outputBuffer;
}

// instance acceleration structure is created to help store triangular meshes
// and save on memory by storing them as instances with different transforms
// (translation, rotation, etc)
// also allows the combining of spheres and surfaces onto a single acceleration structure
static DeviceByteBuffer createInstanceAccelerationStructure(
    OptixDeviceContext ctx, OptixTraversableHandle& handle,
    std::vector<OptixTraversableHandle> combinedHandles) {
	std::vector<OptixInstance> instances = std::vector<OptixInstance>();
	int i = 0;
	for (OptixTraversableHandle handle : combinedHandles) {
		OptixInstance instance = {};
		float transform[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
		memcpy(instance.transform, transform, sizeof(float) * 12);
		instance.flags = OPTIX_INSTANCE_FLAG_NONE;
		instance.instanceId = 0;
		instance.visibilityMask = 255;
		instance.sbtOffset = i;
		instance.traversableHandle = handle;
		instances.push_back(instance);
		i++;
	}

	DeviceBuffer<OptixInstance> instanceBuffer =
	    DeviceBuffer<OptixInstance>(instances.data(), instances.size());

	OptixBuildInput buildInput = {};
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	buildInput.instanceArray.instances = instanceBuffer.handle();
	buildInput.instanceArray.instanceStride = 0;
	buildInput.instanceArray.numInstances = instanceBuffer.count();

	OptixAccelBuildOptions accelerationOptions = {};
	accelerationOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelerationOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes sizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &accelerationOptions,
						 &buildInput, 1, &sizes));

	DeviceByteBuffer tempBuffer = DeviceByteBuffer(sizes.tempSizeInBytes);
	DeviceByteBuffer outputBuffer =
	    DeviceByteBuffer(sizes.outputSizeInBytes);
	DeviceBuffer<size_t> compactedSize = DeviceBuffer<size_t>(1);

	OptixAccelEmitDesc emitProperty = {};
	emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitProperty.result = compactedSize.handle();

	OPTIX_CHECK(optixAccelBuild(
	    ctx, nullptr, &accelerationOptions, &buildInput, 1,
	    tempBuffer.handle(), sizes.tempSizeInBytes, outputBuffer.handle(),
	    sizes.outputSizeInBytes, &handle, &emitProperty, 1));

#ifndef NDEBUG
	printf("\nBuilt an optix acceleration structure of type: instances");
#endif

	size_t compactSize;
	compactedSize.read(&compactSize);

	if (compactSize < sizes.outputSizeInBytes) {
		DeviceByteBuffer buffer = DeviceByteBuffer(compactSize);
		OPTIX_CHECK(optixAccelCompact(ctx, nullptr, handle,
					      buffer.handle(), compactSize,
					      &handle));
		return buffer;
	} else {
		return outputBuffer;
	}
}

// adds curve primitives to a triangular manifold that it interects, onto a
// traversable handle, generates resulting primitives as a devicebytebuffer,
// result. tetrahedral mesh needs to be modified with curve objects that have
// vertex1, vertex2, and widths
static OptixTraversableHandle generateManifoldWithLinearCurves(
    OptixDeviceContext ctx, DeviceBuffer<float3>& triangleVertexBuffer,
    int& primitiveCount, std::vector<DeviceByteBuffer>& result,
    surfmesh smesh, float curveWidthAdjustment,
    float sphereRadiusAdjustment, const mcconfig* cfg) {

	std::vector<uint32_t> sbt = std::vector<uint32_t>();

	// create the curve acceleration structures
	std::vector<float3> curveVertices = std::vector<float3>();
	std::vector<float> curveWidths = std::vector<float>();
	std::vector<unsigned int> curveIndices = std::vector<unsigned int>();
	sbt = std::vector<uint32_t>();

	// Curve vertex coordinates and widths (radius) are inside of
	// tetrahedral manifold
	unsigned int curvecount = 0;
	unsigned int vertices_per_curve = 2;
	for (ImplicitCurve curve : smesh.curves) {
		curveVertices.push_back(curve.vertex1);
		curveVertices.push_back(curve.vertex2);
		curveWidths.push_back(curve.width - curveWidthAdjustment);
		sbt.push_back(0);
		// the index should be the current count of curves*2
		// (vertices per curve)+1
		curveIndices.push_back((curvecount * vertices_per_curve) + 1);
		curvecount = curvecount + 1;
	}

	// specifies starting indices of the vertices of a given curve,
	// which are read in pairs to create linear curves
	DeviceBuffer<unsigned int> curveIndexBuffer =
	    DeviceBuffer<unsigned int>(curveIndices.data(),
				       curveIndices.size());

	DeviceBuffer<uint32_t> sbtBuffer =
	    DeviceBuffer<uint32_t>(sbt.data(), sbt.size());

	OptixTraversableHandle curvesHandle;

	if (smesh.curves.size() > 0) {
		result.push_back(createCurveAccelStructure(
		    ctx, curvesHandle, primitiveCount, curveVertices,
		    curveWidths, curveIndexBuffer, sbtBuffer));
		primitiveCount += smesh.curves.size();
	}

	// create the sphere acceleration structures
	std::vector<float3> sphereCenters = std::vector<float3>();
	std::vector<float> sphereRadii = std::vector<float>();
	sbt = std::vector<uint32_t>();

	// Sphere coordinates and radii is inside of tetrahedral smesh 
	for (ImplicitSphere sphere : smesh.spheres) {
		sphereCenters.push_back(sphere.position);
		sphereRadii.push_back(sphere.radius - sphereRadiusAdjustment);
		sbt.push_back(0);
	}

	DeviceBuffer<float3> centerBuffer =
	    DeviceBuffer<float3>(sphereCenters.data(), sphereCenters.size());
	DeviceBuffer<float> radiiBuffer =
	    DeviceBuffer<float>(sphereRadii.data(), sphereRadii.size());
	sbtBuffer = DeviceBuffer<uint32_t>(sbt.data(), sbt.size());

	OptixTraversableHandle spheresHandle;

	if (smesh.spheres.size() > 0) {
		// this creates the acceleration structures for spheres
		result.push_back(createSphereAccelerationStructure(
		    ctx, spheresHandle, primitiveCount, centerBuffer,
		    radiiBuffer, sbtBuffer));
		primitiveCount += smesh.spheres.size();
	}

	// create the mesh acceleration structures
	std::vector<uint3> indices; 

	for (uint3 triangle_face : smesh.face) {
		indices.push_back(triangle_face);
		sbt.push_back(0);
	}

	OptixTraversableHandle handle;
	DeviceBuffer<uint3> indexBuffer =
	    DeviceBuffer<uint3>(indices.data(), indices.size());
	sbtBuffer = DeviceBuffer<uint32_t>(sbt.data(), sbt.size());

	result.push_back(createAccelerationStructure(
	    ctx, handle, primitiveCount, triangleVertexBuffer, indexBuffer,
	    sbtBuffer));
	primitiveCount += smesh.face.size();

	// combine the handles
	std::vector<OptixTraversableHandle> combinedHandles =
	    std::vector<OptixTraversableHandle>();
	if (smesh.curves.size() > 0) {
		combinedHandles.push_back(curvesHandle);
	}
	if (smesh.spheres.size() > 0){
		combinedHandles.push_back(spheresHandle);
	}
	combinedHandles.push_back(handle);
	result.push_back(
	    createInstanceAccelerationStructure(ctx, handle, combinedHandles));
	return handle;
}

// Prepares and organizes two vectors of traversable handles, one where spheres
// override mesh properties and one where mesh overrides sphere properties Calls
// functions to search for/define manifolds, create traversable handles and
// their acceleration structures.
static std::vector<DeviceByteBuffer> generateTetrahedralAccelerationStructures(
    OptixDeviceContext ctx, tetmesh* mesh,
    std::vector<PrimitiveSurfaceData>& surfaceData,
    std::vector<ImplicitCurve>& curveData, \
			std::vector<OptixTraversableHandle>& handles,
    uint32_t& startTetMedium, OptixTraversableHandle& startHandle,
    bool startInImplicit, unsigned int& num_inside_prims, const float WIDTH_ADJ, const mcconfig* cfg) {
	
    std::cout << "Creating single-material triangle surface-meshes." << std::endl;
	
    // create a c-style array of surface meshes
    surfmesh *smesh = (surfmesh*)calloc((mesh->prop + 1), sizeof(surfmesh));
    prepareSurfMeshArray(mesh, smesh, cfg);

    printf("Num spheres created: %d Num curves created: %d\n", smesh->spheres.size(), smesh->curves.size());    
	std::vector<DeviceByteBuffer> result = std::vector<DeviceByteBuffer>();
	handles = std::vector<OptixTraversableHandle>();
	std::vector<OptixTraversableHandle> insideSphereHandles =
	    std::vector<OptixTraversableHandle>();

    std::vector<float3> nodesvector = convert_mmc_mesh_nodevector(mesh); 
	DeviceBuffer<float3> vertexBuffer =
	    DeviceBuffer<float3>(nodesvector.data(), nodesvector.size());

	std::cout << "Building acceleration structures." << mesh->prop
		  << std::endl;

	int primitiveCount = 0;
	// two separate vectors of traversable handles are created, one for
	// outside spheres
	for (int i = 0; i <= mesh->prop; ++i) {
		 handles.push_back(generateManifoldWithLinearCurves(
		    ctx, vertexBuffer, primitiveCount, result, smesh[i], 0.0,
		    0.0, cfg));
	}

	printf("\nThe number of inside primitives is: %d", primitiveCount);
	num_inside_prims = primitiveCount;
	// one for inside spheres
	for (int i = 0; i <= mesh->prop; ++i) {
		// constant to determine how much wider outside-inside intersection tracking primitives should be
		insideSphereHandles.push_back(generateManifoldWithLinearCurves(
		    ctx, vertexBuffer, primitiveCount, result, smesh[i],
		    WIDTH_ADJ, WIDTH_ADJ, cfg));
	}

	// the following code produces all the "Surface Boundaries"
	// and organizes them, a surface boundary is a 3D surface
	// of connected triangle mesh and any intersecting spheres
	// of the same material
	// This geometric and material data is fed to the device side for
	// closest hit through mcx_params
	//
	// ask douglas about weird ordering of these loops:
	// outside manifolds, inside implicits, inside manifolds,
	// outside implicits
	surfaceData = std::vector<PrimitiveSurfaceData>();

	// record curve info to pass to the device
	curveData = std::vector<ImplicitCurve>();

	std::cout << "Loading surface data." << std::endl;

	// loops through all different surface meshes 
	for (int i = 0; i < mesh->prop; i++) {
		// add all curves to the surface boundaries
		for (ImplicitCurve s : smesh[i].curves) {
		    float4 facenorm_and_mediumid = make_float4(0,0,0, storeuintAsFloat(SPHERE_MATERIAL));	
            surfaceData.push_back(PrimitiveSurfaceData{
			    facenorm_and_mediumid, insideSphereHandles[i]});
			curveData.push_back(s);
		}

#ifndef NDEBUG
			printf("\n number of outside-curves sent to device: %d", smesh[i].curves.size());	
#endif

		// add all spheres to the surface boundaries
		for (ImplicitSphere s : smesh[i].spheres) {
			float4 facenorm_and_mediumid = make_float4(s.position.x, s.position.y, s.position.z,
                       storeuintAsFloat(1)); 
            surfaceData.push_back(PrimitiveSurfaceData{
                facenorm_and_mediumid, insideSphereHandles[i]});
		}

#ifndef NDEBUG
			printf("\n number of outside-spheres sent to device: %d", smesh[i].spheres.size());	
#endif

		// assigns all surface mesh triangles a material
		for (size_t j = 0; j < smesh[i].norm.size(); ++j) {
                // TODO: add actual triangle normals to this
                float4 facenorm_and_mediumid = make_float4(smesh[i].norm[j].x,smesh[i].norm[j].y,smesh[i].norm[j].z,
                            *(float*)&smesh[i].nbtype[j]);
				surfaceData.push_back(PrimitiveSurfaceData{
                    facenorm_and_mediumid,
				    handles[smesh[i].nbtype[j]]});
                //printf("\nTriangle Boundary Material ID is: %f",  *(float*)&smesh[i].nbtype[j]);
		}
#ifndef NDEBUG
			printf("\n number of outside-triangles sent to device: %d", smesh[i].face.size());	
#endif

	}

	for (int i = 0; i < mesh->prop; i++) {
		for (auto& _ : smesh[i].curves) {// iterate for each curve
            (void) _; // suppress compiler warning about unused variable in range based loop	
            float4 facenorm_and_mediumid = make_float4(0,0,0,
                    storeuintAsFloat(1));
            surfaceData.push_back(
			    PrimitiveSurfaceData{
                  facenorm_and_mediumid,
			      handles[i]});
		}
#ifndef NDEBUG
			printf("\n number of inside-curves sent to device: %d", smesh[i].curves.size());	
#endif
		for (ImplicitSphere s : smesh[i].spheres) {
			float4 facenorm_and_mediumid = make_float4(
                    s.position.x, s.position.y, s.position.z, 
                    storeuintAsFloat(SPHERE_MATERIAL));
            surfaceData.push_back(
                			    PrimitiveSurfaceData{
			     facenorm_and_mediumid, handles[i]});
		}

#ifndef NDEBUG
			printf("\n number of inside-spheres sent to device: %d", smesh[i].spheres.size());	
#endif

		for (size_t j = 0; j < smesh[i].norm.size(); ++j) {
                float4 facenorm_and_mediumid = make_float4(smesh[i].norm[j].x,smesh[i].norm[j].y,smesh[i].norm[j].z,       
                    storeuintAsFloat(SPHERE_MATERIAL));
                surfaceData.push_back(PrimitiveSurfaceData{
				    facenorm_and_mediumid,
				    insideSphereHandles[smesh[i].nbtype[j]]});
		}
	#ifndef NDEBUG
			printf("\n number of inside-triangles sent to device: %d", smesh[i].norm.size());	
	#endif
	}

    startTetMedium =
	    mesh->type[cfg->e0-1];

    startHandle =
	    startInImplicit
        //TODO: this needs to refer to the correct surfacemeshes for each id
		? insideSphereHandles[startTetMedium]
		: handles[startTetMedium];

	return result;
}

// constructor
McxContext::McxContext() {

    // ensure any previous device resources are freed

    CUDA_CHECK(cudaFree(nullptr));

	OPTIX_CHECK(optixInit());

	OptixDeviceContextOptions opts = {};
	opts.logCallbackFunction = &McxContext::messageHandler;
	opts.logCallbackData = this;
	opts.logCallbackLevel = 4;

#ifndef NDEBUG
	opts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
	std::cout << "debug mode enabled" << std::endl;
#else
	opts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
	std::cout<< "debug mode not enabled" << std::endl;
#endif

	OPTIX_CHECK(
	    optixDeviceContextCreate(nullptr, &opts, &this->optixContext));

	std::string ptx = std::string(mmcShaderPtx);
	ShaderFunctionSet set = ShaderFunctionSet(
	    "__raygen__rg", "__miss__ms",

	    // this is initializing and feeding a standard vector of hitgroups
	    {ShaderFunctionSet::HitGroup("__closesthit__ch",
					 "__intersection__customlinearcurve"),
	     ShaderFunctionSet::HitGroup("__closesthit__ch", 
			     		"__BUILTIN_SPHERE__"),
	     ShaderFunctionSet::HitGroup("__closesthit__ch", "") 
	     },

	    "gcfg");
	
	unsigned int TOTAL_PARAM_COUNT = 18;

	this->devicePipeline =
	    ShaderPipeline(this->optixContext, ptx, set, TOTAL_PARAM_COUNT, 4);

        // raygen   
        SbtRecord<void*> rrec = SbtRecord<void*>(nullptr);
        OPTIX_CHECK(optixSbtRecordPackHeader(this->devicePipeline.raygenProgram(), &rrec));
        DeviceBuffer <SbtRecord<void*>> raygenRecord = rrec;
        
        // miss
        SbtRecord<void*> mrec = SbtRecord<void*>(nullptr);
        OPTIX_CHECK(optixSbtRecordPackHeader(this->devicePipeline.missProgram(), &mrec));
        DeviceBuffer <SbtRecord<void*>> missRecord = mrec;

        // hit programs
        std::vector<void*> h = {0,0,0};
        std::vector<SbtRecord<void*>> grecs;
        if (h.size() != this->devicePipeline.hitgroupPrograms().size()) {
            throw std::runtime_error("Hitgroup data count was not the same as pipeline hitgroup count");
        }

        for (unsigned int i = 0; i < h.size(); i++)
        {
            grecs.push_back(SbtRecord<void*>(h[i]));
            OPTIX_CHECK(optixSbtRecordPackHeader(
                        this->devicePipeline.hitgroupPrograms()[i],
                        &grecs[i]));
        }

        DeviceBuffer<SbtRecord<void*>> hitgroupRecords = 
            DeviceBuffer<SbtRecord<void*>>(grecs.data(), grecs.size());

        OptixShaderBindingTable sbt = {};

        sbt.raygenRecord = raygenRecord.handle();
        sbt.missRecordBase = missRecord.handle();
        sbt.missRecordStrideInBytes = sizeof(mrec);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = hitgroupRecords.handle();
        sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<void*>);
        sbt.hitgroupRecordCount = grecs.size();

        this->SBT = sbt;
}

// move constructor
McxContext::McxContext(McxContext&& src) {
	this->optixContext = src.optixContext;
	this->devicePipeline = std::move(src.devicePipeline);
	src.optixContext = OptixDeviceContext();
	src.devicePipeline = ShaderPipeline();
}

// calculates if the ray is starting in an implicit structure
// TODO: implement this
bool checkStartInImplicit(tetmesh* mesh, mcconfig* cfg){
    return false;
}

MMCParam prepOptixIMMCLaunchParams(mcconfig* cfg, tetmesh* mesh, const unsigned int num_inside_prims, const OptixTraversableHandle startHandle,
                                    std::vector<PrimitiveSurfaceData> surfaceData, 
                                    std::vector<ImplicitCurve> curveData, float** outputHostBuffer, unsigned int* outputSize, osc::CUDABuffer* outputBuffer){
        MMCParam gcfg;

        // TODO: Implement front-end for MMC instead of temporarily
        // hardcoding optix-MMC variables
        gcfg.tstart = cfg->tstart;
        gcfg.tend = cfg->tend;
        gcfg.Rtstep = 1.0f / cfg->tstep;
        gcfg.maxgate = cfg->maxgate; // this is the last time gate 

        // prepare dual mesh parameters
        // TODO: make this into a function for IMMC with Dual-grid boundaries increased for
        // capsules/spheres outside of mesh
        gcfg.dstep = 1.0f / cfg->unitinmm; // distance step for output is currently hardcoded to 0.01mm
        gcfg.nmin = make_float3(mesh->nmin.x, mesh->nmin.y, mesh->nmin.z);
        gcfg.nmax = make_float3(mesh->nmax.x-mesh->nmin.x,
                                mesh->nmax.y-mesh->nmin.y,
                                mesh->nmax.z-mesh->nmin.z);

        gcfg.crop0.x = cfg->crop0.x;
        gcfg.crop0.y = cfg->crop0.y;
        gcfg.crop0.z = cfg->crop0.z;

        int timeSteps = (int)((cfg->tend - cfg->tstart) / cfg->tstep + 0.5);
        gcfg.crop0.w = cfg->crop0.z * timeSteps;
        
        gcfg.isreflect = cfg->isreflect; // turn reflection settings off for now
        gcfg.outputtype = static_cast<int>(cfg->outputtype);

        // initializing variables for output of data
        *outputSize = (gcfg.crop0.w << 1);
        *outputHostBuffer = (float*) calloc(*outputSize, sizeof(float));
        outputBuffer->alloc_and_upload(*outputHostBuffer, *outputSize);

	    // surfaceData is a vector of surface boundaries
	    DeviceBuffer<PrimitiveSurfaceData> primitive_data = 
            DeviceBuffer<PrimitiveSurfaceData>(surfaceData.data(), surfaceData.size());
	    DeviceBuffer<ImplicitCurve> curves =
	        DeviceBuffer<ImplicitCurve>(curveData.data(), curveData.size());

        // CUdeviceptr for vector of surface boundaries
        gcfg.surfaceBoundaries = primitive_data.handle(); 
        // CUdeviceptr for vector of capsules 
        gcfg.curveData = curves.handle(); 
        // CUdeviceptr for flattened 4D output array
        gcfg.outputbuffer = outputBuffer->d_pointer();
        // float3 for starting position of ray
        gcfg.srcpos = make_float3(cfg->srcpos.x,
                                  cfg->srcpos.y,
                                  cfg->srcpos.z); 
        // float3 for vector of starting ray direction 
        gcfg.srcdir = make_float3(cfg->srcdir.x,
                                  cfg->srcdir.y,
                                  cfg->srcdir.z); 
        
        // starting OptixTraversableHandle 
        gcfg.gashandle0 = startHandle;
       
        // calculate starting medium 
        bool notStartingInImplicit = !checkStartInImplicit(mesh, cfg);
        gcfg.mediumid0 = notStartingInImplicit ? mesh->type[cfg->e0-1] : SPHERE_MATERIAL;
        printf("Starting Medium is: %d\n", gcfg.mediumid0);

        // Media
        for (int i = 0; i <= mesh->prop; ++i) {
            gcfg.medium[i].mua = mesh->med[i].mua;
            gcfg.medium[i].mus = mesh->med[i].mus;
            gcfg.medium[i].g = mesh->med[i].g;
            gcfg.medium[i].n = mesh->med[i].n;
        }

        // get hardware info
        cudaDeviceProp prop;
    	cudaGetDeviceProperties(&prop, 0);
	
    	// Get the number of SMs (streaming multiprocessors)
        unsigned int numSMs = prop.multiProcessorCount;
        // Get the maximum number of threads per SM
        unsigned int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
        // Calculate the total number of threads
        unsigned int launchWidth = (numSMs-1) * maxThreadsPerSM;

	    // set number of threads and photons per thread:
	    unsigned int threadphoton = cfg->nphoton / launchWidth;	
	    unsigned int oddphoton = cfg->nphoton - threadphoton * launchWidth;

        // unsigned int describing number of GAS primitives for out-in tracing 
        gcfg.num_inside_prims = num_inside_prims;
        // float for marginal difference in radii for out-in 
        // vs in-out primitives 
        gcfg.WIDTH_ADJ = 1.0 / 10240.0;
        // unsigned int for number of photons per thread
        gcfg.threadphoton = threadphoton; 
        // unsigned int for remainder after dividing between threads 
        gcfg.oddphoton = oddphoton;

        int totalthread = launchWidth;
        //uint4 hseed[totalthread];
        uint4* hseed = (uint4 *)calloc(totalthread, sizeof(uint4));
        for (int i=0; i<totalthread; ++i){
            hseed[i] = make_uint4(rand(), rand(), rand(), rand());
        }

        // prepare seed buffer
        osc::CUDABuffer seedBuffer;
        seedBuffer.alloc_and_upload(hseed, totalthread);
        gcfg.seedbuffer = seedBuffer.d_pointer(); 
        return gcfg;
}

// this function is run by the main and performs the mmc-optix simulation given
// a mesh, voxel grid size for absorption counts, vector of media optical
// properties, photon count etc.
void McxContext::simulate(tetmesh* mesh, mcconfig* cfg) {

	std::vector<PrimitiveSurfaceData> surfaceData;
	std::vector<ImplicitCurve> curveData;
	std::vector<OptixTraversableHandle> handles;

	OptixTraversableHandle startHandle;
	uint32_t startMedium;

    curveData = mcconfig_to_capsules(cfg);
    std::vector<mcx::ImplicitSphere> spheres = mcconfig_to_spheres(cfg);

	bool startInSphere = insideSphere(make_float3(cfg->srcpos.x, cfg->srcpos.y, cfg->srcpos.z), spheres);
	bool startInCurve = insideCurve(make_float3(cfg->srcpos.x, cfg->srcpos.y, cfg->srcpos.z), curveData);
	// right now set to true if the start is inside either implicit geometry
	bool startInImplicit = startInSphere || startInCurve;

	std::cout << "Starting element is:" << cfg->e0 << std::endl;

	unsigned int num_inside_prims;
	const float WIDTH_ADJ = 1.0 / 10240.0;// Width adjustment to make outer spheres slightly larger than inner spheres
	// function modifies almost everything by reference
	    generateTetrahedralAccelerationStructures(
		this->optixContext, mesh, surfaceData, curveData, handles,
		startMedium, startHandle, startInImplicit, num_inside_prims, WIDTH_ADJ, cfg);

// print surface data for debugging
#ifndef NDEBUG
	for (unsigned int i=0; i<surfaceData.size(); ++i){
		printf("\nThe %dth surface has a material of %d\n", i, storeFloatAsuint(surfaceData[i].fnorm.w));
	}	
#endif	

	// surfaceData is a vector of surface boundaries
	DeviceBuffer<PrimitiveSurfaceData> primitive_data = DeviceBuffer<PrimitiveSurfaceData>(
	    surfaceData.data(), surfaceData.size());
	DeviceBuffer<ImplicitCurve> curves =
	    DeviceBuffer<ImplicitCurve>(curveData.data(), curveData.size());

    // get hardware info
    cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	
	// Get the number of SMs (streaming multiprocessors)
    	unsigned int numSMs = prop.multiProcessorCount;
    	// Get the maximum number of threads per SM
    	unsigned int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;

    	// Calculate the total number of threads
    	unsigned int launchWidth = (numSMs-1) * maxThreadsPerSM;

	printf("\n THE NUMBER OF INSIDE PRIMS BEFORE SENDING TO GPU: %d", num_inside_prims);

    float* outputHostBuffer;
    unsigned int outputSize;
    osc::CUDABuffer outputBuffer; 
    // prepare optix pipeline parameters
	MMCParam gcfg = prepOptixIMMCLaunchParams(cfg, mesh, num_inside_prims, startHandle, surfaceData, curveData, &outputHostBuffer, &outputSize, &outputBuffer);
	    
	DeviceBuffer<MMCParam> paraBuffer(gcfg);

	std::cout << "Beginning simulation." << std::endl;

    MCX_clock timer;

	// OPTIX_CHECK reports if there were errors with the sim
	// optixLaunch launches photons, given a pipeline, stream, pipeline
	// params, pipeline param size, optix shader binding table, width of
	// computations, height of computations, and length of computations From
	// a GPU perspective this is done with a linear set of kernels of length
	// photoncount.
	OPTIX_CHECK(optixLaunch(this->devicePipeline.handle(), nullptr, \
			paraBuffer.handle(),
				sizeof(gcfg), &this->SBT, launchWidth,
				1, 1));

    // download from GPU the outputted data
	outputBuffer.download(outputHostBuffer, outputSize);
    printf("\nsim completed successfully, photons per ms was %f \n", cfg->nphoton/timer.elapse());
    printf("\nTotal kernel time: %f \n", timer.elapse()); 
    printf("\nbuffer downloaded successfully");
    double* TEMPweight;
    TEMPweight = (double*)calloc(sizeof(double) * gcfg.crop0.z, gcfg.maxgate);
    // ==================================================================
    // Save output
    // ==================================================================
    for (size_t i = 0; i < gcfg.crop0.w; i++) {
        // combine two outputs into one
        #pragma omp atomic
        TEMPweight[i] += outputHostBuffer[i] +
            outputHostBuffer[i + gcfg.crop0.w];
    }

    // ==================================================================
    // normalize output
    // ==================================================================
    /*if (cfg->isnormalized) {
        // not used if cfg->method == rtBLBadouelGrid
        float energyabs = 0.0f;
        // for now assume initial weight of each photon is 1.0
        int energytot = pcount;
        mesh_normalize(mesh, cfg, energyabs, energytot, 0);
    }*/

    #pragma omp master
    {
    int datalen = gcfg.crop0.z;
    FILE* fp;

    fp = fopen("optix.bin", "wb");
    fwrite(TEMPweight, sizeof(double), datalen*gcfg.maxgate, fp);
    fclose(fp);
    }
}

void McxContext::onMessageReceived(uint32_t level, const char* tag,
				   const char* message) {
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12)
		  << tag << "]: " << message << "\n";
}

void McxContext::messageHandler(uint32_t level, const char* tag,
				const char* message, void* data) {
	((McxContext*)data)->onMessageReceived(level, tag, message);
}

McxContext::~McxContext() {
	//this->deviceSbt = ShaderBindingTable<void*, void*, void*>();
     
    this->devicePipeline = ShaderPipeline();
	optixDeviceContextDestroy(this->optixContext);
	cudaDeviceSynchronize();
}
}  // namespace mcx
