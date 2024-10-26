#include "mcx_context.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
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
#include "incbin.h"
#include "mcx_launch_params.h"
#include "shader_pipeline.h"
#include "tetrahedral_mesh.h"
#include "util.h"
#include "device_buffer.h"

// this includes lots of optix features
#ifndef NDEBUG
INCTXT(mmcShaderPtx, mmcShaderPtxSize, "mcx_core.ptx")
#else
INCTXT(mmcShaderPtx, mmcShaderPtxSize, "mcx_core.ptx");
#endif

#define SPHERE_MATERIAL 1

namespace mcx {

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
	//buildInput.triangleArray.sbtIndexOffsetBuffer =
	//    sbtIndexOffsets.handle();
	//buildInput.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(int32_t);
	//buildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;
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
	for (int i = 0; i < vertexVector.size(); i += 2) {
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
	//buildInput.sphereArray.sbtIndexOffsetBuffer = sbtIndexOffsets.handle();
	//buildInput.sphereArray.sbtIndexOffsetSizeInBytes = sizeof(int32_t);
	//buildInput.sphereArray.sbtIndexOffsetStrideInBytes = 0;
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
    TetrahedralManifold& manifold, float curveWidthAdjustment,
    float sphereRadiusAdjustment) {

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
	for (ImplicitCurve curve : manifold.curves) {
		curveVertices.push_back(curve.vertex1);
		curveVertices.push_back(curve.vertex2);
		curveWidths.push_back(curve.width + curveWidthAdjustment);
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

	if (manifold.curves.size() > 0) {
		result.push_back(createCurveAccelStructure(
		    ctx, curvesHandle, primitiveCount, curveVertices,
		    curveWidths, curveIndexBuffer, sbtBuffer));
		primitiveCount += manifold.curves.size();
	}

	// create the sphere acceleration structures
	std::vector<float3> sphereCenters = std::vector<float3>();
	std::vector<float> sphereRadii = std::vector<float>();
	sbt = std::vector<uint32_t>();

	// Sphere coordinates and radii is inside of tetrahedral manifold
	for (ImplicitSphere sphere : manifold.spheres) {
		sphereCenters.push_back(sphere.position);
		sphereRadii.push_back(sphere.radius + sphereRadiusAdjustment);
		sbt.push_back(0);
	}

	DeviceBuffer<float3> centerBuffer =
	    DeviceBuffer<float3>(sphereCenters.data(), sphereCenters.size());
	DeviceBuffer<float> radiiBuffer =
	    DeviceBuffer<float>(sphereRadii.data(), sphereRadii.size());
	sbtBuffer = DeviceBuffer<uint32_t>(sbt.data(), sbt.size());

	OptixTraversableHandle spheresHandle;

	if (manifold.spheres.size() > 0) {
		// this creates the acceleration structures for spheres
		result.push_back(createSphereAccelerationStructure(
		    ctx, spheresHandle, primitiveCount, centerBuffer,
		    radiiBuffer, sbtBuffer));
		primitiveCount += manifold.spheres.size();
	}

	// create the mesh acceleration structures
	std::vector<uint3> indices = std::vector<uint3>();

	for (TetrahedronBoundary b : manifold.triangles) {
		indices.push_back(b.indices);
		sbt.push_back(0);
	}

	OptixTraversableHandle handle;
	DeviceBuffer<uint3> indexBuffer =
	    DeviceBuffer<uint3>(indices.data(), indices.size());
	sbtBuffer = DeviceBuffer<uint32_t>(sbt.data(), sbt.size());

	result.push_back(createAccelerationStructure(
	    ctx, handle, primitiveCount, triangleVertexBuffer, indexBuffer,
	    sbtBuffer));
	primitiveCount += manifold.triangles.size();

	// combine the handles
	std::vector<OptixTraversableHandle> combinedHandles =
	    std::vector<OptixTraversableHandle>();
	if (manifold.curves.size() > 0) {
		combinedHandles.push_back(curvesHandle);
	}
	if (manifold.spheres.size() > 0){
		combinedHandles.push_back(spheresHandle);
	}
	combinedHandles.push_back(handle);
	result.push_back(
	    createInstanceAccelerationStructure(ctx, handle, combinedHandles));
	return handle;
}

// creates acceleration structures of manifold and the spheres intersecting with
// that manifold, integrates both onto the vector of optix traversable handles
// called "combinedhandles" (deprecated)
static OptixTraversableHandle generateManifoldWithSpheres(
    OptixDeviceContext ctx, DeviceBuffer<float3>& vertexBuffer, \
		int& primitiveCount,
    std::vector<DeviceByteBuffer>& result, TetrahedralManifold& manifold,
    float sphereRadiusAdjustment) {
	std::vector<uint3> indices = std::vector<uint3>();
	std::vector<uint32_t> sbt = std::vector<uint32_t>();

	for (TetrahedronBoundary b : manifold.triangles) {
		indices.push_back(b.indices);
		sbt.push_back(0);
	}

	// Prepare device buffers for acceleration structure index and shader
	// binding table
	OptixTraversableHandle handle;
	DeviceBuffer<uint3> indexBuffer =
	    DeviceBuffer<uint3>(indices.data(), indices.size());
	DeviceBuffer<uint32_t> sbtBuffer =
	    DeviceBuffer<uint32_t>(sbt.data(), sbt.size());

	// this creates the acceleration structures for manifolds
	result.push_back(createAccelerationStructure(
	    ctx, handle, primitiveCount, vertexBuffer, indexBuffer, sbtBuffer));
	primitiveCount += manifold.triangles.size();

	std::vector<float3> sphereCenters = std::vector<float3>();
	std::vector<float> sphereRadii = std::vector<float>();
	sbt = std::vector<uint32_t>();

	// Sphere coordinates and radii is inside of tetrahedral manifold
	for (ImplicitSphere sphere : manifold.spheres) {
		sphereCenters.push_back(sphere.position);
		sphereRadii.push_back(sphere.radius + sphereRadiusAdjustment);
		sbt.push_back(0);
	}

	DeviceBuffer<float3> centerBuffer =
	    DeviceBuffer<float3>(sphereCenters.data(), sphereCenters.size());
	DeviceBuffer<float> radiiBuffer =
	    DeviceBuffer<float>(sphereRadii.data(), sphereRadii.size());
	sbtBuffer = DeviceBuffer<uint32_t>(sbt.data(), sbt.size());

	OptixTraversableHandle spheresHandle;

	if (manifold.spheres.size() > 0) {
		// this creates the acceleration structures for spheres
		result.push_back(createSphereAccelerationStructure(
		    ctx, spheresHandle, primitiveCount, centerBuffer,
		    radiiBuffer, sbtBuffer));
		primitiveCount += manifold.spheres.size();
	}

	std::vector<OptixTraversableHandle> combinedHandles =
	    std::vector<OptixTraversableHandle>();
	combinedHandles.push_back(handle);
	if (manifold.spheres.size() > 0) {
		combinedHandles.push_back(spheresHandle);
	}
	result.push_back(
	    createInstanceAccelerationStructure(ctx, handle, combinedHandles));


	return handle;
}

// Prepares and organizes two vectors of traversable handles, one where spheres
// override mesh properties and one where mesh overrides sphere properties Calls
// functions to search for/define manifolds, create traversable handles and
// their acceleration structures.
static std::vector<DeviceByteBuffer> generateTetrahedralAccelerationStructures(
    OptixDeviceContext ctx, TetrahedralMesh& mesh,
    std::vector<SurfaceBoundary>& surfaceData,
    std::vector<ImplicitCurve>& curveData, \
			std::vector<OptixTraversableHandle>& handles,
    uint32_t& startTetMedium, OptixTraversableHandle& startHandle,
    bool startInSphere, unsigned int& num_inside_prims, const float WIDTH_ADJ) {
	
    std::cout << "Detecting manifolds." << std::endl;
	std::vector<uint32_t> tetrahedron_to_manifold;
	std::vector<TetrahedralManifold> manifolds =
	    mesh.buildManifold(tetrahedron_to_manifold);

	std::vector<DeviceByteBuffer> result = std::vector<DeviceByteBuffer>();
	handles = std::vector<OptixTraversableHandle>();
	std::vector<OptixTraversableHandle> insideSphereHandles =
	    std::vector<OptixTraversableHandle>();

	DeviceBuffer<float3> vertexBuffer =
	    DeviceBuffer<float3>(mesh.nodes.data(), mesh.nodes.size());

	std::cout << "Building acceleration structures." << manifolds.size()
		  << std::endl;

	int primitiveCount = 0;
	// two separate vectors of traversable handles are created, one for
	// outside spheres
	for (TetrahedralManifold manifold : manifolds) {
//		 handles.push_back(generateManifoldWithSpheres(ctx,
//		 vertexBuffer, primitiveCount, result, manifold, 0.0));
		 handles.push_back(generateManifoldWithLinearCurves(
		    ctx, vertexBuffer, primitiveCount, result, manifold, 0.0,
		    0.0));
	}

	printf("\nThe number of inside primitives is: %d", primitiveCount);
	num_inside_prims = primitiveCount;
	// one for inside spheres
	for (TetrahedralManifold manifold : manifolds) {
//		 insideSphereHandles.push_back(generateManifoldWithSpheres(ctx,
//		 vertexBuffer, primitiveCount, result, manifold, 1.0 /
//		 1024.0));
		// constant to determine how much wider outside-inside intersection tracking primitives should be
		insideSphereHandles.push_back(generateManifoldWithLinearCurves(
		    ctx, vertexBuffer, primitiveCount, result, manifold,
		    WIDTH_ADJ, WIDTH_ADJ));
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
	surfaceData = std::vector<SurfaceBoundary>();

	// record curve info to pass to the device
	curveData = std::vector<ImplicitCurve>();

	std::cout << "Loading surface data." << std::endl;

	// loops through all different manifolds
	for (int i = 0; i < manifolds.size(); i++) {
		// add all curves to the surface boundaries
		for (ImplicitCurve s : manifolds[i].curves) {
			surfaceData.push_back(SurfaceBoundary{
			    SPHERE_MATERIAL, insideSphereHandles[i]});
			curveData.push_back(s);
		}

#ifndef NDEBUG
			printf("\n number of inside-curves sent to device: %d", manifolds[i].curves.size());	
#endif

		// add all spheres to the surface boundaries
		for (ImplicitSphere s : manifolds[i].spheres) {
			surfaceData.push_back(SurfaceBoundary{
			    SPHERE_MATERIAL, insideSphereHandles[i]});
		}

#ifndef NDEBUG
			printf("\n number of inside-spheres sent to device: %d", manifolds[i].spheres.size());	
#endif

		// assigns all manifold triangles a material
		for (TetrahedronBoundary b : manifolds[i].triangles) {
			if (b.manifold > 0) {
				surfaceData.push_back(SurfaceBoundary{
				    manifolds[b.manifold - 1].material,
				    handles[b.manifold - 1]});
			} else {
				surfaceData.push_back(SurfaceBoundary{
				    0, (OptixTraversableHandle) nullptr});
			}
		}
#ifndef NDEBUG
			printf("\n number of inside-triangles sent to device: %d", manifolds[i].triangles.size());	
#endif

	}

	for (int i = 0; i < manifolds.size(); i++) {
		for (ImplicitCurve c : manifolds[i].curves) {
			surfaceData.push_back(
			    SurfaceBoundary{
			      0,
			      handles[i]});

		}
#ifndef NDEBUG
			printf("\n number of outside-curves sent to device: %d", manifolds[i].curves.size());	
#endif
		for (ImplicitSphere s : manifolds[i].spheres) {
			surfaceData.push_back(
			    SurfaceBoundary{
			     0, handles[i]});
		}

#ifndef NDEBUG
			printf("\n number of outside-spheres sent to device: %d", manifolds[i].spheres.size());	
#endif

		for (TetrahedronBoundary b : manifolds[i].triangles) {
			if (b.manifold > 0) {
				surfaceData.push_back(SurfaceBoundary{
				    SPHERE_MATERIAL,
				    insideSphereHandles[b.manifold - 1]});

			} else {
				surfaceData.push_back(SurfaceBoundary{
				    0, (OptixTraversableHandle) nullptr});
			    // TODO: figure out if this should be the mesh material instead of 0
                // this may be to account for surfaces of exiting the domain?	
			}
		}
	#ifndef NDEBUG
			printf("\n number of outside-triangles sent to device: %d", manifolds[i].triangles.size());	
	#endif
	}

    // note that before modification, startTetMedium is the element ID of the starting elem
	startHandle =
	    startInSphere
		? insideSphereHandles[tetrahedron_to_manifold[startTetMedium]]
		: handles[tetrahedron_to_manifold[startTetMedium]];

    startTetMedium =
	    manifolds[tetrahedron_to_manifold[startTetMedium]].material;

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

	    "launchParams");
	
	unsigned int TOTAL_PARAM_COUNT = 18;

	this->devicePipeline =
	    ShaderPipeline(this->optixContext, ptx, set, TOTAL_PARAM_COUNT, 4);



    /* SET UP THE SHADER BINDING TABLE */
/* UNRAVEL THE BELOW ABSTRACTION
	this->deviceSbt = ShaderBindingTable<void*, void*, void*>
        (this->devicePipeline, nullptr, nullptr, {0, 0, 0});
*/
                                                                                        
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

        for (int i = 0; i < h.size(); i++)
        {
            grecs.push_back(SbtRecord<void*>(h[i]));
            OPTIX_CHECK(optixSbtRecordPackHeader(this->devicePipeline.hitgroupPrograms()[i], &grecs[i    ]));
        }

        DeviceBuffer<SbtRecord<void*>> hitgroupRecords = DeviceBuffer<SbtRecord<void*>>(grecs.data(), grecs.size());

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
	//this->deviceSbt = std::move(src.deviceSbt);
	src.optixContext = OptixDeviceContext();
	src.devicePipeline = ShaderPipeline();
//	src.deviceSbt = ShaderBindingTable<void*, void*, void*>();
}

// this function is run by the main and performs the mmc-optix simulation given
// a mesh, voxel grid size for absorption counts, vector of media optical
// properties, photon count etc.
void McxContext::simulate(TetrahedralMesh& mesh, uint3 size,
			  std::vector<Medium> media, uint32_t pcount,
			  float duration, uint32_t timeSteps,
			  float3 pos, float3 dir) {
	if (timeSteps < 1) {
		throw std::runtime_error(
		    "There must be at least one time step.");
	}

	std::vector<SurfaceBoundary> triangleData;
	std::vector<ImplicitCurve> curveData;
	std::vector<OptixTraversableHandle> handles;

	OptixTraversableHandle startHandle;
	uint32_t startMedium;
	bool startInSphere = mesh.insideSphere(pos);
	bool startInCurve = mesh.insideCurve(pos);
	// right now set to true if the start is inside either implicit geometry
	bool startInImplicit = startInSphere || startInCurve;

	// startMedium is passed by reference and modified here.
	std::cout << "Obtaining starting element." << std::endl;
	if (!mesh.surroundingElement(pos, startMedium)) {
		throw std::runtime_error(
		    "Emitter position is not within mesh.");
	}
	std::cout << "Starting element is:" << startMedium << std::endl;

	unsigned int num_inside_prims;
	const float WIDTH_ADJ = 1.0 / 1024.0;
	// accelerationStructures variable isn't actually used anywhere,
	// function modifies almost everything by reference
	    generateTetrahedralAccelerationStructures(
		this->optixContext, mesh, triangleData, curveData, handles,
		startMedium, startHandle, startInImplicit, num_inside_prims, WIDTH_ADJ);

// print surface data for debugging
#ifndef NDEBUG
	for (unsigned int i=0; i<triangleData.size(); ++i){
		printf("\nThe %dth surface has a material of %d\n", i, triangleData[i].medium);
	}	
#endif	

	// triangleData is a vector of surface boundaries
	DeviceBuffer<SurfaceBoundary> boundary = DeviceBuffer<SurfaceBoundary>(
	    triangleData.data(), triangleData.size());
	DeviceBuffer<ImplicitCurve> curves =
	    DeviceBuffer<ImplicitCurve>(curveData.data(), curveData.size());

	// initializing variables for output of data
	size_t ods = size.x * size.y * size.z;
	size_t outputDs = ods * timeSteps;
	float* od = new float[outputDs]();
	DeviceBuffer<float> outputBuffer = DeviceBuffer<float>(od, outputDs);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	
	// Get the number of SMs (streaming multiprocessors)
    	unsigned int numSMs = prop.multiProcessorCount;
    	// Get the maximum number of threads per SM
    	unsigned int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;

    	// Calculate the total number of threads
    	unsigned int launchWidth = (numSMs-1) * maxThreadsPerSM;
	
	// set number of threads and photons per thread:
	unsigned int threadphoton = pcount / launchWidth;	
	unsigned int oddphoton = pcount - threadphoton * launchWidth;

	printf("\n THE NUMBER OF INSIDE PRIMS BEFORE SENDING TO GPU: %d", num_inside_prims);
	// prepare optix pipeline parameters
	McxLaunchParams paras = McxLaunchParams(
	    size, boundary.handle(), curves.handle(), outputBuffer.handle(),
	    duration, timeSteps, pos, dir, media, startHandle,
	    startMedium, num_inside_prims, WIDTH_ADJ, threadphoton, oddphoton);

	DeviceBuffer<McxLaunchParams> paraBuffer =
	    DeviceBuffer<McxLaunchParams>(paras);

	std::cout << "Beginning simulation." << std::endl;
	std::chrono::steady_clock::time_point begin =
	    std::chrono::steady_clock::now();

	// OPTIX_CHECK reports if there were errors with the sim
	// optixLaunch launches photons, given a pipeline, stream, pipeline
	// params, pipeline param size, optix shader binding table, width of
	// computations, height of computations, and length of computations From
	// a GPU perspective this is done with a linear set of kernels of length
	// photoncount.
	OPTIX_CHECK(optixLaunch(this->devicePipeline.handle(), nullptr, \
			paraBuffer.handle(),
				sizeof(paras), &this->SBT, launchWidth,
				1, 1));

	outputBuffer.read(od);

	std::chrono::steady_clock::time_point end =
	    std::chrono::steady_clock::now();
	std::cout << "Simulation time = "
		  << std::chrono::duration_cast<std::chrono::milliseconds>(
			 end - begin)
			 .count()
		  << "[ms]" << std::endl;

	// normalize output:
	// temporary code:
	/*
	isnoramlized = true;
	if (isnormalized){
		mesh_normalize(mesh, cfg, cfg->energyabs, pcount, 0);
	}
	*/

	// loop through and write data to binary file
	for (int qq = 0; qq < ods; qq++) {
		if (od[qq] < 0) {
			float fluff = od[qq];
			throw std::runtime_error("There is an error ");
		}
	}

	for (int i = 0; i < timeSteps; i++) {
		std::ofstream out;
		out.open("attenuation_" + std::to_string(i) + ".bin",
			 std::ios::out | std::ios::binary);
		out.write((char*)(od + i * ods), ods * sizeof(float));
		out.close();
	}

	delete[] od;
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
