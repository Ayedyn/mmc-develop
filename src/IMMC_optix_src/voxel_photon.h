#pragma once

#include <cuda.h>
#include <optix.h>
#include <stdio.h>
#include <vector_types.h>

#include "random.cu"

namespace mcx {
	struct VoxelPhoton {
	public:
		float3 origin;
		float3 direction;
		float scatteringLengthLeft;
		float elapsedTime;
		float scatteringEventCount;
		Random random;
		float energy;
		OptixTraversableHandle manifold;
		uint32_t currentMedium;

		__forceinline__ __device__ void print() {
			printf("[%d] or: (%f %f %f), di: (%f %f %f) %f, sl: %f, rn: (%d %d %d %d), en: %f, vox: %p\n",
				optixGetLaunchIndex().x,
				this->origin.x, this->origin.y, this->origin.z,
				this->direction.x, this->direction.y, this->direction.z, length(this->direction),
				this->scatteringLengthLeft,
				this->random.intSeed.x, this->random.intSeed.y, this->random.intSeed.z, this->random.intSeed.w,
				this->energy,
				this->manifold);
		}
	};

	struct VoxelPhotonPayload {
		float3 origin;
		float3 direction;
		float scatteringLengthLeft;
		Random random;
		float elapsedTime;
		float energy;
		OptixTraversableHandle manifold;
		int currentMedium;

		__forceinline__ __device__ void print() {
			printf("or: (%f %f %f), di: (%f %f %f) %f, sl: %f, rn: (%d %d %d %d), en: %f, vox: %p\n",
				this->origin.x, this->origin.y, this->origin.z,
				this->direction.x, this->direction.y, this->direction.z, length(this->direction),
				this->scatteringLengthLeft,
				this->random.intSeed.x, this->random.intSeed.y, this->random.intSeed.z, this->random.intSeed.w,
				this->energy,
				this->manifold);
		}
	};

	struct VoxelRayAttribute {
		int3 voxel;
		float distance;
	};
}



__device__ __forceinline__ float getScatteringLengthLeftPayload() {
	int value = optixGetPayload_6();
	return *((float*)&value);
}

__device__ __forceinline__ uint32_t getMediumIDPayload() {
	return optixGetPayload_15();
}

__device__ __forceinline__ void getPayload(mcx::VoxelPhotonPayload& payload) {
	int3 origin_data = make_int3(optixGetPayload_0(), optixGetPayload_1(), optixGetPayload_2());
	payload.origin = make_float3(*(float*)&origin_data.x, *(float*)&origin_data.y, *(float*)&origin_data.z);
	payload.direction = optixGetWorldRayDirection();
	unsigned int pf = optixGetPayload_6();
	payload.scatteringLengthLeft = *((float*)&pf);
	payload.random = mcx::Random(make_uint4(optixGetPayload_7(), optixGetPayload_8(), optixGetPayload_9(), optixGetPayload_10()));
	pf = optixGetPayload_11();
	payload.elapsedTime = *((float*)&pf);
	pf = optixGetPayload_12();
	payload.energy = *((float*)&pf);
	payload.manifold = ((OptixTraversableHandle)optixGetPayload_13()) | (((OptixTraversableHandle)optixGetPayload_14()) << 32);
	payload.currentMedium = optixGetPayload_15();
}

__device__ __forceinline__ void setPayload(mcx::VoxelPhotonPayload payload) {
	optixSetPayload_0(*((uint32_t*)&payload.origin.x));
	optixSetPayload_1(*((uint32_t*)&payload.origin.y));
	optixSetPayload_2(*((uint32_t*)&payload.origin.z));
	optixSetPayload_3(*((uint32_t*)&payload.direction.x));
	optixSetPayload_4(*((uint32_t*)&payload.direction.y));
	optixSetPayload_5(*((uint32_t*)&payload.direction.z));
	optixSetPayload_6(*((uint32_t*)&payload.scatteringLengthLeft));
	optixSetPayload_7(payload.random.intSeed.x);
	optixSetPayload_8(payload.random.intSeed.y);
	optixSetPayload_9(payload.random.intSeed.z);
	optixSetPayload_10(payload.random.intSeed.w);
	optixSetPayload_11(*((uint32_t*)&payload.elapsedTime));
	optixSetPayload_12(*(uint32_t*)&payload.energy);
	optixSetPayload_13((uint32_t)(payload.manifold & std::numeric_limits<uint32_t>::max()));
	optixSetPayload_14((uint32_t)((payload.manifold >> 32) & std::numeric_limits<uint32_t>::max()));
	optixSetPayload_15(payload.currentMedium);
}

__device__ __forceinline__ void getAttribute(mcx::VoxelRayAttribute& payload) {
	payload.voxel = make_int3(optixGetAttribute_0(), optixGetAttribute_1(), optixGetAttribute_2());
	*((unsigned int*)&payload.distance) = optixGetAttribute_3();
}
