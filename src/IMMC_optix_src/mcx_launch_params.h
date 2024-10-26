#pragma once

#include <cuda.h>
#include <optix.h>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "mmc_optix_launchparam.h"
#include "surface_boundary.h"

#ifndef IMPLICIT_CURVE_H
#define IMPLICIT_CURVE_H

// Contents of implicit_curve.h go here
#include "implicit_curve.h"

#endif

namespace mcx {
struct McxLaunchParams {
       public:
	static constexpr int MAX_MEDIA = 16384 / sizeof(Medium);
	uint3 dataSize;
	CUdeviceptr surfaceBoundaries;
	CUdeviceptr curveData;
	CUdeviceptr outputBuffer;

	float simulationDuration;
	int timeSteps;
	float inverseTimeStep;
	float3 emitterPosition;
	float3 emitterDirection;
	Medium medium[MAX_MEDIA];
	unsigned int num_inside_prims;
	float WIDTH_ADJ;
	unsigned int threadphoton;
	unsigned int oddphoton;

	OptixTraversableHandle startManifold;
	uint32_t startMedium;

	McxLaunchParams() = default;

	McxLaunchParams(uint3 inSize, CUdeviceptr boundaries,
			CUdeviceptr curves, CUdeviceptr outBuffer, float sd,
			int ts, float3 ep, float3 ed,
			std::vector<Medium> media,
			OptixTraversableHandle startManifold,
			uint32_t startMedium, unsigned int num_inside_prims, float width_adj,
			unsigned int threadphoton, unsigned int oddphoton) {
		this->dataSize = inSize;
		this->surfaceBoundaries = boundaries;
		this->curveData = curves;
		this->outputBuffer = outBuffer;
		this->simulationDuration = sd;
		this->inverseTimeStep = (float)ts / sd;
		this->startManifold = startManifold;
		this->startMedium = startMedium;
		this->timeSteps = ts;
		this->num_inside_prims = num_inside_prims;
		this->WIDTH_ADJ = width_adj;

		while (this->inverseTimeStep * sd > (float)ts) {
			(*((int*)&this->inverseTimeStep))--;
		}

		this->emitterPosition = ep;
		this->emitterDirection = ed;

		if (media.size() > MAX_MEDIA) {
			throw new std::runtime_error(
			    "Maximum number of supported media exceeded.");
		}

		for (int i = 0; i < media.size(); i++) {
			this->medium[i] = media[i];
		}
		for (int i = media.size(); i < MAX_MEDIA; i++) {
			Medium default_medium = {0.0, 0.0, 1.0, 1.0};
            this->medium[i] = default_medium;
		}
		this->threadphoton = threadphoton;
		this->oddphoton = oddphoton;
	}
};
}  // namespace mcx
