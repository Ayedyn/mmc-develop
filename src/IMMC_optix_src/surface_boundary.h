#pragma once

#include <stdint.h>
#include <optix.h>

namespace mcx {
	struct SurfaceBoundary {
	public:
		uint32_t medium;
		OptixTraversableHandle manifold;
	};
}
