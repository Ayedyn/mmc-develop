#pragma once

#include <stdint.h>
#include <optix.h>

#include "medium.h"
#include "shader_pipeline.h"
#include "shader_binding_table.h"
#include "tetrahedral_mesh.h"

namespace mcx {
	class McxContext {
	private:
		
		OptixDeviceContext optixContext;
		ShaderPipeline devicePipeline;
		ShaderBindingTable<void*,void*,void*> deviceSbt;

		McxContext(const McxContext&) = default;
		McxContext& operator= (const McxContext&) = default;

		void onMessageReceived(uint32_t level, const char* tag, const char* message);

		static void messageHandler(uint32_t level, const char* tag, const char* message, void* data);

	public:
		McxContext();
		McxContext(McxContext&& src);

		void simulate(TetrahedralMesh& mesh, uint3 size, std::vector<Medium> media, uint32_t pcount, float duration, uint32_t timeSteps, float3 pos, float3 dir);

		~McxContext();
	};
}
