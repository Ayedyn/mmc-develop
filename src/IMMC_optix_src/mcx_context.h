#pragma once

#include <stdint.h>
#include <optix.h>

#include "mmc_mesh.h"
#include "mmc_optix_launchparam.h"
#include "shader_pipeline.h"
#include "tetrahedral_mesh.h"

namespace mcx {
	class McxContext {
	private:
		
		OptixDeviceContext optixContext;
		ShaderPipeline devicePipeline;
        OptixShaderBindingTable SBT;

		McxContext(const McxContext&) = default;
		McxContext& operator= (const McxContext&) = default;

		void onMessageReceived(uint32_t level, const char* tag, const char* message);

		static void messageHandler(uint32_t level, const char* tag, const char* message, void* data);

        template<typename T>
        struct SbtRecord
        {
            __align__(OPTIX_SBT_RECORD_ALIGNMENT)
            char header[OPTIX_SBT_RECORD_HEADER_SIZE];
            T data;

            SbtRecord(T t);
        };


	public:
		McxContext();
		McxContext(McxContext&& src);
		void simulate(tetmesh* mesh, uint3 size,
                std::vector<Medium> media, uint32_t pcount,
                float duration, uint32_t timeSteps,
                float3 pos, float3 dir, mcconfig* cfg);

		~McxContext();
	};

    // sets up an SbtRecord
    template<typename T>
    McxContext::SbtRecord<T>::SbtRecord(T t) {
        this->data = t;
    }



}
