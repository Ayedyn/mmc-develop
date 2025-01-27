#pragma once

#include <string>
#include <optix.h>
#include <vector>

namespace mcx {
	class ShaderFunctionSet {
	public:
		struct HitGroup {
			std::string closestHitFunction, intersectionFunction;

			HitGroup(std::string chf, std::string iff) {
				this->closestHitFunction = chf;
				this->intersectionFunction = iff;
			}
		};

		std::string raygenFunction, missFunction, launchParamName;
		std::vector<HitGroup> hitgroupFunctions;

		ShaderFunctionSet(std::string rf, std::string mf, std::vector<HitGroup> hgfs, std::string lpn) 
			: raygenFunction(rf),
			  missFunction(mf),
			  launchParamName(lpn),
			  hitgroupFunctions(hgfs)
		{
		}
	};

	class ShaderPipeline {
	private:
		OptixPipeline pipelineHandle;
		OptixModule module;
		OptixProgramGroup raygenProgramGroup, missProgramGroup;
		std::vector<OptixProgramGroup> hitgroupProgramGroups;

		ShaderPipeline(const ShaderPipeline&) = default;
		ShaderPipeline& operator= (const ShaderPipeline&) = default;

		static OptixModule loadShaderModule(std::string ptx, OptixDeviceContext ctx, OptixPipelineCompileOptions& pipelineOpts);
		static OptixPipeline createPipeline(OptixDeviceContext ctx, OptixPipelineCompileOptions& pipelineOpts, OptixProgramGroup* groups, size_t numGroups);

	public:
		ShaderPipeline();
		ShaderPipeline(OptixDeviceContext ctx, std::string& ptx, ShaderFunctionSet& functions, int numPayloads, int numAttribs);
		ShaderPipeline(ShaderPipeline&& src) = default;
		ShaderPipeline& operator= (ShaderPipeline&&);

		OptixPipeline& handle();
		OptixProgramGroup& raygenProgram();
		OptixProgramGroup& missProgram();
		std::vector<OptixProgramGroup>& hitgroupPrograms();

		~ShaderPipeline();
	};
}
