#include "shader_pipeline.h"

#include <optix.h>
#include <optix_stubs.h>
// added this
#include <optix_types.h>

#include "util.h"

namespace mcx {

	// default constructor
	ShaderPipeline::ShaderPipeline() {
		this->pipelineHandle = OptixPipeline();
		this->module = OptixModule();
		this->raygenProgramGroup = this->missProgramGroup = OptixProgramGroup();
		this->hitgroupProgramGroups = std::vector<OptixProgramGroup>();
	}

	// helper function for getting built-in intersection shader for sphere 
	static OptixModule getSphereModule(OptixDeviceContext ctx, OptixPipelineCompileOptions& pipeline_compile_options) {
		OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
		module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

		OptixBuiltinISOptions builtin_is_options = {};

		builtin_is_options.usesMotionBlur = false;
		builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
		OptixModule sphere_module;
		OPTIX_CHECK(optixBuiltinISModuleGet(ctx, &module_compile_options, &pipeline_compile_options,
			&builtin_is_options, &sphere_module));
		return sphere_module;
	}

	// parameterized constructor
	ShaderPipeline::ShaderPipeline(OptixDeviceContext ctx, std::string& ptx, ShaderFunctionSet& functions, int numPayloads, int numAttribs)
	{
		OptixPipelineCompileOptions pipelineOpts = {};
		pipelineOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
		pipelineOpts.numPayloadValues = numPayloads;
		pipelineOpts.numAttributeValues = numAttribs;
		pipelineOpts.pipelineLaunchParamsVariableName = functions.launchParamName.c_str();
		pipelineOpts.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

#ifndef NDEBUG
		pipelineOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
		pipelineOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif

		this->module = loadShaderModule(ptx, ctx, pipelineOpts);
		
		std::vector<OptixProgramGroup> groups;

// RAYGEN PROGRAM GROUP CREATION:
		OptixProgramGroupOptions ropts = {};
		OptixProgramGroupDesc rdesc = {};
		rdesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		rdesc.raygen.module = this->module;
		rdesc.raygen.entryFunctionName = functions.raygenFunction.c_str();
		OPTIX_CHECK(optixProgramGroupCreate(ctx,
				       	&rdesc,
				       	1,
				       	&ropts,
				       	nullptr,
				       	nullptr,
				       	&this->raygenProgramGroup));
		groups.push_back(this->raygenProgramGroup);
// MISS PROGRAM GROUP CREATION:
		OptixProgramGroupOptions mopts = {};
		OptixProgramGroupDesc mdesc = {};
		mdesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		mdesc.miss.module = this->module;
		mdesc.miss.entryFunctionName = functions.missFunction.c_str();
		OPTIX_CHECK(optixProgramGroupCreate(ctx, &mdesc, 1, &mopts, nullptr, nullptr, &this->missProgramGroup));
		groups.push_back(this->missProgramGroup);


// HIT PROGRAM GROUPS CREATION
		for (ShaderFunctionSet::HitGroup& hg : functions.hitgroupFunctions)
		{
			OptixProgramGroupOptions hopts = {};
			OptixProgramGroupDesc hdesc = {};
			hdesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			if (hg.closestHitFunction != "") {
				hdesc.hitgroup.moduleCH = this->module;
				hdesc.hitgroup.entryFunctionNameCH = hg.closestHitFunction.c_str();
			}
			if (hg.intersectionFunction != "") {
				if (hg.intersectionFunction == "__BUILTIN_SPHERE__") {
					hdesc.hitgroup.moduleIS = getSphereModule(ctx, pipelineOpts);
					hdesc.hitgroup.entryFunctionNameIS = nullptr;
				}
				else {
					hdesc.hitgroup.moduleIS = this->module;
					hdesc.hitgroup.entryFunctionNameIS = hg.intersectionFunction.c_str();
				}
			}

			OptixProgramGroup hpg;
			OPTIX_CHECK(optixProgramGroupCreate(ctx, &hdesc, 1, &hopts, nullptr, nullptr, &hpg));
			groups.push_back(hpg);
			this->hitgroupProgramGroups.push_back(hpg);
		}

		this->pipelineHandle = createPipeline(ctx, pipelineOpts, groups.data(), groups.size());
	}

	// Move constructor
	ShaderPipeline& ShaderPipeline::operator= (ShaderPipeline&& a) {
		this->pipelineHandle = a.pipelineHandle;
		this->raygenProgramGroup = a.raygenProgramGroup;
		this->missProgramGroup = a.missProgramGroup;
		this->hitgroupProgramGroups = a.hitgroupProgramGroups;
		this->module = a.module;

		a.pipelineHandle = OptixPipeline();
		a.raygenProgramGroup = OptixProgramGroup();
		a.missProgramGroup = OptixProgramGroup();
		a.hitgroupProgramGroups = std::vector<OptixProgramGroup>();
		a.module = OptixModule();

		return *this;
	}

	OptixProgramGroup& ShaderPipeline::raygenProgram()
	{
		return this->raygenProgramGroup;
	}

	OptixProgramGroup& ShaderPipeline::missProgram()
	{
		return this->missProgramGroup;
	}

	std::vector<OptixProgramGroup>& ShaderPipeline::hitgroupPrograms() {
		return this->hitgroupProgramGroups;
	}

	// loads shader module using the ptx string and "optixModuleCreateFromPTX" function
	OptixModule ShaderPipeline::loadShaderModule(std::string ptx, OptixDeviceContext ctx, OptixPipelineCompileOptions& pipelineOpts)
	{
		OptixModuleCompileOptions opts = {};
		opts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;

#ifndef NDEBUG
		opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
		opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#else
		opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
		opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#endif

		OptixModule mod;
		OPTIX_CHECK(optixModuleCreateFromPTX(ctx, &opts, &pipelineOpts, ptx.c_str(), ptx.length(), nullptr, nullptr, &mod));
		return mod;
	}

	OptixPipeline ShaderPipeline::createPipeline(OptixDeviceContext ctx, OptixPipelineCompileOptions& pipelineOpts, OptixProgramGroup* groups, size_t numGroups)
	{
		OptixPipelineLinkOptions opts = {};
		opts.maxTraceDepth = 1;

#ifndef NDEBUG
		opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
		opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

		OptixPipeline pipeline;
		OPTIX_CHECK(optixPipelineCreate(ctx, &pipelineOpts, &opts, groups, numGroups, nullptr, nullptr, &pipeline));
		return pipeline;
	}

	ShaderPipeline::~ShaderPipeline()
	{
		if (this->pipelineHandle != OptixPipeline()) {
			optixPipelineDestroy(this->pipelineHandle);
			optixProgramGroupDestroy(this->raygenProgramGroup);
			optixProgramGroupDestroy(this->missProgramGroup);
			for (OptixProgramGroup& g : this->hitgroupProgramGroups)
			{
				optixProgramGroupDestroy(g);
			}

			optixModuleDestroy(this->module);
		}
	}

	OptixPipeline& ShaderPipeline::handle()
	{
		return this->pipelineHandle;
	}
}
