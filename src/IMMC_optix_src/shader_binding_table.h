#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <optix.h>

#include "device_buffer.h"
#include "shader_pipeline.h"
#include "util.h"

namespace mcx {

	template<typename R, typename M, typename H>
	class ShaderBindingTable {
	private:
        template<typename T>
        struct SbtRecord
        {
            __align__(OPTIX_SBT_RECORD_ALIGNMENT)
            char header[OPTIX_SBT_RECORD_HEADER_SIZE];
            T data;

            SbtRecord(T t);
        };

		DeviceBuffer<SbtRecord<R>> raygenRecord;
		DeviceBuffer<SbtRecord<M>> missRecord;
		DeviceBuffer<SbtRecord<H>> hitgroupRecords;
        OptixShaderBindingTable shaderTable;

        ShaderBindingTable(const ShaderBindingTable&) = default;
        ShaderBindingTable& operator= (const ShaderBindingTable&) = default;

    public:
		ShaderBindingTable();
        ShaderBindingTable(ShaderPipeline& pipeline, R r, M m, std::vector<H> h);
		ShaderBindingTable(ShaderBindingTable&& src) = default;
		ShaderBindingTable& operator= (ShaderBindingTable&&);

        OptixShaderBindingTable& table() {
            return this->shaderTable;
        }
	};
}

namespace mcx {
	template<typename R, typename M, typename H>
	template<typename T>
	ShaderBindingTable<R, M, H>::SbtRecord<T>::SbtRecord(T t) {
		this->data = t;
	}

	template<typename R, typename M, typename H>
	ShaderBindingTable<R, M, H>::ShaderBindingTable()
	{
		this->raygenRecord = DeviceBuffer<SbtRecord<R>>();
		this->missRecord = DeviceBuffer<SbtRecord<M>>();
		this->hitgroupRecords = DeviceBuffer<SbtRecord<H>>();
		this->shaderTable = OptixShaderBindingTable();
	}

	// organizes raygen (R), miss (M), and hit (H) programs into a shader binding table.
	// Importantly, there is only 1 raygen and 1 hit program,
	// whereas there is an arbitrary amount of hitgroup programs in a vector
	// for different geometries such as triangle, sphere, and curve
	template<typename R, typename M, typename H>
	ShaderBindingTable<R, M, H>::ShaderBindingTable(ShaderPipeline& pipeline, R r, M m, std::vector<H> h) {
	
		// raygen	
		SbtRecord<R> rrec = SbtRecord<R>(r);
		OPTIX_CHECK(optixSbtRecordPackHeader(pipeline.raygenProgram(), &rrec));
		this->raygenRecord = DeviceBuffer<SbtRecord<R>>(rrec);
		
		// miss
		SbtRecord<M> mrec = SbtRecord<M>(m);
		OPTIX_CHECK(optixSbtRecordPackHeader(pipeline.missProgram(), &mrec));
		this->missRecord = DeviceBuffer<SbtRecord<M>>(mrec);

		// hit programs
		std::vector<SbtRecord<H>> grecs;

		if (h.size() != pipeline.hitgroupPrograms().size()) {
			throw std::runtime_error("Hitgroup data count was not the same as pipeline hitgroup count");
		}

		for (int i = 0; i < h.size(); i++)
		{
			grecs.push_back(SbtRecord<H>(h[i]));
			OPTIX_CHECK(optixSbtRecordPackHeader(pipeline.hitgroupPrograms()[i], &grecs[i]));
		}

		this->hitgroupRecords = DeviceBuffer<SbtRecord<H>>(grecs.data(), grecs.size());

		OptixShaderBindingTable sbt = {};

		sbt.raygenRecord = this->raygenRecord.handle();
		sbt.missRecordBase = this->missRecord.handle();
		sbt.missRecordStrideInBytes = sizeof(mrec);
		sbt.missRecordCount = 1;
		sbt.hitgroupRecordBase = this->hitgroupRecords.handle();
		sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<H>);
		sbt.hitgroupRecordCount = grecs.size();

		this->shaderTable = sbt;
	}

	template<typename R, typename M, typename H>
	ShaderBindingTable<R, M, H>& ShaderBindingTable<R, M, H>::operator=(ShaderBindingTable&& a)
	{
		this->raygenRecord = std::move(a.raygenRecord);
		this->missRecord = std::move(a.missRecord);
		this->hitgroupRecords = std::move(a.hitgroupRecords);
		this->shaderTable = a.shaderTable;

		a.raygenRecord = DeviceBuffer<SbtRecord<R>>();
		a.missRecord = DeviceBuffer<SbtRecord<M>>();
		a.hitgroupRecords = DeviceBuffer<SbtRecord<H>>();

		return *this;
	}
}
