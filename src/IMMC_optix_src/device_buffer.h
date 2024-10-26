#pragma once

#include <cuda.h>

#include "util.h"

namespace mcx {
	template<typename T>
	class DeviceBuffer {
	private:
		CUdeviceptr bufferHandle;
		size_t length;

		DeviceBuffer(const DeviceBuffer&) = default;
		DeviceBuffer& operator= (const DeviceBuffer&) = default;

	public:
		DeviceBuffer();
		DeviceBuffer(size_t count);
		DeviceBuffer(T& data);
		DeviceBuffer(T* data);
		DeviceBuffer(T* data, size_t count);
		DeviceBuffer(DeviceBuffer&& src) = default;
		DeviceBuffer& operator= (DeviceBuffer&&);

		CUdeviceptr& handle();
		size_t count();
		void read(T* location);

		~DeviceBuffer();
	};

	typedef DeviceBuffer<uint8_t> DeviceByteBuffer;

	template<typename T>
	DeviceBuffer<T>::DeviceBuffer()
	{
		this->bufferHandle = CUdeviceptr();
		this->length = 0;
	}

	template<typename T>
	DeviceBuffer<T>::DeviceBuffer(size_t count)
	{
		this->length = count;
		count *= sizeof(T);
		CUDA_CHECK(cudaMalloc((void**)&this->bufferHandle, count));
	}

	template<typename T>
	DeviceBuffer<T>::DeviceBuffer(T& data) : DeviceBuffer(&data)
	{
	}

	template<typename T>
	DeviceBuffer<T>::DeviceBuffer(T* data) : DeviceBuffer(data, 1)
	{
	}

	template<typename T>
	DeviceBuffer<T>::DeviceBuffer(T* data, size_t count)
	{
		this->length = count;
		count *= sizeof(T);
		CUDA_CHECK(cudaMalloc((void**)&this->bufferHandle, count));
		CUDA_CHECK(cudaMemcpy((void*)this->bufferHandle, data, count, cudaMemcpyHostToDevice));
	}

	template<typename T>
	DeviceBuffer<T>& DeviceBuffer<T>::operator=(DeviceBuffer&& a)
	{
		this->bufferHandle = a.bufferHandle;
		this->length = a.length;
		a.bufferHandle = CUdeviceptr();
		a.length = 0;
		return *this;
	}

	template<typename T>
	CUdeviceptr& DeviceBuffer<T>::handle()
	{
		return this->bufferHandle;
	}

	template<typename T>
	size_t DeviceBuffer<T>::count()
	{
		return this->length;
	}

	template<typename T>
	void DeviceBuffer<T>::read(T* location)
	{
		CUDA_CHECK(cudaMemcpy((void*)location, (void*)this->bufferHandle, this->length * sizeof(T), cudaMemcpyDeviceToHost));
	}

	template<typename T>
	DeviceBuffer<T>::~DeviceBuffer()
	{
		//cudaFree((void*)this->bufferHandle);
	}

}
