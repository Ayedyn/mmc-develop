// mmc_optix_host.cpp : This file contains the 'main' function. Program
// execution begins and ends there.
//

#include <sutil/vec_math.h>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "device_buffer.h"
#include "mcx_context.h"
#include "medium.h"

#include "tetrahedral_mesh.h"

// Test code generating cube shaped mesh with curve inside it
mcx::TetrahedralMesh sphere_curve_test() {
	std::vector<uint4> elements = {
	    make_uint4(1, 2, 8, 4), make_uint4(1, 3, 4, 8),
	    make_uint4(1, 2, 6, 8), make_uint4(1, 5, 8, 6),
	    make_uint4(1, 3, 8, 7), make_uint4(1, 5, 7, 8),
	};

	std::vector<mcx::Tetrahedron> tets = std::vector<mcx::Tetrahedron>();

	uint32_t k = 0;
	for (uint4 elem : elements) {
		uint32_t label = 0;

		tets.push_back(
		    {make_uint4(elem.x - 1, elem.y - 1, elem.z - 1, elem.w - 1),
		     label});

		k++;
	}
	// define spheres to simulate
	std::vector<mcx::ImplicitSphere> spheres =
	    std::vector<mcx::ImplicitSphere>();
	spheres.push_back({make_float3(30, 30, 40), 10});
	// define curves to simulate
	std::vector<mcx::ImplicitCurve> curves =
	    std::vector<mcx::ImplicitCurve>();
	curves.push_back({make_float3(10, 30, 15), make_float3(40, 30, 15), 5});

	mcx::TetrahedralMesh mesh = mcx::TetrahedralMesh(
	    std::vector<float3>({make_float3(0, 0, 0), make_float3(60, 0, 0),
				 make_float3(0, 60, 0), make_float3(60, 60, 0),
				 make_float3(0, 0, 60), make_float3(60, 0, 60),
				 make_float3(0, 60, 60),
				 make_float3(60, 60, 60)}),
	    tets, spheres, curves);

	return mesh;
}

// main function
int main() {
	try {
		//mcx::TetrahedralMesh mesh = immc_sphere_benchmark();
		
		mcx::TetrahedralMesh mesh = sphere_curve_test();

		std::vector<mcx::Medium> media = {
		    mcx::Medium(0.000458, 0.356541, 0.9, 1.37),
		    mcx::Medium(0.230543, 0.093985, 0.9, 1.37)};
		    //mcx::Medium(0.0458, 35.6541, 0.9, 1.37),
		    //mcx::Medium(23.0543, 9.3985, 0.9, 1.37)};


		uint3 size = make_uint3(60, 60, 60);

		mcx::McxContext ctx = mcx::McxContext();

		// bitshift operation to keep photon counts in factors of 2
		// 1<<18 is 262144
		// 1<<28 is 268435456

		constexpr uint32_t photon_count = 100000000;//1<<26;
		constexpr float duration = 0.005;
		constexpr uint32_t timesteps = 1;

		float3 srcpos = make_float3(30, 30, 0.001);
		float3 srcdir = make_float3(0, 0, 1);

		ctx.simulate(mesh, size, media, photon_count, duration,
			     timesteps, srcpos, srcdir);

		std::cout << "\nSimulation complete.\n";
	} catch (std::runtime_error err) {
		std::cout << "A runtime error occured.\n";
		std::cout << err.what();
	}

	std::getchar();
}
