// mmc_optix_host.cpp : This file contains the 'main' function. Program
// execution begins and ends there.
//

#include <sutil/vec_math.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>

#include "device_buffer.h"
#include "mcx_context.h"
#include "mmc_optix_launchparam.h"

#include "tetrahedral_mesh.h"

struct MCX_clock {
    std::chrono::system_clock::time_point starttime;
    MCX_clock() : starttime(std::chrono::system_clock::now()) {}
    double elapse() {
        std::chrono::duration<double> elapsetime = (std::chrono::system_clock::now() - starttime);
        return elapsetime.count() * 1000.;
    }
};

// helps read binary files for meshes
static std::vector<char> read_all_bytes(char const* filename) {
	std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
	std::ifstream::pos_type pos = ifs.tellg();

	if (pos == 0) {
		return std::vector<char>{};
	}

	std::vector<char> result(pos);

	ifs.seekg(0, std::ios::beg);
	ifs.read(&result[0], pos);

	return result;
}

// helper function: subtracts each element of a 4 dimensional vector
static uint4 sub(uint4 a, uint4 b) {
	return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

// Head atlas tetrahedral mesh
mcx::TetrahedralMesh input_mesh() {
	int elements = 335713;
	int nodes = 59225;

	std::vector<char> bytes = read_all_bytes("input_head_atlas.bin");

	std::vector<float3> nodeList = std::vector<float3>();
	std::vector<mcx::Tetrahedron> tets = std::vector<mcx::Tetrahedron>();

	for (int i = 0; i < nodes; i++) {
		nodeList.push_back(((float3*)bytes.data())[i]);
	}

	for (int i = 0; i < elements; i++) {
		uint16_t* p =
		    (uint16_t*)(bytes.data() + nodes * (3 * sizeof(float)) +
				i * 4 * sizeof(uint16_t));
		tets.push_back(
		    {sub(make_uint4(*p, *(p + 1), *(p + 2), *(p + 3)),
			 make_uint4(1, 1, 1, 1)),
		     (uint32_t)((
			 (uint8_t*)(bytes.data() + nodes * (3 * sizeof(float)) +
				    elements * 4 * sizeof(uint16_t)))[i]) -
			 1});
	}

	return mcx::TetrahedralMesh(nodeList, tets,
				    std::vector<mcx::ImplicitSphere>(),
				    std::vector<mcx::ImplicitCurve>());
}

mcx::TetrahedralMesh basic_cube_test() {
	std::vector<uint4> elements = {
	    make_uint4(1, 2, 8, 4), make_uint4(1, 3, 4, 8),
	    make_uint4(1, 2, 6, 8), make_uint4(1, 5, 8, 6),
	    make_uint4(1, 3, 8, 7), make_uint4(1, 5, 7, 8),
	};

	std::vector<mcx::Tetrahedron> tets = std::vector<mcx::Tetrahedron>();

	uint32_t k = 0;
	for (uint4 elem : elements) {
		uint32_t label = 1;

		tets.push_back(
		    {make_uint4(elem.x - 1, elem.y - 1, elem.z - 1, elem.w - 1),
		     label});

		k++;
	}
	// define placeholder spheres to simulate
	std::vector<mcx::ImplicitSphere> spheres =
	    std::vector<mcx::ImplicitSphere>();
	// define curves to simulate
	std::vector<mcx::ImplicitCurve> curves =
	    std::vector<mcx::ImplicitCurve>();

	mcx::TetrahedralMesh mesh = mcx::TetrahedralMesh(
	    std::vector<float3>({make_float3(0, 0, 0), make_float3(60, 0, 0),
				 make_float3(0, 60, 0), make_float3(60, 60, 0),
				 make_float3(0, 0, 60), make_float3(60, 0, 60),
				 make_float3(0, 60, 60),
				 make_float3(60, 60, 60)}),
	    tets, spheres, curves);

	return mesh;
}

// two layered cube
mcx::TetrahedralMesh two_layered_cube_test() {
	std::vector<uint4> elements = {
	    make_uint4(1, 2, 8, 4), make_uint4(5, 6, 12, 8),
	    make_uint4(1, 3, 4, 8), make_uint4(5, 7, 8, 12),
	    make_uint4(1, 2, 6, 8), make_uint4(5, 6, 10, 12),
        make_uint4(1, 5, 8, 6), make_uint4(5, 9, 12, 10),
        make_uint4(1, 3, 8, 7), make_uint4(5, 7, 12, 11),
        make_uint4(1, 5, 7, 8), make_uint4(5, 9, 11, 12),
	};

	std::vector<mcx::Tetrahedron> tets = std::vector<mcx::Tetrahedron>();

	uint32_t k = 0;
	for (uint4 elem : elements) {	
        uint32_t label = 1+(k%2)*2;

		tets.push_back(
		    {make_uint4(elem.x - 1, elem.y - 1, elem.z - 1, elem.w - 1),
		     label});

		k++;
	}
	// define placeholder spheres to simulate
	std::vector<mcx::ImplicitSphere> spheres =
	    std::vector<mcx::ImplicitSphere>();
	// define curves to simulate
	std::vector<mcx::ImplicitCurve> curves =
	    std::vector<mcx::ImplicitCurve>();

	mcx::TetrahedralMesh mesh = mcx::TetrahedralMesh(
	    std::vector<float3>({make_float3(0, 0, 0), make_float3(60, 0, 0),
				 make_float3(0, 60, 0), make_float3(60, 60, 0),
				 make_float3(0, 0, 30), make_float3(60, 0, 30),
				 make_float3(0, 60, 30),
				 make_float3(60, 60, 30), make_float3(0, 0, 60),
                 make_float3(60, 0, 60), make_float3(0, 60, 60),
                 make_float3(60, 60, 60)
                 }),
	    tets, spheres, curves);

	return mesh;
}


// two layered cube with embedded capsule
mcx::TetrahedralMesh two_layered_cube_capsule_test() {
	std::vector<uint4> elements = {
	    make_uint4(1, 2, 8, 4), make_uint4(5, 6, 12, 8),
	    make_uint4(1, 3, 4, 8), make_uint4(5, 7, 8, 12),
	    make_uint4(1, 2, 6, 8), make_uint4(5, 6, 10, 12),
        make_uint4(1, 5, 8, 6), make_uint4(5, 9, 12, 10),
        make_uint4(1, 3, 8, 7), make_uint4(5, 7, 12, 11),
        make_uint4(1, 5, 7, 8), make_uint4(5, 9, 11, 12),
	};

	std::vector<mcx::Tetrahedron> tets = std::vector<mcx::Tetrahedron>();

	uint32_t k = 0;
	for (uint4 elem : elements) {	
        uint32_t label = 1+(k%2)*2;

		tets.push_back(
		    {make_uint4(elem.x - 1, elem.y - 1, elem.z - 1, elem.w - 1),
		     label});

		k++;
	}
	// define placeholder spheres to simulate
	std::vector<mcx::ImplicitSphere> spheres =
	    std::vector<mcx::ImplicitSphere>();
    //placeholder sphere
    spheres.push_back({make_float3(60, 60, 60), 0.0001});
	// define curves to simulate
	std::vector<mcx::ImplicitCurve> curves =
	    std::vector<mcx::ImplicitCurve>();

    //capsule:
	curves.push_back({make_float3(30, 15, 30), make_float3(30, 45, 30), 5});
    
	mcx::TetrahedralMesh mesh = mcx::TetrahedralMesh(
	    std::vector<float3>({make_float3(0, 0, 0), make_float3(60, 0, 0),
				 make_float3(0, 60, 0), make_float3(60, 60, 0),
				 make_float3(0, 0, 30), make_float3(60, 0, 30),
				 make_float3(0, 60, 30),
				 make_float3(60, 60, 30), make_float3(0, 0, 60),
                 make_float3(60, 0, 60), make_float3(0, 60, 60),
                 make_float3(60, 60, 60)
                 }),
	    tets, spheres, curves);

	return mesh;
}


// Test code generating cube shaped mesh with curve inside it
mcx::TetrahedralMesh sphereshaped_curve_test() {
	std::vector<uint4> elements = {
	    make_uint4(1, 2, 8, 4), make_uint4(1, 3, 4, 8),
	    make_uint4(1, 2, 6, 8), make_uint4(1, 5, 8, 6),
	    make_uint4(1, 3, 8, 7), make_uint4(1, 5, 7, 8),
	};

	std::vector<mcx::Tetrahedron> tets = std::vector<mcx::Tetrahedron>();

	uint32_t k = 0;
	for (uint4 elem : elements) {
		uint32_t label = 1;

		tets.push_back(
		    {make_uint4(elem.x - 1, elem.y - 1, elem.z - 1, elem.w - 1),
		     label});

		k++;
	}
	// define placeholder spheres to simulate
	std::vector<mcx::ImplicitSphere> spheres =
	    std::vector<mcx::ImplicitSphere>();
	// define curves to simulate
	std::vector<mcx::ImplicitCurve> curves =
	    std::vector<mcx::ImplicitCurve>();
	// sphere-like curve
		curves.push_back({make_float3(30, 30, 15), make_float3(30.00001, 30, 15), 10});

	mcx::TetrahedralMesh mesh = mcx::TetrahedralMesh(
	    std::vector<float3>({make_float3(0, 0, 0), make_float3(60, 0, 0),
				 make_float3(0, 60, 0), make_float3(60, 60, 0),
				 make_float3(0, 0, 60), make_float3(60, 0, 60),
				 make_float3(0, 60, 60),
				 make_float3(60, 60, 60)}),
	    tets, spheres, curves);

	return mesh;
}

// Test code generating cube shaped mesh with capsule inside it
mcx::TetrahedralMesh basic_capsule_test() {
	std::vector<uint4> elements = {
	    make_uint4(1, 2, 8, 4), make_uint4(1, 3, 4, 8),
	    make_uint4(1, 2, 6, 8), make_uint4(1, 5, 8, 6),
	    make_uint4(1, 3, 8, 7), make_uint4(1, 5, 7, 8),
	};

	std::vector<mcx::Tetrahedron> tets = std::vector<mcx::Tetrahedron>();

	uint32_t k = 0;
	for (uint4 elem : elements) {
		uint32_t label = 1;

		tets.push_back(
		    {make_uint4(elem.x - 1, elem.y - 1, elem.z - 1, elem.w - 1),
		     label});

		k++;
	}
	// define placeholder spheres to simulate
	std::vector<mcx::ImplicitSphere> spheres =
	    std::vector<mcx::ImplicitSphere>();
    //placeholder sphere
    spheres.push_back({make_float3(60, 60, 60), 0.0001});
	// define curves to simulate
	std::vector<mcx::ImplicitCurve> curves =
	    std::vector<mcx::ImplicitCurve>();

    //capsule:
	curves.push_back({make_float3(30, 15, 30), make_float3(30, 45, 30), 10});
	mcx::TetrahedralMesh mesh = mcx::TetrahedralMesh(
	    std::vector<float3>({make_float3(0, 0, 0), make_float3(60, 0, 0),
				 make_float3(0, 60, 0), make_float3(60, 60, 0),
				 make_float3(0, 0, 60), make_float3(60, 0, 60),
				 make_float3(0, 60, 60),
				 make_float3(60, 60, 60)}),
	    tets, spheres, curves);

	return mesh;
}


// Test code generating cube shaped mesh with capsule inside it
mcx::TetrahedralMesh overlapping_capsule_test() {
	std::vector<uint4> elements = {
	    make_uint4(1, 2, 8, 4), make_uint4(1, 3, 4, 8),
	    make_uint4(1, 2, 6, 8), make_uint4(1, 5, 8, 6),
	    make_uint4(1, 3, 8, 7), make_uint4(1, 5, 7, 8),
	};

	std::vector<mcx::Tetrahedron> tets = std::vector<mcx::Tetrahedron>();

	uint32_t k = 0;
	for (uint4 elem : elements) {
		uint32_t label = 1;

		tets.push_back(
		    {make_uint4(elem.x - 1, elem.y - 1, elem.z - 1, elem.w - 1),
		     label});

		k++;
	}
	// define placeholder spheres to simulate
	std::vector<mcx::ImplicitSphere> spheres =
	    std::vector<mcx::ImplicitSphere>();
    //placeholder sphere
    spheres.push_back({make_float3(60, 60, 60), 0.0001});
	// define curves to simulate
	std::vector<mcx::ImplicitCurve> curves =
	    std::vector<mcx::ImplicitCurve>();

    //capsule:
	curves.push_back({make_float3(30, 15, 30), make_float3(30, 30, 30), 10});
    curves.push_back({make_float3(30, 30, 30), make_float3(30, 45, 30), 10});
	mcx::TetrahedralMesh mesh = mcx::TetrahedralMesh(
	    std::vector<float3>({make_float3(0, 0, 0), make_float3(60, 0, 0),
				 make_float3(0, 60, 0), make_float3(60, 60, 0),
				 make_float3(0, 0, 60), make_float3(60, 0, 60),
				 make_float3(0, 60, 60),
				 make_float3(60, 60, 60)}),
	    tets, spheres, curves);

	return mesh;
}

// benchmark made to match IMMC test
mcx::TetrahedralMesh immc_sphere_benchmark(){
	std::vector<uint4> elements = {
	    make_uint4(1, 2, 8, 4), make_uint4(1, 3, 4, 8),
	    make_uint4(1, 2, 6, 8), make_uint4(1, 5, 8, 6),
	    make_uint4(1, 3, 8, 7), make_uint4(1, 5, 7, 8),
	};

	std::vector<mcx::Tetrahedron> tets = std::vector<mcx::Tetrahedron>();

	uint32_t k = 0;
	for (uint4 elem : elements) {
		uint32_t label = 1;

		tets.push_back(
		    {make_uint4(elem.x - 1, elem.y - 1, elem.z - 1, elem.w - 1),
		     label});

		k++;
	}

	// define spheres to simulate
	std::vector<mcx::ImplicitSphere> spheres =
	    std::vector<mcx::ImplicitSphere>();
	spheres.push_back({make_float3(0.5, 0.5, 0.5), 0.1});
	
	// define placeholder curves to simulate
	std::vector<mcx::ImplicitCurve> curves =
	    std::vector<mcx::ImplicitCurve>();

	curves.push_back({make_float3(1,1,1), make_float3(1, 1, 1.00001), 0.0001});
	
	mcx::TetrahedralMesh mesh = mcx::TetrahedralMesh(
	    std::vector<float3>({make_float3(0, 0, 0), make_float3(1, 0, 0),
				 make_float3(0, 1, 0), make_float3(1, 1, 0),
				 make_float3(0, 0, 1), make_float3(1, 0, 1),
				 make_float3(0, 1, 1),
				 make_float3(1, 1, 1)}),
	    tets, spheres, curves);

	return mesh;
}

// Test code generating a cube shaped mesh with a sphere in it
mcx::TetrahedralMesh basic_sphere_test() {
	std::vector<uint4> elements = {
	    make_uint4(1, 2, 8, 4), make_uint4(1, 3, 4, 8),
	    make_uint4(1, 2, 6, 8), make_uint4(1, 5, 8, 6),
	    make_uint4(1, 3, 8, 7), make_uint4(1, 5, 7, 8),
	};

	std::vector<mcx::Tetrahedron> tets = std::vector<mcx::Tetrahedron>();

	uint32_t k = 0;
	for (uint4 elem : elements) {
		uint32_t label = 1;

		tets.push_back(
		    {make_uint4(elem.x - 1, elem.y - 1, elem.z - 1, elem.w - 1),
		     label});

		k++;
	}

	// define spheres to simulate
	std::vector<mcx::ImplicitSphere> spheres =
	    std::vector<mcx::ImplicitSphere>();
	spheres.push_back({make_float3(30, 30, 30), 10});
	// define placeholder curves to simulate
	std::vector<mcx::ImplicitCurve> curves =
	    std::vector<mcx::ImplicitCurve>();
	curves.push_back({make_float3(55,54,58), make_float3(55,55,58), 0.001});
	
    mcx::TetrahedralMesh mesh = mcx::TetrahedralMesh(
	    std::vector<float3>({make_float3(0, 0, 0), make_float3(60, 0, 0),
				 make_float3(0, 60, 0), make_float3(60, 60, 0),
				 make_float3(0, 0, 60), make_float3(60, 0, 60),
				 make_float3(0, 60, 60),
				 make_float3(60, 60, 60)}),
	    tets, spheres, curves);

	return mesh;
}

// Test many shapes at once:
mcx::TetrahedralMesh complex_test() {
	std::vector<uint4> elements = {
	    make_uint4(1, 2, 8, 4), make_uint4(1, 3, 4, 8),
	    make_uint4(1, 2, 6, 8), make_uint4(1, 5, 8, 6),
	    make_uint4(1, 3, 8, 7), make_uint4(1, 5, 7, 8),
	};

	std::vector<mcx::Tetrahedron> tets = std::vector<mcx::Tetrahedron>();

	uint32_t k = 0;
	for (uint4 elem : elements) {
		uint32_t label = 1;

		tets.push_back(
		    {make_uint4(elem.x - 1, elem.y - 1, elem.z - 1, elem.w - 1),
		     label});

		k++;
	}
	// define spheres to simulate
	std::vector<mcx::ImplicitSphere> spheres =
	    std::vector<mcx::ImplicitSphere>();
	spheres.push_back({make_float3(33, 30, 28), 5});
	spheres.push_back({make_float3(23, 30, 28), 3});
	// define curves to simulate
	std::vector<mcx::ImplicitCurve> curves =
	    std::vector<mcx::ImplicitCurve>();
	curves.push_back({make_float3(10, 30, 15), make_float3(40, 30, 15), 5});
	curves.push_back({make_float3(10, 40, 13), make_float3(30, 40, 20), 3});

	mcx::TetrahedralMesh mesh = mcx::TetrahedralMesh(
	    std::vector<float3>({make_float3(0, 0, 0), make_float3(60, 0, 0),
				 make_float3(0, 60, 0), make_float3(60, 60, 0),
				 make_float3(0, 0, 60), make_float3(60, 0, 60),
				 make_float3(0, 60, 60),
				 make_float3(60, 60, 60)}),
	    tets, spheres, curves);

	return mesh;
}

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
		uint32_t label = 1;

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
	curves.push_back({make_float3(10, 30, 15), make_float3(40, 30, 15), 10});

	mcx::TetrahedralMesh mesh = mcx::TetrahedralMesh(
	    std::vector<float3>({make_float3(0, 0, 0), make_float3(60, 0, 0),
				 make_float3(0, 60, 0), make_float3(60, 60, 0),
				 make_float3(0, 0, 60), make_float3(60, 0, 60),
				 make_float3(0, 60, 60),
				 make_float3(60, 60, 60)}),
	    tets, spheres, curves);

	return mesh;
}

mcx::TetrahedralMesh immc_comparison_sphere() {
	std::vector<uint4> elements = {
	    make_uint4(1, 2, 8, 4), make_uint4(1, 3, 4, 8),
	    make_uint4(1, 2, 6, 8), make_uint4(1, 5, 8, 6),
	    make_uint4(1, 3, 8, 7), make_uint4(1, 5, 7, 8)};

	std::vector<mcx::Tetrahedron> tets = std::vector<mcx::Tetrahedron>();

	// setting the material id to the background material
	uint32_t material_id = 1;
	for (uint4 elem : elements) {
		tets.push_back(
		    {make_uint4(elem.x - 1, elem.y - 1, elem.z - 1, elem.w - 1),
		     material_id});
	}

	std::vector<float3> nodes = {
	    make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 1, 0),
	    make_float3(1, 1, 0), make_float3(0, 0, 1), make_float3(1, 0, 1),
	    make_float3(0, 1, 1), make_float3(1, 1, 1)};

	// define spheres to simulate
	std::vector<mcx::ImplicitSphere> spheres =
	    std::vector<mcx::ImplicitSphere>();
	spheres.push_back({make_float3(0.5, 0.5, 0.5), 0.1});
	// define placeholder curves to simulate
	std::vector<mcx::ImplicitCurve> curves =
	    std::vector<mcx::ImplicitCurve>();

	mcx::TetrahedralMesh mesh =
	    mcx::TetrahedralMesh(nodes, tets, spheres, curves);

	return mesh;
}

mcx::TetrahedralMesh immc_comparison_cylinder() {
	std::vector<uint4> elements = {
	    make_uint4(1, 2, 8, 4), make_uint4(1, 3, 4, 8),
	    make_uint4(1, 2, 6, 8), make_uint4(1, 5, 8, 6),
	    make_uint4(1, 3, 8, 7), make_uint4(1, 5, 7, 8)};

	std::vector<mcx::Tetrahedron> tets = std::vector<mcx::Tetrahedron>();

	// setting the material id to the background material
	uint32_t material_id = 1;
	for (uint4 elem : elements) {
		tets.push_back(
		    {make_uint4(elem.x - 1, elem.y - 1, elem.z - 1, elem.w - 1),
		     material_id});
	}

	std::vector<float3> nodes = {
	    make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 1, 0),
	    make_float3(1, 1, 0), make_float3(0, 0, 1), make_float3(1, 0, 1),
	    make_float3(0, 1, 1), make_float3(1, 1, 1)};

	// define spheres to simulate
	std::vector<mcx::ImplicitSphere> spheres =
	    std::vector<mcx::ImplicitSphere>();
	// define placeholder curves to simulate
	std::vector<mcx::ImplicitCurve> curves =
	    std::vector<mcx::ImplicitCurve>();
	float epsilon = 0.001;
	curves.push_back({make_float3(1 - epsilon, 0.5, 0.5),
			  make_float3(epsilon, 0.5, 0.5), 0.1});

	mcx::TetrahedralMesh mesh =
	    mcx::TetrahedralMesh(nodes, tets, spheres, curves);

	return mesh;
}

// main function
int main() {
	try {

        MCX_clock timer;

		mcx::McxContext ctx = mcx::McxContext();

		constexpr uint32_t photon_count = 100000000;
		constexpr float duration = 0.005;
		constexpr uint32_t timesteps = 10;

        tetmesh mesh;
        mcconfig cfg;

		float3 srcpos = make_float3(30, 30, 0.001);
		float3 srcdir = make_float3(0, 0, 1);

		ctx.simulate(&mesh, size, media, photon_count, duration,
			     timesteps, srcpos, srcdir, &cfg);

		std::cout << "\nSimulation complete.\n";
	} catch (std::runtime_error err) {
		std::cout << "A runtime error occured.\n";
		std::cout << err.what();
	}

}
