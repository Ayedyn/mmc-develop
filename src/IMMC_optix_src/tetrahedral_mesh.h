#pragma once

#include <stdint.h>
#include <optional>
#include <optix.h>
#include <vector>
#include <vector_types.h>

#ifndef IMPLICIT_CURVE_H
#define IMPLICIT_CURVE_H
#include "implicit_curve.h"
#endif

// This file introduces objects to represent several geometries for use in mmc-optix
namespace mcx {

	// tetrahedrons represented with indices of 4 nodes in vector of coordinates from mesh, and material id
	struct Tetrahedron {
	public:
		uint4 nodes;
		// unsigned 32bit integer
		uint32_t material;
	};

	// implicit sphere represented with position and radius
	struct ImplicitSphere {
	public:
		float3 position;
		float radius;
	};

	// Represents the triangle boundary between tetrahedrons belonging to different materials.
	struct TetrahedronBoundary {
	public:
		// indices of nodes of the triangle
		uint3 indices;
		// index for which manifold the boundary is part of
		uint32_t manifold;
	};

	// Represents a continuous surface of triangles enclosing a single material, as well as any intersecting spheres
	struct TetrahedralManifold {
	public:
		std::vector<TetrahedronBoundary> triangles;
		std::vector<ImplicitSphere> spheres;
		std::vector<ImplicitCurve> curves;
		// id for which material this manifold is (uint32_t is unsigned 32bit integer)
		uint32_t material;
	};

	class TetrahedralMesh {
	public:

		// vector of 3D cartesian coordinates of all nodes
		std::vector<float3> nodes;
		// vector of elem materials and node id's
		std::vector<Tetrahedron> elements;
		// vector of sphere structs which include center coords and radius
		std::vector<ImplicitSphere> spheres;
		// vector of curve structs which include vertex1, vertex2, and width
		std::vector<ImplicitCurve> curves;

		// Constructor for meshes
		TetrahedralMesh(std::vector<float3> nodes, std::vector<Tetrahedron> elements,		       std::vector<ImplicitSphere> spheres, std::vector<ImplicitCurve> curves);

		// Constructor for manifolds builds a manifold given a vector list of tetrahedrons indices to make the manifold from?
		std::vector<TetrahedralManifold> buildManifold(std::vector<uint32_t>& tetrahedron_to_manifold);

		bool surroundingElement(float3 position, uint32_t& element);
		bool insideSphere(float3 position);
		bool insideCurve(float3 position);

	private:
		std::vector<std::vector<uint32_t>> node_to_tetrahedra;

		std::vector<uint32_t> adjacentTetrahedra(uint32_t element, std::vector<uint3>& resultFaces, std::vector<uint3>& emptyFaces);
		//
		std::vector<std::vector<uint32_t>> identifyManifolds();
		bool coplanarTetrahedra(uint32_t a, uint32_t b, uint3& result);
	};
}
