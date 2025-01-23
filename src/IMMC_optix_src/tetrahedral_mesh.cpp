#include "tetrahedral_mesh.h"

#include <sutil/vec_math.h>
//#include <vector_functions.h>
//#include <vector_types.h>
#include <exception>
#include <stdexcept>

namespace mcx {
static int nextUnseen(std::vector<bool>& input) {
	for (int i = 0; i < input.size(); i++) {
		if (!input[i]) {
			return i;
		}
	}
	return -1;
}

TetrahedralMesh::TetrahedralMesh(std::vector<float3> nodes,
				 std::vector<Tetrahedron> elements,
				 std::vector<ImplicitSphere> spheres,
				 std::vector<ImplicitCurve> curves) {
	this->nodes = nodes;
	this->elements = elements;
	this->spheres = spheres;
	this->curves = curves;

	this->node_to_tetrahedra = std::vector<std::vector<uint32_t>>();
	for (int i = 0; i < this->nodes.size(); i++) {
		this->node_to_tetrahedra.push_back(std::vector<uint32_t>());
	}

	for (int i = 0; i < this->elements.size(); i++) {
		Tetrahedron t = this->elements[i];
		this->node_to_tetrahedra[t.nodes.x].push_back(i);
		this->node_to_tetrahedra[t.nodes.y].push_back(i);
		this->node_to_tetrahedra[t.nodes.z].push_back(i);
		this->node_to_tetrahedra[t.nodes.w].push_back(i);
	}
}

static uint32_t manifoldOf(uint32_t tet,
			   std::vector<std::vector<uint32_t>> manifolds) {
	for (uint32_t i = 0; i < manifolds.size(); i++) {
		for (uint32_t other : manifolds[i]) {
			if (tet == other) {
				return i;
			}
		}
	}

	throw std::runtime_error(
	    "Tetrahedron not found in detected manifolds.");
}

// checks if a given point is inside any spheres in the tetrahedral mesh
bool TetrahedralMesh::insideSphere(float3 position) {
	for (ImplicitSphere sphere : this->spheres) {
		float3 displacement = position - sphere.position;
		if (dot(displacement, displacement) <= sphere.radius) {
			return true;
		}
	}

	return false;
}

// checks if a given point is inside any linear curves in the tetrahedral mesh,
// using a distance from line segment calculation
bool TetrahedralMesh::insideCurve(float3 position) {
	for (ImplicitCurve curve : this->curves) {
		// vector along the line segment of the curve
		float3 vector_alongcurve =
		    make_float3(curve.vertex2.x - curve.vertex1.x,
				curve.vertex2.y - curve.vertex1.y,
				curve.vertex2.z - curve.vertex1.z);
		// vector from the first vertex to the position
		float3 vector_toposition = make_float3(
		    position.x - curve.vertex1.x, position.y - curve.vertex1.y,
		    position.z - curve.vertex1.z);
		// computes the scalar projection of vector to position onto the
		// line segment
		float scalarproj = dot(vector_alongcurve, vector_toposition) /
				   dot(vector_alongcurve, vector_alongcurve);
		// computes the projection of the vector to position onto the
		// line segment
		float3 projection =
		    make_float3(vector_alongcurve.x * scalarproj,
				vector_alongcurve.y * scalarproj,
				vector_alongcurve.z * scalarproj);
		// gets the vector for the shortest distance from line segment
		float3 distance_vector =
		    make_float3(vector_toposition.x - projection.x,
				vector_toposition.y - projection.y,
				vector_toposition.z - projection.z);
		// get the magnitude of that vector
		float distance = sqrt(distance_vector.x * distance_vector.x +
				      distance_vector.y * distance_vector.y +
				      distance_vector.z * distance_vector.z);
		if (distance < curve.width) {
			return true;
		}
	}
	return false;
}

// Constructor, builds a series of manifolds given a vector list of tetrahedron indices
std::vector<TetrahedralManifold> TetrahedralMesh::buildManifold(
    std::vector<uint32_t>& tetrahedron_to_manifold) {
	std::vector<TetrahedralManifold> result =
	    std::vector<TetrahedralManifold>();

	std::vector<std::vector<uint32_t>> manis = identifyManifolds();

	std::vector<uint32_t>& tet_to_manifold = tetrahedron_to_manifold;
	tet_to_manifold = std::vector<uint32_t>();
	for (int i = 0; i < this->elements.size(); i++) {
		tet_to_manifold.push_back(0);
	}

	for (int i = 0; i < manis.size(); i++) {
		for (uint32_t x : manis[i]) {
			tet_to_manifold[x] = i;
		}
	}

	for (std::vector<uint32_t> tets : manis) {
		std::vector<TetrahedronBoundary> triangles =
		    std::vector<TetrahedronBoundary>();

		for (uint32_t tet : tets) {
			std::vector<uint3> coplanarFaces, emptyFaces;
			std::vector<uint32_t> indices =
			    adjacentTetrahedra(tet, coplanarFaces, emptyFaces);

			for (uint32_t side = 0; side < indices.size(); side++) {
				if (this->elements[tet].material !=
				    this->elements[indices[side]].material) {
					triangles.push_back(TetrahedronBoundary{
					    coplanarFaces[side],
					    tet_to_manifold[indices[side]] +
						1});
				}
			}

			for (uint3 empty : emptyFaces) {
				triangles.push_back(
				    TetrahedronBoundary{empty, 0});
			}
		}

		result.push_back(
		    TetrahedralManifold{triangles, this->spheres, this->curves,
					this->elements[tets[0]].material});
	}

	return result;
}

std::vector<uint32_t> TetrahedralMesh::adjacentTetrahedra(
    uint32_t element, std::vector<uint3>& resultFaces,
    std::vector<uint3>& emptyFaces) {
	resultFaces = std::vector<uint3>();
	std::vector<uint32_t> result = std::vector<uint32_t>();

	uint4 nodes = this->elements[element].nodes;
	for (uint32_t node :
	     std::vector<uint32_t>({nodes.x, nodes.y, nodes.z, nodes.w})) {
		for (uint32_t t : this->node_to_tetrahedra[node]) {
			uint3 faces;

			for (uint32_t alreadySeen : result) {
				if (t == alreadySeen) {
					goto cnt;
				}
			}

			if (coplanarTetrahedra(element, t, faces)) {
				result.push_back(t);
				resultFaces.push_back(faces);
			}
			if (result.size() > 3) {
				break;
			}

		cnt:;
		}
	}

	emptyFaces = std::vector<uint3>();
	uint4 an = this->elements[element].nodes;
	for (uint3 side : std::vector<uint3>({make_uint3(an.x, an.w, an.y),
					      make_uint3(an.w, an.z, an.y),
					      make_uint3(an.x, an.z, an.w),
					      make_uint3(an.x, an.y, an.z)})) {
		bool found = false;
		for (uint3 rf : resultFaces) {
			if (side.x == rf.x && side.y == rf.y &&
			    side.z == rf.z) {
				found = true;
				break;
			}
		}
		if (!found) {
			emptyFaces.push_back(side);
		}
	}

	return result;
}

// identifies and assigns sequential ids to each tetrahedron in a manifold
std::vector<std::vector<uint32_t>> TetrahedralMesh::identifyManifolds() {
	std::vector<std::vector<uint32_t>> result =
	    std::vector<std::vector<uint32_t>>();
	std::vector<bool> alreadySeen = std::vector<bool>();
	for (uint32_t i = 0; i < this->elements.size(); i++) {
		alreadySeen.push_back(false);
	}

	int totalCount = 0;

	int begin;
	while ((begin = nextUnseen(alreadySeen)) > -1) {
		std::vector<uint32_t> currentAdjacents =
		    std::vector<uint32_t>();

		std::vector<uint32_t> examine = std::vector<uint32_t>();
		examine.push_back(begin);
		uint32_t currentMaterial = this->elements[begin].material;

		while (examine.size() > 0) {
			uint32_t tet = examine.back();
			examine.pop_back();
			if (!alreadySeen[tet] &&
			    this->elements[tet].material == currentMaterial) {
				std::vector<uint3> _a, _b;
				for (uint32_t next :
				     adjacentTetrahedra(tet, _a, _b)) {
					examine.push_back(next);
				}
				currentAdjacents.push_back(tet);
				alreadySeen[tet] = true;
				totalCount++;
			}
		}

		result.push_back(currentAdjacents);
	}

	return result;
}

static bool containsNodes(uint3& tri, uint4& nodes) {
	int count = 0;
	for (uint32_t n :
	     std::vector<uint32_t>({nodes.x, nodes.y, nodes.z, nodes.w})) {
		if (tri.x == n || tri.y == n || tri.z == n) {
			count++;
		}
	}
	return count == 3;
}

bool TetrahedralMesh::coplanarTetrahedra(uint32_t a, uint32_t b,
					 uint3& result) {
	if (a == b) {
		return false;
	}

	uint4 an = this->elements[a].nodes;
	uint4 bn = this->elements[b].nodes;

	for (uint3 node : std::vector<uint3>({make_uint3(an.x, an.w, an.y),
					      make_uint3(an.w, an.z, an.y),
					      make_uint3(an.x, an.z, an.w),
					      make_uint3(an.x, an.y, an.z)})) {
		if (containsNodes(node, bn)) {
			result = node;
			return true;
		}
	}

	return false;
}

bool TetrahedralMesh::surroundingElement(float3 position, uint32_t& element) {
	element = 0xffffffff;
	for (int i = 0; i < this->elements.size(); i++) {
		uint4 an = this->elements[i].nodes;
		float3 mid = (this->nodes[an.x] + this->nodes[an.y] +
			      this->nodes[an.z] + this->nodes[an.w]) /
			     4;
		for (uint3 node :
		     std::vector<uint3>({make_uint3(an.x, an.w, an.y),
					 make_uint3(an.w, an.z, an.y),
					 make_uint3(an.x, an.z, an.w),
					 make_uint3(an.x, an.y, an.z)})) {
			float3 norm =
			    cross(this->nodes[node.x] - this->nodes[node.y],
				  this->nodes[node.z] - this->nodes[node.y]);

			if (dot(norm, mid - this->nodes[node.y]) < 0) {
				throw std::runtime_error(
				    "Tetrahedron was not oriented out.");
			}

			if (dot(norm, position - this->nodes[node.y]) < 0) {
				goto cnt;
			}
		}

		if (element == 0xffffffff) {
			element = i;
		} else {
			printf("Found a duplicate\n");
		}

	cnt:;
	}

	return element != 0xffffffff;
}
}  // namespace mcx
