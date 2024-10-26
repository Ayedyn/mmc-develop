#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <optix.h>
#include <optix_device.h>
#include <optix_types.h>
#include <stdint.h>
#include <sutil/vec_math.h>

#include <iostream>
#include <limits>

#ifdef __INTELLISENSE__
#include "intellisense_cuda_intrinsics.h"
#endif

#include "mcx_launch_params.h"
#include "mmc_optix_launchparam.h"
#include "voxel_photon.h"

constexpr float C_MM_PER_US = 299792.458;
constexpr float C_US_PER_MM = 1.0 / C_MM_PER_US;
constexpr float ESCAPE_BIAS = 1.0 / 2048.0;

extern "C" {
__constant__ mcx::McxLaunchParams launchParams;
}

__device__ __forceinline__ float3 mix(float3 a, float3 b, int3 select) {
	return make_float3(select.x ? a.x : b.x, select.y ? a.y : b.y,
			   select.z ? a.z : b.z);
}

__device__ __forceinline__ bool allLessThan(float3 a, float3 b) {
	return a.x < b.x && a.y < b.y && a.z < b.z;
}

__device__ __forceinline__ bool allLessThan(int3 a, int3 b) {
	return a.x < b.x && a.y < b.y && a.z < b.z;
}

__device__ __forceinline__ bool allLessThan(uint3 a, uint3 b) {
	return a.x < b.x && a.y < b.y && a.z < b.z;
}

__device__ __forceinline__ bool allLessThanEqual(float3 a, float3 b) {
	return a.x <= b.x && a.y <= b.y && a.z <= b.z;
}

__device__ __forceinline__ bool allLessThanEqual(int3 a, int3 b) {
	return a.x <= b.x && a.y <= b.y && a.z <= b.z;
}

__device__ __forceinline__ int3 negativeMask(float3 a) {
	constexpr unsigned int SIGN_BIT = 1 << 31;
	return make_int3((*(unsigned int*)&a.x & SIGN_BIT) > 0,
			 (*(unsigned int*)&a.y & SIGN_BIT) > 0,
			 (*(unsigned int*)&a.z & SIGN_BIT) > 0);
}

__device__ __forceinline__ int3 selectStepDirection(float3 extCoeffs,
						    int3 dirCoeffs,
						    float& len) {
	dirCoeffs = make_int3(-2, -2, -2) * dirCoeffs + make_int3(1, 1, 1);
	if (extCoeffs.x < extCoeffs.y && extCoeffs.x < extCoeffs.z) {
		len = extCoeffs.x;
		return make_int3(dirCoeffs.x, 0, 0);
	} else if (extCoeffs.y < extCoeffs.z) {
		len = extCoeffs.y;
		return make_int3(0, dirCoeffs.y, 0);
	} else {
		len = extCoeffs.z;
		return make_int3(0, 0, dirCoeffs.z);
	}
}

//
__device__ __forceinline__ size_t flattenArrayLocationUint3(uint3 loc) {
	return loc.x + loc.y * launchParams.dataSize.x +
	       loc.z * launchParams.dataSize.x * launchParams.dataSize.y;
}

// gets the linear index of simulation data in outputBuffer, given a 4d Index
// with height, width, length, time.
__device__ __forceinline__ size_t flattenArrayLocationUint4(uint4 loc) {
	return loc.x + loc.y * launchParams.dataSize.x +
	       loc.z * launchParams.dataSize.x * launchParams.dataSize.y +
	       loc.w * launchParams.dataSize.x * launchParams.dataSize.y *
		   launchParams.dataSize.z;
}

// get the index for the current time frame into the time array
__device__ __forceinline__ unsigned int getTimeFrame(const float &tof) {
	unsigned int t =
	    min((int)floorf(fminf(tof, launchParams.simulationDuration) *
			    launchParams.inverseTimeStep),
		launchParams.timeSteps - 1);
	return t;
}

// gets a reference to a surface boundary struct via a primitive id in launch
// params, which is a devicebuffer
__device__ __forceinline__ mcx::SurfaceBoundary& getSurfaceBoundary(
    int primIdx) {
	return ((mcx::SurfaceBoundary*)launchParams.surfaceBoundaries)[primIdx];
}

__device__ __forceinline__ Medium& getMediumFromID(int id) {
	return launchParams.medium[id];
}

__device__ __forceinline__ mcx::ImplicitCurve& getCurveFromID(int id) {
	return ((mcx::ImplicitCurve*)launchParams.curveData)[id];
}

// saves a given energy or other simulation output to a linearized 4D buffer
// of x, y, z, space, using its linear index and weight values
__device__ __forceinline__ void saveToBuffer(const uint &eid, const float &w) {
    // to minimize numerical error, use the same trick as MCX
    // becomes much slower when using atomicAdd(*double, double)
    float accum = atomicAdd(&((float*)launchParams.outputBuffer)[eid], w);
}

  /**
  * @brief Accumulate output quantities to a 3D grid
  */
__device__ __forceinline__ void accumulateOutput(const mcx::VoxelPhotonPayload &vp, 
						 const Medium &medium,
     						 const float &lmove) {
     // divide path into segments of equal length
     int segcount = ((int)(lmove) + 1) << 1;
     float seglen = lmove / segcount;
     float segdecay = expf(-medium.mua * seglen);
     // false is a placeholder for statement checking for energy deposition setting
     float segloss = (false) ? vp.energy * (1.0f - segdecay) :
         (medium.mua ? vp.energy * (1.0f - segdecay) / medium.mua : 0.0f);
 
     // deposit weight loss of each segment to the corresponding grid
     float3 step = seglen * vp.direction;
     // in this case we can assume that nmin is 0,0,0, so delete the term:
     // float3 segmid = vp.origin - gcfg.nmin + 0.5f * step; // segment midpoint
     float3 segmid = vp.origin + 0.5f * step; // segment midpoint
     float currtof = vp.elapsedTime + seglen * C_US_PER_MM * medium.n; // current time of flight

     // round segmid to get the corresponding voxel value
     float3 lastVoxel = floor(segmid);
     uint3 midpt_voxel = make_uint3(lastVoxel.x, lastVoxel.y, lastVoxel.z);

     // find the information of the first segment
     unsigned int oldeid = flattenArrayLocationUint4(make_uint4(midpt_voxel, getTimeFrame(currtof)));
     float oldweight = segloss;
 
     // iterater over the rest of the segments
     for (int i = 1; i < segcount; ++i) {
         // update information for the curr segment
         segloss *= segdecay;
         segmid += step;
         currtof += seglen * C_US_PER_MM * medium.n;
	 float3 lastVoxel = floor(segmid);
	 uint3 midpt_voxel = make_uint3(lastVoxel.x, lastVoxel.y, lastVoxel.z);
         unsigned int neweid = flattenArrayLocationUint4(make_uint4(midpt_voxel, getTimeFrame(currtof)));
 
         // save when entering a new element or during the last segment
         if (neweid != oldeid) {
             saveToBuffer(oldeid, oldweight);
             // reset oldeid and weight bucket
             oldeid = neweid;
             oldweight = 0.0f;
         }
         oldweight += segloss;
     }
 
     // save the weight loss of the last segment
     saveToBuffer(oldeid, oldweight);
}



__device__ __forceinline__ size_t accumulateEnergy(uint3 loc, float energy,
						   float time) {
	unsigned int t =
	    min((int)floorf(fminf(time, launchParams.simulationDuration) *
			    launchParams.inverseTimeStep),
		launchParams.timeSteps - 1);
	atomicAdd(
	    &((float*)launchParams
		  .outputBuffer)[flattenArrayLocationUint4(make_uint4(loc, t))],
	    energy);
}

__device__ __forceinline__ size_t accumulateEnergyDeposition(uint3 loc, float energy, float time, float medium_mua) {
	unsigned int t =
	    min((int)floorf(fminf(time, launchParams.simulationDuration) *
			    launchParams.inverseTimeStep),
		launchParams.timeSteps - 1);

	atomicAdd(
	    &((float*)launchParams
		  .outputBuffer)[flattenArrayLocationUint4(make_uint4(loc, t))],
	    energy*medium_mua);	

}


__device__ __forceinline__ float3 rotateVector(float3 vec, float2 zen,
					       float2 azi) {
	if (vec.z > -1.0f + std::numeric_limits<float>::epsilon() &&
	    vec.z < 1.0f - std::numeric_limits<float>::epsilon()) {
		float tmp0 = 1.0f - vec.z * vec.z;
		float tmp1 = zen.x * rsqrtf(tmp0);
		return tmp1 * (azi.y * make_float3(vec.x, vec.y, -tmp0) *
				   make_float3(vec.z, vec.z, 1) +
			       azi.x * make_float3(-vec.y, vec.x, 0.0)) +
		       zen.y * vec;

		return make_float3(
		    tmp1 * (vec.x * vec.z * azi.y - vec.y * azi.x) +
			vec.x * zen.y,
		    tmp1 * (vec.y * vec.z * azi.y + vec.x * azi.x) +
			vec.y * zen.y,
		    -tmp1 * tmp0 * azi.y + vec.z * zen.y);
	} else {
		return make_float3(zen.x * azi.y, zen.x * azi.x,
				   (vec.z > 0.0f) ? zen.y : -zen.y);
	}
}

// Returns the sine and cosine from the Henyey-Greenstein distribution.
__device__ __forceinline__ float2 henyeyGreenstein(float g, mcx::Random& rand) {
	float ctheta;
	if (fabs(g) > std::numeric_limits<float>::epsilon()) {
		ctheta = (1.0f - g * g) /
			 (1.0f - g + 2.0f * g * rand.uniform(0.0f, 1.0f));
		ctheta *= ctheta;
		ctheta = (1.0f + g * g - ctheta) / (2.0f * g);
		ctheta = fmax(-1.0f, fmin(1.0f, ctheta));
	} else {
		ctheta = 2.0f * rand.uniform(0.0f, 1.0f) - 1.0f;
	}
	return make_float2(sinf(acosf(ctheta)), ctheta);
}

__device__ __forceinline__ float3 selectScatteringDirection(float3 dir, float g,
							    mcx::Random& rand) {
	float2 aziScat;
	sincosf(rand.uniform(0.0f, 2.0f * M_PIf), &aziScat.x, &aziScat.y);

	float2 zenScat = henyeyGreenstein(g, rand);

	return rotateVector(dir, zenScat, aziScat);
}

__device__ __forceinline__ void stepPhoton(mcx::VoxelPhoton& vp) {
	optixTrace(vp.manifold, vp.origin - ESCAPE_BIAS * vp.direction,
		   vp.direction, 0.0,
		   vp.scatteringLengthLeft *
			   1/getMediumFromID(vp.currentMedium).mus +
		       ESCAPE_BIAS,
		   0.0f, OptixVisibilityMask(1),
		   OptixRayFlags::OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES, 0,
		   1, 0, *(uint32_t*)&vp.origin.x, *(uint32_t*)&vp.origin.y,
		   *(uint32_t*)&vp.origin.z, *(uint32_t*)&vp.direction.x,
		   *(uint32_t*)&vp.direction.y, *(uint32_t*)&vp.direction.z,
		   *(uint32_t*)&vp.scatteringLengthLeft, vp.random.intSeed.x,
		   vp.random.intSeed.y, vp.random.intSeed.z,
		   vp.random.intSeed.w, *(uint32_t*)&vp.elapsedTime,
		   *(uint32_t*)&vp.energy, *(uint32_t*)&vp.manifold,
		   *(((uint32_t*)&vp.manifold) + 1), vp.currentMedium);
}

// starts a new photon from the ray source
__device__ __forceinline__ void resetPhoton(mcx::VoxelPhoton& vp) {
	vp.origin = launchParams.emitterPosition;
	vp.manifold = launchParams.startManifold;
	vp.direction = launchParams.emitterDirection;
	vp.elapsedTime = 0;
	vp.scatteringEventCount = 0;
	vp.scatteringLengthLeft =
	    vp.random.exponential(1.0, std::numeric_limits<float>::epsilon());
	vp.energy = 1.0f;
	vp.currentMedium = launchParams.startMedium;
}


// helper function to calculate distance of a point from the line segment of a
// capsule
__device__ __forceinline__ float capsule_distance(float3 ctrlpt1,
						  float3 ctrlpt2, float width,
						  float3 point) {
	float h = min(
	    1.0, max(0.0, dot(point - ctrlpt1, ctrlpt2 - ctrlpt1) /
			      dot((ctrlpt2 - ctrlpt1), (ctrlpt2 - ctrlpt1))));
	return length(point - ctrlpt1 - h * (ctrlpt2 - ctrlpt1));
}

#ifndef NDEBUG

__device__ __forceinline__ void printPhotonGeometry(const mcx::VoxelPhoton vp) {
	printf(
	    "\nThe photon is at x: %f y: %f z: %f\nThe current Medium is "
	    "%d\nThe current photon time is: %f\n",
	    vp.origin.x, vp.origin.y, vp.origin.z, vp.currentMedium,
	    vp.elapsedTime);
	printf("Distance from center of the capsule is: %f\n", capsule_distance(make_float3(20, 20, 10), make_float3(40,40,10),4, vp.origin));
}

__device__ __forceinline__ bool checkIfPhotonWithinCurve(
    const mcx::VoxelPhoton vp) {
	
	float distance = capsule_distance(make_float3(20, 20, 10), make_float3(40,40,10),
		       	4, vp.origin);

	if (distance <= 4) {
		//printf(
		//    "\n---Photon is effectively inside the simple 'curve' at "
		//    "%f %f %f with distance of %f from center---\n ",
		//    vp.origin.x, vp.origin.y, vp.origin.z, distance);
		
		return true;

	} else {
		//printf(
		//    "\n---Photon is effectively outside the simple 'curve' at "
		//    "%f %f %f with distance of %f from center---\n",
		//    vp.origin.x, vp.origin.y, vp.origin.z, distance);
		
		return false;

	}
}

#endif

// tests intersection of ray with a sphere and returns intersections as floats of distance from start of ray
// geometric solution taken from: scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html 
__device__ __forceinline__ float2 get_sphere_intersections(const float3 center, const float width, const float3 ray_origin, const float3 ray_dir){

	float3 L = center - ray_origin;
	float tca = dot(L, ray_dir);

	float d2 = dot(L, L) - tca*tca;
	// result if no intersections, set both to negatives
	if(d2 > width*width){
		float2 hitTs = make_float2(-1,-1);
		return hitTs;
	}
	// calculates both intersections
	float thc = sqrt(width*width -d2);
	float2 hitTs;
	hitTs.x = tca-thc;
	hitTs.y = tca+thc;
	return hitTs;
}

// tests intersection of ray with an infinite cylinder (derived from parametric substitution)
// from davidjcobb.github.io/articles/ray-cylinder-intersection
// solve a quadratic equation of at^2+bt+c = 0 for t
__device__ __forceinline__ float2 get_inf_cyl_intersections(const float3 vertex1, const float3 vertex2, const float width, const float3 ray_origin, const float3 ray_dir){
	float3 R1 = ray_origin - vertex2;
	float3 Cs = vertex1-vertex2;
	float  Ch = length(Cs);
	float3 Ca = Cs/Ch;

	float Ca_dot_Rd = dot(Ca, ray_dir);
	float Ca_dot_R1 = dot(Ca, R1);
	float R1_dot_R1 = dot(R1, R1);

	float a = 1 - (Ca_dot_Rd * Ca_dot_Rd);
	float b = 2 * (dot(ray_dir, R1) - Ca_dot_Rd * Ca_dot_R1);
	float c = R1_dot_R1 - Ca_dot_R1 * Ca_dot_R1 - (width * width);

	float2 cyl_hits;
	float discriminant = b*b-4.0f*a*c;
	if(discriminant<0){
		cyl_hits.x = -1;
		cyl_hits.y = -1;
		return cyl_hits;
	}

	cyl_hits.x = (-b-sqrt(discriminant))/(2*a);
	cyl_hits.y = (-b+sqrt(discriminant))/(2*a);
	return cyl_hits;
}

// ray generation, first function to run on kernel
extern "C" __global__ void __raygen__rg() {

	uint3 launchIndex = optixGetLaunchIndex();

	// create a random seed for each launch index
	float3 extents = make_float3(launchParams.dataSize);
	unsigned int sd =
	    (1151 * (launchIndex.x + 1) ^ 937 * (launchIndex.y + 1));
	uint64_t seed = ((uint64_t)sd << 16 ^ sd) | 1;
	mcx::VoxelPhoton vp;
	vp.random = mcx::Random(seed, seed);
	for (int i = 0; i < 4; i++) {
		vp.random.uniform(0.0f, 1.0f);
	}

	// init a ray
	resetPhoton(vp);

	// count of photons run per thread
	int co = 0;

	// perform scalar multiplication to adjust for size of bins
	float3 lastVoxel = floor(vp.origin);
	uint3 vx = make_uint3(lastVoxel.x, lastVoxel.y, lastVoxel.z);

	while (co < launchParams.threadphoton + (launchIndex.x < launchParams.oddphoton)) {
		float originalEnergy = vp.energy;
		stepPhoton(vp);
		//accumulateEnergy(vx, originalEnergy - vp.energy,
		//		 vp.elapsedTime);
		//float mua = getMediumFromID(vp.currentMedium).absorption;		

		//accumulateEnergyDeposition(vx, originalEnergy - vp.energy, vp.elapsedTime, mua);
		// check most recent voxel
		lastVoxel = floor(vp.origin);
		vx = make_uint3(lastVoxel.x, lastVoxel.y, lastVoxel.z);
		
		// if a photon escapes or time of flight reaches the limit
		if (!(vp.manifold != (OptixTraversableHandle) nullptr &&
		      vp.elapsedTime < launchParams.simulationDuration &&
		      allLessThan(vx, make_uint3(256, 256, 256)))) {
			resetPhoton(vp);
			lastVoxel = floor(vp.origin);
			vx = make_uint3(lastVoxel.x, lastVoxel.y, lastVoxel.z);
			co++;
		}
	}
}

// this program executes when the ray doesn't hit any of the mesh or implicit
// geometry
extern "C" __global__ void __miss__ms() {
	// get photon and ray information from payload
	mcx::VoxelPhotonPayload pl;
	getPayload(pl);

	// get medium properties
	Medium& medium = getMediumFromID(pl.currentMedium);

	// equivalent to lmove
	float distanceTraveled =
	    pl.scatteringLengthLeft * 1/medium.mus;

	// accumulate energy
	accumulateOutput(pl, medium, distanceTraveled);

	// update photon position
	pl.origin += distanceTraveled * pl.direction;
	
	// update photon timer
	pl.elapsedTime +=
	    distanceTraveled * C_US_PER_MM * medium.n;

	// scattering event
	pl.direction = selectScatteringDirection(pl.direction,
						 medium.g, pl.random);
	pl.scatteringLengthLeft =
	    pl.random.exponential(1.0, std::numeric_limits<float>::epsilon());

	// update photon weight
	pl.energy *= expf(-medium.mua * distanceTraveled);

	setPayload(pl);
}

// runs upon closest hit, most important for tracking our medium changes
extern "C" __global__ void __closesthit__ch() {
	mcx::VoxelPhotonPayload pl;
	getPayload(pl);
	Medium& medium = getMediumFromID(pl.currentMedium);

	float tm = fmaxf(1.0 / 8192.0, optixGetRayTmax() - ESCAPE_BIAS);
	float scatDist = tm * medium.mus;
	mcx::SurfaceBoundary& boundary =
	    getSurfaceBoundary(optixGetPrimitiveIndex());

	pl.origin += tm * pl.direction;

	uint3 index = optixGetLaunchIndex();	
	int threadnum = index.x;
	if(threadnum<0){
		 printf("\n*** Closest hit found at: x: %f y: %f z: %f t: %f ***\n",
	 pl.origin.x, pl.origin.y, pl.origin.z, pl.elapsedTime);
		 printf("Material updated from %d to %d\n", pl.currentMedium, boundary.medium);	
	}

	pl.manifold = boundary.manifold;
	pl.scatteringLengthLeft -= scatDist;
	pl.elapsedTime += tm * C_US_PER_MM * medium.n;
	// printf("The medium's absorption used was: %f\n", medium.mua);
	pl.energy *= expf(-medium.mua * tm);

	pl.currentMedium = boundary.medium;

	Medium newmedium = getMediumFromID(pl.currentMedium);

	setPayload(pl);
}

// intersection test for curve primitives
extern "C" __global__ void __intersection__customlinearcurve(){
	// 1. initialize variables for geometry
	int primIdx = optixGetPrimitiveIndex();
	unsigned int curveIdx;
	
	float width_offset = 0;
	if (primIdx >= launchParams.num_inside_prims) {
		curveIdx = primIdx - launchParams.num_inside_prims;
		width_offset = launchParams.WIDTH_ADJ;
	} else {
		curveIdx = primIdx;
	}

	const mcx::ImplicitCurve& curve = getCurveFromID(curveIdx);
	
	// vector going from pt2 to pt1:
	float3 lineseg_AB = curve.vertex1-curve.vertex2;
	float width = curve.width + width_offset;
	float t_min = optixGetRayTmin();
	float t_max = optixGetRayTmax();
	// get normalized ray direction
	float3 ray_dir = normalize(optixGetWorldRayDirection());
	float3 ray_origin = optixGetWorldRayOrigin();

	// test for intersections with sphere 1:
	float2 sphere_one_hits = get_sphere_intersections(curve.vertex1, width, ray_origin, ray_dir);
	
	// get coordinates for intersections
	float3 sphere_one_hit_one = ray_origin + ray_dir * sphere_one_hits.x;
	float3 sphere_one_hit_two = ray_origin + ray_dir * sphere_one_hits.y;

	// discard intersections on interior side of sphere 1:
	// (discard if negative when taking dot product with vector AB)
	if(dot(sphere_one_hit_one-curve.vertex1, lineseg_AB)<0){
		sphere_one_hits.x = -1;
	}
	if(dot(sphere_one_hit_two-curve.vertex1, lineseg_AB)<0){
		sphere_one_hits.y = -1;
	}
	optixReportIntersection(sphere_one_hits.x, 1);
	optixReportIntersection(sphere_one_hits.y, 1);

	// test for intersections with sphere 2:
	float2 sphere_two_hits = get_sphere_intersections(curve.vertex2, width, ray_origin, ray_dir);
	// get coordinates for intersections
	float3 sphere_two_hit_one = ray_origin + (ray_dir * sphere_two_hits.x);
	float3 sphere_two_hit_two = ray_origin + (ray_dir * sphere_two_hits.y);

	// discard intersections on interior side of sphere 2:
	if(dot(curve.vertex2-sphere_two_hit_one, lineseg_AB)<0){
		sphere_two_hits.x = -1;
	}
	if(dot(curve.vertex2-sphere_two_hit_two, lineseg_AB)<0){
		sphere_two_hits.y = -1;
	}
	optixReportIntersection(sphere_two_hits.x, 2);
	optixReportIntersection(sphere_two_hits.y, 2);

	// test for intersections with infinite cylinder:
	float2 cyl_hits = get_inf_cyl_intersections(curve.vertex1, curve.vertex2, width, ray_origin, ray_dir);
	
	// discard intersections on exterior of cylinder:
	float3 cyl_hit_one = ray_origin + ray_dir * cyl_hits.x;
	float3 cyl_hit_two = ray_origin + ray_dir * cyl_hits.y;
	if(dot(cyl_hit_one-curve.vertex1, lineseg_AB)>0 || 
	dot(cyl_hit_one-curve.vertex2, lineseg_AB)<0){
		cyl_hits.x = -1;
	}

	if(dot(cyl_hit_two-curve.vertex1, lineseg_AB)>0 || 
	dot(cyl_hit_two-curve.vertex2, lineseg_AB)<0){
		cyl_hits.y = -1;
	}

	optixReportIntersection(cyl_hits.x, 3);
	optixReportIntersection(cyl_hits.y, 3);

	return;

}

