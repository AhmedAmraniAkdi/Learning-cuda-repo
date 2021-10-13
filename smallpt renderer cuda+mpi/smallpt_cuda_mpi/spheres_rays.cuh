#ifndef SPHERES_RAYS_CUH
#define SPHERES_RAYS_CUH

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

struct Ray{
    float3 origin, dir;
    __host__  __device__ Ray(float3 origin, float3 dir) : origin(origin), dir(dir) {}
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

struct Sphere {
	float radius;
	float3 pos, emision, color;
	Refl_t refl;
	Sphere(float r, float3 p, float3 e, float3 c, Refl_t refl) :
		radius(r), pos(p), emision(e), color(c), refl(refl) {}

	__device__ Sphere() {}

	__device__ inline float intersect(const Ray &r) const {
		// solves:
        // (P - position) * (P - position) - radius^2 = 0 and P(t) = Origin + t * Dir
		// returns distance, negative if nohit
		// |r.o+t*r.d-p|^2 = rad^2
		// |r.d|^2*t^2 - 2*<r.d,p-r.o> + |p-r.o|^2 - rad^2 = 0
		// A = |r.d|^2 = 1 (r.d normalized), B = <r.d,p-r.o>, C = |p-r.o|^2 - rad^2, det = B^2 - C
		// t1,2 = B -+ sqrt(det) (as long as det>=0)
        float3 op = pos - r.origin;
		// moved eps to the scene intersect routine, and modified sphere intersect to return negative if no intersection
		float t;
		float b = dot(op, r.dir);
		float det = b * b - dot(op, op) + radius * radius;
		if (det < 0) return -1; else det = sqrt(det);
		// if t<0 -> no intersection
		return (t = b - det) > 0 ? t : b + det;
	}

	__device__ float3 normal(const float3& v) const{
		return (v - pos) / radius;
	}
};

// all threads read the same sphere one by one, all spheres, and store the closest one
//#define numspheres 9
/*Sphere spheres_cpu[] = {//Scene: radius, position, emission, color, material
  Sphere(1e5, make_float3(1e5 + 1,40.8,81.6), make_float3(0),make_float3(.75,.25,.25),DIFF),//Left
  Sphere(1e5, make_float3(-1e5 + 99,40.8,81.6),make_float3(0),make_float3(.25,.25,.75),DIFF),//Rght
  Sphere(1e5, make_float3(50,40.8, 1e5),     make_float3(0),make_float3(.75,.75,.75),DIFF),//Back
  Sphere(1e5, make_float3(50,40.8,-1e5 + 170), make_float3(0),make_float3(0),           DIFF),//Frnt
  Sphere(1e5, make_float3(50, 1e5, 81.6),    make_float3(0),make_float3(.75,.75,.75),DIFF),//Botm
  Sphere(1e5, make_float3(50,-1e5 + 81.6,81.6),make_float3(0),make_float3(.75,.75,.75),DIFF),//Top
  Sphere(16.5,make_float3(27,16.5,47),       make_float3(0),make_float3(1,1,1)*.999, SPEC),//Mirr
  Sphere(16.5,make_float3(73,16.5,78),       make_float3(0),make_float3(1,1,1)*.999, REFR),//Glas
  Sphere(600, make_float3(50,681.6 - .27,81.6),make_float3(12,12,12),  make_float3(0), DIFF) //Lite
};*/

// new scene 
#define numspheres 15
Sphere spheres_cpu[] = {//Scene: radius, position, emission, color, material
  Sphere(1e5, make_float3(1e5 + 1,40.8,81.6), make_float3(0),make_float3(.75,.25,.25),DIFF),//Left
  Sphere(1e5, make_float3(-1e5 + 99,40.8,81.6),make_float3(0),make_float3(.25,.25,.75),DIFF),//Rght
  Sphere(1e5, make_float3(50,40.8, 1e5),     make_float3(0),make_float3(.75,.75,.75),DIFF),//Back
  Sphere(1e5, make_float3(50,40.8,-1e5 + 170), make_float3(0),make_float3(0),           DIFF),//Frnt
  Sphere(1e5, make_float3(50, 1e5, 81.6),    make_float3(0),make_float3(.75,.75,.75),REFR),//Botm
  Sphere(1e5, make_float3(50,-1e5 + 81.6,81.6),make_float3(0),make_float3(.75,.75,.75),DIFF),//Top
  Sphere(16.5,make_float3(27,16.5,47),make_float3(0),make_float3(1,1,1)*.999, SPEC),//Mirr
  Sphere(16.5,make_float3(73,16.5,78),make_float3(0),make_float3(0.75,0,0.25), REFR),//violet
  Sphere(10,make_float3(15,45,112),make_float3(0.01),make_float3(1,0.5,0)*.999, REFR),//whit
  Sphere(15,make_float3(16,16,130),make_float3(0),make_float3(1,1,0)*.999, REFR),// big yello
  Sphere(7.5,make_float3(40,8,120),make_float3(0),make_float3(1,1,0)*.999, REFR),//small yello mid
  Sphere(8.5,make_float3(60,9,110),make_float3(0),make_float3(1,1,0)*.999, REFR),//small yello righto
  Sphere(5,make_float3(50,75,81.6),make_float3(0),make_float3(0, .682, .999), REFR),//blue
  Sphere(600, make_float3(50,681.6 - .27,81.6),make_float3(12,12,12),  make_float3(0), DIFF) //Lite
};

__constant__ Sphere spheres[numspheres];

int toInt(float x) { return int(pow(clamp(x, 0.0, 1.0), 1 / 2.2) * 255 + .5); }

__device__ inline bool intersect(const Ray &r, float &t, int &id) {
	float d;
	float inf = FLT_MAX;
	float eps = 1e-4;
	#pragma unroll
	for (int i = 0; i < numspheres; i++){
		if ((d = spheres[i].intersect(r)) > eps && d < t){ 
			t = d; 
			id = i; 
		}
	}
	return t < inf;
}

#endif
