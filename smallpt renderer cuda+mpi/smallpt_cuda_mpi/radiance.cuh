#ifndef RADIANCE_CUH
#define RADIANCE_CUH

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <curand_kernel.h>
#include "spheres_rays.cuh"

#define DEPTH 10
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

__device__ inline void diff(/*curandStatePhilox4_32_10_t*/ curandState_t *state, int idx, float3 nl, float3 x, float3 albedo, float3 &multiplier, Ray &r){
	
	float phi = 2 * M_PI * curand_uniform(&state[idx]);
	float r2 = curand_uniform(&state[idx]);
	float sinTheta = sqrt(r2);
	float cosTheta = sqrt(1 - r2);
	float3 w = nl;
	float3 u = normalize(cross(fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0), w));
	float3 v = cross(w, u);
	float3 d = normalize(u*cos(phi)*sinTheta + v * sin(phi)*sinTheta + w * cosTheta);

	// https://github.com/straaljager/GPU-path-tracing-with-CUDA-tutorial-1/blob/master/smallptCUDA.cu
	// feels nice, did it correctly
	r.origin = x + 0.05*nl; // offset ray origin slightly to prevent self intersection - i would have never found the error HAHAHAH
	r.dir = d;
	multiplier *= albedo;

}

__device__ inline void spec(/*curandStatePhilox4_32_10_t*/ curandState_t *state, int idx, float3 n, float3 x, float3 albedo, float3 &multiplier, Ray &r){

	float3 refl = r.dir - n * 2 * dot(n, r.dir);

	r.origin = x;
	r.dir = refl;
	multiplier *= albedo;

}

__device__ inline void refr(/*curandStatePhilox4_32_10_t*/ curandState_t *state, int idx, float3 n, float3 nl, float3 x, float3 albedo, float3 &multiplier, Ray &r){

	Ray reflRay(x, r.dir - n * 2 * dot(n,r.dir));
	bool into = dot(n,nl) > 0;
	float nc = 1;
	float nt = 1.5;
	float nnt = into ? nc / nt : nt / nc;
	float cosTheta = dot(r.dir, nl);
	float cosTheta2Sqr;

	if ((cosTheta2Sqr = 1 - nnt * nnt*(1 - cosTheta * cosTheta)) < 0){
		r.origin = reflRay.origin;
		r.dir = reflRay.dir;
		multiplier *= albedo;
		return;
	} 

	float3 tdir = normalize(r.dir*nnt - n * ((into ? 1 : -1)*(cosTheta*cosTheta + sqrt(cosTheta2Sqr))));

	float a = nt - nc;
	float b = nt + nc;
	float R0 = a * a / (b*b);
	float cosTheta2 = dot(tdir, n);
	float c = 1 - (into ? -cosTheta : cosTheta2);
	float Re = R0 + (1 - R0)*c*c*c*c*c;
	float Tr = 1 - Re;

	float P = .25 + .5*Re;
	float RP = Re / P;
	float TP = Tr / (1 - P);

	// Russian roulette decision (between reflected and refracted ray)
	if (curand_uniform(&state[idx]) < P){//reflect
		r.origin = reflRay.origin;
		r.dir = reflRay.dir;
		multiplier = albedo * RP;
	}
	else{//refract
		r.origin = x;
		r.dir = tdir;
		multiplier *= albedo * TP;
	}
}


__device__ inline float3 radiance(Ray r, /*curandStatePhilox4_32_10_t*/ curandState_t *state, int idx){

	float3 result = make_float3(0);
	float3 multiplier = make_float3(1);

	#pragma unroll
	for(int i = 0; i < DEPTH; i++){

		float t = FLT_MAX;
		int id = 0;
		if (!intersect(r, t, id)) return make_float3(0); // miss

		const Sphere &obj = spheres[id];
		float3 x = r.origin + r.dir*t;
		float3 n = obj.normal(x);
		float3 nl = dot(n, r.dir) < 0 ? n : n * -1;
		float3 albedo = obj.color;

		result += obj.emision * multiplier;

		switch(obj.refl){
			case DIFF: diff(state, idx, nl, x, albedo, multiplier, r); break;
			case SPEC: spec(state, idx, n, x, albedo, multiplier, r); break;
			case REFR: refr(state, idx, n, nl, x, albedo, multiplier, r); break;
		}

	}

	return result;
}


/*
__device__ float3 radiance_recursive(const Ray &r, int depth, curandStatePhilox4_32_10_t *state, int idx) {
	// Limit max depth (or you'll run into a stackoverflow on some scenes)
	if (depth > DEPTH) return make_float3(0);


	float t;                                 // distance to intersection
	int id = 0;                               // id of intersected object
	if (!intersect(r, t, id)) return make_float3(0); // if miss, return black


	const Sphere &obj = spheres[id];        // the hit object
	// intersection point
	float3 x = r.origin + r.dir*t;
	// changed to use a generic method to compute the normal
	float3 n = obj.normal(x);
	// find the correct facing normal (if the ray is on the 'inside' when hitting obj, then flip it)
	float3 nl = dot(n, r.dir) < 0 ? n : n * -1;
	// albedo
	float3 albedo = obj.color;

	// Russian Roulette:
	// probability to continue the ray (the less reflective the material, the lower)
	// modified to use luma rather than max(r, max(g, b))
	float russianRouletteProb = luma(albedo); 
	// Apply Russian Roulette only after the 5th bounce, and increment depth by 1
	if (++depth > 5) 
		if (curand_uniform(&state[idx]) < russianRouletteProb) 
			albedo /= russianRouletteProb; // boost the ray to compesnate for the probability of terminating it
		else 
			return obj.emision; // terminate the ray

	if (obj.refl == DIFF) {                  // Ideal DIFFUSE reflection
		// generate cosine weighted points on the upper hemisphere through inverse transform sampling:
		// pdf = cos(theta) * sin(theta) / pi
		// integrating out phi ([0,2pi]), and then integrating over theta yields the marginal CDF for theta: 
		// (1-cos^2(theta))/2 = r2 -> cos(theta) = sqrt(1-r2) ~ sqrt(r2)
		// the last equivalence follows from 1-r2, and r2 being identically distributed
		// since the joint pdf is separable we don't need to use the conditional distribution and can integrate out theta ([0,pi/2])
		// and then integrate over phi: phi / (2*pi) = r1 -> phi = 2 * pi * r1
		float phi = 2 * M_PI * curand_uniform(&state[idx]);
		float r2 = curand_uniform(&state[idx]);
		float sinTheta = sqrt(r2);
		float cosTheta = sqrt(1 - r2);

		// use correct facing normal for the central vector of the hemisphere
		float3 w = nl;
		// build an orthonormal basis from it
		float3 u = normalize(cross(fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0), w));
		float3 v = cross(w, u);
		// transform the generated points (in spherical coordinates) into the orthonormal basis defined by the correct facing normal
		// use the sampled point as a direction for the bounce
		float3 d = normalize(u*cos(phi)*sinTheta + v * sin(phi)*sinTheta + w * cosTheta);

		// From the estimator of the rendering equation (integrand / pdf):
		// radiance += emitted + brdf * radiance * cos(theta)/cos(theta) * PI
		// note that the cosine term in the integrand gets canceled by the cosine weighted pdf from which we sample the random direction
		// we assume that PI was already encoded in the brdf
		return obj.emision + albedo*radiance_recursive(Ray(x, d), depth, state, idx);
	} 
	else if (obj.refl == SPEC)            // Ideal SPECULAR reflection
	{
		// reflection around the normal:
		//v___| n
		// \  |
		//  \ | -dot(d,n)*n
		// d \| 

		// v ___|___ v
		//   \  |  /
		//  d \ | / r
		//     \|/
		//

		// v = d - dot(d,n)*n
		// r = -d + 2*v = -d + 2*d -2*dot(d,n)*n = d - 2*dot(d,n)*n
		// r = r.d - 2.0*dot(r.d,n)*n;

		// reflect the ray around the normal
	    float3 refl = r.dir - n * 2 * dot(n, r.dir);

		// rendering equation estimator (integrand/pdf) for mirror brdf
		// mirror brdf = albedo * delta(theta_refl-theta_in) * delta(phi_refl - phi_in +- pi)  / (cos(theta_in) * sin(theta_in))
		return obj.emision + albedo * radiance_recursive(Ray(x, refl), depth, state, idx);
	}
	
	// refraction in the plane defined by d and n:
	// sin(theta1)*eta1 = sin(theta2)*eta2
	// sin(theta2) = sin(theta1) * eta1 / eta2
	// if eta1/eta2>1, it is possible that |sin(theta1)*eta1/eta2|>1
	// there is no angle theta1 that satisifies this -> total internal reflection
	// otherwise:
	//\  |   eta1
	// \ |      
	//d \| n          theta1 = angle(-d,n)
	//---------
	//   ||   eta2    theta2 = angle(r,-n)
	//   | |
	//-n |  | r

	// r = cos(theta2)*(-n) + sin(theta2)*perp(n)
	// perp(n) = d - dot(d,n)*n / |d - dot(d,n)*n| = (d - cos(theta1)*n)/sin(theta1)
	// cos(theta2) = sqrt(1-sin^2(theta2)) = sqrt(1-(eta1/eta2*sin(theta1))^2)
	// sin(theta2) = eta1/eta2*sin(theta1)
	// r = cos(theta2)*(-n) + eta1/eta2*sin(theta1)/sin(theta1)* (d - cos(theta1)*n)
	// r = -sqrt(1-(eta1/eta2*sin(theta1))^2)*n + eta1/eta2*(d-cos(theta1)*n)
	// r = eta1/eta2*d + (eta1/eta2*dot(d,n)-sqrt(1-(eta1/eta2)^2*(1-dot(d,n)^2)))*n


	Ray reflRay(x, r.dir - n * 2 * dot(n,r.dir));     // Ideal dielectric REFRACTION
	bool into = dot(n,nl) > 0;                    // Ray from outside going in?
	// indices of refraction
	float nc = 1;
	float nt = 1.5;
	// indices of refraction ratio
	float nnt = into ? nc / nt : nt / nc;
	// cosTheta
	float cosTheta = dot(r.dir, nl);
	float cosTheta2Sqr;

	if ((cosTheta2Sqr = 1 - nnt * nnt*(1 - cosTheta * cosTheta)) < 0)    // Total internal reflection
		return obj.emision + albedo*radiance_recursive(reflRay, depth, state, idx);

	// refracted ray direction
	float3 tdir = normalize(r.dir*nnt - n * ((into ? 1 : -1)*(cosTheta*cosTheta + sqrt(cosTheta2Sqr))));

	// Schlick's Fresnel approximation:  Schlick, Christophe, An Inexpensive BDRF Model for Physically based Rendering,
	float a = nt - nc;
	float b = nt + nc;
	float R0 = a * a / (b*b);
	float cosTheta2 = dot(tdir, n);
	float c = 1 - (into ? -cosTheta : cosTheta2);
	float Re = R0 + (1 - R0)*c*c*c*c*c; // reflection weight
	float Tr = 1 - Re; // refraction weight

	// Russian roulette probability (for reflection)
	float P = .25 + .5*Re;
	// reflection weight boosted by the russian roulette probability
	float RP = Re / P;
	// refraction weight boosted by the russian roulette probability
	float TP = Tr / (1 - P);

	// splitting below depth 3
	if (depth < 3)
	{
		return obj.emision + albedo*(radiance_recursive(reflRay, depth, state, idx)*Re + radiance_recursive(Ray(x, tdir), depth, state, idx)*Tr); // weighted relfection + refraction based on fresnel
	}
	else
	{
		// Russian roulette decision (between reflected and refracted ray)
		if (curand_uniform(&state[idx]) < P)
			return obj.emision + albedo * radiance_recursive(reflRay, depth, state, idx)*RP; //reflect
		else
			return obj.emision + albedo * radiance_recursive(Ray(x, tdir), depth, state, idx)*TP; //refract
	}
}
*/
#endif
