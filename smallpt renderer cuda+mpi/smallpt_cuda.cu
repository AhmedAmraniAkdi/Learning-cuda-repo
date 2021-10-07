// https://github.com/vchizhov/smallpt-explained/blob/master/smallpt_explained.cpp

#define GL_GLEXT_PROTOTYPES

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <cuda_gl_interop.h>
#include "GL/glut.h" 
#include <curand_kernel.h>
#include <math.h>

#define W 1024
#define H 768
#define samps 64 // samples per subpixel

#define BLOCKDIM 128
#define DEPTH 10 // max depth of raduance recursion 

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

//// Ray

struct Ray{
    float3 origin, dir;
    __device__ Ray(float3 origin, float3 dir) : origin(origin), dir(dir) {}
};

//// Spheres

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

struct Sphere {
	float radius;
	float3 pos, emision, color;
	Refl_t refl;
	__device__ Sphere(float r, float3 p, float3 e, float3 c, Refl_t refl) :
		radius(r), pos(p), emision(e), color(c), refl(refl) {}
	__device__ float intersect(const Ray &r) const {
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
__constant__ Sphere spheres[] = {//Scene: radius, position, emission, color, material
  Sphere(1e5, make_float3(1e5 + 1,40.8,81.6), make_float3(0),make_float3(.75,.25,.25),DIFF),//Left
  Sphere(1e5, make_float3(-1e5 + 99,40.8,81.6),make_float3(0),make_float3(.25,.25,.75),DIFF),//Rght
  Sphere(1e5, make_float3(50,40.8, 1e5),     make_float3(0),make_float3(.75,.75,.75),DIFF),//Back
  Sphere(1e5, make_float3(50,40.8,-1e5 + 170), make_float3(0),make_float3(0),           DIFF),//Frnt
  Sphere(1e5, make_float3(50, 1e5, 81.6),    make_float3(0),make_float3(.75,.75,.75),DIFF),//Botm
  Sphere(1e5, make_float3(50,-1e5 + 81.6,81.6),make_float3(0),make_float3(.75,.75,.75),DIFF),//Top
  Sphere(16.5,make_float3(27,16.5,47),       make_float3(0),make_float3(1,1,1)*.999, SPEC),//Mirr
  Sphere(16.5,make_float3(73,16.5,78),       make_float3(0),make_float3(1,1,1)*.999, REFR),//Glas
  Sphere(600, make_float3(50,681.6 - .27,81.6),make_float3(12,12,12),  make_float3(0), DIFF) //Lite
};

__device__ int toInt(float x) { return int(pow(clamp(x, 0.0, 1.0), 1 / 2.2) * 255 + .5); }

__device__ bool intersect(const Ray &r, float &t, int &id) {
	int n = sizeof(spheres) / sizeof(Sphere);
	float d;
	float inf = t = FLT_MAX;
	float eps = 1e-4;
	for (int i = n; i--;) 
		if ((d = spheres[i].intersect(r)) > eps && d < t){ 
			t = d; 
			id = i; 
		}
	return t < inf;
}

//// Ray tracing (radiance)

__device__ inline float luma(const float3& color){
	return dot(color, make_float3(0.2126f, 0.7152f, 0.0722f));
}

float3 radiance(const Ray &r, int depth, curandState *state, int idx) {
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
		return obj.emision + albedo*radiance(Ray(x, d), depth, state, idx);
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
		return obj.emision + albedo * radiance(Ray(x, refl), depth, state, idx);
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
		return obj.emision + albedo*radiance(reflRay, depth, state, idx);

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
		return obj.emision + albedo*(radiance(reflRay, depth, state, idx)*Re + radiance(Ray(x, tdir), depth, state, idx)*Tr); // weighted relfection + refraction based on fresnel
	}
	else
	{
		// Russian roulette decision (between reflected and refracted ray)
		if (curand_uniform(&state[idx]) < P)
			return obj.emision + albedo * radiance(reflRay, depth, state, idx)*RP; //reflect
		else
			return obj.emision + albedo * radiance(Ray(x, tdir), depth, state, idx)*TP; //refract
	}
		
}

//// kernel

__global__ void smallpt_kernel(float3 *d_img, curandState_t *state, float3 cx, float3 cy, Ray cam){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int total_idx = idy * W + idx;

    if(total_idx >= H * W) return;

    int i = (H - idy - 1)*W + idx; // img comes reversed

    d_img += i;

    curand_init(total_idx, total_idx, 0, &state[total_idx]);

    float3 r = make_float3(0);

    float3 acum = make_float3(0);

    #pragma unroll
    for(int sy = 0; sy < 2; sy++){

        #pragma unroll
        for(int sx = 0; sx < 2; sx++, r = make_float3(0)){

            #pragma unroll
            for(int s = 0; s < samps ; s++){

                // Bartlett window / tent function / triangular function random sampling
                // f(x) = 1 - |x|, |x|<=1, else f(x) = 0
                // Let F(x) be the indefinite integral of f(x): x in [-1,0], F(x) = x + x^2/2 + C; x in [0,1] F(x) = x - x^2/2 + C
                // We can use the inverse transform sampling method to sample from the tent pdf, we'll first split it into 2 pdfs:
                // in [-1,0] and [0,1], since both intervals have equal probability (1/2) 
                // Then for r1<=0.5 sample from the first, for r1>0.5 from the second

                // set r1' = 2 * r1

                // renormalize pdfs: f'(x) = 2 * f(x)
                // r1'<=1 -> 
                // -1<= x <= 0, F(x) - F(-1) = 2x + x^2 + 2 - 1 = (x+1)^2 = r1', then:
                //  x = sqrt(r1')-1

                // r1' > 1 ->
                // 0<=x<=1, F(x) - F(0) = 2x - x^2 = -(x^2 - 2x + 1) +1 = -(x-1)^2 + 1 = r1' - 1
                // x = 1 - sqrt(2-r1'), note that the negative root is taken since x<=1
                float r1 = 2 * curand_uniform (&state[total_idx]);
                float dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                // the joint pdf is separable, so similarly for y:
                float r2 = 2 * curand_uniform (&state[total_idx]);
                float dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);

                // primary ray direction
                float3 d = cx * (((sx + .5 + dx) / 2 + idx) / W - .5) +
                    cy * (((sy + .5 + dy) / 2 + idy) / H - .5) + cam.dir;
                
                // start the ray from the inside of the cornell 'box'
                // and accumulate the radiance per subpixel
                r = r + radiance(Ray(cam.origin + d * 140, normalize(d)), 0, state, total_idx)*(1. / samps);

            }

            acum = acum + clamp(r, 0, 1) *.25;
        }
    }

    d_img[0] = acum;

}

//// main

int main(){

    // https://en.wikipedia.org/wiki/Ray_tracing_(graphics)#Calculate_rays_for_rectangular_viewport

    Ray cam(make_float3(50, 52 , 295.6), normalize(make_float3(0, -0.042612, -1)));

    float aspectRatio = W/H;

    float vfov = 0.502643;
    float fovScale = 2 * tan(0.5*vfov);

    float3 cx = make_float3(aspectRatio, 0, 0) * fovScale;
    float3 cy = normalize(cross(cx, cam.dir)) * fovScale;

    float3 h_img[W*H];
    memset(h_img, 0, sizeof(float3) * H * W);

    // cuda variables

    curandState_t *devStates;
    cudaMalloc((void **)&devStates, sizeof(curandState) * W * H);

    float3 *d_img;
    cudaMalloc((void **)&d_img, sizeof(float3) * H * W);

    dim3 dimBlock(BLOCKDIM, BLOCKDIM);
    dim3 dimGrid((W + BLOCKDIM - 1)/BLOCKDIM, (H + BLOCKDIM - 1)/BLOCKDIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;

    cudaEventRecord(start, 0);
    smallpt_kernel<<<dimGrid, dimBlock>>>(d_img, devStates, cx, cy, cam);
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "Ray Tracing time: %.8f ms\n", elapsed_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_img);
    cudaFree(devStates);

    return 0;
}