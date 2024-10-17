#ifndef VECTOR_CUH
#define VECTOR_CUH

#include <cmath>
#include <vector>

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3();
    __host__ __device__ Vec3(float x, float y, float z);
};

struct Vec4 {
    float x, y, z, w;
    __host__ __device__ Vec4();
    __host__ __device__ Vec4(float x, float y, float z, float w);
};

#endif