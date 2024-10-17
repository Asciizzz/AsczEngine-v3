#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <Vector.cuh>

struct Mat4 {
    float data[4][4] = {0};
    __host__ __device__ Mat4();
    __host__ __device__ Mat4(float data[4][4]);

    __host__ __device__ Vec4 operator*(const Vec4& vec);
    __host__ __device__ Mat4 operator*(const Mat4& other);
    __host__ __device__ Mat4 operator*(const float scalar);
    __host__ __device__ Mat4 operator+(const Mat4& other);
    __host__ __device__ Mat4 operator-(const Mat4& other);
};

#endif