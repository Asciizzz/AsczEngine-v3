#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <Vector.cuh>

// Note: & is matrix multiplication, * is element-wise multiplication

struct Mat4f {
    float data[4][4] = {0};
    __host__ __device__ Mat4f();
    __host__ __device__ Mat4f(float data[4][4]);

    // Basic operations
    __host__ __device__ Mat4f operator+(const Mat4f &mat);
    __host__ __device__ Mat4f operator-(const Mat4f &mat);
    __host__ __device__ Mat4f operator*(const float scl);
    // Advanced operations
    __host__ __device__ Vec4f operator*(const Vec4f &vec);
    __host__ __device__ Mat4f operator*(const Mat4f &mat);
    __host__ __device__ float det(); // Determinant

    void print() { // Will remove this later
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                printf("%f ", data[i][j]);
            }
            printf("\n");
        }
    }
};

#endif