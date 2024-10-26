#ifndef VECTOR_CUH
#define VECTOR_CUH

#include <iostream>
#include <vector>
#include <cstdio>
#include <cmath>

#define UInt unsigned int
#define ULInt unsigned long int
#define ULLInt unsigned long long int
#define UInts std::vector<UInt>
#define ULLInts std::vector<ULLInt>

#define Vecs2f std::vector<Vec2f>
#define Vecs3f std::vector<Vec3f>
#define Vecs4f std::vector<Vec4f>

#define M_PI 3.14159265358979323846 // 180 degrees
#define M_PI_2 1.57079632679489661923 // 90 degrees
#define M_2_PI 6.28318530717958647692 // 360 degrees

struct Vec2f {
    float x, y;
    __host__ __device__ Vec2f();
    __host__ __device__ Vec2f(float x, float y);

    // Basic operations
    __host__ __device__ Vec2f operator+(const Vec2f &vec);
    __host__ __device__ Vec2f operator+(const float t);
    __host__ __device__ Vec2f operator-(const Vec2f &vec);
    __host__ __device__ Vec2f operator-(const float t);
    __host__ __device__ Vec2f operator*(const float scl);
    __host__ __device__ Vec2f operator/(const float scl);
};

struct Vec4f; // Forward declaration
struct Vec3f {
    float x, y, z;
    __host__ __device__ Vec3f();
    __host__ __device__ Vec3f(float x, float y, float z);
    __host__ __device__ Vec3f(float a);
    __host__ __device__ Vec4f toVec4f();

    // Basic operations
    __host__ __device__ Vec3f operator+(const Vec3f &vec);
    __host__ __device__ Vec3f operator+(const float t);
    __host__ __device__ Vec3f operator-(const Vec3f &vec);
    __host__ __device__ Vec3f operator-(const float t);
    __host__ __device__ Vec3f operator*(const float scl);
    __host__ __device__ Vec3f operator/(const float scl);
    __host__ __device__ void operator+=(const Vec3f &vec);
    __host__ __device__ void operator-=(const Vec3f &vec);
    __host__ __device__ void operator*=(const float scl);
    __host__ __device__ void operator/=(const float scl);
    // Advanced operations
    __host__ __device__ float operator*(const Vec3f &vec); // Dot product
    __host__ __device__ Vec3f operator&(const Vec3f &vec); // Cross product
    __host__ __device__ float mag(); // Magnitude
    __host__ __device__ void norm(); // Normalize
    // Special operations
    __host__ __device__ static Vec3f bary(Vec2f v, Vec2f v0, Vec2f v1, Vec2f v2); // Barycentric coordinates
    __host__ __device__ void limit(float min, float max); // Limit the vector

    // Transformations
    __host__ __device__ static Vec3f translate(Vec3f &vec, const Vec3f &t);
    __host__ __device__ static Vec3f rotate(Vec3f &vec, const Vec3f &origin, const Vec3f &rot);
    __host__ __device__ static Vec3f scale(Vec3f &vec, const Vec3f &origin, const Vec3f &scl);
    __host__ __device__ static Vec3f scale(Vec3f &vec, const Vec3f &origin, const float scl);
    // Transformations but on self
    __host__ __device__ void translate(const Vec3f &t);
    __host__ __device__ void rotate(const Vec3f &origin, const Vec3f &rot);
    __host__ __device__ void scale(const Vec3f &origin, const Vec3f &scl);
    __host__ __device__ void scale(const Vec3f &origin, const float scl);
};

struct Vec4f {
    float x, y, z, w;
    __host__ __device__ Vec4f();
    __host__ __device__ Vec4f(float x, float y, float z, float w);
    __host__ __device__ Vec3f toVec3f(bool norm=true); // From Homogeneous to Cartesian

    __host__ __device__ Vec4f operator+(const Vec4f &vec);
    __host__ __device__ Vec4f operator+(const float t);
    __host__ __device__ Vec4f operator-(const Vec4f &vec);
    __host__ __device__ Vec4f operator-(const float t);
    __host__ __device__ Vec4f operator*(const float scl);
    __host__ __device__ Vec4f operator/(const float scl);

    __host__ __device__ void limit(float min, float max); // Limit the vector
};

// SoA structure Vecs

struct Vecptr2f {
    float *x, *y;
    ULLInt size;

    void malloc(ULLInt size);
    void free();
    void operator+=(Vecptr2f &vec);
};
struct Vecptr3f {
    float *x, *y, *z;
    ULLInt size;

    void malloc(ULLInt size);
    void free();

    void operator+=(Vecptr3f &vec);
};
struct Vecptr4f {
    float *x, *y, *z, *w;
    ULLInt size;

    void malloc(ULLInt size);
    void free();
    void operator+=(Vecptr4f &vec);
};

// Mainly used for faces
struct Vecptr4ulli {
    // Vertex, texture, normal, and objId
    ULLInt *v, *t, *n, *o;
    ULLInt size;

    void malloc(ULLInt size);
    void free();
    void operator+=(Vecptr4ulli &vec);
};

// Atomic functions for float
__device__ bool atomicMinFloat(float* addr, float value);

#endif