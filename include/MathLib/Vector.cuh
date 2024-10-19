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

#define Vecs2f std::vector<Vec2f>
#define Vecs3f std::vector<Vec3f>
#define Vecs4f std::vector<Vec4f>
#define Vecs3uli std::vector<Vec3uli>

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

struct Vec3uli { // For faces indices
    unsigned long int x, y, z;
    __host__ __device__ Vec3uli();
    __host__ __device__ Vec3uli(int x, int y, int z);

    __host__ __device__ void operator+=(unsigned long int d);
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

    // We only need operator + - and scalar multiplication
    __host__ __device__ Vec4f operator+(const Vec4f &vec);
    __host__ __device__ Vec4f operator-(const Vec4f &vec);
    __host__ __device__ Vec4f operator*(const float scl);
    __host__ __device__ Vec4f operator/(const float scl);
};

#endif