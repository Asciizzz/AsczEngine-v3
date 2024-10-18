#include <Vector.cuh>

// VEC2f
__host__ __device__ Vec2f::Vec2f() : x(0), y(0) {}
__host__ __device__ Vec2f::Vec2f(float x, float y) : x(x), y(y) {}

// VEC3uli (unsigned long int)
__host__ __device__ Vec3uli::Vec3uli() : x(0), y(0), z(0) {}
__host__ __device__ Vec3uli::Vec3uli(int x, int y, int z) : x(x), y(y), z(z) {}
__host__ __device__ void Vec3uli::operator+=(unsigned long int d) {
    x += d; y += d; z += d;
}

// VEC3f
__host__ __device__ Vec3f::Vec3f() : x(0), y(0), z(0) {}
__host__ __device__ Vec3f::Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}
__host__ __device__ Vec4f Vec3f::toVec4f() {
    return Vec4f(x, y, z, 1);
}
__host__ __device__ Vec3f Vec3f::operator+(const Vec3f& v) {
    return Vec3f(x + v.x, y + v.y, z + v.z);
}
__host__ __device__ Vec3f Vec3f::operator-(const Vec3f& v) {
    return Vec3f(x - v.x, y - v.y, z - v.z);
}
__host__ __device__ Vec3f Vec3f::operator*(const float scalar) {
    return Vec3f(x * scalar, y * scalar, z * scalar);
}
__host__ __device__ float Vec3f::operator*(const Vec3f& v) {
    return x * v.x + y * v.y + z * v.z;
}
__host__ __device__ Vec3f Vec3f::operator&(const Vec3f& v) {
    return Vec3f(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
}
__host__ __device__ float Vec3f::mag() {
    return sqrt(x * x + y * y + z * z);
}
__host__ __device__ void Vec3f::norm() {
    float m = mag();
    x /= m; y /= m; z /= m;
}

// VEC4
__host__ __device__ Vec4f::Vec4f() : x(0), y(0), z(0), w(0) {}
__host__ __device__ Vec4f::Vec4f(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
__host__ __device__ Vec3f Vec4f::toVec3f() {
    return Vec3f(x, y, z);
}