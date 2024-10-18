#include <Vector.cuh>
#include <Matrix.cuh>

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
__host__ __device__ void Vec3f::operator+=(const Vec3f& v) {
    x += v.x; y += v.y; z += v.z;
}
__host__ __device__ void Vec3f::operator-=(const Vec3f& v) {
    x -= v.x; y -= v.y; z -= v.z;
}
__host__ __device__ void Vec3f::operator*=(const float scalar) {
    x *= scalar; y *= scalar; z *= scalar;
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

// Transformations
__host__ __device__ Vec3f Vec3f::translate(Vec3f& vec, const Vec3f& t) {
    return vec + t;
}
__host__ __device__ Vec3f Vec3f::rotate(Vec3f& vec, const Vec3f& origin, const Vec3f& rot) {
    // Translate to origin
    Vec3f diff = vec - origin;

    float cosX = cos(rot.x), sinX = sin(rot.x);
    float cosY = cos(rot.y), sinY = sin(rot.y);
    float cosZ = cos(rot.z), sinZ = sin(rot.z);

    // Rotation matrices
    float rX[4][4] = {
        {1, 0, 0, 0},
        {0, cosX, -sinX, 0},
        {0, sinX, cosX, 0},
        {0, 0, 0, 1}
    };
    float rY[4][4] = {
        {cosY, 0, sinY, 0},
        {0, 1, 0, 0},
        {-sinY, 0, cosY, 0},
        {0, 0, 0, 1}
    };
    float rZ[4][4] = {
        {cosZ, -sinZ, 0, 0},
        {sinZ, cosZ, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    Mat4f rMat = Mat4f(rX) * Mat4f(rY) * Mat4f(rZ);

    Vec4f rVec4 = rMat * diff.toVec4f();
    Vec3f rVec3 = rVec4.toVec3f();
    rVec3 += origin;

    return rVec3;
}
__host__ __device__ Vec3f Vec3f::scale(Vec3f& vec, const Vec3f& origin, const Vec3f& scl) {
    Vec3f diff = vec - origin;
    return Vec3f(
        origin.x + diff.x * scl.x,
        origin.y + diff.y * scl.y,
        origin.z + diff.z * scl.z
    );
}
__host__ __device__ Vec3f Vec3f::scale(Vec3f& vec, const Vec3f& origin, const float scl) {
    return scale(vec, origin, Vec3f(scl, scl, scl));
}

// Transformations but on self
__host__ __device__ void Vec3f::translate(const Vec3f& t) {
    *this += t;
}
__host__ __device__ void Vec3f::rotate(const Vec3f& origin, const Vec3f& rot) {
    *this = rotate(*this, origin, rot);
}
__host__ __device__ void Vec3f::scale(const Vec3f& origin, const Vec3f& scl) {
    *this = scale(*this, origin, scl);
}
__host__ __device__ void Vec3f::scale(const Vec3f& origin, const float scl) {
    *this = scale(*this, origin, scl);
}

// VEC4
__host__ __device__ Vec4f::Vec4f() : x(0), y(0), z(0), w(0) {}
__host__ __device__ Vec4f::Vec4f(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
__host__ __device__ Vec3f Vec4f::toVec3f() {
    return Vec3f(x / w, y / w, z / w);
}