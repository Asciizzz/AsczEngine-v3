#include <Vector.cuh>
#include <Matrix.cuh>

// VEC2f
Vec2f::Vec2f() : x(0), y(0) {}
Vec2f::Vec2f(float x, float y) : x(x), y(y) {}
Vec2f Vec2f::operator+(const Vec2f& v) {
    return Vec2f(x + v.x, y + v.y);
}
Vec2f Vec2f::operator+(const float t) {
    return Vec2f(x + t, y + t);
}
Vec2f Vec2f::operator-(const Vec2f& v) {
    return Vec2f(x - v.x, y - v.y);
}
Vec2f Vec2f::operator-(const float t) {
    return Vec2f(x - t, y - t);
}
Vec2f Vec2f::operator*(const float scl) {
    return Vec2f(x * scl, y * scl);
}
Vec2f Vec2f::operator/(const float scl) {
    return Vec2f(x / scl, y / scl);
}

// VEC3f
Vec3f::Vec3f() : x(0), y(0), z(0) {}
Vec3f::Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}
Vec3f::Vec3f(float a) : x(a), y(a), z(a) {}
Vec4f Vec3f::toVec4f() {
    return Vec4f(x, y, z, 1);
}

Vec3f Vec3f::operator+(const Vec3f& v) {
    return Vec3f(x + v.x, y + v.y, z + v.z);
}
Vec3f Vec3f::operator+(const float t) {
    return Vec3f(x + t, y + t, z + t);
}
Vec3f Vec3f::operator-(const Vec3f& v) {
    return Vec3f(x - v.x, y - v.y, z - v.z);
}
Vec3f Vec3f::operator-(const float t) {
    return Vec3f(x - t, y - t, z - t);
}
Vec3f Vec3f::operator*(const float scl) {
    return Vec3f(x * scl, y * scl, z * scl);
}
Vec3f Vec3f::operator/(const float scl) {
    return Vec3f(x / scl, y / scl, z / scl);
}
void Vec3f::operator+=(const Vec3f& v) {
    x += v.x; y += v.y; z += v.z;
}
void Vec3f::operator-=(const Vec3f& v) {
    x -= v.x; y -= v.y; z -= v.z;
}
void Vec3f::operator*=(const float scl) {
    x *= scl; y *= scl; z *= scl;
}
void Vec3f::operator/=(const float scl) {
    x /= scl; y /= scl; z /= scl;
}

float Vec3f::operator*(const Vec3f& v) {
    return x * v.x + y * v.y + z * v.z;
}
Vec3f Vec3f::operator&(const Vec3f& v) {
    return Vec3f(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
}
float Vec3f::mag() {
    return sqrt(x * x + y * y + z * z);
}
void Vec3f::norm() {
    float m = mag();
    x /= m; y /= m; z /= m;
}

Vec3f Vec3f::bary(Vec2f v, Vec2f v0, Vec2f v1, Vec2f v2) {
    float d = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
    float a = ((v1.y - v2.y) * (v.x - v2.x) + (v2.x - v1.x) * (v.y - v2.y)) / d;
    float b = ((v2.y - v0.y) * (v.x - v2.x) + (v0.x - v2.x) * (v.y - v2.y)) / d;
    float c = 1 - a - b;
    return Vec3f(a, b, c);
}
void Vec3f::limit(float min, float max) {
    x = std::max(min, std::min(x, max));
    y = std::max(min, std::min(y, max));
    z = std::max(min, std::min(z, max));
}

// Transformations
Vec3f Vec3f::translate(Vec3f& vec, const Vec3f& t) {
    return vec + t;
}

Vec3f Vec3f::rotateX(Vec3f &vec, const Vec3f &origin, const float rx) {
    Vec3f diff = vec - origin;
    Vec4f diff4 = diff.toVec4f();

    float cosX = cos(rx), sinX = sin(rx);
    float rX[4][4] = {
        {1, 0, 0, 0},
        {0, cosX, -sinX, 0},
        {0, sinX, cosX, 0},
        {0, 0, 0, 1}
    };

    Vec4f rVec4 = Mat4f(rX) * diff4;
    Vec3f rVec3 = rVec4.toVec3f();
    rVec3 += origin;

    return rVec3;
}
Vec3f Vec3f::rotateY(Vec3f &vec, const Vec3f &origin, const float ry) {
    Vec3f diff = vec - origin;
    Vec4f diff4 = diff.toVec4f();

    float cosY = cos(ry), sinY = sin(ry);
    float rY[4][4] = {
        {cosY, 0, sinY, 0},
        {0, 1, 0, 0},
        {-sinY, 0, cosY, 0},
        {0, 0, 0, 1}
    };

    Vec4f rVec4 = Mat4f(rY) * diff4;
    Vec3f rVec3 = rVec4.toVec3f();
    rVec3 += origin;

    return rVec3;
}
Vec3f Vec3f::rotateZ(Vec3f &vec, const Vec3f &origin, const float rz) {
    Vec3f diff = vec - origin;
    Vec4f diff4 = diff.toVec4f();

    float cosZ = cos(rz), sinZ = sin(rz);
    float rZ[4][4] = {
        {cosZ, -sinZ, 0, 0},
        {sinZ, cosZ, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };

    Vec4f rVec4 = Mat4f(rZ) * diff4;
    Vec3f rVec3 = rVec4.toVec3f();
    rVec3 += origin;

    return rVec3;
}

Vec3f Vec3f::scale(Vec3f& vec, const Vec3f& origin, const Vec3f& scl) {
    Vec3f diff = vec - origin;
    return Vec3f(
        origin.x + diff.x * scl.x,
        origin.y + diff.y * scl.y,
        origin.z + diff.z * scl.z
    );
}
Vec3f Vec3f::scale(Vec3f& vec, const Vec3f& origin, const float scl) {
    return scale(vec, origin, Vec3f(scl, scl, scl));
}

// Transformations but on self
void Vec3f::translate(const Vec3f& t) {
    *this += t;
}

void Vec3f::rotateX(const Vec3f& origin, const float rx) {
    *this = rotateX(*this, origin, rx);
}
void Vec3f::rotateY(const Vec3f& origin, const float ry) {
    *this = rotateY(*this, origin, ry);
}
void Vec3f::rotateZ(const Vec3f& origin, const float rz) {
    *this = rotateZ(*this, origin, rz);
}

void Vec3f::scale(const Vec3f& origin, const Vec3f& scl) {
    *this = scale(*this, origin, scl);
}
void Vec3f::scale(const Vec3f& origin, const float scl) {
    *this = scale(*this, origin, scl);
}

// VEC4
Vec4f::Vec4f() : x(0), y(0), z(0), w(0) {}
Vec4f::Vec4f(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
Vec3f Vec4f::toVec3f(bool norm) {
    if (norm) return Vec3f(x / w, y / w, z / w);
    return Vec3f(x, y, z);
}
Vec4f Vec4f::operator+(const Vec4f& v) {
    return Vec4f(x + v.x, y + v.y, z + v.z, w + v.w);
}
Vec4f Vec4f::operator+(const float t) {
    return Vec4f(x + t, y + t, z + t, w + t);
}
Vec4f Vec4f::operator-(const Vec4f& v) {
    return Vec4f(x - v.x, y - v.y, z - v.z, w - v.w);
}
Vec4f Vec4f::operator-(const float t) {
    return Vec4f(x - t, y - t, z - t, w - t);
}
Vec4f Vec4f::operator*(const float scl) {
    return Vec4f(x * scl, y * scl, z * scl, w * scl);
}
Vec4f Vec4f::operator/(const float scl) {
    return Vec4f(x / scl, y / scl, z / scl, w / scl);
}
void Vec4f::limit(float min, float max) {
    x = std::max(min, std::min(x, max));
    y = std::max(min, std::min(y, max));
    z = std::max(min, std::min(z, max));
    w = std::max(min, std::min(w, max));
}

// SoA structure Vecs

void Vec2f_ptr::malloc(ULLInt size) {
    this->size = size;
    cudaMalloc(&x, size * sizeof(float));
    cudaMalloc(&y, size * sizeof(float));
}
void Vec2f_ptr::free() {
    this->size = 0;
    cudaFree(x);
    cudaFree(y);
}
void Vec2f_ptr::operator+=(Vec2f_ptr& vec) {
    Vec2f_ptr newVec;
    newVec.malloc(size + vec.size);

    // Copy original data
    cudaMemcpy(newVec.x, x, size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.y, y, size * sizeof(float), cudaMemcpyDeviceToDevice);

    // Copy new data
    cudaMemcpy(newVec.x + size, vec.x, vec.size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.y + size, vec.y, vec.size * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free datas
    free();
    vec.free();

    // Update
    *this = newVec;
}

void Vec3f_ptr::malloc(ULLInt size) {
    this->size = size;
    cudaMalloc(&x, size * sizeof(float));
    cudaMalloc(&y, size * sizeof(float));
    cudaMalloc(&z, size * sizeof(float));
}
void Vec3f_ptr::free() {
    this->size = 0;
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
}
void Vec3f_ptr::operator+=(Vec3f_ptr& vec) {
    Vec3f_ptr newVec;
    newVec.malloc(size + vec.size);

    // Copy original data
    cudaMemcpy(newVec.x, x, size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.y, y, size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.z, z, size * sizeof(float), cudaMemcpyDeviceToDevice);

    // Copy new data
    cudaMemcpy(newVec.x + size, vec.x, vec.size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.y + size, vec.y, vec.size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.z + size, vec.z, vec.size * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free datas
    free();
    vec.free();

    // Update
    *this = newVec;
}

void Vec4f_ptr::malloc(ULLInt size) {
    this->size = size;
    cudaMalloc(&x, size * sizeof(float));
    cudaMalloc(&y, size * sizeof(float));
    cudaMalloc(&z, size * sizeof(float));
    cudaMalloc(&w, size * sizeof(float));
}
void Vec4f_ptr::free() {
    this->size = 0;
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(w);
}
void Vec4f_ptr::operator+=(Vec4f_ptr& vec) {
    Vec4f_ptr newVec;
    newVec.malloc(size + vec.size);

    // Copy original data
    cudaMemcpy(newVec.x, x, size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.y, y, size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.z, z, size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.w, w, size * sizeof(float), cudaMemcpyDeviceToDevice);

    // Copy new data
    cudaMemcpy(newVec.x + size, vec.x, vec.size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.y + size, vec.y, vec.size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.z + size, vec.z, vec.size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.w + size, vec.w, vec.size * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free datas
    free();
    vec.free();

    // Update
    *this = newVec;
}

void Vec3ulli_ptr::malloc(ULLInt size) {
    this->size = size;
    cudaMalloc(&v, size * sizeof(ULLInt));
    cudaMalloc(&t, size * sizeof(ULLInt));
    cudaMalloc(&n, size * sizeof(ULLInt));
}
void Vec3ulli_ptr::free() {
    this->size = 0;
    cudaFree(v);
    cudaFree(t);
    cudaFree(n);
}
void Vec3ulli_ptr::operator+=(Vec3ulli_ptr& vec) {
    Vec3ulli_ptr newVec;
    newVec.malloc(size + vec.size);

    // Copy original data
    cudaMemcpy(newVec.v, v, size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.t, t, size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.n, n, size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);

    // Copy new data
    cudaMemcpy(newVec.v + size, vec.v, vec.size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.t + size, vec.t, vec.size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.n + size, vec.n, vec.size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);

    // Free datas
    free();
    vec.free();

    // Update
    *this = newVec;
}

// Atomics
__device__ bool atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old) > value;
}

__device__ bool atomicMinDouble(double* addr, double value) {
    unsigned long long int* addr_as_ull = (unsigned long long int*)addr;
    unsigned long long int old = *addr_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(fmin(value, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old) > value;
}