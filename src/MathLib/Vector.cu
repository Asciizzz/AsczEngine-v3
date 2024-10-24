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

// VEC3ulli (unsigned long long int)
Vec3ulli::Vec3ulli() : x(0), y(0), z(0) {}
Vec3ulli::Vec3ulli(ULLInt x, ULLInt y, ULLInt z) : x(x), y(y), z(z) {}
Vec3ulli::Vec3ulli(ULLInt a) : x(a), y(a), z(a) {}
void Vec3ulli::operator+=(ULLInt t) {
    x += t; y += t; z += t;
}
void Vec3ulli::operator-=(ULLInt t) {
    x -= t; y -= t; z -= t;
}

// VEC4ulli
Vec4ulli::Vec4ulli() : x(0), y(0), z(0), w(0) {}
Vec4ulli::Vec4ulli(ULLInt x, ULLInt y, ULLInt z, ULLInt w) : x(x), y(y), z(z), w(w) {}
Vec4ulli::Vec4ulli(ULLInt a) : x(a), y(a), z(a), w(a) {}

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
Vec3f Vec3f::rotate(Vec3f& vec, const Vec3f& origin, const Vec3f& rot) {
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
void Vec3f::rotate(const Vec3f& origin, const Vec3f& rot) {
    *this = rotate(*this, origin, rot);
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

void Vecptr2f::malloc(ULLInt size) {
    this->size = size;
    cudaMalloc(&x, size * sizeof(float));
    cudaMalloc(&y, size * sizeof(float));
}
void Vecptr2f::free() {
    this->size = 0;
    cudaFree(x);
    cudaFree(y);
}
void Vecptr2f::operator+=(Vecptr2f& vec) {
    Vecptr2f newVec;
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

void Vecptr3f::malloc(ULLInt size) {
    this->size = size;
    cudaMalloc(&x, size * sizeof(float));
    cudaMalloc(&y, size * sizeof(float));
    cudaMalloc(&z, size * sizeof(float));
}
void Vecptr3f::free() {
    this->size = 0;
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
}
void Vecptr3f::operator+=(Vecptr3f& vec) {
    Vecptr3f newVec;
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

void Vecptr4f::malloc(ULLInt size) {
    this->size = size;
    cudaMalloc(&x, size * sizeof(float));
    cudaMalloc(&y, size * sizeof(float));
    cudaMalloc(&z, size * sizeof(float));
    cudaMalloc(&w, size * sizeof(float));
}
void Vecptr4f::free() {
    this->size = 0;
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(w);
}
void Vecptr4f::operator+=(Vecptr4f& vec) {
    Vecptr4f newVec;
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

void Vecptr4ulli::malloc(ULLInt size) {
    this->size = size;
    cudaMalloc(&v, size * sizeof(ULLInt));
    cudaMalloc(&t, size * sizeof(ULLInt));
    cudaMalloc(&n, size * sizeof(ULLInt));
    cudaMalloc(&o, size * sizeof(ULLInt));
}
void Vecptr4ulli::free() {
    this->size = 0;
    cudaFree(v);
    cudaFree(t);
    cudaFree(n);
    cudaFree(o);
}
void Vecptr4ulli::operator+=(Vecptr4ulli& vec) {
    Vecptr4ulli newVec;
    newVec.malloc(size + vec.size);

    // Copy original data
    cudaMemcpy(newVec.v, v, size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.t, t, size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.n, n, size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.o, o, size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);

    // Copy new data
    cudaMemcpy(newVec.v + size, vec.v, vec.size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.t + size, vec.t, vec.size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.n + size, vec.n, vec.size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newVec.o + size, vec.o, vec.size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);

    // Free datas
    free();
    vec.free();

    // Update
    *this = newVec;
}