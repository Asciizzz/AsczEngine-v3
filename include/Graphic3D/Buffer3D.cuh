#ifndef BUFFER3D_CUH
#define BUFFER3D_CUH

#include <Mesh3D.cuh>

struct Vec3 {
    float x, y, z;

    Vec3 operator+(const Vec3 &v) const {
        return {x + v.x, y + v.y, z + v.z};
    }
    Vec3 operator-(const Vec3 &v) const {
        return {x - v.x, y - v.y, z - v.z};
    }
    Vec3 operator*(float s) const {
        return {x * s, y * s, z * s};
    }
    Vec3 operator*(const Vec3 &v) const {
        return {x * v.x, y * v.y, z * v.z};
    }
};

class Buffer3D {
public:
    float *depth;
    Vec3 *normal;
    Vec3 *color;
    Vec3 *world;
};

#endif