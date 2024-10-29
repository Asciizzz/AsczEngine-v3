#ifndef PLANE3D_CUH
#define PLANE3D_CUH

#include <Matrix.cuh>

class Plane3D { // Ax + By + Cz + D = 0
public:
    float a, b, c, d;
    Vec3f n;

    __host__ __device__ Plane3D(Vec3f n=Vec3f(), Vec3f v=Vec3f()) {
        n.norm();
        this->n = n;
        a = n.x;
        b = n.y;
        c = n.z;
        d = -(a * v.x + b * v.y + c * v.z);
    }

    __host__ __device__ float equation(Vec3f v) {
        return a * v.x + b * v.y + c * v.z + d;
    }

    __host__ __device__ Vec3f intersect(Vec3f v1, Vec3f v2) {
        Vec3f dir = v2 - v1;
        float t = -(a * v1.x + b * v1.y + c * v1.z + d) / (a * dir.x + b * dir.y + c * dir.z);
        return v1 + dir * t;
    }
};

#endif