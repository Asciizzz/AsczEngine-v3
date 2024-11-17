#ifndef MATERIAL3D_CUH
#define MATERIAL3D_CUH

#include <Vector.cuh>

class Material3D {
public:
    // For the time being we will only focus on diffuse
    Vec3f_ptr ambient;
    Vec3f_ptr diffuse;
    Vec3f_ptr specular;
    float *shininess;
};

#endif