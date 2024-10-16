#ifndef CAMERA3D_CUH
#define CAMERA3D_CUH

#include <cmath>
#include <vector>
#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923
#define M_2_PI 6.28318530717958647692

class Camera3D {
public:
    // Window properties
    int width, height;
    float centerX, centerY;
    void resizeWindow(int width, int height);

    // Projection properties
    float x, y, z;
    float pitch, yaw, roll; // Roll is rarely used

    float fov = 90.0f;
    float screenDist;
    void setFov(float f);

    // Other
    float vel = 1;
    float mSens = .1f;

    Camera3D();

    void movement(float dTimeSec);

    __host__ __device__
    void project(float x, float y, float z, float& px, float& py, float& pz);
};

#endif