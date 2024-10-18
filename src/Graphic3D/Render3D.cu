#include <Render3D.cuh>

void Render3D::setResolution(float w, float h) {
    RES = {w, h};
    RES_HALF = {w / 2, h / 2};
    CAMERA.setResolution(w, h);
}