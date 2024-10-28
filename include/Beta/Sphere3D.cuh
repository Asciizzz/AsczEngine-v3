#ifndef SPHERE3D_CUH
#define SPHERE3D_CUH

#include <Playground.cuh>

#include <FpsHandler.cuh>

class Sphere3D {
public:
    Vec3f center;
    float radius;

    Vec3f vel;

    Mesh mesh;

    Sphere3D(Vec3f center, float radius) : center(center), radius(radius) {
        mesh = Playground::readObjFile(
            "assets/Models/Shapes/Sphere.obj", 1, 1, true
        );
        mesh.translateIni(center); // Translate first
        mesh.scaleIni(center, Vec3f(radius / 2.47)); // Scale second
    }

    void setpos(Vec3f pos) {
        Vec3f diff = pos - center;
        center = pos;
        mesh.translateRuntime(diff);
    }

    void addpos(Vec3f pos) {
        center += pos;
        mesh.translateRuntime(pos);
    }

    void movement() {
        FpsHandler &fps = FpsHandler::instance();

        vel.y -= .1 * fps.dTimeSec;
        addpos(vel);

        if (center.y - radius < 0) {
            setpos(Vec3f(center.x, radius, center.z));
            vel.y = -vel.y * 0.8;
        }

        if (center.x + radius > 4) {
            setpos(Vec3f(4 - radius, center.y, center.z));
            vel.x = -vel.x * 0.7;
        } else if (center.x - radius < -4) {
            setpos(Vec3f(-4 + radius, center.y, center.z));
            vel.x = -vel.x * 0.7;
        }

        if (center.z + radius > 4) {
            setpos(Vec3f(center.x, center.y, 4 - radius));
            vel.z = -vel.z * 0.7;
        } else if (center.z - radius < -4) {
            setpos(Vec3f(center.x, center.y, -4 + radius));
            vel.z = -vel.z * 0.7;
        }

        // Just for fun: rotate the sphere based on the magnitude of the velocity
        float mag = vel.mag();
        mesh.rotateRuntime(center, Vec3f(mag));
    }
};

#endif