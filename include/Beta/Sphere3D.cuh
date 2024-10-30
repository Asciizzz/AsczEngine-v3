#ifndef SPHERE3D_CUH
#define SPHERE3D_CUH

#include <Utility.cuh>

#include <FpsHandler.cuh>

class Sphere3D {
public:
    Vec3f center;
    float radius;

    Vec3f vel;
    Vec3f angvel;

    Mesh mesh;

    Sphere3D(Vec3f center, float radius) : center(center), radius(radius) {
        mesh = Utils::readObjFile(
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

            // Random ratio in range 0.5 - 0.8
            float ratio = 0.5 + (rand() % 30) / 100.0;
            vel.y = -vel.y * ratio;
        }

        if (center.x + radius > 4) {
            float ratio = 0.5 + (rand() % 20) / 100.0;
            vel.x = -vel.x * ratio;
            angvel.z = -angvel.z * ratio;
            
            setpos(Vec3f(4 - radius, center.y, center.z));
        } else if (center.x - radius < -4) {
            float ratio = 0.5 + (rand() % 20) / 100.0;
            vel.x = -vel.x * ratio;
            angvel.z = -angvel.z * ratio;
            
            setpos(Vec3f(-4 + radius, center.y, center.z));
        }

        if (center.z + radius > 4) {
            float ratio = 0.5 + (rand() % 20) / 100.0;
            vel.z = -vel.z * ratio;
            angvel.x = -angvel.x * ratio;

            setpos(Vec3f(center.x, center.y, 4 - radius));
        } else if (center.z - radius < -4) {
            float ratio = 0.5 + (rand() % 20) / 100.0;
            vel.z = -vel.z * ratio;
            angvel.x = -angvel.x * ratio;
            
            setpos(Vec3f(center.x, center.y, -4 + radius));
        }

        mesh.rotateRuntime(center, angvel * fps.dTimeSec);
    }
};

#endif