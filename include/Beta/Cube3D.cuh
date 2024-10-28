#ifndef CUBE3D_CUH
#define CUBE3D_CUH

#include <Playground.cuh>

#include <FpsHandler.cuh>

class Cube3D {
public:
    Vec3f center;
    Vec3f world[8];
    Mesh mesh;

    Cube3D(Vec3f center, float h_size) {
        this->center = center;

        world[0] = center + Vec3f(-h_size, -h_size, -h_size);
        world[1] = center + Vec3f(h_size, -h_size, -h_size);
        world[2] = center + Vec3f(-h_size, h_size, -h_size);
        world[3] = center + Vec3f(h_size, h_size, -h_size);
        world[4] = center + Vec3f(-h_size, -h_size, h_size);
        world[5] = center + Vec3f(h_size, -h_size, h_size);
        world[6] = center + Vec3f(-h_size, h_size, h_size);
        world[7] = center + Vec3f(h_size, h_size, h_size);

        mesh = Playground::readObjFile(
            "assets/Models/Shapes/Cube.obj", 1, 1, true
        ); // Cube file x, y, z = +-1
        mesh.translateIni(center); // Translate first
        mesh.scaleIni(center, Vec3f(h_size)); // Scale second
    }

    bool collideHorizontal(Vec3f point) {
        return (
            point.x > world[0].x && point.x < world[7].x &&
            point.z > world[0].z && point.z < world[7].z
        );
    }

    bool collideTop(Vec3f point) {
        return point.y < world[7].y;
    }

    void physic() {
        Camera3D &cam = Graphic3D::instance().camera;
        
        // To be added
    }
};

#endif