#include <FpsHandler.cuh>
#include <Mesh3D.cuh>

int main() {
    // Create a mesh of a square
    std::vector<Vec3f> pos = {
        Vec3f(-1, -1, 0),
        Vec3f(1, -1, 0),
        Vec3f(1, 1, 0),
        Vec3f(-1, 1, 0)
    };
    std::vector<Vec3f> normal = {
        Vec3f(0, 0, 1),
        Vec3f(0, 0, 1),
        Vec3f(0, 0, 1),
        Vec3f(0, 0, 1)
    };
    std::vector<Vec2f> tex = {
        Vec2f(0, 0),
        Vec2f(1, 0),
        Vec2f(1, 1),
        Vec2f(0, 1)
    };
    std::vector<Vec3uli> faces = {
        Vec3uli(0, 1, 2),
        Vec3uli(0, 2, 3)
    };

    Mesh3D mesh(2, pos, normal, tex, faces);
    Mesh3D mesh2(1, pos, normal, tex, faces);

    return 0;
}