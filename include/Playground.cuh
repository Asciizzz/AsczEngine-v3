#ifndef PLAYGROUND_CUH
#define PLAYGROUND_CUH

#include <Graphic3D.cuh>

#include <fstream>
#include <sstream>
#include <string>

// Create a function that would read an .obj file and return a Mesh3D object

class Playground {
public:
    static Mesh3D readObjFile(UInt meshId, std::string path, bool rainbow=false, bool center=true) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << path << std::endl;
            exit(1);
        }

        std::string line;
        Vecs3f world;
        Vecs3f normal;
        Vecs2f texture;
        Vecs4f color;
        Vecs3x3uli faces;

        // We will use these value to shift the mesh to the origin
        float minX = INFINITY, minY = INFINITY, minZ = INFINITY;
        float maxX = -INFINITY, maxY = -INFINITY, maxZ = -INFINITY;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string type;
            ss >> type;

            if (type == "v") {
                Vec3f v;
                ss >> v.x >> v.y >> v.z;

                // Update the min and max values
                minX = std::min(minX, v.x);
                minY = std::min(minY, v.y);
                minZ = std::min(minZ, v.z);
                maxX = std::max(maxX, v.x);
                maxY = std::max(maxY, v.y);
                maxZ = std::max(maxZ, v.z);

                world.push_back(v);
            } else if (type == "vn") {
                Vec3f n;
                ss >> n.x >> n.y >> n.z;
                normal.push_back(n);
            } else if (type == "vt") {
                Vec2f t;
                ss >> t.x >> t.y;
                texture.push_back(t);
            } else if (type == "f") {
                /* Note:
                Faces index in .obj files are 1-based
                Faces format: f v1/t1/n1 v2/t2/n2 v3/t3/n3
                */

                std::string vtn1, vtn2, vtn3, vtn4 = "";
                ss >> vtn1 >> vtn2 >> vtn3 >> vtn4;

                Vec3uli v, t, n;
                std::stringstream ss1(vtn1), ss2(vtn2), ss3(vtn3);
                ss1 >> v.x; ss1.ignore(1); ss1 >> t.x; ss1.ignore(1); ss1 >> n.x;
                ss2 >> v.y; ss2.ignore(1); ss2 >> t.y; ss2.ignore(1); ss2 >> n.y;
                ss3 >> v.z; ss3.ignore(1); ss3 >> t.z; ss3.ignore(1); ss3 >> n.z;

                v -= 1; t -= 1; n -= 1;

                if (vtn4 != "") {
                    ULInt v4, t4, n4;
                    std::stringstream ss4(vtn4);
                    ss4 >> v4; ss4.ignore(1); ss4 >> t4; ss4.ignore(1); ss4 >> n4;
                    v4 -= 1; t4 -= 1; n4 -= 1;

                    // Create an additional face to triangulate the quad
                    faces.push_back(Vec3x3uli(
                        Vec3uli(v.x, v.z, v4),
                        Vec3uli(t.x, t.z, t4),
                        Vec3uli(n.x, n.z, n4)
                    ));
                }

                faces.push_back(Vec3x3uli(v, t, n));
            }
        }

        for (Vec3f &v : world) {

            if (rainbow) {
                // Set the color based on the ratio of x, y, and z
                float r = (v.x - minX) / (maxX - minX);
                float g = (v.y - minY) / (maxY - minY);
                float b = (v.z - minZ) / (maxZ - minZ);
                color.push_back(Vec4f(255 - r * 255, g * 255, b * 255, 255));
            } else {
                // Just set it to white
                color.push_back(Vec4f(255, 255, 255, 255));
            }
        }

        Mesh3D mesh = Mesh3D(meshId, world, normal, texture, color, faces);

        // Shift the mesh to the origin
        Vec3f shift = Vec3f(-(minX + maxX) / 2, -(minY + maxY) / 2, -(minZ + maxZ) / 2);
        if (!center) shift.y = -minY;
        mesh.translate(meshId, shift);

        return mesh;
    }

};

#endif