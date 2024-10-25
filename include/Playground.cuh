#ifndef PLAYGROUND_CUH
#define PLAYGROUND_CUH

#include <Graphic3D.cuh>

#include <fstream>
#include <sstream>
#include <string>

#include <omp.h>

// Create a function that would read an .obj file and return a Mesh3D object

class Playground {
public:
    static Mesh readObjFile(UInt objId, std::string path, short fIdxBased=1, bool rainbow=true, bool center=true) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << path << std::endl;
            exit(1);
        }

        std::string line;

        std::vector<float> wx, wy, wz;
        std::vector<float> nx, ny, nz;
        std::vector<float> tu, tv;
        std::vector<float> cr, cg, cb, ca;
        std::vector<ULLInt> fw, ft, fn;

        // We will use these value to shift the mesh to the origin
        float minX = INFINITY, minY = INFINITY, minZ = INFINITY;
        float maxX = -INFINITY, maxY = -INFINITY, maxZ = -INFINITY;

        std::vector<std::string> lines;

        while (std::getline(file, line)) {
            lines.push_back(line);
        }

        #pragma omp parallel for
        for (size_t i = 0; i < lines.size(); i++) {
            std::stringstream ss(lines[i]);
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

                wx.push_back(v.x);
                wy.push_back(v.y);
                wz.push_back(v.z);
            } else if (type == "vn") {
                Vec3f n;
                ss >> n.x >> n.y >> n.z;
                nx.push_back(n.x);
                ny.push_back(n.y);
                nz.push_back(n.z);
            } else if (type == "vt") {
                Vec2f t;
                ss >> t.x >> t.y;
                tu.push_back(t.x);
                tv.push_back(t.y);
            } else if (type == "f") {
                /* Note:
                Faces index in .obj files are 1-based
                Faces format: f v1/t1/n1 v2/t2/n2 v3/t3/n3
                */

                std::string vtn1, vtn2, vtn3, vtn4 = "";
                ss >> vtn1 >> vtn2 >> vtn3 >> vtn4;

                Vec3ulli v, t, n;
                std::stringstream ss1(vtn1), ss2(vtn2), ss3(vtn3);
                ss1 >> v.x; ss1.ignore(1); ss1 >> t.x; ss1.ignore(1); ss1 >> n.x;
                ss2 >> v.y; ss2.ignore(1); ss2 >> t.y; ss2.ignore(1); ss2 >> n.y;
                ss3 >> v.z; ss3.ignore(1); ss3 >> t.z; ss3.ignore(1); ss3 >> n.z;

                v -= fIdxBased; t -= fIdxBased; n -= fIdxBased;

                if (vtn4 != "") {
                    ULLInt v4, t4, n4;
                    std::stringstream ss4(vtn4);
                    ss4 >> v4; ss4.ignore(1); ss4 >> t4; ss4.ignore(1); ss4 >> n4;
                    v4 -= fIdxBased; t4 -= fIdxBased; n4 -= fIdxBased;

                    fw.push_back(v.x); fw.push_back(v.z); fw.push_back(v4);
                    fn.push_back(n.x); fn.push_back(n.z); fn.push_back(n4);
                    ft.push_back(t.x); ft.push_back(t.z); ft.push_back(t4);
                }

                fw.push_back(v.x); fw.push_back(v.y); fw.push_back(v.z);
                fn.push_back(n.x); fn.push_back(n.y); fn.push_back(n.z);
                ft.push_back(t.x); ft.push_back(t.y); ft.push_back(t.z);
            }
        }

        for (size_t i = 0; i < wx.size(); i++) {
            if (rainbow) {
                // Set the color based on the ratio of x, y, and z
                float r = (wx[i] - minX) / (maxX - minX);
                float g = (wy[i] - minY) / (maxY - minY);
                float b = (wz[i] - minZ) / (maxZ - minZ);
                cr.push_back(255 - r * 255);
                cg.push_back(g * 255);
                cb.push_back(b * 255);
                ca.push_back(255);

                // Shift the mesh to the origin
                wx[i] -= (minX + maxX) / 2;
                wy[i] -= (minY + maxY) / 2;
                wz[i] -= (minZ + maxZ) / 2;

                if (!center) wy[i] = -minY;
            } else {
                // Just set it to white
                cr.push_back(255);
                cg.push_back(255);
                cb.push_back(255);
                ca.push_back(255);
            }
        }

        Mesh mesh = {
            wx, wy, wz,
            nx, ny, nz,
            tu, tv,
            cr, cg, cb, ca,
            fw, ft, fn
        };

        return mesh;
    }

};

#endif