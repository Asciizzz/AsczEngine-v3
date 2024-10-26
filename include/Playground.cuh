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
    // Note Placement: 0: original, 1: center, 2: floor

    static Mesh readObjFile(std::string path, short fIdxBased=1, short placement=0, bool rainbow=true) {
        std::ifstream file(path);
        if (!file.is_open()) return Mesh();

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

        #pragma omp parallel for collapse(2)
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
                float mag = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
                n.x /= mag; n.y /= mag; n.z /= mag;

                nx.push_back(n.x);
                ny.push_back(n.y);
                nz.push_back(n.z);
            } else if (type == "vt") {
                Vec2f t;
                ss >> t.x >> t.y;
                tu.push_back(t.x);
                tv.push_back(t.y);
            } else if (type == "vc") { // NOTE: This is not in a standard .obj file format
                Vec4f c;
                ss >> c.x >> c.y >> c.z >> c.w;
                cr.push_back(c.x * 255);
                cg.push_back(c.y * 255);
                cb.push_back(c.z * 255);
                ca.push_back(c.w * 255);
            } else if (type == "f") {
                std::vector<ULLInt> vs, ts, ns;
                while (ss.good()) {
                    std::string vtn;
                    ss >> vtn;

                    ULLInt v, t, n;
                    std::stringstream ss2(vtn);
                    ss2 >> v; ss2.ignore(1); ss2 >> t; ss2.ignore(1); ss2 >> n;

                    vs.push_back(v - fIdxBased);
                    ts.push_back(t - fIdxBased);
                    ns.push_back(n - fIdxBased);
                }

                /* For n points, we will construct n - 2 triangles
                
                Example: (A B C D E F):
                - Use A as anchor point
                    => (A B C), (A C D), (A D E), (A E F)

                Note:  .obj files are assumed to organized the points
                        in a clockwise (or counter-clockwise) order
                */

                for (int i = 1; i < vs.size() - 1; i++) {
                    fw.push_back(vs[0]); fw.push_back(vs[i]); fw.push_back(vs[i + 1]);
                    ft.push_back(ts[0]); ft.push_back(ts[i]); ft.push_back(ts[i + 1]);
                    fn.push_back(ns[0]); fn.push_back(ns[i]); fn.push_back(ns[i + 1]);
                }
            }
        }

        if (cr.size() == 0) {
            #pragma omp parallel
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

                    // Shift to center of xz plane
                    if (placement > 0) {
                        wx[i] -= (minX + maxX) / 2;
                        wz[i] -= (minZ + maxZ) / 2;
                    }

                    if (placement == 1) { // Shift to center
                        wy[i] -= (minY + maxY) / 2;
                    } else if (placement == 2) { // Shift to floor
                        wy[i] -= minY;
                    }
                } else {
                    // Just set it to white
                    cr.push_back(255);
                    cg.push_back(255);
                    cb.push_back(255);
                    ca.push_back(255);
                }
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