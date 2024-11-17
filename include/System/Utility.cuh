#ifndef UTILITY_CUH
#define UTILITY_CUH

#include <Graphic3D.cuh>
#include <fstream>
#include <sstream>

#include <unordered_map>

#define MTLs std::unordered_map<std::string, MTLMaterial>

struct MTLMaterial {
    Vec3f ka, kd, ks;
    float ns, ni, d, tr;
    int illum;
    std::string mapKa, mapKd, mapKs, mapNs, mapD, mapBump;
};

class Utils {
public:
    static MTLs readMtlFile(std::string path);
    static Mesh readObjFile(std::string path, short fIdxBased=1, short placement=0, bool rainbow=false);

    static void applyTransformation(std::vector<Mesh> objs);
};

#endif