#ifndef UTILITY_CUH
#define UTILITY_CUH

#include <Graphic3D.cuh>
#include <fstream>
#include <sstream>

#include <unordered_map>

struct MTL {
    ULLInt mIdx;
    float kdr, kdg, kdb;
};

class Utils {
public:
    static Mesh readObjFile(std::string path, short fIdxBased=1, short placement=0, bool rainbow=false);

    static void applyTransformation(std::vector<Mesh> objs);
};

#endif