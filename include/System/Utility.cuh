#ifndef UTILITY_CUH
#define UTILITY_CUH

#include <Graphic3D.cuh>
#include <fstream>
#include <sstream>

#include <unordered_map>

struct MTL {
    ULLInt idx;
    float kar = 1, kag = 1, kab = 1;
    float kdr = 1, kdg = 1, kdb = 1;
    float ksr = 1, ksg = 1, ksb = 1;
    LLInt mkd = -1;

    MTL(ULLInt idx=0) : idx(idx) {}
};

class Utils {
public:
    static Mesh readObjFile(std::string path, short fIdxBased=1, short placement=0, bool rainbow=false);

    static void applyTransformation(std::vector<Mesh> objs);
};

#endif