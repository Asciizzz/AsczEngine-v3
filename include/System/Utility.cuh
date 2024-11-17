#ifndef UTILITY_CUH
#define UTILITY_CUH

#include <Graphic3D.cuh>
#include <fstream>
#include <sstream>

struct ObjPath {
    std::string dotObj;
    std::string dotMtl;

    bool mtllib = false;

    ObjPath(std::string dotObj, std::string dotMtl) {
        this->dotObj = dotObj;
        this->dotMtl = dotMtl;
        mtllib = true;
    }
    ObjPath(std::string dotObj) {
        this->dotObj = dotObj;
    }
};

class Utils {
public:
    static Mesh readObjFile(ObjPath path, short fIdxBased=1, short placement=0, bool rainbow=false);

    static void applyTransformation(std::vector<Mesh> objs);
};

#endif