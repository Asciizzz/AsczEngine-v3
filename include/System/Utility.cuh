#ifndef UTILITY_CUH
#define UTILITY_CUH

#include <Graphic3D.cuh>
#include <fstream>
#include <sstream>

class Utils {
public:
    static Mesh readObjFile(std::string path, short fIdxBased=1, short placement=0, bool rainbow=true);

    static void applyTransformation(std::vector<Mesh> objs);
};

#endif