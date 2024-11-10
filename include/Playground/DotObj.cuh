#ifndef DOTOBJ_CUH
#define DOTOBJ_CUH

#include <Utility.cuh>

class DotObj {
public:
    std::vector<Mesh> objs;
    int objCount = 0;

    DotObj(std::string path) {
        std::ifstream objsFile(path);
        std::string line;

        while (std::getline(objsFile, line)) {
            // If line start with #, it's a comment
            if (line[0] == '#' || line.empty()) continue;
            // If line start with ~, it's the end of the file
            if (line[0] == '~') break;

            std::string objPath = "";
            float scale = 1;
            Vec3f translate;
            Vec3f rotate;

            std::stringstream ss(line);

            ss >> objPath >> scale;
            ss >> rotate.x >> rotate.y >> rotate.z;
            ss >> translate.x >> translate.y >> translate.z;
            rotate *= M_PI / 180;

            Mesh obj = Utils::readObjFile(objPath, 1, 1, true);
            obj.scaleIni(Vec3f(), Vec3f(scale));
            obj.rotateIni(Vec3f(), rotate.x, 0);
            obj.rotateIni(Vec3f(), rotate.y, 1);
            obj.rotateIni(Vec3f(), rotate.z, 2);
            obj.translateIni(translate);

            objs.push_back(obj);

            objCount++;
        }
    }
};

#endif