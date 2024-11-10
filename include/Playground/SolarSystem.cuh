#ifndef SOLARSYSTEM_CUH
#define SOLARSYSTEM_CUH

#include <Utility.cuh>

class SolarSystem {
public:
    std::vector<Mesh> stars;

    void setStars(int batch, int perBatch, float rIn, float rOut, float scl) {
        stars.resize(batch);

        for (int i = 0; i < batch * perBatch; i++) {
            std::string starPath = "assets/Models/Shapes/Star.obj";
            Mesh star = Utils::readObjFile(starPath, 1, 1, false);

            star.scaleIni(Vec3f(), Vec3f(scl));

            float rotx = rand() % 360 * M_PI / 180;
            float roty = rand() % 360 * M_PI / 180;
            float rotz = rand() % 360 * M_PI / 180;

            star.rotateIni(Vec3f(), rotx, 0);
            star.rotateIni(Vec3f(), roty, 1);
            star.rotateIni(Vec3f(), rotz, 2);

            int rDist = (rOut - rIn) * 100;
            float r = (rand() % rDist) / 100 + rIn;
            if (rand() % 2) r *= -1;
            Vec3f pos = Vec3f(0, r, 0);
            pos.rotateX(Vec3f(), rand() % 360 * M_PI / 180);
            pos.rotateY(Vec3f(), rand() % 360 * M_PI / 180);
            pos.rotateZ(Vec3f(), rand() % 360 * M_PI / 180);

            star.translateIni(pos);

            stars[i % batch].push(star);
        }
    }
};

#endif