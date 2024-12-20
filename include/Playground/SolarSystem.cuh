#ifndef SOLARSYSTEM_CUH
#define SOLARSYSTEM_CUH

#include <Utility.cuh>

class SolarSystem {
public:
    Mesh stars; // Stars decoration

    void setStars(int batch, int perBatch, float rIn, float rOut, float scl) {
        // Constructing the stars mesh using the new technique
        VectF wx, wy, wz;
        VectF tu, tv;
        VectF nx, ny, nz;
        VectULLI fw; VectLLI ft, fn; // x3
        VectLLI fm; // x1

        // Default material properties
        VectF kar = {1}, kag = {1}, kab = {1};
        VectF kdr = {1}, kdg = {1}, kdb = {1};
        VectF ksr = {1}, ksg = {1}, ksb = {1};
        VectLLI mkd = {0};

        for (int i = 0; i < batch * perBatch; i++) {
            std::string starPath = "assets/Models/Shapes/Star.obj";
            /* Important note:

            The only relevant data in Mesh star is the vertex and face data
            We only need to apply offset to the face data
            
            */
            Mesh star = Utils::readObjFile("", starPath, 1, 1);

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

            for (int j = 0; j < star.fw.size(); j++) {
                // Data x3
                fw.push_back(star.fw[j] + wx.size());
                ft.push_back(-1);
                fn.push_back(star.fn[j] + nx.size());

                // Data x1
                if (j % 3 == 0) {
                    fm.push_back(-1);
                }
            }
            for (int j = 0; j < star.wx.size(); j++) {
                wx.push_back(star.wx[j]);
                wy.push_back(star.wy[j]);
                wz.push_back(star.wz[j]);
                nx.push_back(star.nx[j]);
                ny.push_back(star.ny[j]);
                nz.push_back(star.nz[j]);
            }
        }

        // Default mesh range map
        ObjRangeMap objmap;
        VectStr objmapKs;
        for (int b = 0; b < batch; b++) {
            std::string key = "Batch" + std::to_string(b);

            ULLInt wxperbatch = wx.size() / batch;
            ULLInt nxperbatch = nx.size() / batch;
            ULLInt fperbatch = fm.size() / batch; // x1

            objmap[key] = ObjRange(
                wxperbatch * b, wxperbatch * (b + 1),
                0, 0, // No texture data
                nxperbatch * b, nxperbatch * (b + 1),
                fperbatch * b, fperbatch * (b + 1)
            );

            objmapKs.push_back(key);
        }

        stars = Mesh(
            wx, wy, wz,
            tu, tv,
            nx, ny, nz,
            fw, ft, fn,
            fm,
            kar, kag, kab,
            kdr, kdg, kdb,
            ksr, ksg, ksb,
            mkd,
            // No texture data
            VectF(), VectF(), VectF(),
            VectI(), VectI(), VectLLI(),
            // Mesh map data
            objmap, objmapKs
        );
        stars.name = "SolarSystem_Stars";
    }
};

#endif