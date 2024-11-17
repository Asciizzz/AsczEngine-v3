#ifndef GRAPHIC3D_CUH
#define GRAPHIC3D_CUH

#include <Mesh3D.cuh> 
#include <Camera3D.cuh>
#include <Buffer3D.cuh>

// BETA: LightSrc
struct LightSrc {
    Vec3f dir = {0, 0, 1};
    float ambient = 0.1;
    float specular = 1.1;
    Vec3f color = {1, 1, 1};

    std::string data() {
        std::string str = "LightSrc:\n";
        str += "| Dir: " + std::to_string(dir.x) + " " + std::to_string(dir.y) + " " + std::to_string(dir.z) + "\n";
        str += "| Ambient: " + std::to_string(ambient) + "\n";
        str += "| Specular: " + std::to_string(specular) + "\n";
        str += "| Color: " + std::to_string(color.x) + " " + std::to_string(color.y) + " " + std::to_string(color.z) + "\n";
        return str;
    }
};

struct Face3D {
    // x3
    float *sx, *sy, *sz, *sw;
    float *wx, *wy, *wz;
    float *nx, *ny, *nz;
    float *tu, *tv;
    float *cr, *cg, *cb, *ca;
    // x1
    bool *active;
    long long *mat;
    float *area;

    ULInt size = 0;

    void malloc(ULLInt size);
    void free();
};

class Graphic3D {
public:
    // Singleton
    static Graphic3D& instance() {
        static Graphic3D instance;
        return instance;
    }
    Graphic3D(const Graphic3D&) = delete;
    Graphic3D &operator=(const Graphic3D&) = delete;

    // Resolution
    Vec2f res = {800, 600};
    Vec2f res_half = {400, 300};
    int pixelSize = 4;
    void setResolution(float w, float h, float ps=4);

    // Free everything
    void free();

    // Mesh3D
    Mesh3D mesh;

    // For runtime faces
    Face3D rtFaces;

    // For indexing runtime faces
    ULLInt rtCount1, *d_rtCount1, *rtIndex1;
    ULLInt rtCount2, *d_rtCount2, *rtIndex2;

    void mallocRuntimeFaces();
    void freeRuntimeFaces();
    void resizeRuntimeFaces();

    cudaStream_t rtStreams[4];
    void createRuntimeStreams();
    void destroyRuntimeStreams();

    // Camera3D and Buffers3D
    Camera3D camera;
    Buffer3D buffer;

    // ========== BETAs SECTION ==========

    // BETA: Texture mapping
    int textureWidth, textureHeight;
    Vec3f *d_texture; // Device texture
    bool textureSet = false;
    void createTexture(const std::string &path);
    void freeTexture();

    // BETA: Lighting
    LightSrc light;

    // BETA: Shadow mapping
    float *shadowDepth;
    int shdwWidth, shdwHeight;
    int shdwTileSizeX, shdwTileSizeY;
    int shdwTileNumX, shdwTileNumY, shdwTileNum;
    void createShadowMap(int w, int h, int tw, int th);
    void freeShadowMap();

private:
    Graphic3D() {}
};

#endif