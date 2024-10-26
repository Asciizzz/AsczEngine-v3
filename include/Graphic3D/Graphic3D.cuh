#ifndef GRAPHIC3D_CUH
#define GRAPHIC3D_CUH

#include <Mesh3D.cuh> 
#include <Camera3D.cuh>
#include <Buffer3D.cuh>

// BETA: LightSrc
struct LightSrc {
    Vec3f dir = {0, 0, 1};
    float ambient = 0.1;
    float specular = 1.2;
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
    float *wx, *wy, *wz;
    float *nx, *ny, *nz;
    float *tu, *tv;
    float *cr, *cg, *cb, *ca;
    float *sx, *sy, *sz, *sw;

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
    int pixelSize = 8;
    void setResolution(float w, float h, float ps=4);

    // For tile-based rasterization
    int tileSizeX = 20;
    int tileSizeY = 20;
    int tileNumX, tileNumY, tileNum;
    void setTileSize(int tw, int th);

    // Free everything
    void free();

    // Mesh3D
    Mesh3D mesh;

    // For runtime faces
    ULLInt faceCounter;
    ULLInt *d_faceCounter;
    Face3D rtFaces;

    void mallocRuntimeFaces();
    void freeRuntimeFaces();
    void resizeRuntimeFaces();

    // Face stream for chunking
    cudaStream_t *faceStreams;
    size_t chunkSize = 5e6;
    int chunkNum;
    void mallocFaceStreams();
    void freeFaceStreams();
    void resizeFaceStreams();

    // Camera3D and Buffer3D
    Camera3D camera;
    Buffer3D buffer;

    // ========== BETAs SECTION ==========

    // BETA: Texture mapping
    int textureWidth, textureHeight;
    Vec3f *d_texture; // Device texture
    bool textureSet = false;
    void createTexture(const std::string &path);
    void freeTexture();

    // BETA: LightSrc
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