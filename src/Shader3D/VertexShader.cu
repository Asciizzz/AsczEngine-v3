#include <VertexShader.cuh>

// Render functions

Vec4f VertexShader::toScreenSpace(Camera3D &camera, Vec3f world, int buffWidth, int buffHeight) {
    Vec4f v4 = world.toVec4f();
    Vec4f t4 = camera.mvp * v4;
    Vec3f t3 = t4.toVec3f(); // Convert to NDC [-1, 1]
    Vec4f p = t3.toVec4f();
    p.w = camera.isInsideFrustum(world);

    return p;
}

// Render pipeline

void VertexShader::cameraProjection() {
    Graphic3D &graphic = Graphic3D::instance();
    Mesh3D &mesh = graphic.mesh;
    Camera3D &camera = graphic.camera;
    Buffer3D &buffer = graphic.buffer;
    Vec4f *projection = graphic.projection;

    cameraProjectionKernel<<<mesh.blockNumVs, mesh.blockSize>>>(
        projection, mesh.world, camera, buffer.width, buffer.height, mesh.numVs
    );
    cudaDeviceSynchronize();
}

void VertexShader::createDepthMap() {
    Graphic3D &graphic = Graphic3D::instance();
    Mesh3D &mesh = graphic.mesh;
    Buffer3D &buffer = graphic.buffer;
    Vec4f *projection = graphic.projection;

    buffer.clearBuffer();
    buffer.nightSky(); // Cool effect

    for (int i = 0; i < 2; i++)
        createDepthMapKernel<<<mesh.blockNumFs, mesh.blockSize>>>(
            projection, mesh.world, mesh.faces, mesh.numFs,
            buffer.active, buffer.depth, buffer.faceID, buffer.bary, buffer.width, buffer.height
        );
    cudaDeviceSynchronize();
}

void VertexShader::rasterization() {
    Graphic3D &graphic = Graphic3D::instance();
    Mesh3D &mesh = graphic.mesh;
    Buffer3D &buffer = graphic.buffer;

    rasterizationKernel<<<buffer.blockCount, buffer.blockSize>>>(
        mesh.color, mesh.world, mesh.normal, mesh.texture, mesh.meshID, mesh.faces,
        buffer.active, buffer.color, buffer.world, buffer.normal, buffer.texture, buffer.meshID,
        buffer.faceID, buffer.bary, buffer.width, buffer.height
    );
    cudaDeviceSynchronize();
}

// Kernels
__global__ void cameraProjectionKernel(
    Vec4f *projection, Vec3f *world, Camera3D camera, int buffWidth, int buffHeight, ULLInt numVs
) {
    ULLInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVs) return;

    Vec4f p = VertexShader::toScreenSpace(camera, world[i], buffWidth, buffHeight);
    projection[i] = p;
}

__global__ void createDepthMapKernel(
    Vec4f *projection, Vec3f *world, Vec3uli *faces, ULLInt numFs,
    bool *buffActive, float *buffDepth, ULLInt *buffFaceId, Vec3f *buffBary,
    int buffWidth, int buffHeight
) {
    ULLInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numFs) return;

    Vec3uli f = faces[i];
    Vec4f p0 = projection[f.x];
    Vec4f p1 = projection[f.y];
    Vec4f p2 = projection[f.z];
    Vec4f p3 = p0; // In case

    // Entirely outside the frustum
    if (p0.w <= 0 && p1.w <= 0 && p2.w <= 0) return;

    p0.x = (p0.x + 1) * buffWidth / 2;
    p0.y = (1 - p0.y) * buffHeight / 2;
    p1.x = (p1.x + 1) * buffWidth / 2;
    p1.y = (1 - p1.y) * buffHeight / 2;
    p2.x = (p2.x + 1) * buffWidth / 2;
    p2.y = (1 - p2.y) * buffHeight / 2;
    p3.x = (p3.x + 1) * buffWidth / 2;
    p3.y = (1 - p3.y) * buffHeight / 2;

    // Bounding box
    int minX = min(min(p0.x, p1.x), p2.x);
    int maxX = max(max(p0.x, p1.x), p2.x);
    int minY = min(min(p0.y, p1.y), p2.y);
    int maxY = max(max(p0.y, p1.y), p2.y);

    // Clip the bounding box
    minX = max(minX, 0);
    maxX = min(maxX, buffWidth - 1);
    minY = max(minY, 0);
    maxY = min(maxY, buffHeight - 1);

    for (int x = minX; x <= maxX; x++)
    for (int y = minY; y <= maxY; y++) {
        int bIdx = x + y * buffWidth;

        Vec3f bary = Vec3f::bary(
            Vec2f(x, y), Vec2f(p0.x, p0.y), Vec2f(p1.x, p1.y), Vec2f(p2.x, p2.y)
        );

        if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

        float zDepth = bary.x * p0.z + bary.y * p1.z + bary.z * p2.z;

        if (atomicMinFloat(&buffDepth[bIdx], zDepth)) {
            buffActive[bIdx] = true;
            buffDepth[bIdx] = zDepth;
            buffFaceId[bIdx] = i;
            buffBary[bIdx] = bary;
        }
    }
}

__global__ void rasterizationKernel(
    Vec4f *color, Vec3f *world, Vec3f *normal, Vec2f *texture, UInt *meshID, Vec3uli *faces,
    bool *buffActive, Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture,
    UInt *buffMeshId, ULLInt *buffFaceId, Vec3f *buffBary, int buffWidth, int buffHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight || !buffActive[i]) return;

    ULLInt fIdx = buffFaceId[i];

    // Set vertex index
    ULLInt vIdx0 = faces[fIdx].x;
    ULLInt vIdx1 = faces[fIdx].y;
    ULLInt vIdx2 = faces[fIdx].z;

    // Get barycentric coordinates
    float alp = buffBary[i].x;
    float bet = buffBary[i].y;
    float gam = buffBary[i].z;

    // Set color
    Vec4f c0 = color[vIdx0];
    Vec4f c1 = color[vIdx1];
    Vec4f c2 = color[vIdx2];
    buffColor[i] = c0 * alp + c1 * bet + c2 * gam;

    // Set world position
    Vec3f w0 = world[vIdx0];
    Vec3f w1 = world[vIdx1];
    Vec3f w2 = world[vIdx2];
    buffWorld[i] = w0 * alp + w1 * bet + w2 * gam;

    // Set normal
    Vec3f n0 = normal[vIdx0];
    Vec3f n1 = normal[vIdx1];
    Vec3f n2 = normal[vIdx2];
    n0.norm(); n1.norm(); n2.norm();
    buffNormal[i] = n0 * alp + n1 * bet + n2 * gam;

    // Set texture
    Vec2f t0 = texture[vIdx0];
    Vec2f t1 = texture[vIdx1];
    Vec2f t2 = texture[vIdx2];
    buffTexture[i] = t0 * alp + t1 * bet + t2 * gam;

    // Set mesh ID
    buffMeshId[i] = meshID[vIdx0];
}