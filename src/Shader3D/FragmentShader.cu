#include <FragmentShader.cuh>

// Phong Shading

void FragmentShader::phongShading() {
    Graphic3D &graphic = Graphic3D::instance();
    Buffer3D &buffer = graphic.buffer;

    phongShadingKernel<<<buffer.blockNum, buffer.blockSize>>>(
        graphic.light,
        buffer.active, buffer.color, buffer.world, buffer.normal, buffer.texture,
        buffer.width, buffer.height
    );
    cudaDeviceSynchronize();
}

__global__ void phongShadingKernel(
    LightSrc light,
    bool *buffActive, Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture,
    int buffWidth, int buffHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight || !buffActive[i]) return;

    // Apply colored light
    buffColor[i].x *= light.color.x;
    buffColor[i].y *= light.color.y;
    buffColor[i].z *= light.color.z;
    buffColor[i].limit(0, 255);

    // Find the light direction
    Vec3f lightDir = light.dir * -1;
    Vec3f n = buffNormal[i];

    // Calculate the cosine of the angle between the normal and the light direction
    float dot = n * lightDir;
    
    float cosA = dot / (n.mag() * lightDir.mag());
    if (cosA < 0) cosA = 0;

    float diff = light.ambient * (1 - cosA) + light.specular * cosA;

    // Apply the light
    buffColor[i].x *= diff;
    buffColor[i].y *= diff;
    buffColor[i].z *= diff;
    buffColor[i].limit(0, 255);
}

// Custom Fragment Shader

void FragmentShader::customFragmentShader() {
    Graphic3D &graphic = Graphic3D::instance();
    Buffer3D &buffer = graphic.buffer;
    Mesh3D &mesh = graphic.mesh;

    customFragmentShaderKernel<<<buffer.blockNum, buffer.blockSize>>>(
        mesh.world, buffer.world, mesh.wObjId, buffer.wObjId,
        mesh.normal, buffer.normal, mesh.nObjId, buffer.nObjId,
        mesh.texture, buffer.texture, mesh.tObjId, buffer.tObjId,
        mesh.color, buffer.color,
        mesh.faceWs, mesh.faceNs, mesh.faceTs, buffer.faceID,
        buffer.bary, buffer.bary,
        buffer.active, buffer.depth, buffer.width, buffer.height
    );
    cudaDeviceSynchronize();
}

__global__ void customFragmentShaderKernel(
    Vec3f *world, Vec3f *buffWorld, UInt *wObjId, UInt *buffWObjId,
    Vec3f *normal, Vec3f *buffNormal, UInt *nObjId, UInt *buffNObjId,
    Vec2f *texture, Vec2f *buffTexture, UInt *tObjId, UInt *buffTObjId,
    Vec4f *color, Vec4f *buffColor,
    Vec3ulli *faceWs, Vec3ulli *faceNs, Vec3ulli *faceTs, ULLInt *buffFaceId,
    Vec3f *bary, Vec3f *buffBary,
    bool *buffActive, float *buffDepth, int buffWidth, int buffHeight
) {
    int bIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bIdx >= buffWidth * buffHeight || !buffActive[bIdx]) return;

    return;

    int bx = bIdx % buffWidth;
    int by = bIdx / buffWidth;

    ULLInt fIdx = buffFaceId[bIdx];

    // Set vertex, texture, and normal indices
    Vec3ulli vIdx = faceWs[fIdx];
    Vec3ulli tIdx = faceTs[fIdx];
    Vec3ulli nIdx = faceNs[fIdx];

    // Get barycentric coordinates
    float alp = buffBary[bIdx].x;
    float bet = buffBary[bIdx].y;
    float gam = buffBary[bIdx].z;

    // Have fun with the custom fragment shader

    // If the z is in range 0.9 to 1, reduce the opacity
    // 0.9: 100% opacity, 1: 0% opacity
    float ratio = 1 - (buffDepth[bIdx] - 0.9) * 10;
    ratio = ratio < 0 ? 0 : ratio > 1 ? 1 : ratio;

    // Set color
    if (ratio != 1) {
        buffColor[bIdx].x *= ratio;
        buffColor[bIdx].y *= ratio;
        buffColor[bIdx].z *= ratio;
    }
}