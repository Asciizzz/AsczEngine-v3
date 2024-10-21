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
        mesh.world, buffer.world, mesh.wMeshId, buffer.wMeshId,
        mesh.normal, buffer.normal, mesh.nMeshId, buffer.nMeshId,
        mesh.texture, buffer.texture, mesh.tMeshId, buffer.tMeshId,
        mesh.color, buffer.color,
        mesh.faces, buffer.faceID, buffer.bary, buffer.bary,
        buffer.active, buffer.width, buffer.height
    );
    cudaDeviceSynchronize();
}

__global__ void customFragmentShaderKernel(
    Vec3f *world, Vec3f *buffWorld, UInt *wMeshId, UInt *buffWMeshId,
    Vec3f *normal, Vec3f *buffNormal, UInt *nMeshId, UInt *buffNMeshId,
    Vec2f *texture, Vec2f *buffTexture, UInt *tMeshId, UInt *buffTMeshId,
    Vec4f *color, Vec4f *buffColor,
    Vec3x3uli *faces, ULLInt *buffFaceId, Vec3f *bary, Vec3f *buffBary,
    bool *buffActive, int buffWidth, int buffHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight || !buffActive[i]) return;

    return;

    int bx = i % buffWidth;
    int by = i / buffWidth;

    ULLInt fIdx = buffFaceId[i];

    // Set vertex, texture, and normal indices
    Vec3uli vIdx = faces[fIdx].v;
    Vec3uli tIdx = faces[fIdx].t;
    Vec3uli nIdx = faces[fIdx].n;

    // Get barycentric coordinates
    float alp = buffBary[i].x;
    float bet = buffBary[i].y;
    float gam = buffBary[i].z;

    // Have fun with the custom fragment shader

    bool even = (bx + by) % 2 == 0;

    if (even) {
        buffColor[i].x *= 0.5;
    }
}