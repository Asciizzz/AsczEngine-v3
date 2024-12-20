#include <SFMLTexture.cuh>

SFMLTexture::SFMLTexture(int width, int height) { resize(width, height); }

void SFMLTexture::free() {
    delete[] sfPixel;
    cudaFree(d_sfPixel);
}

void SFMLTexture::resize(int width, int height) {
    texture.create(width, height);
    sprite.setTexture(texture);

    // Allocate memory for the Pixel buffer
    sfPixel = new sf::Uint8[width * height * 4];
    cudaMalloc(&d_sfPixel, width * height * 4 * sizeof(sf::Uint8));

    pixelCount = width * height * 4;
    blockNum = (width * height + blockSize - 1) / blockSize;
}

void SFMLTexture::updateTexture(
    float *cr, float *cg, float *cb, float *ca,
    int b_w, int b_h, int p_s
) {
    int bCount = (b_w * b_h + blockSize - 1) / blockSize;

    updateTextureKernel<<<bCount, blockSize>>>(
        d_sfPixel, cr, cg, cb, ca, b_w, b_h, p_s
    );
    cudaMemcpy(sfPixel, d_sfPixel, pixelCount * sizeof(sf::Uint8), cudaMemcpyDeviceToHost);
    texture.update(sfPixel);
}

// Kernel for updating the texture
__global__ void updateTextureKernel(
    sf::Uint8 *d_sfPixel, float *cr, float *cg, float *cb, float *ca,
    int b_w, int b_h, int p_s
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b_w * b_h) return;

    int x = i % b_w;
    int y = i / b_w;
    int b_i = x + y * b_w;

    for (int i = 0; i < p_s; i++)
    for (int j = 0; j < p_s; j++) {
        int p_i = (x * p_s + i) + (y * p_s + j) * b_w * p_s;
        p_i *= 4;

        d_sfPixel[p_i + 0] = (sf::Uint8)(cr[b_i]);
        d_sfPixel[p_i + 1] = (sf::Uint8)(cg[b_i]);
        d_sfPixel[p_i + 2] = (sf::Uint8)(cb[b_i]);
        d_sfPixel[p_i + 3] = (sf::Uint8)(ca[b_i]);
    }
}