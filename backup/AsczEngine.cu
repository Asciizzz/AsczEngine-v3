#include <Render3D.cuh>
#include <FpsHandler.cuh>

#include <SFML/Graphics.hpp>
#include <thread>

int main() {
    // Initializer
    FpsHandler &FPS = FpsHandler::instance();
    Render3D &RENDER = Render3D::instance();

    // Create example mesh data of a cube
    std::vector<float> x1 = {1, 1, 1, 1, -1, -1, -1, -1};
    std::vector<float> y1 = {1, 1, -1, -1, 1, 1, -1, -1};
    std::vector<float> z1 = {1, -1, -1, 1, 1, -1, -1, 1};
    std::vector<float> nx1 = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> ny1 = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> nz1 = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> u1 = {0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<float> v1 = {0, 1, 1, 0, 0, 1, 1, 0};
    std::vector<uint32_t> indices1 = {0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 5, 6, 1, 6, 2, 0, 1, 5, 0, 5, 4, 3, 2, 6, 3, 6, 7};

    for (size_t i = 0; i < x1.size(); i++) {
        x1[i] *= 10;
        y1[i] *= 10;
        z1[i] *= 10;

        z1[i] += 40;
    }

    // Create a mesh object
    Mesh3D mesh1(x1.size(), indices1.size(), 0);
    mesh1.upload(x1, y1, z1, nx1, ny1, nz1, u1, v1, indices1);

    // Create another cube
    std::vector<float> x2 = {1, 1, 1, 1, -1, -1, -1, -1};
    std::vector<float> y2 = {1, 1, -1, -1, 1, 1, -1, -1};
    std::vector<float> z2 = {1, -1, -1, 1, 1, -1, -1, 1};
    std::vector<float> nx2 = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> ny2 = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> nz2 = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> u2 = {0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<float> v2 = {0, 1, 1, 0, 0, 1, 1, 0};
    std::vector<uint32_t> indices2 = {0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 5, 6, 1, 6, 2, 0, 1, 5, 0, 5, 4, 3, 2, 6, 3, 6, 7};

    for (size_t i = 0; i < x2.size(); i++) {
        x2[i] *= 10;
        y2[i] *= 10;
        z2[i] *= 10;

        z2[i] += 40;
        x2[i] += 20;
    }

    Mesh3D mesh2(x2.size(), indices2.size(), 1);
    mesh2.upload(x2, y2, z2, nx2, ny2, nz2, u2, v2, indices2);

    RENDER.MESH += mesh1;
    RENDER.MESH += mesh2;

    // Very mediocre drawing method
    float *px = new float[RENDER.MESH.numVtxs];
    float *py = new float[RENDER.MESH.numVtxs];
    float *pz = new float[RENDER.MESH.numVtxs];
    uint32_t *idx = new uint32_t[RENDER.MESH.numIdxs];

    cudaMemcpy(idx, RENDER.MESH.idxs.vertexId, RENDER.MESH.numIdxs * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Create an SFML window
    sf::RenderWindow window(
        sf::VideoMode(RENDER.CAMERA.width, RENDER.CAMERA.height),
        "3D Mesh Projection"
    );
    // Hide the cursor
    window.setMouseCursorVisible(false);

    // Main loop
    while (window.isOpen()) {
        // FPS handling
        FPS.startFrame();

        // Event handling
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                (event.type == sf::Event::KeyPressed &&
                event.key.code == sf::Keyboard::Escape)) {
                window.close();
            }
        }

        // Mouse movement handling
        sf::Vector2i mousePos = sf::Mouse::getPosition(window);
        sf::Mouse::setPosition(sf::Vector2i(
            RENDER.CAMERA.centerX, RENDER.CAMERA.centerY
        ), window);

        // Move from center
        int dMx = mousePos.x - RENDER.CAMERA.centerX;
        int dMy = mousePos.y - RENDER.CAMERA.centerY;

        // Camera look around
        RENDER.CAMERA.pitch -= dMy * RENDER.CAMERA.mSens * FPS.dTimeSec;
        RENDER.CAMERA.yaw += dMx * RENDER.CAMERA.mSens * FPS.dTimeSec;

        // Restrict the angle
        RENDER.CAMERA.pitch = std::max(-M_PI_2, std::min(M_PI_2, double(RENDER.CAMERA.pitch)));

        if (RENDER.CAMERA.yaw > M_2_PI) RENDER.CAMERA.yaw -= M_2_PI;
        if (RENDER.CAMERA.yaw < 0) RENDER.CAMERA.yaw += M_2_PI;

        bool m_left = sf::Mouse::isButtonPressed(sf::Mouse::Left);
        bool m_right = sf::Mouse::isButtonPressed(sf::Mouse::Right);

        // Mouse Click = move forward
        if (m_left && !m_right)      RENDER.CAMERA.vel = 20;
        else if (m_right && !m_left) RENDER.CAMERA.vel = -20;
        else                         RENDER.CAMERA.vel = 0;
        RENDER.CAMERA.movement(FPS.dTimeSec);

        // Clear the window
        window.clear();

        mesh1.rotate(0, 0, 40, 0, 0, M_PI_2 * FPS.dTimeSec);

        // Project the mesh
        RENDER.toCameraProjection();

        cudaMemcpy(px, RENDER.MESH.prjs.x, RENDER.MESH.numVtxs * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(py, RENDER.MESH.prjs.y, RENDER.MESH.numVtxs * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(pz, RENDER.MESH.prjs.z, RENDER.MESH.numVtxs * sizeof(float), cudaMemcpyDeviceToHost);

        // Draw the mesh based on the indices
        int red = 100;
        for (size_t i = 0; i < RENDER.MESH.numIdxs; i += 3) {
            sf::ConvexShape triangle(3);

            if (pz[idx[i]] < 0 && pz[idx[i + 1]] < 0 && pz[idx[i + 2]] < 0) {
                continue;
            }

            triangle.setPoint(0, sf::Vector2f(px[idx[i]], py[idx[i]]));
            triangle.setPoint(1, sf::Vector2f(px[idx[i + 1]], py[idx[i + 1]]));
            triangle.setPoint(2, sf::Vector2f(px[idx[i + 2]], py[idx[i + 2]]));
            triangle.setFillColor(sf::Color(red, 100, 100));
            window.draw(triangle);

            red = red == 100 ? 200 : 100;
        }

        // Display the window
        window.display();

        // End of frame
        FPS.endFrame();
    }

    // Clean up
    delete[] px;
    delete[] py;

    return 0;
}