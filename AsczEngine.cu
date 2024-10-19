#include <FpsHandler.cuh>
#include <CsLogHandler.cuh>
#include <Render3D.cuh>

#include <SFMLTexture.cuh>

int main() {
    // Initialize Default stuff
    FpsHandler &FPS = FpsHandler::instance();
    CsLogHandler LOG = CsLogHandler();

    Render3D &RENDER = Render3D::instance();
    RENDER.setResolution(1600, 900);

    SFMLTexture SFTex = SFMLTexture(1600, 900);

    sf::RenderWindow window(sf::VideoMode(1600, 900), "AsczEngine");
    window.setMouseCursorVisible(false);

    Vecs3f cubeWorld = {
        Vec3f(-1, -1, -1), Vec3f(1, -1, -1),
        Vec3f(1, 1, -1), Vec3f(-1, 1, -1),
        Vec3f(-1, -1, 1), Vec3f(1, -1, 1),
        Vec3f(1, 1, 1), Vec3f(-1, 1, 1)
    };
    Vecs3f cubeNormal = {
        Vec3f(-1, -1, -1), Vec3f(1, -1, -1),
        Vec3f(1, 1, -1), Vec3f(-1, 1, -1),
        Vec3f(-1, -1, 1), Vec3f(1, -1, 1),
        Vec3f(1, 1, 1), Vec3f(-1, 1, 1)
    };
    Vecs2f cubeTexture = {
        Vec2f(0, 0), Vec2f(1, 0),
        Vec2f(1, 1), Vec2f(0, 1),
        Vec2f(0, 0), Vec2f(1, 0),
        Vec2f(1, 1), Vec2f(0, 1)
    };
    Vecs4f cube0Color(8, Vec4f(255, 255, 255, 255));
    // {
    //     Vec4f(255, 0, 0, 255), Vec4f(0, 255, 0, 255),
    //     Vec4f(255, 255, 0, 255), Vec4f(0, 0, 255, 255), 
    //     Vec4f(0, 255, 255, 255), Vec4f(255, 0, 255, 255),
    //     Vec4f(255, 125, 0, 255), Vec4f(125, 0, 255, 255)
    // };
    Vecs4f cube1Color(8, Vec4f(255, 255, 255, 255));
    // {
    //     Vec4f(0, 255, 0, 255), Vec4f(255, 0, 0, 255), 
    //     Vec4f(0, 0, 255, 255), Vec4f(255, 255, 0, 255),
    //     Vec4f(0, 255, 255, 255), Vec4f(255, 0, 255, 255),
    //     Vec4f(255, 125, 0, 255), Vec4f(125, 0, 255, 255)
    // };
    Vecs3uli cubeFaces = {
        Vec3uli(0, 1, 2), Vec3uli(0, 2, 3),
        Vec3uli(4, 5, 6), Vec3uli(4, 6, 7),
        Vec3uli(0, 4, 7), Vec3uli(0, 7, 3),
        Vec3uli(1, 5, 6), Vec3uli(1, 6, 2),
        Vec3uli(0, 1, 5), Vec3uli(0, 5, 4),
        Vec3uli(3, 2, 6), Vec3uli(3, 6, 7)
    };

    // A cube
    Mesh3D cube0(0, cubeWorld, cubeNormal, cubeTexture, cube0Color, cubeFaces);
    Mesh3D cube1(1, cubeWorld, cubeNormal, cubeTexture, cube1Color, cubeFaces);

    float cubeScale = 10;
    cube0.scale(0, Vec3f(), Vec3f(cubeScale));
    cube1.scale(1, Vec3f(), Vec3f(cubeScale));

    RENDER.mesh += cube0;
    // RENDER.mesh += cube1;
    // RENDER.mesh += star;
    RENDER.allocateProjection();

    cube0.freeMemory();
    cube1.freeMemory();

    while (window.isOpen()) {
        // Frame start
        FPS.startFrame();

        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
                window.close();
            }

            if (event.type == sf::Event::KeyPressed) {
                // Press f1 to toggle focus
                if (event.key.code == sf::Keyboard::F1) {
                    RENDER.camera.focus = !RENDER.camera.focus;
                    window.setMouseCursorVisible(!RENDER.camera.focus);
                    sf::Mouse::setPosition(sf::Vector2i(
                        RENDER.res_half.x, RENDER.res_half.y
                    ), window);
                }
            }
        }

        if (RENDER.camera.focus) {
            // Mouse movement handling
            sf::Vector2i mousepos = sf::Mouse::getPosition(window);
            sf::Mouse::setPosition(sf::Vector2i(
                RENDER.res_half.x, RENDER.res_half.y
            ), window);

            // Move from center
            int dMx = mousepos.x - RENDER.res_half.x;
            int dMy = mousepos.y - RENDER.res_half.y;

            // Camera look around
            RENDER.camera.rot.x -= dMy * RENDER.camera.mSens * FPS.dTimeSec;
            RENDER.camera.rot.y -= dMx * RENDER.camera.mSens * FPS.dTimeSec;
            RENDER.camera.restrictRot();
            RENDER.camera.updateMVP();

            // Mouse Click = move forward
            float vel = 0;
            bool m_left = sf::Mouse::isButtonPressed(sf::Mouse::Left);
            bool m_right = sf::Mouse::isButtonPressed(sf::Mouse::Right);
            bool k_ctrl = sf::Keyboard::isKeyPressed(sf::Keyboard::LControl);
            bool k_shift = sf::Keyboard::isKeyPressed(sf::Keyboard::LShift);
            // Move forward/backward
            if (m_left && !m_right)      vel = 20;
            else if (m_right && !m_left) vel = -20;
            else                         vel = 0;
            // Move slower/faster
            if (k_ctrl && !k_shift)      vel *= 0.2;
            else if (k_shift && !k_ctrl) vel *= 4;
            // Update camera World pos
            RENDER.camera.pos += RENDER.camera.forward * vel * FPS.dTimeSec;
        }

        // Rotate the cube
        float rotX = M_PI_2 / 12 * FPS.dTimeSec;
        float rotZ = M_PI_2 / 7 * FPS.dTimeSec;
        RENDER.mesh.rotate(0, Vec3f(), Vec3f(rotX, 0, rotZ));

        // Render Pipeline
        RENDER.vertexProjection();
        RENDER.createDepthMap();
        RENDER.rasterization();
        RENDER.lighting();

        // From buffer to texture
        // (clever way to incorporate CUDA into SFML)
        SFTex.updateTexture(
            RENDER.buffer.color,
            RENDER.buffer.width,
            RENDER.buffer.height,
            RENDER.pixelSize
        );
        window.clear(sf::Color(0, 0, 0));
        window.draw(SFTex.sprite);

        // Log handling
        // FPS <= 10: Fully Red
        // FPS >= 60: Fully Green
        double gRatio = double(FPS.fps - 10) / 50;
        gRatio = std::max(0.0, std::min(gRatio, 1.0));
        sf::Color fpsColor((1 - gRatio) * 255, gRatio * 255, 0);
        LOG.addLog("FPS: " + std::to_string(FPS.fps), fpsColor);
        LOG.drawLog(window);

        window.display();

        // Frame end
        FPS.endFrame();
    }

    // Clean up
    RENDER.freeProjection();
    RENDER.mesh.freeMemory();
    RENDER.buffer.free();

    return 0;
}