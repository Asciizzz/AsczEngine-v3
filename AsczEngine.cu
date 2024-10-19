#include <FpsHandler.cuh>
#include <CsLogHandler.cuh>
#include <Lighting3D.cuh>

#include <SFMLTexture.cuh>

int main() {
    // Initialize Default stuff
    FpsHandler &FPS = FpsHandler::instance();
    CsLogHandler LOG = CsLogHandler();

    Render3D &RENDER = Render3D::instance();
    RENDER.setResolution(1600, 900);

    Camera3D &CAMERA = RENDER.camera;
    CAMERA.pos = Vec3f(0, 0, -24);
    CAMERA.rot = Vec3f(0, 0, 0);

    SFMLTexture SFTex = SFMLTexture(1600, 900);
    sf::RenderWindow window(sf::VideoMode(1600, 900), "AsczEngine");
    window.setMouseCursorVisible(false);
    sf::Mouse::setPosition(sf::Vector2i(
        RENDER.res_half.x, RENDER.res_half.y
    ), window);

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
    Vecs4f cubeColor = {
        Vec4f(255, 0, 0, 255), Vec4f(0, 255, 0, 255),
        Vec4f(255, 255, 0, 255), Vec4f(0, 0, 255, 255), 
        Vec4f(0, 255, 255, 255), Vec4f(255, 0, 255, 255),
        Vec4f(255, 125, 0, 255), Vec4f(125, 0, 255, 255)
    };
    Vecs3uli cubeFaces = {
        Vec3uli(0, 1, 2), Vec3uli(0, 2, 3),
        Vec3uli(4, 5, 6), Vec3uli(4, 6, 7),
        Vec3uli(0, 4, 7), Vec3uli(0, 7, 3),
        Vec3uli(1, 5, 6), Vec3uli(1, 6, 2),
        Vec3uli(0, 1, 5), Vec3uli(0, 5, 4),
        Vec3uli(3, 2, 6), Vec3uli(3, 6, 7)
    };
    Mesh3D cube(0, cubeWorld, cubeNormal, cubeTexture, cubeColor, cubeFaces);
    cube.scale(0, Vec3f(), Vec3f(4));

    // Create a white wall behind the cube
    float wallSize = 12;
    Vecs3f wallWorld = {
        Vec3f(-wallSize, -wallSize, wallSize), Vec3f(wallSize, -wallSize, wallSize),
        Vec3f(wallSize, wallSize, wallSize), Vec3f(-wallSize, wallSize, wallSize)
    };
    Vecs3f wallNormal = { // Facing towards the cube
        Vec3f(0, 0, -1), Vec3f(0, 0, -1),
        Vec3f(0, 0, -1), Vec3f(0, 0, -1)
    };
    Vecs2f wallTexture = {
        Vec2f(0, 0), Vec2f(1, 0),
        Vec2f(1, 1), Vec2f(0, 1)
    };
    Vecs4f wallColor = {
        Vec4f(255, 125, 125, 255), Vec4f(125, 255, 125, 255),
        Vec4f(125, 125, 255, 255), Vec4f(255, 255, 125, 255)
    };
    Vecs3uli wallFaces = {
        Vec3uli(0, 1, 2), Vec3uli(0, 2, 3)
    };
    Mesh3D wall(1, wallWorld, wallNormal, wallTexture, wallColor, wallFaces);

    // Create an equallateral triangle
    // This is 
    Mesh equTri(2,
        Vecs3f{
            Vec3f(-0.5, -sqrt(3) / 4, 0), Vec3f(0.5, - sqrt(3) / 4, 0), Vec3f(0, sqrt(3) / 4, 0)
        },
        Vecs3f{
            Vec3f(0, 0, 1), Vec3f(0, 0, 1), Vec3f(0, 0, 1)
        },
        Vecs2f{
            Vec2f(0, 0), Vec2f(1, 0), Vec2f(0.5, 1)
        },
        Vecs4f{
            Vec4f(255, 0, 0, 255), Vec4f(0, 255, 0, 255), Vec4f(0, 0, 255, 255)
        },
        Vecs3uli{
            Vec3uli(0, 1, 2)
        }
    );
    Mesh3D tri(equTri);

    RENDER.mesh += cube;
    RENDER.mesh += wall;
    // RENDER.mesh += tri;
    RENDER.allocateProjection();

    // Free memory
    cube.freeMemory();
    wall.freeMemory();

    Lighting3D &LIGHT = Lighting3D::instance();
    LIGHT.allocateShadowMap(800, 800);
    LIGHT.allocateLightProj();

    // To avoid floating point errors
    // We will use a float that doesnt have a lot of precision
    float fovDeg = 90;

    // Cool rainbow effect for title
    double rainbowR = 255;
    double rainbowG = 0;
    double rainbowB = 0;
    short cycle = 0;

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
                    CAMERA.focus = !CAMERA.focus;
                    window.setMouseCursorVisible(!CAMERA.focus);
                    sf::Mouse::setPosition(sf::Vector2i(
                        RENDER.res_half.x, RENDER.res_half.y
                    ), window);
                }
            }

            // Scroll to zoom in/out
            if (event.type == sf::Event::MouseWheelScrolled) {
                if (event.mouseWheelScroll.delta > 0) fovDeg -= 10;
                else                                  fovDeg += 10;

                if (fovDeg < 10) fovDeg = 10;
                if (fovDeg > 170) fovDeg = 170;

                float fovRad = fovDeg * M_PI / 180;
                CAMERA.fov = fovRad;
            }
        }

        if (CAMERA.focus) {
            // Mouse movement handling
            sf::Vector2i mousepos = sf::Mouse::getPosition(window);
            sf::Mouse::setPosition(sf::Vector2i(
                RENDER.res_half.x, RENDER.res_half.y
            ), window);

            // Move from center
            int dMx = mousepos.x - RENDER.res_half.x;
            int dMy = mousepos.y - RENDER.res_half.y;

            // Camera look around
            CAMERA.rot.x -= dMy * CAMERA.mSens * FPS.dTimeSec;
            CAMERA.rot.y -= dMx * CAMERA.mSens * FPS.dTimeSec;
            CAMERA.restrictRot();
            CAMERA.updateMVP();

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
            CAMERA.pos += CAMERA.forward * vel * FPS.dTimeSec;
        }

        // Rotate the cube
        float rot1 = M_PI / 6 * FPS.dTimeSec;
        float rot2 = M_PI / 3 * FPS.dTimeSec;
        RENDER.mesh.rotate(0, Vec3f(), Vec3f(rot1, 0, rot2));

        // ========== Render Pipeline ==========

        RENDER.cameraProjection();
        RENDER.createDepthMap();
        RENDER.rasterization();

        // Beta feature
        // LIGHT.phongShading();
        LIGHT.lightProjection();
        // LIGHT.resetShadowMap();
        // LIGHT.createShadowMap();
        // LIGHT.applyShadowMap();

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

        // ========== Log handling ==========

        // Rainbow color
        double step = 120 * FPS.dTimeSec;
        if (cycle == 0) {
            rainbowG += step; rainbowR -= step;
            if (rainbowG >= 255) cycle = 1;
        } else if (cycle == 1) {
            rainbowB += step; rainbowG -= step;
            if (rainbowB >= 255) cycle = 2;
        } else if (cycle == 2) {
            rainbowR += step; rainbowB -= step;
            if (rainbowR >= 255) cycle = 0;
        }
        sf::Color rainbow = sf::Color(rainbowR, rainbowG, rainbowB);
        LOG.addLog("Welcome to AsczEngine 3.0", rainbow, 1);

        double gRatio = double(FPS.fps - 10) / 50;
        gRatio = std::max(0.0, std::min(gRatio, 1.0));
        sf::Color fpsColor((1 - gRatio) * 255, gRatio * 255, 0);
        LOG.addLog("FPS: " + std::to_string(FPS.fps), fpsColor);

        // Camera data
        LOG.addLog(CAMERA.data(), sf::Color::White);

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