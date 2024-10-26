#include <FpsHandler.cuh>
#include <CsLogHandler.cuh>

#include <VertexShader.cuh>
#include <FragmentShader.cuh>
#include <SFMLTexture.cuh>

#include <Playground.cuh>

int main() {
    // Initialize Default stuff
    FpsHandler &FPS = FpsHandler::instance();
    CsLogHandler LOG = CsLogHandler();

    int width, height, pixelSize, tileSizeX, tileSizeY;
    // Note: higher pixelSize = lower resolution
    std::ifstream("cfg/resolution.txt")
        >> width >> height >> pixelSize >> tileSizeX >> tileSizeY;

    Graphic3D &GRAPHIC = Graphic3D::instance();
    GRAPHIC.setResolution(width, height, pixelSize);
    GRAPHIC.setTileSize(tileSizeX, tileSizeY);

    Camera3D &CAMERA = GRAPHIC.camera;
    std::ifstream("cfg/cameraPos.txt") >> CAMERA.pos.x >> CAMERA.pos.y >> CAMERA.pos.z;
    std::ifstream("cfg/cameraSpd.txt") >> CAMERA.slowFactor >> CAMERA.fastFactor;

    SFMLTexture SFTex = SFMLTexture(width, height);
    sf::RenderWindow window(sf::VideoMode(width, height), "AsczEngine");
    window.setMouseCursorVisible(false);
    sf::Mouse::setPosition(sf::Vector2i(
        GRAPHIC.res_half.x, GRAPHIC.res_half.y
    ), window);

    // ===================== INITIALIZATION =====================

    std::string objPath = "";
    float objScale = 1;
    // File: <path> <scale>
    std::ifstream file("cfg/model.txt");
    file >> objPath >> objScale;
    Mesh obj = Playground::readObjFile(objPath, 1, 1, true);
    #pragma omp parallel for
    for (size_t i = 0; i < obj.wx.size(); i++) {
        // Rotate in the z axis by 180 degrees
        Vec3f v = obj.w3f(i);
        v.rotate(Vec3f(0), Vec3f(0, 0, 0));
        obj.wx[i] = v.x;
        obj.wy[i] = v.y;
        obj.wz[i] = v.z;

        obj.wx[i] *= objScale;
        obj.wy[i] *= objScale;
        obj.wz[i] *= objScale;
    }

    // A wall span x +- wallSize, y +- wallSize
    float wallSize = 2;
    Mesh wall = Playground::readObjFile("assets/Models/Shapes/Wall.obj", 1, 1, true);
    wall.scale(Vec3f(), Vec3f(wallSize));
    wall.translate(Vec3f(0, 0, wallSize));

    // A cube span x +- 1, y +- 1, z +- 1
    Mesh cube = Playground::readObjFile("assets/Models/Shapes/Cube.obj", 1, 1, true);
    cube.scale(Vec3f(), Vec3f(.4));
    cube.translate(Vec3f(.4, 0, -2));

    // Append all the meshes here
    GRAPHIC.mesh += obj;
    GRAPHIC.mesh += wall;
    GRAPHIC.mesh += cube;

    GRAPHIC.mallocRuntimeFaces();
    GRAPHIC.mallocFaceStreams();
    GRAPHIC.createShadowMap(800, 800, 80, 80);

    std::string texturePath = "";
    std::ifstream("cfg/texture.txt") >> texturePath;
    GRAPHIC.createTexture(texturePath);

    // To avoid floating point errors
    // We will use a float that doesnt have a lot of precision
    float fovDeg = 90;

    // Cool rainbow effect for title
    double rainbowR = 255;
    double rainbowG = 0;
    double rainbowB = 0;
    short cycle = 0;

    // Turn on/off texture mode
    bool textureMode = true;

    // Gif animation texture
    int gifFrame = 0;
    int gifMaxFrame = 26;
    float gifTime = 0;
    float gifMaxTime = 0.03;

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
                        GRAPHIC.res_half.x, GRAPHIC.res_half.y
                    ), window);
                }

                // Press L to read light.txt file and set its prop
                if (event.key.code == sf::Keyboard::L) {
                    std::ifstream dir("cfg/lightDir.txt");
                    dir >> GRAPHIC.light.dir.x >> GRAPHIC.light.dir.y >> GRAPHIC.light.dir.z;

                    std::ifstream color("cfg/lightColor.txt");
                    color >> GRAPHIC.light.color.x >> GRAPHIC.light.color.y >> GRAPHIC.light.color.z;
                }
                
                // Press f2 to read texture.txt file and set its prop
                if (event.key.code == sf::Keyboard::F2) {
                    std::string texturePath = "";
                    std::ifstream("cfg/texture.txt") >> texturePath;
                    GRAPHIC.createTexture(texturePath);
                }

                // Press T to toggle texture mode
                if (event.key.code == sf::Keyboard::T) {
                    textureMode = !textureMode;
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

        bool m_left = sf::Mouse::isButtonPressed(sf::Mouse::Left);
        bool m_right = sf::Mouse::isButtonPressed(sf::Mouse::Right);
        bool k_ctrl = sf::Keyboard::isKeyPressed(sf::Keyboard::LControl);
        bool k_shift = sf::Keyboard::isKeyPressed(sf::Keyboard::LShift);
        bool k_r = sf::Keyboard::isKeyPressed(sf::Keyboard::R);
        bool k_q = sf::Keyboard::isKeyPressed(sf::Keyboard::Q);
        bool k_e = sf::Keyboard::isKeyPressed(sf::Keyboard::E);

        if (CAMERA.focus) {
            // Mouse movement handling
            sf::Vector2i mousepos = sf::Mouse::getPosition(window);
            sf::Mouse::setPosition(sf::Vector2i(
                GRAPHIC.res_half.x, GRAPHIC.res_half.y
            ), window);

            // Move from center
            int dMx = mousepos.x - GRAPHIC.res_half.x;
            int dMy = mousepos.y - GRAPHIC.res_half.y;

            // Camera look around
            CAMERA.rot.x -= dMy * CAMERA.mSens * FPS.dTimeSec;
            CAMERA.rot.y -= dMx * CAMERA.mSens * FPS.dTimeSec;
            CAMERA.restrictRot();
            CAMERA.updateMVP();

            // Mouse Click = move forward
            float vel = 0;
            // Move forward/backward
            if (m_left && !m_right)      vel = 20;
            else if (m_right && !m_left) vel = -20;
            else                         vel = 0;
            // Move slower/faster
            if (k_ctrl && !k_shift)      vel *= CAMERA.slowFactor;
            else if (k_shift && !k_ctrl) vel *= CAMERA.fastFactor;
            // Update camera World pos
            CAMERA.pos += CAMERA.forward * vel * FPS.dTimeSec;
        }

        // Press Q to rotate light source in x axis
        if (k_q) {
            float rot = M_PI / 3 * FPS.dTimeSec;
            if (k_ctrl) rot *= -1;
            if (k_shift) rot *= 3;

            GRAPHIC.light.dir.rotate(Vec3f(0), Vec3f(rot, 0, 0));
        }
        // Press E to rotate light source in y axis
        if (k_e) {
            float rot = M_PI / 3 * FPS.dTimeSec;
            if (k_ctrl) rot *= -1;
            if (k_shift) rot *= 3;

            GRAPHIC.light.dir.rotate(Vec3f(0), Vec3f(0, rot, 0));
        }

        // ========== Playgrounds ==============

        // 3 digit frame number (add additional 0s if needed)
        std::string frameStr;
        if (gifFrame < 10) frameStr = "00" + std::to_string(gifFrame);
        else if (gifFrame < 100) frameStr = "0" + std::to_string(gifFrame);
        else frameStr = std::to_string(gifFrame);

        std::string gifPath = "assets/Gif/frame_" + frameStr + ".png";
        if (gifTime < gifMaxTime) {
            gifTime += FPS.dTimeSec;
        } else {
            gifTime = 0;
            gifFrame++;

            GRAPHIC.createTexture(gifPath);

            if (gifFrame >= gifMaxFrame) {
                gifFrame = 0;
            }
        }

        // ========== Render Pipeline ==========

        VertexShader::cameraProjection();
        VertexShader::createRuntimeFaces();
        VertexShader::createDepthMapBeta();
        VertexShader::rasterization();

        if (textureMode) FragmentShader::applyTexture();
        FragmentShader::phongShading();
        
        FragmentShader::resetShadowMap();
        FragmentShader::createShadowMap();
        FragmentShader::applyShadowMap();

        // From buffer to texture
        // (clever way to incorporate CUDA into SFML)
        SFTex.updateTexture(
            GRAPHIC.buffer.color.x,
            GRAPHIC.buffer.color.y,
            GRAPHIC.buffer.color.z,
            GRAPHIC.buffer.color.w,
            GRAPHIC.buffer.width,
            GRAPHIC.buffer.height,
            GRAPHIC.pixelSize
        );

        // ========== Log handling ==========

        // Rainbow title
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

        // Dynamic FPS color
        double gRatio = double(FPS.fps - 10) / 50;
        gRatio = std::max(0.0, std::min(gRatio, 1.0));
        sf::Color fpsColor((1 - gRatio) * 255, gRatio * 255, 0);

        // Log all the data
        LOG.addLog("Welcome to AsczEngine 3.0", rainbow, 1);
        LOG.addLog("FPS: " + std::to_string(FPS.fps), fpsColor);
        LOG.addLog(
            "Screen:\n| Res: " + std::to_string(width) +
            " x " + std::to_string(height) +
            " | Pixel Size: " + std::to_string(pixelSize) + "\n" +
            "| Tile Size: " + std::to_string(tileSizeX) + " x " + std::to_string(tileSizeY) + "\n" +
            "| Visible Face: " + std::to_string(GRAPHIC.faceCounter) +
            " / " + std::to_string(GRAPHIC.mesh.faces.size / 3),
            sf::Color(255, 160, 160)
        );
        LOG.addLog(CAMERA.data(), sf::Color(160, 160, 255));
        LOG.addLog(GRAPHIC.light.data(), sf::Color(160, 255, 160));

        // Displays
        window.clear(sf::Color(0, 0, 0));
        window.draw(SFTex.sprite);
        LOG.drawLog(window);
        window.display();

        // Frame end
        FPS.endFrame();
    }

    // Clean up
    GRAPHIC.free();
    SFTex.free();

    return 0;
}