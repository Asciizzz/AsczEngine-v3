#include <FpsHandler.cuh>
#include <CsLogHandler.cuh>

#include <VertexShader.cuh>
#include <FragmentShader.cuh>
#include <SFMLTexture.cuh>

#include <Playground.cuh>
#include <Sphere3D.cuh>

// Main
int main() {
    // Initialize Default stuff
    FpsHandler &FPS = FpsHandler::instance();
    CsLogHandler LOG = CsLogHandler();

    int width, height, pixelSize, tileSizeX, tileSizeY;
    // Note: higher pixelSize = lower resolution
    std::ifstream("assets/cfg/resolution.txt")
        >> width >> height >> pixelSize >> tileSizeX >> tileSizeY;

    Graphic3D &GRAPHIC = Graphic3D::instance();
    GRAPHIC.setResolution(width, height, pixelSize);
    GRAPHIC.setTileSize(tileSizeX, tileSizeY);

    Camera3D &CAMERA = GRAPHIC.camera;
    std::ifstream("assets/cfg/cameraPos.txt") >> CAMERA.pos.x >> CAMERA.pos.y >> CAMERA.pos.z;
    std::ifstream("assets/cfg/cameraSpd.txt") >> CAMERA.slowFactor >> CAMERA.fastFactor;

    SFMLTexture SFTex = SFMLTexture(width, height);
    sf::RenderWindow window(sf::VideoMode(width, height), "AsczEngine");
    window.setMouseCursorVisible(false);
    sf::Mouse::setPosition(sf::Vector2i(
        GRAPHIC.res_half.x, GRAPHIC.res_half.y
    ), window);

    // ===================== INITIALIZATION =====================
    // Each model in models.txt will contain:
    // src scl rotX rotY rotZ transX transY transZ
    std::ifstream objsFile("assets/cfg/models.txt");
    std::string line;
    std::vector<Mesh> objs;

    std::string objsTxt = "";
    int objsCount = 0;
    while (std::getline(objsFile, line)) {
        // If line start with #, it's a comment
        if (line[0] == '#' || line.empty()) continue;

        std::string objPath = "";
        float scale = 1;
        Vec3f translate;
        Vec3f rotate;

        std::stringstream ss(line);

        ss >> objPath >> scale;
        ss >> rotate.x >> rotate.y >> rotate.z;
        ss >> translate.x >> translate.y >> translate.z;
        rotate *= M_PI / 180;

        Mesh obj = Playground::readObjFile(objPath, 1, 1, true);
        obj.scaleIni(Vec3f(), Vec3f(scale));
        obj.rotateIni(Vec3f(), rotate);
        obj.translateIni(translate);

        GRAPHIC.mesh += obj;
        objs.push_back(obj);

        // Write to log
        objsTxt += "Obj " + std::to_string(objsCount) + " - " + objPath + "\n";
        objsCount++;
    }
    std::cout << objsTxt;

    // Create a test sphere
    Sphere3D sphere(Vec3f(0, 8, 0), 1);
    sphere.vel = Vec3f(0.3, 0, 0.1);
    sphere.angvel = Vec3f(M_PI * 2.8, 0, M_PI * 2.4);
    GRAPHIC.mesh += sphere.mesh;

    GRAPHIC.mallocRuntimeFaces();

    std::string texturePath = "";
    std::ifstream("assets/cfg/texture.txt") >> texturePath;
    GRAPHIC.createTexture(texturePath);

    int shdwWidth, shdwHeight, shdwTileSizeX, shdwTileSizeY;
    std::ifstream("assets/cfg/shadow.txt") >> shdwWidth >> shdwHeight >> shdwTileSizeX >> shdwTileSizeY;
    GRAPHIC.createShadowMap(shdwWidth, shdwHeight, shdwTileSizeX, shdwTileSizeY);

    // To avoid floating point errors
    // We will use a float that doesnt have a lot of precision
    float fovDeg = 90;

    // Cool rainbow effect for title
    Vec3f rainbow;
    short cycle = 0;

    // Turn on/off texture mode
    bool textureMode = true;
    bool shadowMode = true;
    bool shadeMode = true;

    // Gif animation texture
    Vec2ulli gifFrame = {0, 26};
    Vec2f gifTime = {0, 0.03};

    // Other miscellaneus stuff
    bool k_t_hold = false;

    Vec3f cam_vel;
    bool moving = false;

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
                    std::ifstream dir("assets/cfg/lightDir.txt");
                    dir >> GRAPHIC.light.dir.x >> GRAPHIC.light.dir.y >> GRAPHIC.light.dir.z;

                    std::ifstream color("assets/cfg/lightColor.txt");
                    color >> GRAPHIC.light.color.x >> GRAPHIC.light.color.y >> GRAPHIC.light.color.z;
                }
                
                // Press f2 to read texture.txt file and set its prop
                if (event.key.code == sf::Keyboard::F2) {
                    std::string texturePath = "";
                    std::ifstream("assets/cfg/texture.txt") >> texturePath;
                    GRAPHIC.createTexture(texturePath);
                }

                // Press 1 to toggle texture mode
                if (event.key.code == sf::Keyboard::Num1)
                    textureMode = !textureMode;
                // Press 2 to toggle shadow mode
                if (event.key.code == sf::Keyboard::Num2)
                    shadowMode = !shadowMode;
                // Press 3 to toggle shade mode
                if (event.key.code == sf::Keyboard::Num3)
                    shadeMode = !shadeMode;
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
        bool k_q = sf::Keyboard::isKeyPressed(sf::Keyboard::Q);
        bool k_e = sf::Keyboard::isKeyPressed(sf::Keyboard::E);
        bool k_t = sf::Keyboard::isKeyPressed(sf::Keyboard::T);

        bool k_w = sf::Keyboard::isKeyPressed(sf::Keyboard::W);
        bool k_a = sf::Keyboard::isKeyPressed(sf::Keyboard::A);
        bool k_s = sf::Keyboard::isKeyPressed(sf::Keyboard::S);
        bool k_d = sf::Keyboard::isKeyPressed(sf::Keyboard::D);
        bool k_space = sf::Keyboard::isKeyPressed(sf::Keyboard::Space);

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
            CAMERA.rot.y += dMx * CAMERA.mSens * FPS.dTimeSec;
            CAMERA.restrictRot();
            CAMERA.updateMVP();

            // // Mouse Click = move forward
            // float vel = 0;
            // // Move forward/backward
            // if (m_left && !m_right)      vel = 20;
            // else if (m_right && !m_left) vel = -20;
            // else                         vel = 0;
            // // Move slower/faster
            // if (k_ctrl && !k_shift)      vel *= CAMERA.slowFactor;
            // else if (k_shift && !k_ctrl) vel *= CAMERA.fastFactor;
            // // Update camera World pos
            // CAMERA.pos += CAMERA.forward * vel * FPS.dTimeSec;
        }

        // Gravity
        cam_vel.y -= 1.15 * FPS.dTimeSec;

        // On ground
        if (CAMERA.pos.y < 1.5) {
            cam_vel.y = 0;
            CAMERA.pos.y = 1.5;
        }

        // Jump
        if (k_space && cam_vel.y == 0) cam_vel.y = .3;

        float vel_xz = sqrt(cam_vel.x * cam_vel.x + cam_vel.z * cam_vel.z);

        // Move
        moving = false;
        if (k_w && !k_s) {
            moving = true;
            cam_vel.x += CAMERA.forward.x * FPS.dTimeSec;
            cam_vel.z += CAMERA.forward.z * FPS.dTimeSec;
        }
        if (k_s && !k_w) {
            moving = true;
            cam_vel.x -= CAMERA.forward.x * FPS.dTimeSec;
            cam_vel.z -= CAMERA.forward.z * FPS.dTimeSec;
        }
        if (k_a && !k_d) {
            moving = true;
            cam_vel.x += CAMERA.right.x * FPS.dTimeSec;
            cam_vel.z += CAMERA.right.z * FPS.dTimeSec;
        }
        if (k_d && !k_a) {
            moving = true;
            cam_vel.x -= CAMERA.right.x * FPS.dTimeSec;
            cam_vel.z -= CAMERA.right.z * FPS.dTimeSec;
        }
        if (vel_xz > 0 && !moving) {
            cam_vel.x /= 1.5;
            cam_vel.z /= 1.5;
        }

        // Limit and restrict horizontal speed
        if (vel_xz > 1) {
            cam_vel.x /= vel_xz;
            cam_vel.z /= vel_xz;
        }
        
        // Set postition
        CAMERA.pos += Vec3f(
            cam_vel.x / 1.2, cam_vel.y, cam_vel.z / 1.2
        );

        // Press T to read an transform.txt file and apply it
        // Note: hold ctrl to switch keyT from hold to tap
        if (k_t && (!k_t_hold || !k_ctrl)) {
            k_t_hold = true;

            Playground::applyTransformation(objs);
        }
        if (!k_t) k_t_hold = false;

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
        if (gifFrame.x < 10) frameStr = "00" + std::to_string(gifFrame.x);
        else if (gifFrame.x < 100) frameStr = "0" + std::to_string(gifFrame.x);
        else frameStr = std::to_string(gifFrame.x);
        std::string gifPath = "assets/Gif/frame_" + frameStr + ".png";

        if (gifTime.x < gifTime.y) {
            gifTime.x += FPS.dTimeSec;
        } else {
            gifTime.x = 0;

            gifFrame.x++;
            if (gifFrame.x >= gifFrame.y) gifFrame.x = 0;

            // GRAPHIC.createTexture(gifPath);
        }

        sphere.movement();

        // ========== Render Pipeline ==========

        // Vertex Shader
        VertexShader::cameraProjection();
        VertexShader::createRuntimeFaces();
        VertexShader::createDepthMap();
        VertexShader::rasterization();

        // Fragment Shader (bunch of beta features)
        if (textureMode) FragmentShader::applyTexture();
        if (shadowMode) {
            FragmentShader::resetShadowMap();
            FragmentShader::createShadowMap();
            FragmentShader::applyShadowMap();
        }
        if (shadeMode) FragmentShader::phongShading();

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
            rainbow.y += step; rainbow.x -= step;
            if (rainbow.y >= 255) cycle = 1;
        } else if (cycle == 1) {
            rainbow.z += step; rainbow.y -= step;
            if (rainbow.z >= 255) cycle = 2;
        } else if (cycle == 2) {
            rainbow.x += step; rainbow.z -= step;
            if (rainbow.x >= 255) cycle = 0;
        }
        sf::Color rainbowColor = sf::Color(rainbow.x, rainbow.y, rainbow.z);

        // Dynamic FPS color
        double gRatio = double(FPS.fps - 10) / 50;
        gRatio = std::max(0.0, std::min(gRatio, 1.0));
        sf::Color fpsColor((1 - gRatio) * 255, gRatio * 255, 0);

        // Log all the data
        LOG.addLog("Welcome to AsczEngine 3.0", rainbowColor, 1);
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
        LOG.addLog(CAMERA.data(), sf::Color(160, 255, 160));
        LOG.addLog(GRAPHIC.light.data(), sf::Color(160, 160, 255));
        LOG.addLog("Shader (BETA)", sf::Color(255, 255, 255), 1);
        LOG.addLog(
            "| Texture: " + std::to_string(textureMode),
            sf::Color(textureMode ? 255 : 100, 50, 50)
        );
        LOG.addLog(
            "| Shadow: " + std::to_string(shadowMode),
            sf::Color(50, shadowMode ? 255 : 100, 50)
        );
        LOG.addLog(
            "| Shade: " + std::to_string(shadeMode),
            sf::Color(50, 50, shadeMode ? 255 : 100)
        );

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