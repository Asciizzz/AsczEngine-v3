#include <FpsHandler.cuh>
#include <CsLogHandler.cuh>
#include <Utility.cuh>

#include <VertexShader.cuh>
#include <FragmentShader.cuh>
#include <SFMLTexture.cuh>

// Playgrounds
#include <SolarSystem.cuh>
#include <DotObj.cuh>

// Main
int main() {
    // Initialize Default stuff
    FpsHandler &FPS = FpsHandler::instance();
    CsLogHandler LOG = CsLogHandler();

    int width, height, pixelSize;
    // Note: higher pixelSize = lower resolution
    std::ifstream("assets/cfg/resolution.txt")
        >> width >> height >> pixelSize;

    Graphic3D &GRAPHIC = Graphic3D::instance();
    GRAPHIC.setResolution(width, height, pixelSize);
    GRAPHIC.createRuntimeStreams();

    Camera3D &CAMERA = GRAPHIC.camera;
    std::ifstream("assets/cfg/cameraPos.txt")
        >> CAMERA.pos.x >> CAMERA.pos.y >> CAMERA.pos.z;
    std::ifstream("assets/cfg/cameraSpd.txt")
        >> CAMERA.velSpec >> CAMERA.slowFactor >> CAMERA.fastFactor;
    std::ifstream("assets/cfg/cameraView.txt")
        >> CAMERA.near >> CAMERA.far;

    SFMLTexture SFTex = SFMLTexture(width, height);
    sf::RenderWindow window(sf::VideoMode(width, height), "AsczEngine");
    window.setMouseCursorVisible(false);
    sf::Mouse::setPosition(sf::Vector2i(
        GRAPHIC.res_half.x, GRAPHIC.res_half.y
    ), window);

    // ===================== INITIALIZATION =====================
    // Each model in models.txt will contain:
    // src scl rotX rotY rotZ transX transY transZ
    DotObj dotObjs("assets/cfg/models.txt");
    GRAPHIC.mesh.push(dotObjs.objs);

    SolarSystem solarSystem;
    solarSystem.setStars(4, 400, 6000, 8000, 6);
    GRAPHIC.mesh.push(solarSystem.stars);


    GRAPHIC.mallocRuntimeFaces();

    // Debug purposes
    GRAPHIC.mesh.logMeshMap(24);

    // Beta: shadow mapping
    int shdwWidth, shdwHeight, shdwTileSizeX, shdwTileSizeY;
    std::ifstream("assets/cfg/shadow.txt") >> shdwWidth >> shdwHeight >> shdwTileSizeX >> shdwTileSizeY;
    GRAPHIC.createShadowMap(shdwWidth, shdwHeight, shdwTileSizeX, shdwTileSizeY);

    // To avoid floating point errors
    // We will use a float that doesnt have a lot of precision
    float fovDeg = 90;

    // Cool rainbow effect for title
    Vec3f rainbow;
    short cycle = 0;

    // Turn on/off features
    bool materialMode = true;
    bool shadowMode = false;
    bool shadeMode = true;
    bool customMode = false;

    bool moveMode = true;

    // Other miscellaneus stuff
    short logmode = 0;

    // =====================================================
    // ===================== MAIN LOOP =====================
    // =====================================================

    while (window.isOpen()) {
        // Frame start
        FPS.startFrame();

        // Setting input activities
        bool m_left = sf::Mouse::isButtonPressed(sf::Mouse::Left);
        bool m_right = sf::Mouse::isButtonPressed(sf::Mouse::Right);
        bool k_ctrl = sf::Keyboard::isKeyPressed(sf::Keyboard::LControl);
        bool k_shift = sf::Keyboard::isKeyPressed(sf::Keyboard::LShift);

        bool k_w = sf::Keyboard::isKeyPressed(sf::Keyboard::W);
        bool k_a = sf::Keyboard::isKeyPressed(sf::Keyboard::A);
        bool k_s = sf::Keyboard::isKeyPressed(sf::Keyboard::S);
        bool k_d = sf::Keyboard::isKeyPressed(sf::Keyboard::D);
        bool k_space = sf::Keyboard::isKeyPressed(sf::Keyboard::Space);

        bool k_q = sf::Keyboard::isKeyPressed(sf::Keyboard::Q);
        bool k_e = sf::Keyboard::isKeyPressed(sf::Keyboard::E);
        bool k_t = sf::Keyboard::isKeyPressed(sf::Keyboard::T);

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

                // Press 1 to toggle texture mode
                if (event.key.code == sf::Keyboard::Num1)
                    materialMode = !materialMode;
                // Press 2 to toggle shadow mode
                if (event.key.code == sf::Keyboard::Num2)
                    shadowMode = !shadowMode;
                // Press 3 to toggle shade mode
                if (event.key.code == sf::Keyboard::Num3)
                    shadeMode = !shadeMode;
                // Press 4 to toggle custom mode
                if (event.key.code == sf::Keyboard::Num4)
                    customMode = !customMode;

                // Press tab (without ctrl and shift) to toggle log mode
                if (event.key.code == sf::Keyboard::Tab &&
                    !k_ctrl && !k_shift) {
                    logmode = (logmode + 1) % 3;
                }

                if (event.key.code == sf::Keyboard::Tab && logmode == 2) {
                    int &curlogpart = GRAPHIC.mesh.curlogpart;
                    int &maxlogpart = GRAPHIC.mesh.maxlogpart;

                    if (k_ctrl) curlogpart ++;
                    if (k_shift) curlogpart --;
                    // Wrap around
                    curlogpart = (curlogpart + maxlogpart) % maxlogpart;
                }

                // Press Z to toggle move mode
                if (event.key.code == sf::Keyboard::Z)
                    moveMode = !moveMode;

                // Press L to read light.txt file and set its prop
                if (event.key.code == sf::Keyboard::L) {
                    std::ifstream dir("assets/cfg/lightDir.txt");
                    dir >> GRAPHIC.light.dir.x >> GRAPHIC.light.dir.y >> GRAPHIC.light.dir.z;

                    std::ifstream color("assets/cfg/lightColor.txt");
                    color >> GRAPHIC.light.color.x >> GRAPHIC.light.color.y >> GRAPHIC.light.color.z;
                }

                // Press ~ to enter active status setting mode
                if (event.key.code == sf::Keyboard::Tilde) {
                    MeshMap &meshmap = GRAPHIC.mesh.meshmap;

                    std::string mesh;
                    std::string obj;
                    bool active;

                    std::cout << "\nGet mesh: ";
                    std::cin >> mesh; 
                    if (meshmap.find(mesh) == meshmap.end()) {
                        std::cout << "Mesh not found" << std::endl;
                        continue;
                    }

                    std::cout << GRAPHIC.mesh.meshmap[mesh].getObjRtMapLog();
                    
                    std::cout << "\nGet obj: ";
                    std::cin >> obj;
                    if (GRAPHIC.mesh.meshmap[mesh].objmapRT.find(obj) ==
                        GRAPHIC.mesh.meshmap[mesh].objmapRT.end()) {
                        std::cout << "Obj not found" << std::endl;
                        continue;
                    }

                    std::cout << "\nSet active status (0/1): ";
                    std::cin >> active;

                    GRAPHIC.mesh.meshmap[mesh].setActiveStatus(obj, active);

                    std::cout << "Active status set" << std::endl;
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

        // Mouse movement => Look around
        if (CAMERA.focus) {
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
        }

        // Update camera
        CAMERA.update();

        // Csgo perspective mode
        if (CAMERA.focus && !moveMode) {
            float vel = CAMERA.velSpec;
            // Hold ctrl to go slow, hold shift to go fast
            if (k_ctrl && !k_shift)      vel *= CAMERA.slowFactor;
            else if (k_shift && !k_ctrl) vel *= CAMERA.fastFactor;
            // Press W/S to move forward/backward
            if (k_w && !k_s) CAMERA.pos += CAMERA.forward * vel * FPS.dTimeSec;
            if (k_s && !k_w) CAMERA.pos -= CAMERA.forward * vel * FPS.dTimeSec;
            // Press A/D to move left/right
            if (k_a && !k_d) CAMERA.pos -= CAMERA.right * vel * FPS.dTimeSec;
            if (k_d && !k_a) CAMERA.pos += CAMERA.right * vel * FPS.dTimeSec;
        }

        if (CAMERA.focus && moveMode) {
            // Gravity
            CAMERA.vel.y -= 1.15 * FPS.dTimeSec;

            // On ground
            if (CAMERA.pos.y < 1.5) {
                CAMERA.vel.y = 0;
                CAMERA.pos.y = 1.5;
            }

            // Jump
            if (k_space && abs(CAMERA.vel.y) < 0.01) CAMERA.vel.y = .3;

            float vel_xz = sqrt(
                CAMERA.vel.x * CAMERA.vel.x + CAMERA.vel.z * CAMERA.vel.z
            );

            // Move
            bool moving = false;
            if (k_w && !k_s) {
                moving = true;
                CAMERA.vel.x += CAMERA.forward.x * FPS.dTimeSec;
                CAMERA.vel.z += CAMERA.forward.z * FPS.dTimeSec;
            }
            if (k_s && !k_w) {
                moving = true;
                CAMERA.vel.x -= CAMERA.forward.x * FPS.dTimeSec;
                CAMERA.vel.z -= CAMERA.forward.z * FPS.dTimeSec;
            }
            if (k_a && !k_d) {
                moving = true;
                CAMERA.vel.x -= CAMERA.right.x * FPS.dTimeSec;
                CAMERA.vel.z -= CAMERA.right.z * FPS.dTimeSec;
            }
            if (k_d && !k_a) {
                moving = true;
                CAMERA.vel.x += CAMERA.right.x * FPS.dTimeSec;
                CAMERA.vel.z += CAMERA.right.z * FPS.dTimeSec;
            }
            if (vel_xz > 0 && !moving) {
                CAMERA.vel.x /= 1.5;
                CAMERA.vel.z /= 1.5;
            }

            // Limit and restrict horizontal speed
            if (vel_xz > 1) {
                CAMERA.vel.x /= vel_xz;
                CAMERA.vel.z /= vel_xz;
            }

            CAMERA.pos += CAMERA.vel * .1;
        }

        // ========== Playgrounds ==============

        // Set light position to camera position
        GRAPHIC.light.dir = CAMERA.pos;

        // Rotate stars
        if (GRAPHIC.mesh.meshmap.find("SolarSystem_Stars") !=
            GRAPHIC.mesh.meshmap.end()) {
            Mesh &stars = GRAPHIC.mesh.meshmap["SolarSystem_Stars"];

            stars.rotateRuntime("star0", Vec3f(), M_PI_2 * FPS.dTimeSec / 190, 1);
            stars.rotateRuntime("star1", Vec3f(), M_PI_2 * FPS.dTimeSec / 210, 1);
            stars.rotateRuntime("star2", Vec3f(), M_PI_2 * FPS.dTimeSec / 230, 1);
            stars.rotateRuntime("star3", Vec3f(), M_PI_2 * FPS.dTimeSec / 340, 1);

            stars.rotateRuntime("star0", Vec3f(), M_PI_2 * FPS.dTimeSec / 990, 0);
            stars.rotateRuntime("star1", Vec3f(), M_PI_2 * FPS.dTimeSec / 810, 0);
            stars.rotateRuntime("star2", Vec3f(), M_PI_2 * FPS.dTimeSec / 1030, 0);
            stars.rotateRuntime("star3", Vec3f(), M_PI_2 * FPS.dTimeSec / 740, 0);
        }

        // ========== Render Pipeline ==========

        // Vertex Shader
        VertexShader::cameraProjection();
        VertexShader::frustumCulling();
        VertexShader::createDepthMap();
        VertexShader::rasterization();

        // Fragment Shader (bunch of beta features)
        if (materialMode) FragmentShader::applyMaterial();
        if (shadowMode) {
            FragmentShader::resetShadowMap();
            FragmentShader::createShadowMap();
            FragmentShader::applyShadowMap();
        }
        if (shadeMode) FragmentShader::phongShading();
        if (customMode) FragmentShader::customShader();

        // From buffer to SFMLtexture
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
        switch (logmode) {
        case 0:
            LOG.addLog("[Tab] to toggle log", sf::Color(255, 245, 55), 1);
            break;

        case 1:
            LOG.addLog(
                "Screen:\n| Res: " + std::to_string(width) +
                " x " + std::to_string(height) + "\n" +
                "| Pixel Size: " + std::to_string(pixelSize) + "\n" +
                "| RTFace1: " + std::to_string(GRAPHIC.rtCount1) + " / "
                + std::to_string(GRAPHIC.mesh.f.size / 3) + "\n" +
                "| RTFace2: " + std::to_string(GRAPHIC.rtCount2) + " / "
                + std::to_string(GRAPHIC.mesh.f.size / 3),
                sf::Color(255, 160, 160)
            );
            LOG.addLog(CAMERA.data(), sf::Color(160, 255, 160));
            LOG.addLog(GRAPHIC.light.data(), sf::Color(160, 160, 255));
            LOG.addLog("Shader (BETA)", sf::Color(255, 255, 255), 1);
            LOG.addLog(
                "| Material: " + std::to_string(materialMode),
                sf::Color(materialMode ? 255 : 100, 50, 50)
            );
            LOG.addLog(
                "| Shadow: " + std::to_string(shadowMode),
                sf::Color(50, shadowMode ? 255 : 100, 50)
            );
            LOG.addLog(
                "| Shade: " + std::to_string(shadeMode),
                sf::Color(50, 50, shadeMode ? 255 : 100)
            );
            LOG.addLog(
                "| Custom: " + std::to_string(customMode),
                sf::Color(customMode ? 255 : 100, 50, customMode ? 255 : 100)
            );
            break;

        case 2:
            // Also debug purposes
            int curlogpart = GRAPHIC.mesh.curlogpart;
            int maxlogpart = GRAPHIC.mesh.maxlogpart;

            LOG.addLog(
                "[Mesh Map Page " +
                std::to_string(curlogpart + 1) + " / " +
                std::to_string(maxlogpart) + "]" +
                "(ctrl/shift + tab to navigate)",
                sf::Color(255, 100, 100), 1
            );

            if (curlogpart != 0) LOG.addLog("  . . .", sf::Color(255, 255, 255), 1);
            LOG.addLog(GRAPHIC.mesh.getMeshMapLog(), sf::Color(255, 255, 255));
            if (curlogpart != maxlogpart - 1) LOG.addLog("  . . .", sf::Color(255, 255, 255), 1);
            
            break;
        }

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