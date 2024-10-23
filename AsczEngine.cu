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

    int width, height, pixelSize, tileWidth, tileHeight;
    // Note: higher pixelSize = lower resolution
    std::ifstream("cfg/resolution.txt")
        >> width >> height >> pixelSize >> tileWidth >> tileHeight;

    Graphic3D &GRAPHIC = Graphic3D::instance();
    GRAPHIC.setResolution(width, height, pixelSize);
    GRAPHIC.setTileSize(tileWidth, tileHeight);

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

    // Create a .obj mesh (Work in progress)
    Mesh3D obj = Playground::readObjFile(0, objPath, true);
    obj.scale(Vec3f(), Vec3f(objScale));
    // obj.rotate(0, Vec3f(), Vec3f(0, 0, 0));

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
    Vecs3ulli cubeFaces = {
        Vec3ulli(0, 1, 2), Vec3ulli(0, 2, 3),
        Vec3ulli(4, 5, 6), Vec3ulli(4, 6, 7),
        Vec3ulli(0, 4, 7), Vec3ulli(0, 7, 3),
        Vec3ulli(1, 5, 6), Vec3ulli(1, 6, 2),
        Vec3ulli(0, 1, 5), Vec3ulli(0, 5, 4),
        Vec3ulli(3, 2, 6), Vec3ulli(3, 6, 7)
    };
    Mesh3D cube(1, cubeWorld, cubeNormal, cubeTexture, cubeColor, cubeFaces);
    cube.scale(Vec3f(), Vec3f(4));

    // Create a white wall behind the cube
    float wallSize = 10;
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
    Vecs3ulli wallFaces = {
        Vec3ulli(0, 1, 2), Vec3ulli(0, 2, 3)
    };
    Mesh3D wall(2, wallWorld, wallNormal, wallTexture, wallColor, wallFaces);

    // Graphing calculator for y = f(x, z)
    Vecs3f world;
    Vecs3f normal;
    Vecs2f texture;
    Vecs4f color;
    Vecs3ulli faces;

    // Append points to the grid
    Vec2f rangeX(-500, 500);
    Vec2f rangeZ(-500, 500);
    Vec2f step(1, 1);

    int sizeX = (rangeX.y - rangeX.x) / step.x + 1;
    int sizeZ = (rangeZ.y - rangeZ.x) / step.y + 1;

    float maxY = -INFINITY;
    float minY = INFINITY;
    int numX = 0;
    int numZ = 0;
    for (float x = rangeX.x; x <= rangeX.y; x += step.x) {
        numX++;

        for (float z = rangeZ.x; z <= rangeZ.y; z += step.y) {
            numZ++;

            // World pos of the point
            float y = sin(x / 50) * cos(z / 50) * 50;
            // float y = rand() % 30 - 10;

            maxY = std::max(maxY, y);
            minY = std::min(minY, y);

            world.push_back(Vec3f(x, y, z));

            // x and z ratio (0 - 1)
            float ratioX = (x - rangeX.x) / (rangeX.y - rangeX.x);
            float ratioZ = (z - rangeZ.x) / (rangeZ.y - rangeZ.x);
            // Texture
            texture.push_back(Vec2f(ratioX, ratioZ));
        }
    }
    numZ /= numX;

    for (ULLInt i = 0; i < world.size(); i++) {
        // Set color based on ratio
        float r = (world[i].x - rangeX.x) / (rangeX.y - rangeX.x);
        float g = (world[i].y - minY) / (maxY - minY);
        float b = (world[i].z - rangeZ.x) / (rangeZ.y - rangeZ.x);
        color.push_back(Vec4f(255 - r * 255, g * 255, b * 255, 255));

        // Set normal based on the triangle of surrounding points
        int x = i / numZ;
        int z = i % numZ;

        int edge = 10;
        if (x < edge || x >= numX - edge || z < edge || z >= numZ - edge) {
            normal.push_back(Vec3f(0, 1, 0));
            continue;
        }

        if (x % 100 == 0 || z % 100 == 0) {
            normal.push_back(Vec3f(0, 1, 0));
            continue;
        }

        int idxLeft = x * numZ + z - 1;
        int idxRight = x * numZ + z + 1;
        int idxUp = (x - 1) * numZ + z;
        int idxDown = (x + 1) * numZ + z;

        std::vector<int> idxDir = {
            idxLeft, idxRight, idxUp, idxDown
        };

        // Triangle group: mid left up, mid up right, mid right down, mid down left
        std::vector<Vec3f> triNormals;

        for (int j = 0; j < 4; j++) {
            int idx = idxDir[j];
            Vec3f mid = world[i];
            Vec3f left = world[idxLeft];
            Vec3f right = world[idxRight];
            Vec3f up = world[idxUp];
            Vec3f down = world[idxDown];

            if (j == 0) triNormals.push_back((mid - left) & (up - left));
            if (j == 1) triNormals.push_back((mid - up) & (right - up));
            if (j == 2) triNormals.push_back((mid - right) & (down - right));
            if (j == 3) triNormals.push_back((mid - down) & (left - down));
        }

        Vec3f avgNormal = Vec3f();
        for (Vec3f triNormal : triNormals) {
            avgNormal += triNormal;
        }
        avgNormal.norm();
        normal.push_back(avgNormal);
    }

    // Append faces to the grid
    for (ULLInt x = 0; x < sizeX - 1; x++) {
        for (ULLInt z = 0; z < sizeZ - 1; z++) {
            ULLInt i = x * sizeZ + z;
            faces.push_back(Vec3ulli(i, i + 1, i + sizeZ));
            faces.push_back(Vec3ulli(i + 1, i + sizeZ + 1, i + sizeZ));
        }
    }

    Mesh3D graph(3, world, normal, texture, color, faces);

    GRAPHIC += obj;
    // GRAPHIC.mesh += cube;
    // GRAPHIC.mesh += wall;
    // GRAPHIC += graph;

    GRAPHIC.mallocGFaces();
    GRAPHIC.mallocFaceStreams();

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

                // Press f2 to set the resolution and pixel size again
                // Do not use this as I just introduced tile-based rasterization
                // if (event.key.code == sf::Keyboard::F2) {
                //     std::ifstream("cfg/resolution.txt") >> width >> height >> pixelSize;
                //     GRAPHIC.setResolution(width, height, pixelSize);
                //     SFTex.free();
                //     SFTex.resize(width, height);
                    
                //     window.setSize(sf::Vector2u(width, height));
                // }
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

        // Press R to rotate the object
        if (k_r) {
            float rot = M_PI / 3 * FPS.dTimeSec;
            if (k_ctrl) rot *= -1;
            if (k_shift) rot *= 3;
            GRAPHIC.mesh.rotate(0, Vec3f(), Vec3f(0, rot, 0));
        }
        // Press Q to rotate light source in x axis
        if (k_q) {
            float rot = M_PI / 3 * FPS.dTimeSec;
            if (k_ctrl) rot *= -1;
            if (k_shift) rot *= 3;

            GRAPHIC.light.dir.rotate(Vec3f(0), Vec3f(rot, 0, 0));
        }
        // Press E to rotate light source in z axis
        if (k_e) {
            float rot = M_PI / 3 * FPS.dTimeSec;
            if (k_ctrl) rot *= -1;
            if (k_shift) rot *= 3;

            GRAPHIC.light.dir.rotate(Vec3f(0), Vec3f(0, 0, rot));
        }

        // ========== Render Pipeline ==========

        VertexShader::cameraProjection();
        VertexShader::filterVisibleFaces();
        VertexShader::createDepthMap();
        VertexShader::rasterization();

        FragmentShader::phongShading();

        // Custom Fragment Shader
        FragmentShader::customFragmentShader();

        // From buffer to texture
        // (clever way to incorporate CUDA into SFML)
        SFTex.updateTexture(
            GRAPHIC.buffer.color,
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
            "| Tile Size: " + std::to_string(tileWidth) + " x " + std::to_string(tileHeight) + "\n" +
            "| Visible Face: " + std::to_string(GRAPHIC.numVisibFs) + " / " + std::to_string(GRAPHIC.mesh.numFs),
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