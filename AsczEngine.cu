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

    // A cube
    Mesh3D cube(1,
        Vecs3f({
            Vec3f(-1, -1, -1), Vec3f(1, -1, -1),
            Vec3f(1, 1, -1), Vec3f(-1, 1, -1),
            Vec3f(-1, -1, 1), Vec3f(1, -1, 1),
            Vec3f(1, 1, 1), Vec3f(-1, 1, 1)
        }),
        Vecs3f({
            Vec3f(-1, -1, -1), Vec3f(1, -1, -1),
            Vec3f(1, 1, -1), Vec3f(-1, 1, -1),
            Vec3f(-1, -1, 1), Vec3f(1, -1, 1),
            Vec3f(1, 1, 1), Vec3f(-1, 1, 1)
        }),
        Vecs2f({
            Vec2f(0, 0), Vec2f(1, 0),
            Vec2f(1, 1), Vec2f(0, 1),
            Vec2f(0, 0), Vec2f(1, 0),
            Vec2f(1, 1), Vec2f(0, 1)
        }),
        Vecs4f({ 
            // Red Green Blue Yellow Cyan Magenta Orange Purple 
            Vec4f(255, 0, 0, 255), Vec4f(0, 255, 0, 255),
            Vec4f(0, 0, 255, 255), Vec4f(255, 255, 0, 255),
            Vec4f(0, 255, 255, 255), Vec4f(255, 0, 255, 255),
            Vec4f(255, 125, 0, 255), Vec4f(125, 0, 255, 255)

            // // Top side cyan, bottom side yellow
            // Vec4f(0, 255, 255, 255), Vec4f(0, 255, 255, 255),
            // Vec4f(255, 255, 0, 255), Vec4f(255, 255, 0, 255),
            // Vec4f(0, 255, 255, 255), Vec4f(0, 255, 255, 255),
            // Vec4f(255, 255, 0, 255), Vec4f(255, 255, 0, 255)
        }),
        Vecs3uli({
            Vec3uli(0, 1, 2), Vec3uli(0, 2, 3),
            Vec3uli(4, 5, 6), Vec3uli(4, 6, 7),
            Vec3uli(0, 4, 7), Vec3uli(0, 7, 3),
            Vec3uli(1, 5, 6), Vec3uli(1, 6, 2),
            Vec3uli(0, 1, 5), Vec3uli(0, 5, 4),
            Vec3uli(3, 2, 6), Vec3uli(3, 6, 7)
        })
    );

    cube.translate(1, Vec3f(0, 1, 0));

    float cubeScale = 30;
    cube.scale(0, Vec3f(), Vec3f(cubeScale));

    // Graphing calculator for y = f(x, z)
    Vecs3f world;
    Vecs3f normal;
    Vecs2f texture;
    Vecs4f color;

    Vecs3uli faces;

    // Append points to the grid
    Vec2f rangeX(-500, 500);
    Vec2f rangeZ(-500, 500);
    Vec2f step(1, 1);

    int sizeX = (rangeX.y - rangeX.x) / step.x + 1;
    int sizeZ = (rangeZ.y - rangeZ.x) / step.y + 1;

    float maxY = -INFINITY;
    float minY = INFINITY;
    for (float x = rangeX.x; x <= rangeX.y; x += step.x) {
        for (float z = rangeZ.x; z <= rangeZ.y; z += step.y) {
            // World pos of the point
            // float y = sin(x / 10) * cos(z / 10) * 10;
            float y = rand() % 20 - 10;

            maxY = std::max(maxY, y);
            minY = std::min(minY, y);

            world.push_back(Vec3f(x, y, z));
            normal.push_back(Vec3f(0, 0, -1));

            // x and z ratio (0 - 1)
            float ratioX = (x - rangeX.x) / (rangeX.y - rangeX.x);
            float ratioZ = (z - rangeZ.x) / (rangeZ.y - rangeZ.x);

            // Texture
            texture.push_back(Vec2f(ratioX, ratioZ));

            // Cool color
            color.push_back(Vec4f(40 * ratioX + 130, 255, 40 * ratioX + 130, 255));
        }
    }

    for (ULLInt i = 0; i < world.size(); i++) {
        // Set green color based on y value
        float ratioY = (world[i].y - minY) / (maxY - minY);
        color[i].y = 150 + 100 * ratioY;

        // Random ratio range from 0.8 to 1.2
        color[i].y *= 0.8 + 0.4 * (rand() % 100) / 100;
        color[i].y = std::min(255.0f, std::max(0.0f, color[i].y));
    }

    // Append faces to the grid
    for (ULLInt x = 0; x < sizeX - 1; x++) {
        for (ULLInt z = 0; z < sizeZ - 1; z++) {
            ULLInt i = x * sizeZ + z;
            faces.push_back(Vec3uli(i, i + 1, i + sizeZ));
            faces.push_back(Vec3uli(i + 1, i + sizeZ + 1, i + sizeZ));
        }
    }

    Mesh3D graph(1, world, normal, texture, color, faces);

    // STAR GENERATOR ALGORITHM
    /* Explanation:
    - We will generate a bunch of triangles(faces) very far away from the camera
    - Star will be generated randomly in a sphere area 600 < r < 1000
    - Star will have a random color
    */
    Vecs3f starWorld;
    Vecs3f starNormal;
    Vecs2f starTexture;
    Vecs4f starColor;
    Vecs3uli starFaces;

    int numStars = 2000;
    int curStars = 0;
    while (curStars < numStars) {
        Vec3f v0(
            rand() % 2000 - 1000,
            rand() % 2000 - 1000,
            rand() % 2000 - 1000
        );

        if (v0.mag() < 900 || v0.mag() > 1000) continue;

        Vec3f v1 = v0 + Vec3f(rand() % 16 - 8, rand() % 16 - 8, rand() % 16 - 8);
        Vec3f v2 = v0 + Vec3f(rand() % 16 - 8, rand() % 16 - 8, rand() % 16 - 8);

        starWorld.push_back(v0);
        starWorld.push_back(v1);
        starWorld.push_back(v2);

        starNormal.push_back(Vec3f());
        starNormal.push_back(Vec3f());
        starNormal.push_back(Vec3f());

        starTexture.push_back(Vec2f(0, 0));
        starTexture.push_back(Vec2f(0, 0));
        starTexture.push_back(Vec2f(0, 0));

        starColor.push_back(Vec4f(255, 255, 255, 255));
        starColor.push_back(Vec4f(255, 255, 255, 255));
        starColor.push_back(Vec4f(255, 255, 255, 255));

        starFaces.push_back(Vec3uli(curStars * 3, curStars * 3 + 1, curStars * 3 + 2));

        curStars++;
    }

    Mesh3D star(2, starWorld, starNormal, starTexture, starColor, starFaces);

    RENDER.mesh += cube;
    // RENDER.mesh += graph;
    // RENDER.mesh += star;
    RENDER.allocateProjection();

    graph.freeMemory();
    cube.freeMemory();
    star.freeMemory();

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

        // Rotate the star
        float rotY = M_PI_2 / 120 * FPS.dTimeSec;
        RENDER.mesh.rotate(2, Vec3f(), Vec3f(0, -rotY, 0));

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