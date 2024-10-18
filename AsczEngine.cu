#include <FpsHandler.cuh>
#include <CsLogHandler.cuh>
#include <Mesh3D.cuh>
#include <Camera3D.cuh>

#include <SFML/Graphics.hpp>

int main() {
    // Initialize Default stuff
    FpsHandler &FPS = FpsHandler::instance();
    CsLogHandler LOG = CsLogHandler();

    // Graphing calculator for y = f(x, z)

    Vecs3f pos;
    Vecs3f normal;
    Vecs2f tex;
    Vecs3f color;

    Vecs3uli faces;

    // Append points to the grid
    Vec2f rangeX(-100, 100);
    Vec2f rangeZ(-100, 100);
    Vec2f step(1, 1);

    int sizeX = (rangeX.y - rangeX.x) / step.x + 1;
    int sizeZ = (rangeZ.y - rangeZ.x) / step.y + 1;

    for (float x = rangeX.x; x <= rangeX.y; x += step.x) {
        for (float z = rangeZ.x; z <= rangeZ.y; z += step.y) {
            // Position of the point
            float y = sin(x / 10) * cos(z / 10) * 10;
            pos.push_back(Vec3f(x, y, z));
            // Not important for now
            normal.push_back(Vec3f(0, 1, 0));
            tex.push_back(Vec2f(0, 0));

            // Cool color
            float ratioX = (x - rangeX.x) / (rangeX.y - rangeX.x);
            float ratioY = (y - rangeX.x) / (rangeX.y - rangeX.x);
            float ratioZ = (z - rangeZ.x) / (rangeZ.y - rangeZ.x);
            color.push_back(Vec3f(255 * ratioX, 255 * ratioZ, 255));
        }
    }

    // Append faces to the grid
    for (ULLInt x = 0; x < sizeX - 1; x++) {
        for (ULLInt z = 0; z < sizeZ - 1; z++) {
            ULLInt i = x * sizeZ + z;
            faces.push_back(Vec3uli(i, i + 1, i + sizeZ));
            faces.push_back(Vec3uli(i + 1, i + sizeZ + 1, i + sizeZ));
        }
    }

    // Create a cube mesh with 8 vertices and 12 faces
    Mesh test(0, pos, normal, tex, color, faces);

    // For the time being we gonna just use for loop to transform vertices
    Mesh3D MESH(test); 

    Vecs3f transformedVs(MESH.numVs);

    Camera3D camera;

    int width = 1600;
    int height = 900;
    sf::RenderWindow window(sf::VideoMode(width, height), "AsczEngine");
    window.setMouseCursorVisible(false);

    camera.aspect = float(width) / float(height);

    while (window.isOpen()) {
        // Frame start
        FPS.startFrame();

        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
                window.close();
            }

            // Press f1 to toggle focus
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::F1) {
                    camera.focus = !camera.focus;
                    window.setMouseCursorVisible(!camera.focus);
                    sf::Mouse::setPosition(sf::Vector2i(width/2, height/2), window);
                }
            }
        }

        if (camera.focus) {
            // Mouse movement handling
            sf::Vector2i mousePos = sf::Mouse::getPosition(window);
            sf::Mouse::setPosition(sf::Vector2i(width/2, height/2), window);

            // Move from center
            int dMx = mousePos.x - width/2;
            int dMy = mousePos.y - height/2;

            // Camera look around
            camera.rot.x -= dMy * camera.mSens * FPS.dTimeSec;
            camera.rot.y -= dMx * camera.mSens * FPS.dTimeSec;
            camera.restrictRot();
            camera.updateMVP();

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
            // Update camera position
            camera.pos += camera.forward * vel * FPS.dTimeSec;
        }

        // Perform transformation
        for (ULLInt i = 0; i < test.pos.size(); i++) {
            // Fun functions
            // test.pos[i].rotate(Vec3f(0, 0, 0), Vec3f(0, M_PI * FPS.dTimeSec, 0));

            // Project vertices to NDC
            Vec4f v = test.pos[i].toVec4f();
            v = camera.mvp * v;
            v.y *= -1; // Invert Y axis
            transformedVs[i] = v.toVec3f();
        }

        window.clear(sf::Color::Black);
        // Draw mesh based on transformed vertices
        for (ULLInt i = 0; i < MESH.numFs; i++) {
            Vec3uli f = test.faces[i];

            // If a single point of the test outside of the frustum, skip drawing
            if (!camera.isInsideFrustum(test.pos[f.x]) ||
                !camera.isInsideFrustum(test.pos[f.y]) ||
                !camera.isInsideFrustum(test.pos[f.z])) {
                continue;
            }

            // NDC coordinates
            Vec3f v0 = transformedVs[f.x];
            Vec3f v1 = transformedVs[f.y];
            Vec3f v2 = transformedVs[f.z];
            // Screen coordinates
            Vec2f p0 = Vec2f((v0.x + 1) * width/2, (v0.y + 1) * height/2);
            Vec2f p1 = Vec2f((v1.x + 1) * width/2, (v1.y + 1) * height/2);
            Vec2f p2 = Vec2f((v2.x + 1) * width/2, (v2.y + 1) * height/2);

            sf::Color colorA = sf::Color(test.color[f.x].x, test.color[f.x].y, test.color[f.x].z);
            sf::Color colorB = sf::Color(test.color[f.y].x, test.color[f.y].y, test.color[f.y].z);
            sf::Color colorC = sf::Color(test.color[f.z].x, test.color[f.z].y, test.color[f.z].z);

            // Create 3 lines for each face to draw wireframe
            sf::Vertex line01[] = {
                sf::Vertex(sf::Vector2f(p0.x, p0.y), colorA),
                sf::Vertex(sf::Vector2f(p1.x, p1.y), colorB)
            };
            sf::Vertex line12[] = {
                sf::Vertex(sf::Vector2f(p1.x, p1.y), colorB),
                sf::Vertex(sf::Vector2f(p2.x, p2.y), colorC)
            };
            sf::Vertex line02[] = {
                sf::Vertex(sf::Vector2f(p0.x, p0.y), colorA),
                sf::Vertex(sf::Vector2f(p2.x, p2.y), colorC)
            };

            window.draw(line01, 2, sf::Lines);
            window.draw(line12, 2, sf::Lines);
            window.draw(line02, 2, sf::Lines);
        }

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

    return 0;
}