#include <FpsHandler.cuh>
#include <Mesh3D.cuh>
#include <Camera3D.cuh>

#include <SFML/Graphics.hpp>

int main() {
    // Create a cube mesh with 8 vertices and 12 faces
    Mesh test(1,// ID
        Vecs3f{ // Position
            Vec3f(-1, -1, -1), Vec3f(1, -1, -1), Vec3f(1, 1, -1), Vec3f(-1, 1, -1),
            Vec3f(-1, -1, 1), Vec3f(1, -1, 1), Vec3f(1, 1, 1), Vec3f(-1, 1, 1)
        },
        Vecs3f{ // Normal
            Vec3f(-1, -1, -1), Vec3f(1, -1, -1), Vec3f(1, 1, -1), Vec3f(-1, 1, -1),
            Vec3f(-1, -1, 1), Vec3f(1, -1, 1), Vec3f(1, 1, 1), Vec3f(-1, 1, 1)
        },
        Vecs2f{ // Texture
            Vec2f(0, 0), Vec2f(1, 0), Vec2f(1, 1), Vec2f(0, 1),
            Vec2f(0, 0), Vec2f(1, 0), Vec2f(1, 1), Vec2f(0, 1)
        },
        Vecs3f{ // Color
            Vec3f(255, 0, 0), Vec3f(0, 255, 0), Vec3f(0, 0, 255), Vec3f(255, 255, 0),
            Vec3f(255, 0, 0), Vec3f(0, 255, 0), Vec3f(0, 0, 255), Vec3f(255, 255, 0)
        },
        Vecs3uli{ // Faces
            Vec3uli(0, 1, 2), Vec3uli(0, 2, 3),
            Vec3uli(4, 5, 6), Vec3uli(4, 6, 7),
            Vec3uli(0, 4, 7), Vec3uli(0, 7, 3),
            Vec3uli(1, 5, 6), Vec3uli(1, 6, 2),
            Vec3uli(0, 1, 5), Vec3uli(0, 5, 4),
            Vec3uli(2, 3, 7), Vec3uli(2, 7, 6)
        }
    );

    // For the time being we gonna just use for loop to transform vertices
    Mesh3D MESH(test);
    MESH.printVertices();
    MESH.printFaces();

    Vecs3f transformedVs(MESH.numVs);

    Camera3D camera;

    int width = 1600;
    int height = 900;
    sf::RenderWindow window(sf::VideoMode(width, height), "AsczEngine");
    window.setMouseCursorVisible(false);

    camera.aspect = float(width) / float(height);

    FpsHandler &FPS = FpsHandler::instance();
    while (window.isOpen()) {
        // Frame start
        FPS.startFrame();

        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
                window.close();
            }
        }

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

        // Perform transformation
        for (ULLInt i = 0; i < test.pos.size(); i++) {
            // Fun functions
            test.pos[i].rotate(Vec3f(0, 0, 0), Vec3f(0, M_PI * FPS.dTimeSec, 0));

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

        window.display();

        // Frame end
        FPS.endFrame();
    }

    return 0;
}