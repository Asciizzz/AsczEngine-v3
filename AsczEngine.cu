#include <FpsHandler.cuh>
#include <Mesh3D.cuh>
#include <Camera3D.cuh>

#include <SFML/Graphics.hpp>

int main() {
    Mesh mesh(
        0, // ID
        Vecs3f({ // Position
            Vec3f(-1, -1, 0), Vec3f(1, -1, 0), Vec3f(1, 1, 0), Vec3f(-1, 1, 0)
        }),
        Vecs3f({ // Normal
            Vec3f(0, 0, 1), Vec3f(0, 0, 1), Vec3f(0, 0, 1), Vec3f(0, 0, 1)
        }),
        Vecs2f({ // Texture
            Vec2f(0, 0), Vec2f(1, 0), Vec2f(1, 1), Vec2f(0, 1)
        }),
        Vecs3uli({ // Faces
            Vec3uli(0, 1, 2), Vec3uli(0, 2, 3)
        })
    );

    // For the time being we gonna just use for loop to transform vertices
    Vecs3f transformedVs(mesh.pos.size());

    Mesh3D MESH(mesh);

    MESH.printVertices();
    MESH.printFaces();

    Camera3D camera;

    sf::RenderWindow window(sf::VideoMode(800, 600), "AsczEngine");
    window.setMouseCursorVisible(false);

    camera.aspect = window.getSize().x / window.getSize().y;

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
        sf::Mouse::setPosition(sf::Vector2i(400, 300), window);

        // Move from center
        int dMx = mousePos.x - 400;
        int dMy = mousePos.y - 300;

        // Camera look around
        camera.rot.x += dMy * camera.mSens * FPS.dTimeSec;
        camera.rot.y -= dMx * camera.mSens * FPS.dTimeSec;
        camera.restrictRot();
        camera.updateMVP();

        // Mouse Click = move forward
        float vel = 0;
        bool m_left = sf::Mouse::isButtonPressed(sf::Mouse::Left);
        bool m_right = sf::Mouse::isButtonPressed(sf::Mouse::Right);
        if (m_left && !m_right)      vel = 20;
        else if (m_right && !m_left) vel = -20;
        else                         vel = 0;
        camera.pos += camera.forward * vel * FPS.dTimeSec;
        camera.pos.print();
        camera.rot.print();
        std::cout << "---\n";

        // Perform transformation
        for (ULLInt i = 0; i < mesh.pos.size(); i++) {
            Vec4f v = mesh.pos[i].toVec4f();
            v = camera.mvp * v;
            transformedVs[i] = v.toVec3f();
        }

        window.clear(sf::Color::Black);
        // Draw mesh based on transformed vertices
        for (ULLInt i = 0; i < mesh.faces.size(); i++) {
            Vec3uli f = mesh.faces[i];
            sf::ConvexShape triangle;
            triangle.setPointCount(3);
            // NDC coordinates
            Vec3f v0 = transformedVs[f.x];
            Vec3f v1 = transformedVs[f.y];
            Vec3f v2 = transformedVs[f.z];
            // Screen coordinates
            Vec2f p0 = Vec2f((v0.x + 1) * 400, (v0.y + 1) * 300);
            Vec2f p1 = Vec2f((v1.x + 1) * 400, (v1.y + 1) * 300);
            Vec2f p2 = Vec2f((v2.x + 1) * 400, (v2.y + 1) * 300);

            triangle.setPoint(0, sf::Vector2f(p0.x, p0.y));
            triangle.setPoint(1, sf::Vector2f(p1.x, p1.y));
            triangle.setPoint(2, sf::Vector2f(p2.x, p2.y));

            sf::Color color = i % 2 ? sf::Color::Red : sf::Color::Blue;
            triangle.setFillColor(color);

            window.draw(triangle);
        }

        window.display();

        // Frame end
        FPS.endFrame();
    }

    return 0;
}