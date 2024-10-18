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
        camera.rot.x -= dMy * camera.mSens * FPS.dTimeSec;
        camera.rot.y += dMx * camera.mSens * FPS.dTimeSec;
        camera.restrictRot();
        camera.updateMVP();

        bool m_left = sf::Mouse::isButtonPressed(sf::Mouse::Left);
        bool m_right = sf::Mouse::isButtonPressed(sf::Mouse::Right);

        // Mouse Click = move forward
        float vel = 0;
        if (m_left && !m_right)      vel = 20;
        else if (m_right && !m_left) vel = -20;
        else                         vel = 0;
        camera.pos += camera.forward * vel * FPS.dTimeSec;

        camera.rot.print();
        camera.pos.print();

        window.clear(sf::Color::Black);

        window.display();

        // Frame end
        FPS.endFrame();
    }

    return 0;
}