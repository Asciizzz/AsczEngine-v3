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
        Vecs3f({ // Color
            Vec3f(255, 0, 0), Vec3f(0, 255, 0), Vec3f(0, 0, 255), Vec3f(255, 255, 0)
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
            // NDC coordinates
            Vec3f v0 = transformedVs[f.x];
            Vec3f v1 = transformedVs[f.y];
            Vec3f v2 = transformedVs[f.z];
            // Screen coordinates
            Vec2f p0 = Vec2f((v0.x + 1) * width/2, (v0.y + 1) * height/2);
            Vec2f p1 = Vec2f((v1.x + 1) * width/2, (v1.y + 1) * height/2);
            Vec2f p2 = Vec2f((v2.x + 1) * width/2, (v2.y + 1) * height/2);

            sf::Color colorA = sf::Color(mesh.color[f.x].x, mesh.color[f.x].y, mesh.color[f.x].z);
            sf::Color colorB = sf::Color(mesh.color[f.y].x, mesh.color[f.y].y, mesh.color[f.y].z);
            sf::Color colorC = sf::Color(mesh.color[f.z].x, mesh.color[f.z].y, mesh.color[f.z].z);

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