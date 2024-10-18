#include <FpsHandler.cuh>
#include <CsLogHandler.cuh>
#include <Render3D.cuh>

#include <SFML/Graphics.hpp>

struct Line {
    Vec3f p0, p1, p2;
    Vec4f color0, color1, color2;
    bool in0, in1, in2;
};

sf::Color vec4fToColor(Vec4f v) {
    return sf::Color(v.x, v.y, v.z, v.w);
}

__global__ void toLines(Vec4f *projection, Vec4f *color, Vec3uli *faces, Line *lines, ULLInt numFs) {
    ULLInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numFs) return;

    Vec3uli f = faces[i];

    // Screen pos
    Vec4f v0 = projection[f.x];
    Vec4f v1 = projection[f.y];
    Vec4f v2 = projection[f.z];
    // Color
    Vec4f c0 = color[f.x];
    Vec4f c1 = color[f.y];
    Vec4f c2 = color[f.z];
    // Inside frustum
    bool in0 = v0.w > 0;
    bool in1 = v1.w > 0;
    bool in2 = v2.w > 0;

    Vec3f p0 = Vec3f(v0.x, v0.y, v0.z);
    Vec3f p1 = Vec3f(v1.x, v1.y, v1.z);
    Vec3f p2 = Vec3f(v2.x, v2.y, v2.z);

    lines[i].p0 = p0; lines[i].p1 = p1; lines[i].p2 = p2;
    lines[i].color0 = c0; lines[i].color1 = c1; lines[i].color2 = c2;
    lines[i].in0 = in0; lines[i].in1 = in1; lines[i].in2 = in2;
}

int main() {
    // Initialize Default stuff
    FpsHandler &FPS = FpsHandler::instance();
    CsLogHandler LOG = CsLogHandler();

    Render3D &RENDER = Render3D::instance();
    RENDER.setResolution(1600, 900);

    sf::RenderWindow window(sf::VideoMode(1600, 900), "AsczEngine");
    window.setMouseCursorVisible(false);

    // Graphing calculator for y = f(x, z)
    Vecs3f world;
    Vecs3f normal;
    Vecs2f texture;
    Vecs4f color;

    Vecs3uli faces;

    // Append points to the grid
    Vec2f rangeX(-100, 100);
    Vec2f rangeZ(-100, 100);
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
            normal.push_back(Vec3f(0, 1, 0));

            // x and z ratio (0 - 1)
            float ratioX = (x - rangeX.x) / (rangeX.y - rangeX.x);
            float ratioZ = (z - rangeZ.x) / (rangeZ.y - rangeZ.x);

            // Texture
            texture.push_back(Vec2f(ratioX, ratioZ));

            // Cool color
            color.push_back(Vec4f(255 * ratioX, 125, 125 * ratioZ, 255));
        }
    }

    for (ULLInt i = 0; i < world.size(); i++) {
        // Set opacity based on the height
        float ratioY = (world[i].y - minY) / (maxY - minY);
        color[i].w = 100 + 150 * ratioY;
    }

    // Append faces to the grid
    for (ULLInt x = 0; x < sizeX - 1; x++) {
        for (ULLInt z = 0; z < sizeZ - 1; z++) {
            ULLInt i = x * sizeZ + z;
            faces.push_back(Vec3uli(i, i + 1, i + sizeZ));
            faces.push_back(Vec3uli(i + 1, i + sizeZ + 1, i + sizeZ));
        }
    }

    Mesh test(0, world, normal, texture, color, faces);
    RENDER.mesh += Mesh3D(test);
    RENDER.allocateProjection();

    // Device memory for lines
    Line *d_lines = new Line[RENDER.mesh.numFs];
    cudaMalloc(&d_lines, RENDER.mesh.numFs * sizeof(Line));
    // Host memory for lines
    Line *lines = new Line[RENDER.mesh.numFs];

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

        // Rotate the mesh
        // float rotY = M_PI_2 / 6 * FPS.dTimeSec;
        // RENDER.mesh.rotate(0, Vec3f(0, 0, 0), Vec3f(0, rotY, 0));

        // Clear buffer
        RENDER.buffer.clearBuffer();
        RENDER.vertexProjection();

        // Turn faces into lines for wireframe
        toLines<<<RENDER.mesh.blockNumFs, RENDER.mesh.blockSize>>>(
            RENDER.projection, RENDER.mesh.color, RENDER.mesh.faces, d_lines, RENDER.mesh.numFs
        );

        // Copy lines from device to host
        cudaMemcpy(lines, d_lines, RENDER.mesh.numFs * sizeof(Line), cudaMemcpyDeviceToHost);

        window.clear(sf::Color::Black);
        // Draw mesh based on transformed vertices
        for (ULLInt i = 0; i < RENDER.mesh.numFs; i++) {
            Line l = lines[i];
            if (!l.in0 || !l.in1 || !l.in2) continue;

            sf::Color c0 = vec4fToColor(l.color0);
            sf::Color c1 = vec4fToColor(l.color1);
            sf::Color c2 = vec4fToColor(l.color2);

            sf::Vertex v1(sf::Vector2f(l.p0.x, l.p0.y), c0);
            sf::Vertex v2(sf::Vector2f(l.p1.x, l.p1.y), c1);
            sf::Vertex v3(sf::Vector2f(l.p2.x, l.p2.y), c2);

            sf::Vertex line[] = {v1, v2, v3, v1};
            window.draw(line, 4, sf::LineStrip);
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