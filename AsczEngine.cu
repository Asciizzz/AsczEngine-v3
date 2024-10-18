#include <FpsHandler.cuh>
#include <CsLogHandler.cuh>
#include <Render3D.cuh>

#include <SFML/Graphics.hpp>

struct Line {
    Vec3f p0, p1, p2;
    Vec3f color0, color1, color2;
    bool in0, in1, in2;
};

struct Point2D {
    Vec3f pos;
    Vec3f color;
    bool isInsideFrustum;
};

sf::Color vec3fToColor(Vec3f v) {
    return sf::Color(v.x, v.y, v.z);
}

__global__ void toPoint2D(
    Point2D *point2D, Camera3D camera,
    Vec3f *pos, Vec3f *color, ULLInt numVs
) {
    ULLInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVs) return;

    Vec4f v4 = pos[i].toVec4f();
    Vec4f t4 = camera.mvp * v4;
    Vec3f t3 = t4.toVec3f();

    // Check if the point is inside the frustum
    point2D[i].pos = t3;
    point2D[i].color = color[i];
    point2D[i].isInsideFrustum = camera.isInsideFrustum(pos[i]);
}

__global__ void toLines(Point2D *point2D, Vec3uli *faces, Line *lines, Vec2f resHalf, ULLInt numFs) {
    ULLInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numFs) return;

    Vec3uli f = faces[i];

    // NDC
    Vec3f v0 = point2D[f.x].pos;
    Vec3f v1 = point2D[f.y].pos;
    Vec3f v2 = point2D[f.z].pos;
    // Screen space
    v0.x = (v0.x + 1) * resHalf.x;
    v0.y = (1 - v0.y) * resHalf.y;
    v1.x = (v1.x + 1) * resHalf.x;
    v1.y = (1 - v1.y) * resHalf.y;
    v2.x = (v2.x + 1) * resHalf.x;
    v2.y = (1 - v2.y) * resHalf.y;
    // Color
    Vec3f c0 = point2D[f.x].color;
    Vec3f c1 = point2D[f.y].color;
    Vec3f c2 = point2D[f.z].color;
    // Inside frustum
    bool in0 = point2D[f.x].isInsideFrustum;
    bool in1 = point2D[f.y].isInsideFrustum;
    bool in2 = point2D[f.z].isInsideFrustum;

    lines[i].p0 = v0; lines[i].p1 = v1; lines[i].p2 = v2;
    lines[i].color0 = c0; lines[i].color1 = c1; lines[i].color2 = c2;
    lines[i].in0 = in0; lines[i].in1 = in1; lines[i].in2 = in2;
}

int main() {
    // Initialize Default stuff
    FpsHandler &FPS = FpsHandler::instance();
    CsLogHandler LOG = CsLogHandler();

    Render3D &RENDER = Render3D::instance();
    RENDER.setResolution(1600, 900);
    RENDER.BUFFER.resize(int(RENDER.RES.x), int(RENDER.RES.y), RENDER.PIXEL_SIZE);

    sf::RenderWindow window(sf::VideoMode(1600, 900), "AsczEngine");
    window.setMouseCursorVisible(false);

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
            float ratioZ = (z - rangeZ.x) / (rangeZ.y - rangeZ.x);
            color.push_back(Vec3f(255 * ratioX, 125, 125 * ratioZ));
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

    Mesh test(0, pos, normal, tex, color, faces);
    RENDER.MESH += Mesh3D(test);

    // Device memory for transformed vertices
    Point2D *d_point2D = new Point2D[RENDER.MESH.numVs];
    cudaMalloc(&d_point2D, RENDER.MESH.numVs * sizeof(Point2D));

    // Device memory for lines
    Line *d_lines = new Line[RENDER.MESH.numFs];
    cudaMalloc(&d_lines, RENDER.MESH.numFs * sizeof(Line));
    // Host memory for lines
    Line *lines = new Line[RENDER.MESH.numFs];

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
                    RENDER.CAMERA.focus = !RENDER.CAMERA.focus;
                    window.setMouseCursorVisible(!RENDER.CAMERA.focus);
                    sf::Mouse::setPosition(sf::Vector2i(
                        RENDER.RES_HALF.x, RENDER.RES_HALF.y
                    ), window);
                }
            }
        }

        if (RENDER.CAMERA.focus) {
            // Mouse movement handling
            sf::Vector2i mousePos = sf::Mouse::getPosition(window);
            sf::Mouse::setPosition(sf::Vector2i(
                RENDER.RES_HALF.x, RENDER.RES_HALF.y
            ), window);

            // Move from center
            int dMx = mousePos.x - RENDER.RES_HALF.x;
            int dMy = mousePos.y - RENDER.RES_HALF.y;

            // Camera look around
            RENDER.CAMERA.rot.x -= dMy * RENDER.CAMERA.mSens * FPS.dTimeSec;
            RENDER.CAMERA.rot.y -= dMx * RENDER.CAMERA.mSens * FPS.dTimeSec;
            RENDER.CAMERA.restrictRot();
            RENDER.CAMERA.updateMVP();

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
            RENDER.CAMERA.pos += RENDER.CAMERA.forward * vel * FPS.dTimeSec;
        }

        // Rotate the mesh
        // float rotY = M_PI_2 / 6 * FPS.dTimeSec;
        // RENDER.MESH.rotate(0, Vec3f(0, 0, 0), Vec3f(0, rotY, 0));

        // Perform 2D projection
        toPoint2D<<<RENDER.MESH.blockNumVs, RENDER.MESH.blockSize>>>(
            d_point2D, RENDER.CAMERA, RENDER.MESH.pos, RENDER.MESH.color, RENDER.MESH.numVs
        );

        // Turn faces into lines for wireframe
        toLines<<<RENDER.MESH.blockNumFs, RENDER.MESH.blockSize>>>(
            d_point2D, RENDER.MESH.faces, d_lines, RENDER.RES_HALF, RENDER.MESH.numFs
        );

        // Copy lines from device to host
        cudaMemcpy(lines, d_lines, RENDER.MESH.numFs * sizeof(Line), cudaMemcpyDeviceToHost);

        window.clear(sf::Color::Black);
        // Draw mesh based on transformed vertices
        for (ULLInt i = 0; i < RENDER.MESH.numFs; i++) {
            Line l = lines[i];
            if (!l.in0 || !l.in1 || !l.in2) continue;

            sf::Color c0 = vec3fToColor(l.color0);
            sf::Color c1 = vec3fToColor(l.color1);
            sf::Color c2 = vec3fToColor(l.color2);

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