#include <Utility.cuh>

#include <SFML/Graphics.hpp>

Mesh Utils::readObjFile(std::string path, short fIdxBased, short placement, bool rainbow) {
    std::ifstream file(path);
    if (!file.is_open()) return Mesh();

    std::string line;

    MeshMap objmap;
    VectStr objmapKeys; // To ensure the order of the objects
    std::string curObj;

    VectF wx, wy, wz;
    VectF tu, tv;
    VectF nx, ny, nz;
    VectULLI fw;
    VectLLI ft, fn, fm;

    std::map<std::string, int> mtlMap;
    int matIdx = -1;
    int matSize = 0;
    VectF kar, kag, kab;
    VectF ksr, ksg, ksb;
    VectF kdr, kdg, kdb;
    VectLLI mkd;

    std::map<std::string, int> txMap;
    ULLInt txSize = 0;
    int txCount = 0;
    VectF txr, txg, txb; // Color
    VectI txw, txh; // Size
    VectLLI txof; // Offset

    // We will use these value to shift the mesh to the origin
    float minX = INFINITY, minY = INFINITY, minZ = INFINITY;
    float maxX = -INFINITY, maxY = -INFINITY, maxZ = -INFINITY;

    // Extract the lines from the file
    VectStr lines;
    while (std::getline(file, line)) {
        if (line.size() == 0 || line[0] == '#') continue;
        lines.push_back(line);
    }

    bool off = false; // Mostly for debugging purposes

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < lines.size(); i++) {
        std::stringstream ss(lines[i]);
        std::string type;
        ss >> type;

        if (type == "off") off = true;
        if (type == "on") off = false;
        if (off) continue;

        if (type == "mtllib") {
            std::string mtlPath;
            ss >> mtlPath;

            // Get the relative path of the .mtl file by removing the file name
            std::string mtlDir = path.substr(0, path.find_last_of("/\\") + 1);
            std::ifstream mtlFile(mtlDir + mtlPath);
            if (!mtlFile.is_open()) continue;

            std::string mtlLine;
            while (std::getline(mtlFile, mtlLine)) {
                if (mtlLine.size() == 0 || mtlLine[0] == '#') continue;

                std::stringstream mtlSS(mtlLine);
                std::string mtlType;
                mtlSS >> mtlType;

                if (mtlType == "newmtl") {
                    std::string mtlName;
                    mtlSS >> mtlName;

                    mtlMap[mtlName] = matSize++;

                    kar.push_back(1); kag.push_back(1); kab.push_back(1);
                    kdr.push_back(1); kdg.push_back(1); kdb.push_back(1);
                    ksr.push_back(1); ksg.push_back(1); ksb.push_back(1);
                    mkd.push_back(-1);
                }

                if (mtlType == "Ka") {
                    float r, g, b; mtlSS >> r >> g >> b;
                    kar.back() = r; kag.back() = g; kab.back() = b;
                } else if (mtlType == "Kd") {
                    float r, g, b; mtlSS >> r >> g >> b;
                    kdr.back() = r; kdg.back() = g; kdb.back() = b;
                } else if (mtlType == "Ks") {
                    float r, g, b; mtlSS >> r >> g >> b;
                    ksr.back() = r; ksg.back() = g; ksb.back() = b;
                } else if (mtlType == "map_Kd") {
                    std::string txPath; mtlSS >> txPath;

                    // If already loaded, assign the index and continue
                    if (txMap.find(txPath) != txMap.end()) {
                        mkd.back() = txMap[txPath];
                        continue;
                    } else {
                    // If not, add it to the map
                        txMap[txPath] = txCount;
                    }

                    // Get the texture data using SFML
                    sf::Image txImage;
                    if (!txImage.loadFromFile(mtlDir + txPath)) continue;

                    mkd.back() = txCount++;

                    int tw = txImage.getSize().x;
                    int th = txImage.getSize().y;

                    txw.push_back(tw); txh.push_back(th);
                    txof.push_back(txSize);

                    txSize += tw * th;

                    for (int y = 0; y < th; y++) {
                        for (int x = 0; x < tw; x++) {
                            int ix = x;
                            int iy = th - y - 1; // Flip the y-axis
                            sf::Color color = txImage.getPixel(ix, iy);

                            txr.push_back(float(color.r));
                            txg.push_back(float(color.g));
                            txb.push_back(float(color.b));
                        }
                    }
                }
            }
        }

        if (type == "o") {
            std::string name;
            ss >> name;

            // Append the key
            objmapKeys.push_back(name);

            if (curObj == "") {
                curObj = name;

                objmap[curObj].w1 = 0;
                objmap[curObj].t1 = 0;
                objmap[curObj].n1 = 0;
            }

            if (curObj != name) {
                objmap[curObj].w2 = wx.size();
                objmap[curObj].t2 = tu.size();
                objmap[curObj].n2 = nx.size();

                curObj = name;

                objmap[curObj].w1 = wx.size();
                objmap[curObj].t1 = tu.size();
                objmap[curObj].n1 = nx.size();
            }
        }

        if (type == "usemtl") {
            std::string mtlName;
            ss >> mtlName;

            // If not in the map, set it to -1
            if (mtlMap.find(mtlName) == mtlMap.end()) {
                matIdx = -1;
            } else {
            // Get the idx of the material based on the mtlMap
                matIdx = mtlMap[mtlName];
            }
        }

        if (type == "v") {
            Vec3f v;
            ss >> v.x >> v.y >> v.z;

            // Update the min and max values
            minX = std::min(minX, v.x);
            minY = std::min(minY, v.y);
            minZ = std::min(minZ, v.z);
            maxX = std::max(maxX, v.x);
            maxY = std::max(maxY, v.y);
            maxZ = std::max(maxZ, v.z);

            wx.push_back(v.x);
            wy.push_back(v.y);
            wz.push_back(v.z);
        } else if (type == "vt") {
            Vec2f t;
            ss >> t.x >> t.y;
            tu.push_back(t.x);
            tv.push_back(t.y);
        } else if (type == "vn") {
            Vec3f n;
            ss >> n.x >> n.y >> n.z;
            float mag = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
            n.norm();

            nx.push_back(n.x);
            ny.push_back(n.y);
            nz.push_back(n.z);
        } else if (type == "f") {
            VectULLI vs;
            VectLLI ts, ns;
            while (ss.good()) {
                std::string vtn;
                ss >> vtn;

                ULLInt v;
                LLInt t = 0, n = 0;
                std::stringstream ss2(vtn);

                // Read vertex index
                ss2 >> v;

                // Check for texture index (skip if missing)
                if (ss2.peek() == '/') {
                    ss2.ignore(1); // Ignore the first '/'
                    if (ss2.peek() != '/') {
                        ss2 >> t; // Read texture index if present
                    } else {
                        t = fIdxBased - 1; // No texture index provided
                    }
                } else {
                    t = fIdxBased - 1; // No slashes, so no texture coordinate
                }

                // Check for normal index
                if (ss2.peek() == '/') {
                    ss2.ignore(1); // Ignore the second '/'
                    ss2 >> n; // Read normal index
                } else {
                    n = fIdxBased - 1 ; // No normal index provided
                }

                vs.push_back(v - fIdxBased); // Adjust to 0-based index
                ts.push_back(t - fIdxBased); // Adjust to 0-based index
                ns.push_back(n - fIdxBased); // Adjust to 0-based index
            }

            /* For n points, we will construct n - 2 triangles
            
            Example: (A B C D E F):
            - Use A as anchor point
                => (A B C), (A C D), (A D E), (A E F)

            Note:  .obj files are assumed to organized the points
                    in a clockwise (or counter-clockwise) order
                    If they don't, well, sucks to be you
            */

            for (int i = 1; i < vs.size() - 1; i++) {
                fw.push_back(vs[0]); fw.push_back(vs[i]); fw.push_back(vs[i + 1]);
                ft.push_back(ts[0]); ft.push_back(ts[i]); ft.push_back(ts[i + 1]);
                fn.push_back(ns[0]); fn.push_back(ns[i]); fn.push_back(ns[i + 1]);
                fm.push_back(matIdx); fm.push_back(matIdx); fm.push_back(matIdx);
            }
        }
    }
    // If there's no object, set default "obj"
    if (objmap.size() == 0) {
        objmap["obj"].w1 = 0; objmap["obj"].w2 = wx.size();
        objmap["obj"].t1 = 0; objmap["obj"].t2 = tu.size();
        objmap["obj"].n1 = 0; objmap["obj"].n2 = nx.size();
    } else {
        // Set the end of the last object
        objmap[curObj].w2 = wx.size();
        objmap[curObj].t2 = tu.size();
        objmap[curObj].n2 = nx.size();
    }

    #pragma omp parallel for
    for (size_t i = 0; i < wx.size(); i++) {
        // Shift to center of xz plane
        if (placement > 0) {
            wx[i] -= (minX + maxX) / 2;
            wz[i] -= (minZ + maxZ) / 2;
        }

        if (placement == 1) { // Shift to center
            wy[i] -= (minY + maxY) / 2;
        } else if (placement == 2) { // Shift to floor
            wy[i] -= minY;
        }
    }

    Mesh mesh = {
        wx, wy, wz,
        tu, tv,
        nx, ny, nz,
        fw, ft, fn, fm,
        kar, kag, kab,
        kdr, kdg, kdb,
        ksr, ksg, ksb,
        mkd,
        txr, txg, txb,
        txw, txh, txof,
        objmap, objmapKeys
    };

    return mesh;
}

void Utils::applyTransformation(std::vector<Mesh> objs) {
    // Read an transform.txt file and apply it
    /* Format:
        <idx> t <x> <y> <z>
        <idx> r <ox> <oy> <oz> <x> <y> <z>
        <idx> s <ox> <oy> <oz> <x> <y> <z>
    */
    std::ifstream file("assets/cfg/transform.txt");
    if (!file.is_open()) return;

    std::string line;
    while (std::getline(file, line)) {
        if (line.size() == 0 || line[0] == '#') continue;

        std::stringstream ss(line);
        ULLInt idx; std::string type;
        ss >> idx >> type;

        if (idx >= objs.size()) continue;

        // if (type == "t") {
        //     Vec3f t; ss >> t.x >> t.y >> t.z;

        //     objs[idx].translateRuntime(t);
        // } else if (type == "rx") {
        //     Vec3f origin; ss >> origin.x >> origin.y >> origin.z;
        //     float r; ss >> r;
        //     r = r * M_PI / 180;

        //     objs[idx].rotateRuntime(origin, r, 0);
        // } else if (type == "ry") {
        //     Vec3f origin; ss >> origin.x >> origin.y >> origin.z;
        //     float r; ss >> r;
        //     r = r * M_PI / 180;

        //     objs[idx].rotateRuntime(origin, r, 1);
        // } else if (type == "rz") {
        //     Vec3f origin; ss >> origin.x >> origin.y >> origin.z;
        //     float r; ss >> r;
        //     r = r * M_PI / 180;

        //     objs[idx].rotateRuntime(origin, r, 2);
        // } else if (type == "s") {
        //     Vec3f origin; ss >> origin.x >> origin.y >> origin.z;
        //     Vec3f scl; ss >> scl.x >> scl.y >> scl.z;

        //     objs[idx].scaleRuntime(origin, scl);
        // }
    }
}