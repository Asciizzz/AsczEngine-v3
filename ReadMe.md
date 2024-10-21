# AsczEngine v3

### Overview

- A custom 3D engine made entirely from scratch, without the help of any externel dedicated graphic libraries.
- The only things used are:
  - SFML for the window display
  - CUDA for parallel kernels

### Progress:

- Created `Matrix` and `Vector` as well as their operations.
- Created dynamic `Camera3D` with fps-like movement and `mat4 MVP`(Model-View-Projection) transformation matrix for converting vertex from 3D world $\rightarrow$ 3D view $\rightarrow$ 2D screen space.
- Created `Mesh3D` system with `Vertices` and `Faces` data
  - *`Vertices`*: structured using a SoA-AoS hybrid approach: Vertex with 8 attributes `x`, `y`, `z`, `nx`, `ny`, `nz`, `u`, `v` are grouped into 3 groups: `position`, `normal`, `texture`.
  - *`Faces`*: data type `Vec3x3ulli` (`Vec3x3 unsigned long int`), acting as faces (or triangles) with 3 vertices. Each `face` will have 3 `Vec3ulli`:
    - `v`: 3 indices of the world coordinate
    - `t`: 3 indices of the texture map
    - `n`: 3 indices of the normal
  - *`operator+=`* combines datas of 2 (or more) meshes to itself, with indices offset to make sure faces access the correct vertices.
- Created CUDA kernels for graphic pipeline:
  - *Vertices Projection*: uses `Camera3D`'s `mat4 MVP` to project `vertices` world coord into screen coord
  - *Frustum Clipping*: WIP!
  - *Rasterization*: uses the projected `faces` to interpolate and fill in `buffers`: `color`, `depth`, `world`, `normal`, `texture`, etc.
  - *Phong Shading*: uses the `buffers`' datas to apply Phong Shading to each pixel.

### Future Addition:

- Shadow Map
- Physic Engine (very ambitious)

### Personal Goal

- I want to get a bit personal here, this 3D engine is a project aimed at creating an abstract, liminal-space-like experience. I've always been drawn to those eerie, empty, and strangely nostalgic places. More than just experiencing them, I’ve always wanted to create such spaces. It’s ironic, isn’t it? A low-level project, built with minimal abstraction, yet aimed at evoking a sense of maximum abstraction and surrealism.