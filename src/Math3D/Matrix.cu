#include <Matrix.cuh>

__host__ __device__ Mat4::Mat4() {
    // Identity matrix
    data[0][0] = 1;
    data[1][1] = 1;
    data[2][2] = 1;
    data[3][3] = 1;
}

__host__ __device__ Mat4::Mat4(float data[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            this->data[i][j] = data[i][j];
        }
    }
}

__host__ __device__ Vec4 Mat4::operator*(const Vec4& vec) {
    Vec4 result;
    result.x = data[0][0] * vec.x + data[0][1] * vec.y + data[0][2] * vec.z + data[0][3] * vec.w;
    result.y = data[1][0] * vec.x + data[1][1] * vec.y + data[1][2] * vec.z + data[1][3] * vec.w;
    result.z = data[2][0] * vec.x + data[2][1] * vec.y + data[2][2] * vec.z + data[2][3] * vec.w;
    result.w = data[3][0] * vec.x + data[3][1] * vec.y + data[3][2] * vec.z + data[3][3] * vec.w;

    return result;
}

__host__ __device__ Mat4 Mat4::operator*(const Mat4& other) {
    Mat4 result;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.data[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                result.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }

    return result;
}

__host__ __device__ Mat4 Mat4::operator*(const float scalar) {
    Mat4 result;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.data[i][j] = data[i][j] * scalar;
        }
    }

    return result;
}

__host__ __device__ Mat4 Mat4::operator+(const Mat4& other) {
    Mat4 result;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }

    return result;
}

__host__ __device__ Mat4 Mat4::operator-(const Mat4& other) {
    Mat4 result;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }

    return result;
}