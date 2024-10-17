#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <Vector.cuh>

struct Mat4 {
    float data[4][4] = {0};

    Vec4 multiplyVec4(const Vec4& v) {
        Vec4 result = {
            data[0][0] * v.x + data[0][1] * v.y + data[0][2] * v.z + data[0][3] * v.w,
            data[1][0] * v.x + data[1][1] * v.y + data[1][2] * v.z + data[1][3] * v.w,
            data[2][0] * v.x + data[2][1] * v.y + data[2][2] * v.z + data[2][3] * v.w,
            data[3][0] * v.x + data[3][1] * v.y + data[3][2] * v.z + data[3][3] * v.w
        };

        return result;
    }

    Mat4 operator*(const Mat4& other) {
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

    Mat4 operator*(const float scalar) {
        Mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }

        return result;
    }

    Mat4 operator+(const Mat4& other) {
        Mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }

        return result;
    }

    Mat4 operator-(const Mat4& other) {
        Mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }

        return result;
    }
};

#endif