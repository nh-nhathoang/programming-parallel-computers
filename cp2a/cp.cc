/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <cmath>
#include <cstdlib>

typedef double double4_t __attribute__((vector_size(4*sizeof(double))));

static inline double sum4(double4_t v) {
    return v[0] + v[1] + v[2] + v[3];
}

void correlate(int ny, int nx, const float *data, float *result) {

    int nb = (nx + 3) / 4;
    double4_t* X = nullptr;
    int r = posix_memalign((void**)&X, 32, nb * ny * sizeof(double4_t));
    if (r != 0) {
        return;
    }

    for (int y = 0; y < ny; y++) {

        double sum = 0.0;

        for (int x = 0; x < nx; x++) {
            sum += (double) data[x + y*nx];
        }

        double avg = sum/nx;

        double sq = 0.0;

        for (int b = 0; b < nb; b++) {
            double4_t v;
            for (int i = 0; i < 4; i++) {
                int x = b*4 + i;
                v[i] = (x < nx) ? (double) data[x + y*nx] - avg : 0.0;
                sq += v[i] * v[i];
            }
            X[y*nb + b] = v;
        }

        double scale = 1.0 / sqrt(sq);

        for (int b = 0; b < nb; b++) {
            X[y*nb + b] *= scale;
        }
    }

    for (int i = ny - 1; i >= 0; i--) {
        for (int j = 0; j <= i; j++) {
            double4_t sum = {0.0, 0.0, 0.0, 0.0};
            for (int b = 0; b < nb; b++) {
                sum += X[i*nb + b] * X[j*nb + b];
            }
            
            result[i + j*ny] = (float) sum4(sum);
        }
    }
    free(X);
}
