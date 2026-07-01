/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- the correlation between rows i and j has to be stored in result[i + j*ny]
- only elements with 0 <= j <= i < ny need to be filled
*/
#include <cmath>
#include <cstdlib>
#include <omp.h>

typedef double v8df __attribute__((vector_size(64)));

static inline double hsum(v8df v) {
    return v[0] + v[1] + v[2] + v[3]
         + v[4] + v[5] + v[6] + v[7];
}

void correlate(int ny, int nx, const float *data, float *result)
{
    constexpr int VS = 8;

    int nb = (nx + VS - 1) / VS;
    int nx_pad = nb * VS;

    double *X = nullptr;

    if (posix_memalign((void**)&X, 64,
        (size_t)ny * nx_pad * sizeof(double)) != 0) {
        return;
    }

    // normalize rows
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; y++) {

        const float *src = data + (size_t)y * nx;
        double *dst = X + (size_t)y * nx_pad;

        double sum = 0.0;
        double sq = 0.0;

        for (int x = 0; x < nx; x++) {
            double v = (double)src[x];
            sum += v;
            sq += v * v;
        }

        double mean = sum / (double)nx;
        double norm = sq - sum * mean;

        double scale = 0.0;

        if (norm > 0.0) {
            scale = 1.0 / std::sqrt(norm);
        }

        for (int i = 0; i < nx; i++) {
            dst[i] = ((double)src[i] - mean) * scale;
        }

        for (int i = nx; i < nx_pad; i++) {
            dst[i] = 0.0;
        }
    }

    constexpr int BI = 8;
    constexpr int BJ = 8;

    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < ny; ii += BI) {

        for (int jj = 0; jj <= ii; jj += BJ) {

            double acc[BI][BJ] = {};

            int imax = (ii + BI < ny) ? BI : (ny - ii);
            int jmax = (jj + BJ < ny) ? BJ : (ny - jj);

            for (int k = 0; k < nx_pad; k += VS) {

                v8df a[BI];
                v8df b[BJ];

                for (int i = 0; i < imax; i++) {
                    a[i] = *(v8df*)(X + (size_t)(ii + i) * nx_pad + k);
                }

                for (int j = 0; j < jmax; j++) {
                    b[j] = *(v8df*)(X + (size_t)(jj + j) * nx_pad + k);
                }

                for (int i = 0; i < imax; i++) {
                    for (int j = 0; j < jmax; j++) {
                        acc[i][j] += hsum(a[i] * b[j]);
                    }
                }
            }

            for (int i = 0; i < imax; i++) {
                for (int j = 0; j < jmax; j++) {

                    int r = ii + i;
                    int c = jj + j;

                    if (c <= r) {
                        result[r + (size_t)c * ny]
                            = (float)acc[i][j];
                    }
                }
            }
        }
    }

    free(X);
}