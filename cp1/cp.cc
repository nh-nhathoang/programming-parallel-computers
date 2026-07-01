/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <cmath>
#include <vector>

void correlate(int ny, int nx, const float *data, float *result) {

    std::vector<double> normalize_data(nx * ny);

    for (int y = 0; y < ny; y++) {

        double sum = 0.00;

        for (int x = 0; x < nx; x++) {
            sum += (double) data[x + y*nx];
        }

        double avg = (double) (sum/nx);

        for (int x = 0; x < nx; x++) {
            normalize_data[x + y*nx] = (double) data[x + y*nx] - avg ;
        }

        double sum2 = 0.00;

        for (int x = 0; x < nx; x++) {
            sum2 += (double) normalize_data[x + y*nx] * normalize_data[x + y*nx];
        }

        for (int x = 0; x < nx; x++) {
            normalize_data[x + y*nx] = (double) normalize_data[x + y*nx] / sqrt(sum2);
        }

       
    }

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double p = 0.0;
            for (int x = 0; x < nx; x++) {
                p += (double) normalize_data[x + i*nx] * normalize_data[x + j*nx]; 
            }
            result[i + j*ny] = (float) p;
        }
    }
}
