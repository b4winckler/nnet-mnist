#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>


uint8_t *idx_ubyte_data(const char *path)
{
    int fd = open(path, O_RDONLY);
    struct stat s;
    fstat(fd, &s);
    uint8_t *p = (uint8_t *)mmap(0, s.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    return p + 4 + 4 * p[3];
}

void convert_input(double *dst, uint8_t *src, int length)
{
    for (int i = 0; i < length; ++i)
        dst[i] = src[i] / 255.0;    // normalize input to lie in [0,1]
}

void feed_forward(double w[2][100][785], double x[3][785], int dims[3])
{
    for (int l = 1; l < 3; ++l)
        for (int j = 0; j < dims[l]; ++j) {
            double acc = 0;
            for (int i = 0; i <= dims[l - 1]; ++i)
                acc += w[l - 1][j][i] * x[l - 1][i];
            x[l][j] = 1.0 / (1.0 + exp(-acc));
        }
}

int main()
{
    srand(8310);

    // load test and train data
    uint8_t *x0 = idx_ubyte_data("mnist/train-images-idx3-ubyte");
    uint8_t *y0 = idx_ubyte_data("mnist/train-labels-idx1-ubyte");
    uint8_t *x1 = idx_ubyte_data("mnist/t10k-images-idx3-ubyte");
    uint8_t *y1 = idx_ubyte_data("mnist/t10k-labels-idx1-ubyte");

    // draw initial network weights from uniform distribution on [-1,1]
    double w[2][100][785];
    int dims[3] = { 784, 100, 10 };
    for (int l = 0; l < 2; ++l)
        for (int j = 0; j < dims[l + 1]; ++j)
            for (int i = 0; i < dims[l]; ++i)
                w[l][j][i] = 1.0 - 2.0 * rand() / (double) RAND_MAX;

    // cached data
    double x[3][785];
    double d[2][100];
    double g[2][100][785];
    x[0][dims[0]] = x[1][dims[1]] = 1.0;    // convention: input to bias is 1.0

    // init training indices
    int perm[60000];
    for (int i = 0; i < 60000; ++i)
        perm[i] = i;

    // train network
    for (int epoch = 0; epoch < 4; ++epoch) {
        // shuffle training indices
        for (int i = 0; i < 60000; ++i) {
            int swap = perm[i];
            int n = rand() % 60000;
            perm[i] = perm[n];
            perm[n] = swap;
        }

        for (int n = 0; n < 60000; ) {
            // clear gradient
            for (int l = 0; l < 2; ++l)
                for (int j = 0; j < dims[l + 1]; ++j)
                    for (int i = 0; i <= dims[l]; ++i)
                        g[l][j][i] = 0;

            for (int batch = 0; batch < 10; ++batch, ++n) {
                convert_input(x[0], x0 + perm[n] * dims[0], dims[0]);
                feed_forward(w, x, dims);

                // back propagate
                for (int j = 0; j < dims[2]; ++j)
                    d[1][j] = (x[2][j] - (y0[perm[n]] == j)) * x[2][j]
                            * (1.0 - x[2][j]);
                for (int i = 0; i < dims[1]; ++i) {
                    double acc = 0;
                    for (int j = 0; j < dims[2]; ++j)
                        acc += d[1][j] * w[1][j][i];
                    d[0][i] = acc * x[1][i] * (1.0 - x[1][i]);
                }

                // accumulate gradient
                for (int l = 1; l < 3; ++l)
                    for (int j = 0; j < dims[l]; ++j)
                        for (int i = 0; i <= dims[l - 1]; ++i)
                            g[l - 1][j][i] += d[l - 1][j] * x[l - 1][i];
            }

            // take gradient step
            for (int l = 1; l < 3; ++l)
                for (int j = 0; j < dims[l]; ++j)
                    for (int i = 0; i <= dims[l - 1]; ++i)
                        w[l - 1][j][i] -= 3.0 * g[l - 1][j][i] / 10;
        }
    }

    // test network by counting number of matching predictions
    int nmatch = 0;
    for (int n = 0; n < 10000; ++n) {
        convert_input(x[0], x1 + n * dims[0], dims[0]);
        feed_forward(w, x, dims);

        int match = 1;
        for (int i = 0; match && i < dims[2]; ++i)
            match = match && (x[2][y1[n]] >= x[2][i]);
        nmatch += match;
    }
    printf("%d correct predictions out of 10000\n", nmatch);

    return 0;
}
