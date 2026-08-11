// Naive row-wise softmax for HLS experiments (attention weights).
// Prefer a vendor-optimized softmax when moving to production.

#include <cmath>

extern "C" {

void softmax_kernel(const float *scores, float *probs, int rows, int cols) {
#pragma HLS INTERFACE m_axi port = scores offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = probs offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = rows
#pragma HLS INTERFACE s_axilite port = cols
#pragma HLS INTERFACE s_axilite port = return

    for (int r = 0; r < rows; ++r) {
        const float *row = scores + r * cols;
        float *out = probs + r * cols;

        float m = row[0];
        for (int c = 1; c < cols; ++c) {
            if (row[c] > m)
                m = row[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            out[c] = std::exp(row[c] - m);
            sum += out[c];
        }
        const float inv = 1.0f / sum;
        for (int c = 0; c < cols; ++c)
            out[c] *= inv;
    }
}

} // extern "C"
