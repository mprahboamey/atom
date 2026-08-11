// HLS-oriented score kernel (digital stand-in for hybrid optical scores).
// Binary-phase optical scores == Q @ K^T / sqrt(D). This kernel computes that.
//
// Intended flow: Vitis HLS -> xo -> xclbin (AWS F1/F2 or Alveo).
// Not synthesised in-repo; treat as the functional contract for the FPGA path.
//
// Interface (flattened row-major):
//   q[B*S*D], k[B*S*D], out[B*S*S]

#include <cmath>

extern "C" {

void score_kernel(
    const float *q,
    const float *k,
    float *out,
    int B,
    int S,
    int D
) {
#pragma HLS INTERFACE m_axi port = q offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = k offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = B
#pragma HLS INTERFACE s_axilite port = S
#pragma HLS INTERFACE s_axilite port = D
#pragma HLS INTERFACE s_axilite port = return

    const float scale = std::sqrt(static_cast<float>(D));

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < S; ++j) {
#pragma HLS PIPELINE II = 1
                float acc = 0.0f;
                for (int d = 0; d < D; ++d) {
                    const int q_idx = (b * S + i) * D + d;
                    const int k_idx = (b * S + j) * D + d;
                    acc += q[q_idx] * k[k_idx];
                }
                out[(b * S + i) * S + j] = acc / scale;
            }
        }
    }
}

} // extern "C"
