#include "matrix.h"

#include <immintrin.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
//#include <bits/confname.h>
//#include <time.h>

#define L1_CACHE_SIZE (32 * 1024)
bool track_time = false;
namespace chrono = std::chrono;
chrono::time_point<chrono::system_clock> start;

void naive_matrix_multiply(float *a, float *b, float *c, int n) {
    if (track_time) {
        start = chrono::high_resolution_clock::now();
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i * n + j] = 0.0f;
            for (int k = 0; k < n; k++) {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }

    if (track_time) {
        auto end = chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "optimized end with: " << dur.count() << " ms.\n";
    }
}
// helper function----------------------------------
static uint32_t get_cache_size() {
    #ifdef _SC_LEVEL1_DCACHE_SIZE
        long size = __sysconf(_SC_LEVEL1_DCACHE_SIZE);
        if (size != -1) {
            return static_cast<uint32_t>(size);
        }
    #endif
        return L1_CACHE_SIZE; // slides
}

static void transpose_matrix_blocked(const float *src, float *dst, int n, int block_size) {
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            int i_end = std::min(i + block_size, n);
            int j_end = std::min(j + block_size, n);
            for (int ii = i; ii < i_end; ++ii) {
                for (int jj = j; jj < j_end; ++jj) {
                    dst[jj * n + ii] = src[ii * n + jj];
                }
            }
        }
    }
}

static void blocked_threaded_simd_multiply(const float *a, const float *bT, float *c, int n, int block_size) {
    // Anzahl Threads (max. 4 laut Aufgabenstellung)
    int num_threads = 4;

    auto worker = [&](int thread_id) {
        int rows_per_thread = n / num_threads;
        int start_row = thread_id * rows_per_thread;
        int end_row = (thread_id == num_threads - 1) ? n : start_row + rows_per_thread;

        // Innere Schleifen mit Blocking
        for (int i_block = start_row; i_block < end_row; i_block += block_size) {
            int i_end = std::min(i_block + block_size, end_row);
            for (int j_block = 0; j_block < n; j_block += block_size) {
                int j_end = std::min(j_block + block_size, n);
                for (int k_block = 0; k_block < n; k_block += block_size) {
                    int k_end = std::min(k_block + block_size, n);

                    for (int i = i_block; i < i_end; ++i) {
                        for (int j = j_block; j < j_end; ++j) {
                            // SIMD-optimierte innere Schleife über k
                            __m128 sum_vec = _mm_setzero_ps();
                            int k;
                            // SIMD-Loop (verarbeitet 4 Elemente pro Iteration)
                            for (k = k_block; k + 4 <= k_end; k += 4) {
                                __m128 a_vec = _mm_loadu_ps(&a[i * n + k]);
                                __m128 b_vec = _mm_loadu_ps(&bT[j * n + k]);
                                __m128 mul = _mm_mul_ps(a_vec, b_vec);
                                sum_vec = _mm_add_ps(sum_vec, mul);
                            }

                            // Summe in ein Register holen
                            float temp[4];
                            _mm_storeu_ps(temp, sum_vec);
                            float sum_val = temp[0] + temp[1] + temp[2] + temp[3];

                            // Rest falls n nicht vielfaches von 4
                            for (; k < k_end; ++k) {
                                sum_val += a[i * n + k] * bT[j * n + k];
                            }

                            c[i * n + j] += sum_val;
                        }
                    }
                }
            }
        }
    };

    // Threads starten
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto &th : threads) {
        th.join();
    }
}

void optimized_matrix_mul(float *a, float *b, float *c, int n) {
    if (track_time) {
        auto start = chrono::high_resolution_clock::now();
    }

    std::fill(c, c + n*n, 0.0f);

    size_t cache_size = get_cache_size();

    int block_size = static_cast<int>(std::sqrt(cache_size / 12.0));
    if (block_size < 16) {
        block_size = 32;
    }

    std::vector<float> b_transposed(n * n, 0.0f);
    transpose_matrix_blocked(b, b_transposed.data(), n, block_size);

    blocked_threaded_simd_multiply(a, b_transposed.data(), c, n, block_size);

    if (track_time) {
        auto end = chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "optimized end with: " << dur.count() << " ms.\n";
    }
}

//--------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

void matrix_multiply(float *a, float *b, float *c, int n) {
    //naive_matrix_multiply(a, b, c, n);
    optimized_matrix_mul(a, b, c, n);
}

#ifdef __cplusplus
}
#endif

// int main() {
//     int n = 1024;
//
//
//     return 0;
// }
