#include "mandelbrot.h"
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int MAX_THREADS = 4;
#define L1_CACHE_SIZE (32 * 1024) // time: 2.46
#define L2_CACHE_SIZE (256 * 1024) // time: 2.51
#define L3_CACHE_SIZE (8 * 1024 * 1024) // time: 2.46

uint32_t calculate_block_size() {
    #ifdef _SC_LEVEL1_CACHE_SIZE
        long size = __sysconf(_SC_LEVEL1_CACHE_SIZE);
        if (size != -1) {
            return static_cast<uint32_t>(size);
        }
    #endif
        return L1_CACHE_SIZE;
}

// naive berechnung-----------------------------------------
int mandelbrot_calc_base(float x, float y) {
    auto re = x;
    auto im = y;

    for (auto i = 0; i < LOOP; i++) {
        float re2 = re * re;
        float im2 = im * im;

        if (re2 + im2 > 4.0f) return i;

        im = 2.0f * re * im + y;
        re = re2 - im2 + x;
    }

    return LOOP;
}

void naive_mandelbrot(int width, int height, int* plot) {
    float dx = (X_END - X_START) / (width - 1);
    float dy = (Y_END - Y_START) / (height - 1);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float x = X_START + j * dx;
            float y = Y_END - i * dy;

            auto result = mandelbrot_calc_base(x, y);
            plot[i * width + j] = result;
        }
    }
}
// optimized berechnung-------------------------------------
void thread_worker(int start_row, int end_row, int width, int height, int* plot, float dx, float dy, int block_size) {
    __m128 four = _mm_set1_ps(4.0f);
    __m128 two = _mm_set1_ps(2.0f);
    __m128i one = _mm_set1_epi32(1);
    for (int i_block = start_row; i_block < end_row; i_block += block_size) {
        int i_end = std::min(i_block + block_size, end_row);
        for (int j_block = 0; j_block < width; j_block += block_size) {
            int j_end = std::min(j_block + block_size, width);
            for (int i = i_block; i < i_end; i++) {
                float y = Y_END - i * dy;
                __m128 y_vec = _mm_set1_ps(y);
                int j = j_block;
                for (; j <= j_end - 4; j += 4) {
                    float x0 = X_START + j * dx;
                    float x1 = X_START + (j + 1) * dx;
                    float x2 = X_START + (j + 2) * dx;
                    float x3 = X_START + (j + 3) * dx;
                    __m128 x = _mm_set_ps(x3, x2, x1, x0);
                    __m128 re = x;
                    __m128 im = y_vec;
                    __m128i count = _mm_setzero_si128();
                    for (int k = 0; k < LOOP; k++) {
                        __m128 re2 = _mm_mul_ps(re, re);
                        __m128 im2 = _mm_mul_ps(im, im);
                        __m128 mag2 = _mm_add_ps(re2, im2);
                        __m128 cmp = _mm_cmple_ps(mag2, four);
                        int mask = _mm_movemask_ps(cmp);
                        if (mask == 0) break;
                        __m128i add = _mm_and_si128(_mm_castps_si128(cmp), one);
                        count = _mm_add_epi32(count, add);
                        __m128 re_im = _mm_mul_ps(re, im);
                        __m128 mul2 = _mm_mul_ps(two, re_im);
                        im = _mm_add_ps(mul2, y_vec);
                        __m128 diff = _mm_sub_ps(re2, im2);
                        re = _mm_add_ps(diff, x);
                    }
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&plot[i * width + j]), count);
                }
                for (; j < j_end; j++) {
                    float x = X_START + j * dx;
                    plot[i * width + j] = mandelbrot_calc_base(x, y);
                }
            }
        }
    }
}
void optimized_mandelbrot(int width, int height, int* plot) {
    float dx = (X_END - X_START) / (width - 1);
    float dy = (Y_END - Y_START) / (height - 1);
    uint32_t block_size = calculate_block_size();
    std::vector<std::thread> threads;
    int blocks = height / MAX_THREADS;
    for (int t = 0; t < MAX_THREADS; t++) {
        int start_row = t * blocks;
        int end_row = (t == MAX_THREADS - 1) ? height : start_row + blocks;
        threads.emplace_back(thread_worker, start_row, end_row, width, height, plot, dx, dy, block_size);
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

// -----------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

void mandelbrot(int width, int height, int* plot) {
    //naive_mandelbrot(width, height, plot);
    optimized_mandelbrot(width, height, plot);
}

#ifdef __cplusplus
}
#endif

// void print_arrays(int* a, int size_a, int* b) {
//     for (int i = 0; i < size_a; i++) {
//         if (a[i] != b[i]) {
//             std::cout << i << ": " << "naive: " << a[i] << " | optimized: " << b[i] << "\n";
//             break;
//         }
//     }
// }
//
// int main() {
//     int width = 1920;
//     int height = 1080;
//     int size = width * height;
//     int* plot_naive = new int[size];
//     int* plot_optimized = new int[size];
//
//     auto start_naive = std::chrono::high_resolution_clock::now();
//     naive_mandelbrot(width, height, plot_naive);
//     auto end_naive = std::chrono::high_resolution_clock::now();
//     auto duration_naive = std::chrono::duration_cast<std::chrono::milliseconds>(end_naive - start_naive).count();
//     std::cout << "Naive Mandelbrot duration: " << duration_naive << " ms\n";
//
//     auto start_optimized = std::chrono::high_resolution_clock::now();
//     optimized_mandelbrot(width, height, plot_optimized);
//     auto end_optimized = std::chrono::high_resolution_clock::now();
//     auto duration_optimized = std::chrono::duration_cast<std::chrono::milliseconds>(end_optimized - start_optimized).count();
//     std::cout << "Optimized Mandelbrot duration: " << duration_optimized << " ms\n";
//
//     print_arrays(plot_naive, size, plot_optimized);
//
//     delete[] plot_naive;
//     delete[] plot_optimized;
//
//     return 0;
// }
