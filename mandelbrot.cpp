#include "mandelbrot.h"
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <cmath>

int MAX_THREADS = 4;
#define L1_CACHE_SIZE (32 * 1024)
#define L2_CACHE_SIZE (256 * 1024)
#define L3_CACHE_SIZE (8 * 1024 * 1024)

uint32_t calculate_block_size(int cache_level) {
    int cache_size = (cache_level == 1) ? L1_CACHE_SIZE :
                     (cache_level == 2) ? L2_CACHE_SIZE : L3_CACHE_SIZE;
    int bytes_per_pixel = sizeof(float) * 2; // Assume x, y coordinates
    return static_cast<int>(std::sqrt(cache_size / bytes_per_pixel));
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
void thread_worker(int start_x, int end_x, int start_y, int end_y, int width, int* plot, float dx, float dy) {
    __m128 four = _mm_set1_ps(4.0f);
    __m128 two = _mm_set1_ps(2.0f);
    __m128i one = _mm_set1_epi32(1);

    for (int i = start_y; i < end_y; i++) {
        float y = Y_END - i * dy;
        __m128 y_vec = _mm_set1_ps(y);

        for (int j = start_x; j < end_x; j += 4) {
            alignas(16) float x_vals[4] = {
                X_START + j * dx,
                X_START + (j + 1) * dx,
                X_START + (j + 2) * dx,
                X_START + (j + 3) * dx
            };

            __m128 x = _mm_load_ps(x_vals); // Aligned load
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

            _mm_store_si128(reinterpret_cast<__m128i*>(&plot[i * width + j]), count); // Aligned store
        }
    }
}

void optimized_mandelbrot(int width, int height, int* plot) {
    float dx = (X_END - X_START) / (width - 1);
    float dy = (Y_END - Y_START) / (height - 1);

    int cache_level = 3; // Change this to 1, 2, or 3 for L1, L2, or L3 cache
    uint32_t block_size = calculate_block_size(cache_level);

    std::vector<std::thread> threads;

    int num_blocks = std::max(width, height) / block_size;
    if (num_blocks < MAX_THREADS) num_blocks = MAX_THREADS; // Ensure enough blocks for threads

    for (int t = 0; t < num_blocks; t++) {
        int start_x = (width > height) ? (t * block_size) : 0;
        int end_x = (width > height) ? std::min(static_cast<int>((t + 1) * block_size), width) : width;
        int start_y = (width > height) ? 0 : (t * block_size);
        int end_y = (width > height) ? height : std::min(static_cast<int>((t + 1) * block_size), height);

        if (start_x < end_x && start_y < end_y) {
            threads.emplace_back(thread_worker, start_x, end_x, start_y, end_y,
                width, plot, dx, dy);
        }
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
    optimized_mandelbrot(width, height, plot);
}

#ifdef __cplusplus
}
#endif
