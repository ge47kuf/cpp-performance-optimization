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
#include <mutex>
#include <queue>
#include <condition_variable>

#define L1_CACHE_SIZE (32 * 1024) // time: 0.8
#define L3_CACHE_SIZE (8 * 1024 * 1024) // time: 1.3

namespace chrono = std::chrono;

// naive ansatz-------------------------------
void naive_matrix_multiply(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i * n + j] = 0.0f;
            for (int k = 0; k < n; k++) {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}

// optimized ansatz---------------------------
class ThreadPool {
public:
    ThreadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template <class F>
    void enqueue(F&& task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(task));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
            worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;
};

static uint32_t get_cache_size() {
    #ifdef _SC_LEVEL1_CACHE_SIZE
        long size = __sysconf(_SC_LEVEL1_CACHE_SIZE);
        if (size != -1) {
            return static_cast<uint32_t>(size);
        }
    #endif
        return L1_CACHE_SIZE;
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
    ThreadPool thread_pool(4);

    for (int i_block = 0; i_block < n; i_block += block_size) {
        for (int j_block = 0; j_block < n; j_block += block_size) {
            thread_pool.enqueue([=] {
                for (int k_block = 0; k_block < n; k_block += block_size) {
                    int i_end = std::min(i_block + block_size, n);
                    int j_end = std::min(j_block + block_size, n);
                    int k_end = std::min(k_block + block_size, n);

                    for (int i = i_block; i < i_end; ++i) {
                        for (int j = j_block; j < j_end; ++j) {
                            __m128 sum_vec = _mm_setzero_ps();
                            int k;
                            for (k = k_block; k + 4 <= k_end; k += 4) {
                                __m128 a_vec = _mm_load_ps(&a[i * n + k]);
                                __m128 b_vec = _mm_load_ps(&bT[j * n + k]);
                                __m128 mul = _mm_mul_ps(a_vec, b_vec);
                                sum_vec = _mm_add_ps(sum_vec, mul);
                            }

                            float temp[4];
                            _mm_store_ps(temp, sum_vec);
                            float sum_val = temp[0] + temp[1] + temp[2] + temp[3];

                            for (; k < k_end; ++k) {
                                sum_val += a[i * n + k] * bT[j * n + k];
                            }

                            c[i * n + j] += sum_val;
                        }
                    }
                }
            });
        }
    }
}

void optimized_matrix_mul(float *a, float *b, float *c, int n) {
    std::fill(c, c + n*n, 0.0f);

    size_t cache_size = get_cache_size();

    int block_size = static_cast<int>(std::sqrt(cache_size / 12.0));
    if (block_size < 16) {
        block_size = 32;
    }

    std::vector<float> b_transposed(n * n, 0.0f);
    transpose_matrix_blocked(b, b_transposed.data(), n, block_size);

    blocked_threaded_simd_multiply(a, b_transposed.data(), c, n, block_size);
}

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
//     const int n = 512;
//     std::vector<float> a(n * n), b(n * n), c_naive(n * n), c_optimized(n * n);
//
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<float> dist(0.0f, 1.0f);
//
//     for (int i = 0; i < n * n; ++i) {
//         a[i] = dist(gen);
//         b[i] = dist(gen);
//     }
//
//     auto start_naive = chrono::high_resolution_clock::now();
//     naive_matrix_multiply(a.data(), b.data(), c_naive.data(), n);
//     auto end_naive = chrono::high_resolution_clock::now();
//     auto duration_naive = chrono::duration_cast<chrono::milliseconds>(end_naive - start_naive).count();
//     std::cout << "Naive matrix multiplication took " << duration_naive << " ms\n";
//
//     auto start_optimized = chrono::high_resolution_clock::now();
//     optimized_matrix_mul(a.data(), b.data(), c_optimized.data(), n);
//     auto end_optimized = chrono::high_resolution_clock::now();
//     auto duration_optimized = chrono::duration_cast<chrono::milliseconds>(end_optimized - start_optimized).count();
//     std::cout << "Optimized matrix multiplication took " << duration_optimized << " ms\n";
//
//     for (int i = 0; i < n * n; ++i) {
//         if (std::abs(c_naive[i] - c_optimized[i]) > 1e-2f) {
//             std::cerr << "Mismatch at index " << i << ": naive=" << c_naive[i] << ", optimized=" << c_optimized[i] << "\n";
//             return 1;
//         }
//     }
//
//     std::cout << "Results match!\n";
//     return 0;
// }
