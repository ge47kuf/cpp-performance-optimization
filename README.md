# C++ Performance Optimization Project

## Overview

This project, was a part of a university exercise, focuses on analyzing and optimizing the performance of C++ code for several tasks. The primary tool used for performance analysis was `perf`. The goal was to apply optimization techniques to achieve speedups compared to baseline implementations.

## Tasks & Optimizations

The project involved optimizing three main components:

### 1. Matrix Multiplication (`matrix.cpp`)

* **Goal:** Accelerate the multiplication of large floating-point matrices.
* **Optimizations Implemented:**
    * **Multithreading:** Utilized a custom `ThreadPool` implemented with `std::thread`, `std::mutex`, and `std::condition_variable` to parallelize the computation across multiple cores. `NUM_THREADS` was set to 4.
    * **AVX Vectorization:** Employed AVX intrinsics (`_mm_loadu_ps`, `_mm_mul_ps`, `_mm_add_ps`, etc.) to perform calculations on multiple floating-point numbers simultaneously within the multiplication loop.
    * **Cache Blocking:** Implemented blocking based on L1 cache size (`L1_CACHE_SIZE`) to improve data locality and reduce cache misses during matrix access.
    * **Matrix Transposition:** Transposed the second matrix (`B`) before multiplication to achieve more contiguous memory access patterns when multiplying rows of `A` with columns of `B`.

### 2. Mandelbrot Set Calculation (`mandelbrot.cpp`)

* **Goal:** Speed up the generation of the Mandelbrot set fractal for a given image resolution (e.g., 16384x16384).
* **Optimizations Implemented:**
    * **AVX2 Vectorization:** Leveraged AVX2 intrinsics (`__m256`, `_mm256_...`) to perform calculations on 8 floating-point numbers (representing complex numbers) in parallel.
    * **Multithreading:** Divided the image rows among multiple threads (`std::thread`, `MAX_THREADS = 4`) to compute different parts of the fractal concurrently.
    * **Cache Blocking:** Calculated block sizes based on cache parameters (L1, L2, L3) to potentially improve data locality.

### 3. Map Application (`map_baseline.cpp`)

* **Goal:** Optimize an application that processes a large number of commands (2,000,000) from the `records.txt` file, involving key-value storage and retrieval (`set`, `value` commands) and string parameter replacements (`params` command).
* **Optimizations Implemented:**
    * **Data Structure Selection:** Replaced the initial `std::map` with `std::unordered_map` to benefit from its average O(1) time complexity for insertions and lookups, compared to `std::map`'s O(log n).
    * **Memory Pre-allocation:** Used `entries.reserve(RESERVED_SIZE)` with a large size (`1000000`) to minimize hash table reallocations and copying during the processing of the numerous `set` commands.
    * **Performance Analysis:** Comments in the code suggest `perf` was used to identify initial bottlenecks (e.g., in `service`, `__memcmp_sse2`) which likely guided the optimization efforts.

## Performance Results

Performance was measured using `perf stat`, comparing the optimized versions against baseline implementations. Key metrics analyzed included CPU cycles, instructions executed, cache misses, and branch misses.

* **Matrix Multiplication:**
    * Achieved **[User: Insert Factor]x** speedup in CPU cycles.
    * Reduced instructions executed by **[User: Insert Factor/Percentage]**.
    * Decreased L1 cache misses by **[User: Insert Factor/Percentage]**.
    * *(Add other relevant metrics like cpu-clock speedup)*
* **Mandelbrot Set Calculation:**
    * Achieved **[User: Insert Factor]x** speedup in CPU cycles.
    * *(Add other relevant metrics like instruction count, branch misses, cpu-clock speedup)*
* **Map Application:**
    * Achieved **[User: Insert Factor/Percentage]** improvement in processing time for `records.txt`.
    * *(Add relevant perf metrics if available)*

*(Note: The specific report files `matrix_basic_stat`, `matrix_optimized_stat`, `mandelbrot_basic_stat`, `mandelbrot_optimized_stat` in the `perf_reports` directory contain the detailed profiling data)*

## Building and Running

This section explains how to compile the project, run the optimized programs, and execute the provided tests.

### Prerequisites

Before you begin, ensure you have the following installed:

* A C++ compiler supporting C++17 and AVX/AVX2 instructions (like a recent version of GCC or Clang).
* The `make` build automation tool.
* Python 3 (for running the test suite).
* The `perf` command-line tool (usually available on Linux) if you want to run detailed performance analysis or if the performance tests require it.
* The pthreads library (`libpthread`, usually installed by default with the compiler).

### Building the Project

### Built Project
```bash
    make all
```
### Running Tests
```bash
    make check
```
### Cleaning Up 
```bash
    make clean
```

### Running the Optimized Programs

* **Matrix Multiplication (`./matrix_profiling [size]`):** Performs optimized multiplication of two square matrices of a given size.
    ```bash
    ./matrix_profiling [size]
    ```
* **Mandelbrot Set Generator (`./mandelbrot_profiling [width] [height]`):** Calculates the Mandelbrot set...
    ```bash
    ./mandelbrot_profiling [width] [height]
    ```
* **Map Application (`./map`):** Processes a sequence of commands...
    ```bash
    ./map
    ```

### Technologies Used

- C++ (C++17)
- Make
- Linux 'perf' Tool
- AVX & AVX2 Intrinsics
- std::thread & Custom Thread Pool
- std::unordered_map
- Python 3 (for testing)

### References

- [Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication)
- [Mandelbrot Set](https://en.wikipedia.org/wiki/Mandelbrot_set)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [Perf Usage](https://perf.wiki.kernel.org/index.php/Tutorial)
