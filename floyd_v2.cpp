#include <iostream>
#include <omp.h>
#include <vector>
#include <random>
#include <chrono>
#include <immintrin.h> // For AVX intrinsics
using namespace std;

// Custom aligned allocator for SIMD
template <typename T>
struct aligned_allocator
{
    using value_type = T;
    T *allocate(size_t n)
    {
        return reinterpret_cast<T *>(_mm_malloc(n * sizeof(T), 32)); // Align to 32 bytes for AVX
    }
    void deallocate(T *p, size_t)
    {
        _mm_free(p);
    }
};

void FloyD(int *r, const int *d, const int n)
{
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            r[i * n + j] = d[i * n + j];
        }
    }
// Floyd-Warshall with AVX2
#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < n; i++)
        {
            __m256i x_vec = _mm256_set1_epi32(r[i * n + k]); // Broadcast x
            __m256i max_vec = _mm256_set1_epi32(INT_MAX / 2);
            int j; // Declare j outside the loop
            for (j = 0; j <= n - 8; j += 8)
            { // Process 8 elements at a time
                __m256i y_vec = _mm256_load_si256((__m256i *)&r[k * n + j]);
                __m256i curr_vec = _mm256_load_si256((__m256i *)&r[i * n + j]);

                // Compute z = x + y, handling INT_MAX
                __m256i x_valid = _mm256_cmpgt_epi32(max_vec, x_vec);
                __m256i y_valid = _mm256_cmpgt_epi32(max_vec, y_vec);
                __m256i valid = _mm256_and_si256(x_valid, y_valid);
                __m256i z_vec = _mm256_add_epi32(x_vec, y_vec);
                z_vec = _mm256_blendv_epi8(_mm256_set1_epi32(INT_MAX), z_vec, valid);

                // Compute min(r[i * n + j], z)
                __m256i min_vec = _mm256_min_epi32(curr_vec, z_vec);
                _mm256_store_si256((__m256i *)&r[i * n + j], min_vec);
            }
            // Handle remaining elements
            for (; j < n; j++)
            {
                int x = r[i * n + k];
                int y = r[k * n + j];
                int z = (x >= INT_MAX / 2 || y >= INT_MAX / 2) ? INT_MAX : x + y;
                r[i * n + j] = min(r[i * n + j], z);
            }
        }
    }
}

int main()
{
    int n = 3000;
    // Use aligned vectors for SIMD
    vector<int, aligned_allocator<int>> d(n * n, INT_MAX);
    vector<int, aligned_allocator<int>> r(n * n, INT_MAX);

    for (int i = 0; i < n; i++)
    {
        d[i * n + i] = 0;
    }
#pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();
        mt19937 rng(seed);
        uniform_int_distribution<int> dist(1, 100);
#pragma omp for collapse(2)
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                int val = dist(rng);
                d[i * n + j] = val;
                d[j * n + i] = val;
            }
        }
    }

    auto start = chrono::high_resolution_clock::now();
    FloyD(r.data(), d.data(), n);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken: " << duration.count() << " milliseconds" << endl;
    return 0;
}