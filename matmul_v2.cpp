#include <iostream>
#include <omp.h>
#include <vector>
#include <random>
#include <chrono>
#include <immintrin.h> // For AVX intrinsics

using namespace std;
constexpr int M = 2000;
constexpr int N = 1000;
constexpr int K = 3000;
template <typename T>
struct aligned_allocator{
    using value_type = T;
    T* allocate(size_t n)
    {
        return reinterpret_cast<T*>(_mm_malloc(n * sizeof(T), 32)); // Align to 32 bytes for AVX
    }
    void deallocate(T* p, size_t)
    {
        _mm_free(p);
    }
};

// Res[i][j] = sum(A[i][d] * B[d][j])
void matmul(const int *A, const int *B, int *res)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            __m256i sum_vec = _mm256_setzero_si256(); // Initialize sum vector
            int d;
            for (d = 0; d < N; d+=8)
            {
                __m256i a_vec = _mm256_set1_epi32(A[i * N + d]); // broadcast A[i][d]
                __m256i b_vec = _mm256_load_si256((__m256i*)&B[d * K + j]); // Load B[d][j]
                sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(a_vec, b_vec)); // Multiply and accumulate Res[i][j] = sum(A[i][d] * B[d][j]);
            }
             _mm256_store_si256((__m256i*)&res[i * K + j], sum_vec); // store the result
            for(; d < N ; d++)
            {
                res[i * K + j] += A[i * N + d] * B[d * K + j];
            }
        }
    }
}

int main()
{
    vector<int,aligned_allocator<int>> A(M * N, 100);
    vector<int,aligned_allocator<int>>B(N * K, 50);
    vector<int,aligned_allocator<int>> res(M * K, 0);

    auto start = chrono::high_resolution_clock::now();
    matmul(A.data(), B.data(), res.data());
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken: " << duration.count() << " milliseconds" << endl;
}