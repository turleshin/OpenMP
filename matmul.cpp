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
void matmul(const int *A, const int *B, int *res)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            for (int d = 0; d < N; d++)
            {
                res[i * K + j] += A[i * N + d] * B[d * K + j];
            }
        }
    }
}

int main()
{
    vector<int> A(M * N, 100);
    vector<int> B(N * K, 50);
    vector<int> res(M * K, 0);
    auto start = chrono::high_resolution_clock::now();
    matmul(A.data(), B.data(), res.data());
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken: " << duration.count() << " milliseconds" << endl;
}