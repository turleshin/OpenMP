#include <iostream>
#include <omp.h>
#include <vector>
#include <random>
#include <chrono>
using namespace std;

void FloyD(int *r, const int *d, const int n) {
    // Copy d to r
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            r[i * n + j] = d[i * n + j];
        }
    }
    // Standard Floyd-Warshall
    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            #pragma omp simd
            for (int j = 0; j < n; j++) {
                int x = r[i * n + k];
                int y = r[k * n + j];
                int z = (x == INT_MAX || y == INT_MAX) ? INT_MAX : x + y;
                r[i * n + j] = min(r[i * n + j], z);
            }
        }
    }
}

int main() {
    int n = 3000;
    vector<int> d(n * n, INT_MAX);
    vector<int> r(n * n, INT_MAX);
    for (int i = 0; i < n; i++) {
        d[i * n + i] = 0;
    }

    #pragma omp parallel
    {
        //unsigned int seed = omp_get_thread_num();
        mt19937 rng(8);
        uniform_int_distribution<int> dist(1, 100);
        #pragma omp for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
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