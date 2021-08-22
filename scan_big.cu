// O(N) operations

#include <stdio.h>
#include <iostream>
using namespace std;

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(1);                                               \
        }                                                          \
    }

const int BS = 1 << 5;
const int N = 1 << 10;

__global__ void up_sweep(float *g_idata, const int chunk)
{
    int threadId = threadIdx.x;
    int entireId = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = chunk;

    for (int d = blockDim.x >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (threadId < d)
        {
            // B
            int ai = offset * (2 * (entireId / blockDim.x * d + threadId) + 1) - 1;
            int bi = offset * (2 * (entireId / blockDim.x * d + threadId) + 2) - 1;
            /*
            if (chunk > 512)
            {
                printf("##%d: %d <- %d (%d <- %d)\n", entireId, 0, 0, bi, ai // (int)g_idata[bi], (int)g_idata[ai], bi, ai);
            }
            */
            if (!((0 <= ai && ai <= N) && (0 <= bi && bi <= N)))
                continue;

            /*
            if (chunk >= 512)
            {
                printf("#%d: %lld <- %lld (%d <- %d)\n", entireId, (long long)g_idata[bi], (long long)g_idata[ai], bi, ai);
            }
            */
            g_idata[bi] += g_idata[ai];
        }
        offset <<= 1;
    }
}

__global__ void assign_zero(float *g_idata, const long long pow2)
{
    // C
    g_idata[pow2 - 1] = 0; // clear the last element
}

__global__ void down_sweep(float *g_idata, const int chunk)
{
    int threadId = threadIdx.x;
    int entireId = blockIdx.x * blockDim.x + threadIdx.x;

    int offset = chunk;
    for (int d = blockDim.x >> 1; d > 0; d >>= 1)
        offset <<= 1;

    for (int d = 1; d < blockDim.x; d <<= 1) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (threadId < d)
        {
            // D
            int ai = offset * (2 * (entireId / blockDim.x * d + threadId) + 1) - 1;
            int bi = offset * (2 * (entireId / blockDim.x * d + threadId) + 2) - 1;
            if (!((0 <= ai && ai <= N) && (0 <= bi && bi <= N)))
                continue;

            float t = g_idata[ai];
            g_idata[ai] = g_idata[bi];
            g_idata[bi] += t;

            //printf("#%d: %d <-> %d (%d <-> %d)\n", entireId, (int)g_idata[bi], (int)g_idata[ai], bi, ai);
        }
    }
    __syncthreads();
}

int main()
{
    long long pow2 = 1;
    while (pow2 < N)
        pow2 <<= 1;

    float *v;
    v = (float *)malloc(sizeof(float) * pow2);
    for (int i = 0; i < N; i++)
        v[i] = i + 1;
    float *g_idata;
    cudaMalloc((void **)&g_idata, sizeof(float) * pow2), cudaMemcpy(g_idata, v, sizeof(float) * pow2, cudaMemcpyDefault);

    { // calc
        long long chunkSize;
        for (chunkSize = 1; chunkSize < pow2; chunkSize *= BS)
        {
            up_sweep<<<(pow2 + BS - 1) / BS, BS>>>(g_idata, chunkSize);
            CHECK(cudaDeviceSynchronize());
            //cout << "#" << chunkSize << endl;
        }

        /*
        assign_zero<<<1, 1>>>(g_idata, pow2);
        CHECK(cudaDeviceSynchronize());

        for (chunkSize /= BS; chunkSize > 0; chunkSize /= BS)
        {
            down_sweep<<<(pow2 + BS - 1) / BS, BS>>>(g_idata, chunkSize);
            CHECK(cudaDeviceSynchronize());
        }
        */
    }

    float *res;
    res = (float *)malloc(sizeof(float) * pow2);
    cudaMemcpy(res, g_idata, sizeof(float) * pow2, cudaMemcpyDefault);

    for (int i = N - 10; i < N; i++)
    {
        cout << i << " " << (long long)res[i] << "\n";
    }
    cout << flush;
}
