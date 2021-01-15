#include <cuda.h>
#include <cuda_runtime.h>


template <unsigned int blockSize, typename T>
__device__ void 
warpReduceSum(volatile T *sdata, unsigned int tid) {
    if (blockSize >=  64){
        if(tid < 32) sdata[tid] += sdata[tid + 32];
    }
    if (blockSize >=  32)
        if (tid < 16) sdata[tid] +=  sdata[tid + 16];
    if (blockSize >=  16)
        if (tid < 8) sdata[tid] +=  sdata[tid +  8];
    if (blockSize >=   8)
        if (tid < 4) sdata[tid] += sdata[tid +  4];
    if (blockSize >=   4)
        if (tid < 2) sdata[tid] += sdata[tid +  2];   
    if (blockSize >=   2)
        if (tid < 1) sdata[tid] += sdata[tid +  1];
}

