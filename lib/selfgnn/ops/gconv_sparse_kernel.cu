#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include "gconv.h"
#include "common.cuh"


template <int blockSize, typename scalar_t>
__global__ void gconv_sparse_forward_kernel(
    const scalar_t* __restrict__ feature,
    const int* __restrict__ edge_type,
    const scalar_t* __restrict__ filter,
    scalar_t* __restrict__ res,
    size_t nin_feature,
    size_t nedge_type,
    size_t elist_length)
{
    extern __shared__ int s [];
    int batch_sz = blockIdx.x;
    int node_id = blockIdx.y;
    int nout = blockIdx.z;

    int nnodes = gridDim.y;

    int tid = threadIdx.x;

    volatile scalar_t * partial_res = (scalar_t *) s;
    const scalar_t* cfeature = feature + batch_sz * (nnodes * nin_feature);
    const scalar_t* kernels = filter + nin_feature * nedge_type * nout;
    const int nneibours = edge_type[(batch_sz * nnodes + node_id) * (elist_length)];
    const int *edge_list = edge_type + (batch_sz * nnodes + node_id) * (elist_length) + 1;

    scalar_t* cres = res + batch_sz * nnodes * gridDim.z + node_id * gridDim.z + nout;

    
    partial_res[tid] = 0.0;

    if(tid >= nin_feature) return;
    for(int edge_id = 0; edge_id < nneibours; edge_id++){
        int cnode_id = edge_list[2 * edge_id];
        int edge_type_ = edge_list[2 * edge_id + 1];
        for(int cfeature_id = tid; cfeature_id < nin_feature; cfeature_id+= blockDim.x){
            partial_res[tid] += kernels[cfeature_id * nedge_type + edge_type_] * cfeature[cnode_id * nin_feature + cfeature_id];
        }
    }

    __syncthreads();
    if (blockSize >= 512) { 
        if (tid < 256) { partial_res[tid] += partial_res[tid + 256]; }
        __syncthreads(); }
    if (blockSize >= 256) { 
        if (tid < 128) { partial_res[tid] += partial_res[tid + 128]; }
        __syncthreads(); }
    if (blockSize >= 128) { 
        if (tid <  64) { partial_res[tid] += partial_res[tid + 64]; }
        __syncthreads(); }
    if (tid < 32) warpReduceSum<blockSize, scalar_t>(partial_res, tid);

    if (tid == 0){
        cres[0] = partial_res[0];
    }
}


#define run_kernel(bsize, type) gconv_sparse_forward_kernel<bsize, type><<<blocks, threads, threads * sizeof(type)>>> (feature, edge_type,  filter,  res, nin_features, nedge_type, elist_length)
    
template <typename scalar_t>
void gconv_sparse_forward_runner(const scalar_t * feature,
                                 const int * edge_type,
                                 const scalar_t * filter,
                                 scalar_t * res,
                                 size_t batch_size,
                                 size_t nnodes,
                                 size_t nin_features,
                                 size_t nout_features,
                                 size_t nedge_type,
                                 size_t elist_length)
{
    int best_num_threads = pow(2, ceil(log(nin_features)/log(2)));
    int threads = best_num_threads > 512?512:best_num_threads ;

    const dim3 blocks(batch_size, nnodes, nout_features);

    switch(threads){

    case 512:
        run_kernel(512, scalar_t);
        break;
    case 256:
        run_kernel(256, scalar_t);
        break;
    case 128:
        run_kernel(128, scalar_t);
        break;
    case 64:
        run_kernel(64, scalar_t);
        break;
    case 32:
        run_kernel(32, scalar_t);
        break;
    case 16:
        run_kernel(16, scalar_t);
        break;
    case 8:
        run_kernel(8, scalar_t);
        break;
    case 4:
        run_kernel(4, scalar_t);
        break;
    case 2:
        run_kernel(2, scalar_t);
        break;
    case 1:
        run_kernel(1, scalar_t);
        break;
    }
}

#undef run_kernel


#define scalar_t double
template 
void gconv_sparse_forward_runner(const scalar_t * feature,
                                 const int * edge_type,
                                 const scalar_t * filter,
                                 scalar_t * res,
                                 size_t batch_size,
                                 size_t nnodes,
                                 size_t nin_features,
                                 size_t nout_features,
                                 size_t nedge_type,
                                 size_t elist_length);
#undef scalar_t

#define scalar_t float
template
void gconv_sparse_forward_runner(const scalar_t * feature,
                                 const int * edge_type,
                                 const scalar_t * filter,
                                 scalar_t * res,
                                 size_t batch_size,
                                 size_t nnodes,
                                 size_t nin_features,
                                 size_t nout_features,
                                 size_t nedge_type,
                                 size_t elist_length);
#undef scalar_t


