#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include "common.cuh"

/**
 *
 * @param grad_res gradient of 
 */
template <int blockSize, typename scalar_t>
__global__ void gconv_sparse_backward_kernel_filter(
    const scalar_t* __restrict__ grad_res,
    const scalar_t* __restrict__ feature,
    const int* __restrict__ edge_type,
    scalar_t * __restrict__ filter_grad,
    size_t batch_sz,
    size_t nnodes,
    size_t elist_length
    )
{
    extern __shared__ int s [];
    int ou_id = blockIdx.x;
    int in_id = blockIdx.y;
    int filter_id = blockIdx.z;

    int nou_feature = gridDim.x;
    int nin_feature = gridDim.y;
    int nedge_type = gridDim.z;

    
    volatile scalar_t * partial_res = (scalar_t *) s;
    scalar_t * cres = filter_grad + (ou_id * nin_feature + in_id)* nedge_type + filter_id;

    int tid = threadIdx.x;
    partial_res[tid] = 0.0;

    if (tid >= nnodes * batch_sz) return;

    for(int i = tid; i < nnodes * batch_sz; i += blockDim.x){
        int batch_id = i / nnodes;
        int ou_node_id = i % nnodes;
        int nedges = edge_type[((batch_id * nedge_type + filter_id) * nnodes + ou_node_id) * (elist_length)];
        const int *cedge_type = edge_type + ((batch_id * nedge_type + filter_id) * nnodes + ou_node_id) * (elist_length) + 1;
        

        int cou_id = batch_id * (nnodes * nou_feature) + ou_node_id * nou_feature + ou_id;
        for(int edge_id = 0; edge_id < nedges; edge_id++){
            int in_node_id = cedge_type[edge_id];
            int cin_id = batch_id * (nnodes * nin_feature) + in_node_id * nin_feature + in_id;
            partial_res[tid] += grad_res[cou_id] * feature[cin_id];
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

template <int blockSize, typename scalar_t>
__global__ void gconv_sparse_backward_kernel_filter_batchwise(
    const scalar_t* __restrict__ grad_res,
    const scalar_t* __restrict__ feature,
    const int* __restrict__ edge_type,
    scalar_t * __restrict__ filter_grad,
    size_t batch_sz,
    size_t nnodes,
    size_t elist_length
    )
{
    extern __shared__ int s [];
    int ou_id = blockIdx.x;
    int in_id = blockIdx.y;
    int filter_batch_id = blockIdx.z;
    int nou_feature = gridDim.x;
    int nin_feature = gridDim.y;
    int nedge_type = gridDim.z;

    int filter_id = filter_batch_id % nedge_type;
    int batch_id = filter_batch_id / nedge_type;
    
    volatile scalar_t * partial_res = (scalar_t *) s;
    scalar_t * cres = filter_grad + ((batch_id * nou_feature + ou_id) * nin_feature + in_id)* nedge_type + filter_id;
    const scalar_t * cfeature = feature + batch_id * nnodes * nin_feature;
    const scalar_t * cgrad_res = grad_res + batch_id * nnodes * nou_feature;
    const int *cetype = edge_type + (batch_id * nedge_type + filter_id) * nnodes * elist_length;

    int tid = threadIdx.x;
    partial_res[tid] = 0.0;

    if (tid >= nnodes) return;

    for(int i = tid; i < nnodes; i += blockDim.x){
        int ou_node_id = i;
        int nedges = cetype[ou_node_id * elist_length];
        const int *cedge_type = cetype + ou_node_id * elist_length + 1;
        

        int cou_id = ou_node_id * nou_feature + ou_id;
        for(int edge_id = 0; edge_id < nedges; edge_id++){
            int in_node_id = cedge_type[edge_id];
            int cin_id = in_node_id * nin_feature + in_id;
            partial_res[tid] += cgrad_res[cou_id] * cfeature[cin_id];
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


#define run_kernel(bsize, type) gconv_sparse_backward_kernel_filter_batchwise<bsize, type><<< blocks, bsize, bsize * sizeof(type)  >>> (grad_res, feature, edge_type, filter_grad, batch_size, nnodes, nedge_length)


template <typename scalar_t>
void gconv_sparse_backward_filter_runner(
    const scalar_t * grad_res,
    const scalar_t *feature,
    const int * edge_type,
    scalar_t * filter_grad,
    size_t batch_size,
    size_t nnodes,
    size_t nin_features,
    size_t nout_features,
    size_t nedge_type,
    size_t nedge_length
    )
{
    int best_num_threads = pow(2, ceil(log(nnodes)/log(2)));
    int threads = best_num_threads > 512?512:best_num_threads ;
    const dim3 blocks(nout_features, nin_features, nedge_type * batch_size);
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

template 
void gconv_sparse_backward_filter_runner(const double * grad_res,
                                       const double *feature,
                                       const int * edge_type,
                                       double * filter_grad,
                                       size_t batch_size,
                                       size_t nnodes,
                                       size_t nin_features,
                                       size_t nout_features,
                                         size_t nedge_type,
                                         size_t nelist_length
    );


template 
void gconv_sparse_backward_filter_runner(const float * grad_res,
                                       const float *feature,
                                       const int * edge_type,
                                       float * filter_grad,
                                       size_t batch_size,
                                       size_t nnodes,
                                       size_t nin_features,
                                       size_t nout_features,
                                         size_t nedge_type,
                                         size_t nelist_length
    );



template <int blockSize, typename scalar_t>
__global__ void gconv_sparse_backward_kernel_indata(
    const scalar_t* __restrict__ grad_res,
    const scalar_t* __restrict__ filter,
    const int* __restrict__ edge_type,
    scalar_t * __restrict__ feature_grad,
    size_t nou_feature,
    size_t nedge_type,
    size_t elist_length)
{
    
    extern __shared__ int s [];
    int batch_id = blockIdx.x;
    int node_id = blockIdx.y;
    int nin = blockIdx.z;

    
    int nnodes = gridDim.y;
    int nin_feature = gridDim.z;

    
    volatile scalar_t * partial_res = (scalar_t *) s;

    const scalar_t* out_feature_grad = grad_res + batch_id * nnodes * nou_feature;
    int nedges = edge_type[(batch_id * nnodes + node_id) * (elist_length)];
    const int * cedge_type = edge_type + (batch_id * nnodes + node_id) * (elist_length) + 1;

    scalar_t * cres = feature_grad + (batch_id * nnodes + node_id) * nin_feature + nin;
    int tid = threadIdx.x;

    partial_res[tid] = 0.0;
    if(tid >= nou_feature) return;
    for(int eid = 0; eid < nedges; eid ++){
        int cnode_id = cedge_type[2 * eid];
        int edge_type_ = cedge_type[2 * eid + 1];
        for(int cfeature_id = tid; cfeature_id < nou_feature; cfeature_id += blockDim.x){
            partial_res[tid] += filter[(cfeature_id * nin_feature + nin) * nedge_type + edge_type_] * out_feature_grad[cnode_id * nou_feature + cfeature_id];
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


#define run_kernel(bsize, type) gconv_sparse_backward_kernel_indata<bsize, type>  <<<blocks, bsize, bsize * sizeof(scalar_t)  >>>(grad_res, filter, edge_type, feature_grad, nout_features, nedge_type, elist_length)


template <typename scalar_t>
void gconv_sparse_backward_indata_runner(
    const scalar_t *grad_res,
    const scalar_t *filter,
    const int * edge_type,
    scalar_t * feature_grad,
    size_t batch_size,
    size_t nnodes,
    size_t nin_features,
    size_t nout_features,
    size_t nedge_type,
    size_t elist_length
    )
{
    int best_num_threads = pow(2, ceil(log(nout_features)/ log(2)));
    int threads = best_num_threads > 512?512:best_num_threads ;
    const dim3 blocks(batch_size, nnodes, nin_features);
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



template
void gconv_sparse_backward_indata_runner(
    const double *grad_res,
    const double *filter,
    const int * edge_type,
    double * feature_grad,
    size_t batch_size,
    size_t nnodes,
    size_t nin_features,
    size_t nout_features,
    size_t nedge_type,
    size_t nelist_length
    );



template
void gconv_sparse_backward_indata_runner(
    const float *grad_res,
    const float *filter,
    const int * edge_type,
    float * feature_grad,
    size_t batch_size,
    size_t nnodes,
    size_t nin_features,
    size_t nout_features,
    size_t nedge_type,
    size_t nelist_length
    );
