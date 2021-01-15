#include <torch/extension.h>
#include "gconv.h"
#include <cmath>
#include <vector>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


template <typename scalar_t>
void gconv_cuda_forward_runner(const scalar_t * feature,
                               const int * edge_type,
                               const scalar_t * filter,
                               scalar_t * res,
                               size_t batch_size,
                               size_t nnodes,
                               size_t nin_features,
                               size_t nout_features,
                               size_t nedge_type
    );

torch::Tensor gconv_cuda_forward(torch::Tensor feature,
                                torch::Tensor edge_type,
                                torch::Tensor filter);

torch::Tensor gconv_forward(torch::Tensor feature,
                           torch::Tensor edge_type,
                           torch::Tensor filter){
    
    CHECK_INPUT(feature);
    CHECK_INPUT(edge_type);
    CHECK_INPUT(filter);


    

    return gconv_cuda_forward(feature, edge_type, filter);
}


torch::Tensor gconv_cuda_forward(torch::Tensor feature,
                                torch::Tensor edge_type,
                                torch::Tensor filter)
{
    const auto batch_size = feature.size(0);
    const auto nnodes = feature.size(1);
    const auto nin_features = feature.size(2);
    const auto nout_features = filter.size(0);
    const auto nedge_type = filter.size(2);
    assert(nin_features == filter.size(1));
    
    

    torch::Tensor res = torch::zeros({batch_size, nnodes, nout_features},
                                   torch::dtype(feature.type().scalarType()).requires_grad(true)).cuda();
    

    switch(feature.type().scalarType()){
    case at::ScalarType::Double:
        gconv_cuda_forward_runner<double>(feature.data<double>(),
                                          edge_type.data<int>(),
                                          filter.data<double>(),
                                          res.data<double>(),
                                          batch_size,
                                          nnodes,
                                          nin_features,
                                          nout_features,
                                          nedge_type);
        
        break;
    case at::ScalarType::Float:
        gconv_cuda_forward_runner<float>(feature.data<float>(),
                                         edge_type.data<int>(),
                                         filter.data<float>(),
                                         res.data<float>(),
                                         batch_size,
                                         nnodes,
                                         nin_features,
                                         nout_features,
                                         nedge_type);
    }
    
    
    return res;
}
template <typename scalar_t>
void gconv_cuda_backward_filter_runner(const scalar_t * grad_res,
                                       const scalar_t *feature,
                                       const int * edge_type,
                                       scalar_t * filter_grad,
                                       size_t batch_size,
                                       size_t nnodes,
                                       size_t nin_features,
                                       size_t nout_features,
                                       size_t nedge_type
    );
template <typename scalar_t>
void gconv_cuda_backward_kernel_indata_runner(
    const scalar_t *grad_res,
    const scalar_t *filter,
    const int * edge_type,
    scalar_t * feature_grad,
    size_t batch_size,
    size_t nnodes,
    size_t nin_features,
    size_t nout_features,
    size_t nedge_type
    );
std::vector<torch::Tensor> gconv_cuda_backward(torch::Tensor grad_out,
                                             torch::Tensor feature,
                                             torch::Tensor edge_type,
                                             torch::Tensor filter)
{
    const auto batch_size = feature.size(0);
    const auto nnodes = feature.size(1);
    const auto nin_features = feature.size(2);
    const auto nout_features = filter.size(0);
    const auto nedge_type = filter.size(2);
    assert(nin_features == filter.size(1));
    assert(nout_features == grad_out.size(2));

    
    torch::Tensor feature_grad = torch::zeros_like(feature).cuda();
    torch::Tensor filter_grad = torch::zeros_like(filter).cuda();

    switch(feature.type().scalarType()){
    case at::ScalarType::Double:
        gconv_cuda_backward_filter_runner(grad_out.data<double>(), feature.data<double>(), edge_type.data<int>(), filter_grad.data<double>(), batch_size, nnodes, nin_features, nout_features, nedge_type);
        gconv_cuda_backward_kernel_indata_runner(grad_out.data<double>(), filter.data<double>(), edge_type.data<int>(), feature_grad.data<double>(), batch_size, nnodes, nin_features, nout_features, nedge_type);
        break;
    case at::ScalarType::Float:
        gconv_cuda_backward_filter_runner(grad_out.data<float>(), feature.data<float>(), edge_type.data<int>(), filter_grad.data<float>(), batch_size, nnodes, nin_features, nout_features, nedge_type);
        gconv_cuda_backward_kernel_indata_runner(grad_out.data<float>(), filter.data<float>(), edge_type.data<int>(), feature_grad.data<float>(), batch_size, nnodes, nin_features, nout_features, nedge_type);
        break;
    }
    return {feature_grad, filter_grad};
}


std::vector<torch::Tensor> gconv_backward(torch::Tensor grad_out,
                                          torch::Tensor feature,
                                          torch::Tensor edge_type,
                                          torch::Tensor filter){
    
    CHECK_INPUT(feature);
    CHECK_INPUT(edge_type);
    CHECK_INPUT(filter);


  
    return gconv_cuda_backward(grad_out, feature, edge_type, filter);
}

extern

std::vector<torch::Tensor> gconv_sparse_backward(torch::Tensor grad_out,
                                                 torch::Tensor feature,
                                                 torch::Tensor edge_list,
                                                 torch::Tensor inv_edge_list,
                                                 torch::Tensor filter);
extern
torch::Tensor gconv_sparse_forward(torch::Tensor feature,
                                   torch::Tensor edge_type,
                                   torch::Tensor filter);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gconv_forward", &gconv_forward, "GConv forward (CUDA)");
    m.def("gconv_backward", &gconv_backward, "GConv backward (CUDA)");
    m.def("gconv_sparse_forward", &gconv_sparse_forward, "GConv forward (CUDA) Sparse");
    m.def("gconv_sparse_backward", &gconv_sparse_backward, "GConv backward (CUDA) Sparse");
}
