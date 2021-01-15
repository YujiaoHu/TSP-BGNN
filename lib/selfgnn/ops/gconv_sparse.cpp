#include <torch/extension.h>
#include <cmath>
#include <vector>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
void gconv_sparse_forward_runner(
    const scalar_t * feature,
    const int * edge_type,
    const scalar_t * filter,
    scalar_t * res,
    size_t batch_size,
    size_t nnodes,
    size_t nin_features,
    size_t nout_features,
    size_t nedge_type,
    size_t nelist_length);

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
    size_t nelist_length);


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
    size_t nelist_length
    );

torch::Tensor gconv_sparse_forward(torch::Tensor feature,
                                   torch::Tensor edge_type,
                                   torch::Tensor filter){
    CHECK_INPUT(feature);
    CHECK_INPUT(edge_type);
    CHECK_INPUT(filter);

    const auto batch_size = feature.size(0);
    const auto nnodes = feature.size(1);
    const auto nin_features = feature.size(2);
    const auto nout_features = filter.size(0);
    const auto nedge_type = filter.size(2);

    assert(nnodes == edge_type.size(1));
    assert(nin_features == filter.size(1));

    torch::Tensor res = torch::zeros({batch_size, nnodes, nout_features}, torch::dtype(feature.type().scalarType()).requires_grad(true)).cuda();
    switch(feature.type().scalarType()){
    case at::ScalarType::Double:
        gconv_sparse_forward_runner<double>(feature.data<double>(),
                                            edge_type.data<int>(),
                                            filter.data<double>(),
                                            res.data<double>(),
                                            batch_size,
                                            nnodes,
                                            nin_features,
                                            nout_features,
                                            nedge_type,
                                            edge_type.size(2));
        break;
    case at::ScalarType::Float:
        gconv_sparse_forward_runner<float>(feature.data<float>(),
                                           edge_type.data<int>(),
                                           filter.data<float>(),
                                           res.data<float>(),
                                           batch_size,
                                           nnodes,
                                           nin_features,
                                           nout_features,
                                           nedge_type,
                                           edge_type.size(2));
        break;
    }

    return res;
}



std::vector<torch::Tensor> gconv_sparse_backward(torch::Tensor grad_out,
                                                 torch::Tensor feature,
                                                 torch::Tensor inv_edge_list,
                                                 torch::Tensor etype_list,
                                                 torch::Tensor filter)
{
    CHECK_INPUT(grad_out);
    CHECK_INPUT(feature);
    CHECK_INPUT(etype_list);
    CHECK_INPUT(inv_edge_list);
    CHECK_INPUT(filter);

    
    const auto batch_size = feature.size(0);
    const auto nnodes = feature.size(1);
    const auto nin_features = feature.size(2);
    const auto nout_features = filter.size(0);
    const auto nedge_type = filter.size(2);
    assert(nin_features == filter.size(1));
    assert(nout_features == grad_out.size(2));
    assert(batch_size == grad_out.size(0));
    // assert(nout_features == filter.size(0))
    
    torch::Tensor feature_grad = torch::zeros_like(feature).cuda();
    torch::Tensor filter_grad = torch::zeros({batch_size, nout_features, nin_features, nedge_type}, torch::dtype(feature.type().scalarType()).requires_grad(false)).cuda();
    
    switch(feature.type().scalarType()){
    case at::ScalarType::Double:
        gconv_sparse_backward_filter_runner(grad_out.data<double>(), feature.data<double>(), etype_list.data<int>(), filter_grad.data<double>(), batch_size, nnodes, nin_features, nout_features, nedge_type, etype_list.size(3));
        gconv_sparse_backward_indata_runner(grad_out.data<double>(), filter.data<double>(), inv_edge_list.data<int>(), feature_grad.data<double>(), batch_size, nnodes, nin_features, nout_features, nedge_type, inv_edge_list.size(2));
        break;
    case at::ScalarType::Float:
        gconv_sparse_backward_filter_runner(grad_out.data<float>(), feature.data<float>(), etype_list.data<int>(), filter_grad.data<float>(), batch_size, nnodes, nin_features, nout_features, nedge_type, etype_list.size(3));
        gconv_sparse_backward_indata_runner(grad_out.data<float>(), filter.data<float>(), inv_edge_list.data<int>(), feature_grad.data<float>(), batch_size, nnodes, nin_features, nout_features, nedge_type, inv_edge_list.size(2));
        break;
    }
    return {feature_grad, filter_grad.sum(0)};
}
