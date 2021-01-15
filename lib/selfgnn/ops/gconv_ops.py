import math
import torch
from .gconv_cuda import gconv_forward, gconv_backward, gconv_sparse_backward, gconv_sparse_forward


class gconv_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *input):
        feature, edge_type, filters = input
        res = gconv_forward(feature, edge_type, filters)
        ctx.save_for_backward(*input)
        return res

    @staticmethod
    def backward(ctx, grad_res):
        grad_feature, grad_filter = gconv_backward(grad_res,
                                                   *ctx.saved_variables)
        if not ctx.needs_input_grad[0]:
            grad_feature = None
        if not ctx.needs_input_grad[2]:
            grad_filter = None

        return grad_feature, None, grad_filter


class gconv_sparse_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *input):
        feature, edge_list, inv_edge_list, etype_link_list, filters = input
        res = gconv_sparse_forward(feature, edge_list, filters)
        ctx.save_for_backward(feature, inv_edge_list, etype_link_list, filters)
        return res

    @staticmethod
    def backward(ctx, grad_res):
        grad_feature, grad_filter = gconv_sparse_backward(
            grad_res.contiguous(), *ctx.saved_variables)

        if not ctx.needs_input_grad[0]:
            grad_feature = None
        if not ctx.needs_input_grad[4]:
            grad_filter = None

        return grad_feature, None, None, None, grad_filter
