import torch
import ops.gconv_cuda as gconv_cuda
import numpy as np
from torch.autograd import gradcheck
from .gconv_ops import gconv_function
bsz = 1
n = 20
nin = 10
nout = 5

feature = torch.randn(
    bsz, n, nin, requires_grad=True, dtype=torch.float64).cuda()
edge_type = torch.randint(
    0, 9, [bsz, n, n], dtype=torch.int32, requires_grad=False).cuda()
filter = torch.randn(
    nout, nin, 9, requires_grad=True, dtype=torch.float64).cuda()

# print("here")
# res = gconv_cuda.gconv_forward(feature, edge_type, filter)
# print("end")
# print(res)
# orig_res = res
# nfeature = feature.detach().cpu().numpy()
# nedge_type = edge_type.cpu().numpy()
# nfilter = edge_type.detach().cpu().numpy()

# nres = np.zeros([bsz, n, nout])

# for i in range(bsz):
#     for node in range(n):
#         for nou in range(nout):
#             res = 0.0
#             for nn in range(n):
#                 etype = nedge_type[i, node, nn]
#                 for d in range(nin):
#                     scale = filter[nou, d, etype]
#                     res += scale * feature[i, nn, d]
#             nres[i, node, nou] = res

# print(nres)
# diff = orig_res.cpu().detach().numpy() - nres
# print(np.max(np.abs(diff)))

gconv_layer = gconv_function.apply

test = gradcheck(
    gconv_layer, [feature, edge_type, filter], eps=1e-4, atol=1e-2)

print(test)
