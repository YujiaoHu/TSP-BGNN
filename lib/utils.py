# utils.py ---
# Copyright (C) 2018
# Author:  <ZHANG, Zhen <zhen@zzhang.org>>
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.

#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see https://www.gnu.org/licenses/.

import numpy as np
import torch
import torch.nn.functional as F


def edge_to_eadj_mul_d(nnodes, edges, use_cuda=True):
    """Convert edge tensor to extended sparse adj matrix

    :param nnodes: number of nodes
    :param edge: a nx2 tensor of integer
    :returns: an sparse adj tensor
    :rtype: tf.SparseTensor

    """
    visited = dict()

    for edge in edges:
        idx = edge[0] * nnodes + edge[1]
        idx1 = edge[1] * nnodes + edge[0]
        visited[idx] = 1

    for i in range(nnodes):
        idx = i * nnodes + i
        visited[idx] = 1

    degree = np.zeros([nnodes])
    res = []
    for idx in visited.keys():
        idx0 = int(idx % nnodes)
        idx1 = int(idx / nnodes)
        degree[idx0] += 1
        degree[idx1] += 1
        res.append([idx0, idx1])

    degree = 1.0 / np.sqrt(degree)  # note that degree must be at least one

    final_values = np.ones([len(res)])
    for eidx, edge in enumerate(res):
        final_values[eidx] *= degree[edge[0]]
        final_values[eidx] *= degree[edge[1]]
    if use_cuda:
        i = torch.LongTensor(res).cuda()
        v = torch.FloatTensor(final_values).cuda()
        res = torch.cuda.sparse.FloatTensor(
            i.t(), v, torch.Size([nnodes, nnodes]))
    else:
        i = torch.LongTensor(res)
        v = torch.FloatTensor(final_values)
        res = torch.cuda.sparse.FloatTensor(
            i.t(), v, torch.Size([nnodes, nnodes]))
    return res


def edge_to_adj(nnodes, edges):
    """Convert edge tensor to sparse adj matrix

    :param nnodes: number of nodes
    :param edge: a nx2 tensor of integer
    :returns: an sparse adj tensor
    :rtype: tf.SparseTensor

    """
    return tf.SparseTensor(indices=edges,
                           values=np.ones([len(edges)]),
                           dense_shape=[nnodes, nnodes])


def construct_edge_graph(nnodes, edges):
    nedges = len(edges)
    node_to_edge_idx = [[] for n in range(nnodes)]
    for eidx, edge in enumerate(edges):
        i = edge[0]
        j = edge[1]
        node_to_edge_idx[i].append(eidx)
        node_to_edge_idx[j].append(-eidx)

    adj_matrix = np.zeros([nedges, nedges], dtype=np.int16)

    for eidx, edge in enumerate(edges):
        i = edge[0]
        j = edge[1]

        adj_edges = node_to_edge_idx[i] + node_to_edge_idx[j]

        for k in adj_edges:
            k = int(np.abs(k))
            if(k == eidx):
                continue
            adj_matrix[eidx, k] = 1
            adj_matrix[k, eidx] = 1

    res = []
    for ei in range(nedges):
        for ej in range(nedges):
            if(adj_matrix[ei][ej]):
                res.append([ei, ej])

    return res


def edge_feature_to_node_feature(nnodes, edges, use_cuda=True):

    to_first_node_idx = [[edge[0], eidx] for eidx, edge in enumerate(edges)]
    to_second_node_idx = [[edge[1], eidx] for eidx, edge in enumerate(edges)]

    if use_cuda:
        base = torch.cuda
    else:
        base = torch
    itensor1 = base.LongTensor(to_first_node_idx).t()
    itensor2 = base.LongTensor(to_second_node_idx).t()

    vtensor = base.FloatTensor(np.ones(len(edges)))
    size = torch.Size([nnodes, len(edges)])
    first_tensor = base.sparse.FloatTensor(itensor1, vtensor, size)
    second_tensor = base.sparse.FloatTensor(itensor2, vtensor, size)

    return first_tensor, second_tensor


if __name__ == '__main__':
    nnodes = 4
    edges = [[1, 3], [3, 1], [3, 2], [2, 3]]

    res = edge_to_eadj_mul_d(nnodes, edges)

    print(res)

    ftensor, stensor = edge_feature_to_node_feature(nnodes, edges)

    print(ftensor, stensor)
