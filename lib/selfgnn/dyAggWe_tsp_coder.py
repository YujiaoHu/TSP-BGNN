import torch
from .utils import get_edge_feature, get_nn_node_feature
# 现在train的30-90有relu的用的不是这个mp_conv, 而是dynamic_mp_conv.py
from .dyAggWe_mp_conv import etype_net, mp_conv_v2
from torch.nn import functional as F
import math

SyncBatchNorm = torch.nn.BatchNorm2d
Dropout = torch.nn.Dropout
drop_para = 0.5


class gconv_residual(torch.nn.Module):
    def __init__(self, nin, nmed, win, wmed, netype, with_residual=True, dropout=False):
        super(gconv_residual, self).__init__()
        if dropout is False:
            self.nconv1 = torch.nn.Sequential(
                torch.nn.Conv2d(nin, nmed, 1), SyncBatchNorm(nmed),
                torch.nn.ReLU(inplace=True))
            self.wconv1 = torch.nn.Sequential(
                torch.nn.Conv2d(win, wmed, 1), SyncBatchNorm(wmed),
                torch.nn.ReLU(inplace=True)
            )
            self.mp_conv = mp_conv_v2(nin=nmed, nout=nmed, win=wmed, wout=wmed, nedge_types=netype)
            self.nconv2 = torch.nn.Sequential(
                torch.nn.Conv2d(nmed, nin, 1), SyncBatchNorm(nin),
                torch.nn.ReLU(inplace=True))
            self.wconv2 = torch.nn.Sequential(
                torch.nn.Conv2d(wmed, win, 1), SyncBatchNorm(win),
                torch.nn.ReLU(inplace=True)
            )
            self.with_residual = with_residual
        else:
            self.nconv1 = torch.nn.Sequential(
                torch.nn.Conv2d(nin, nmed, 1), Dropout(drop_para),
                torch.nn.ReLU(inplace=True))
            self.wconv1 = torch.nn.Sequential(
                torch.nn.Conv2d(win, wmed, 1), Dropout(drop_para),
                torch.nn.ReLU(inplace=True)
            )
            self.mp_conv = mp_conv_v2(nin=nmed, nout=nmed, win=wmed, wout=wmed, nedge_types=netype)
            self.nconv2 = torch.nn.Sequential(
                torch.nn.Conv2d(nmed, nin, 1), Dropout(drop_para),
                torch.nn.ReLU(inplace=True))
            self.wconv2 = torch.nn.Sequential(
                torch.nn.Conv2d(wmed, win, 1), Dropout(drop_para),
                torch.nn.ReLU(inplace=True)
            )
            self.with_residual = with_residual

    def forward(self, node_feature, weight_feature, etype, nn_idx):
        nfeature = self.nconv1(node_feature)
        wfeature = self.wconv1(weight_feature)
        nfeature, wfeature = self.mp_conv(nfeature, wfeature, nn_idx, etype)
        nfeature = self.nconv2(nfeature)
        wfeature = self.wconv2(wfeature)

        if self.with_residual:
            nfeature = nfeature + node_feature
            wfeature = wfeature + weight_feature
        return nfeature, wfeature


class attention_decoding(torch.nn.Module):
    def __init__(self, fsize):
        super(attention_decoding, self).__init__()
        self.decoder = []
        self.glb_fixed_Q = None
        self.node_fixed_K = None
        self.node_fixed_V = None
        self.node_fixed_logit_K = None

        self.glb_embedding = torch.nn.Linear(fsize, fsize)
        self.node_embedding = torch.nn.Conv2d(fsize, 3 * fsize, 1, 1)
        self.last_current_embedding = torch.nn.Conv2d(2 * fsize, fsize, 1, 1)
        self.project_out = torch.nn.Linear(fsize, fsize)

    def forward(self, nfeature, last_current):
        batch_size = nfeature.size(0)
        glbfeature, _ = torch.max(nfeature, dim=2)

        glbfeature = glbfeature.contiguous().view(batch_size, -1)
        self.glb_fixed_Q = self.glb_embedding(glbfeature)
        self.glb_fixed_Q = self.glb_fixed_Q.unsqueeze(2)

        self.node_fixed_K, self.node_fixed_V, self.node_fixed_logit_K \
            = self.node_embedding(nfeature).chunk(3, dim=1)
        self.node_fixed_K = self.node_fixed_K.squeeze(3)
        self.node_fixed_V = self.node_fixed_V.squeeze(3)
        self.node_fixed_logit_K = self.node_fixed_logit_K.squeeze(3)

        atour = last_current
        last = atour[:, 0].view(batch_size, 1, 1, 1)
        current = atour[:, 1].view(batch_size, 1, 1, 1)

        index = last.repeat(1, nfeature.size(1), 1, nfeature.size(3))
        lastcity = torch.gather(nfeature, dim=2, index=index)
        index = current.repeat(1, nfeature.size(1), 1, nfeature.size(3))
        currentcity = torch.gather(nfeature, dim=2, index=index)

        context = torch.cat((lastcity, currentcity), dim=1)
        context_Q = self.last_current_embedding(context)
        context_Q = context_Q.squeeze(3) + self.glb_fixed_Q

        ucj = torch.bmm(context_Q.transpose(1, 2), self.node_fixed_K) / math.sqrt(self.node_fixed_K.size(1))
        new_context = torch.bmm(F.softmax(ucj, dim=2), self.node_fixed_V.transpose(1, 2))
        new_context = self.project_out(new_context.squeeze(1)).unsqueeze(1)
        logits = torch.bmm(new_context, self.node_fixed_logit_K) / math.sqrt(self.node_fixed_K.size(1))
        logits = logits.squeeze(1)  # logits: [batch, n]

        logits = torch.tanh(logits) * 10
        return logits


class tsp_coder_attention(torch.nn.Module):
    def __init__(self, with_residual=True,
                 with_global=False,
                 nodeFeature=2,
                 weightFeature=1,
                 with_gnn_decode=False,
                 dropout=False):
        super(tsp_coder_attention, self).__init__()
        self.dropout = dropout

        self.with_global = with_global
        self.with_gnn_decode = with_gnn_decode
        self.etype_net = etype_net(16, 64, nodeFeature, wfeature_size=weightFeature)

        self.mp_conv1 = mp_conv_v2(nin=nodeFeature, nout=64, win=weightFeature, wout=16, nedge_types=16)

        self.mp_residual1 = gconv_residual(
            nin=64, nmed=64, win=16, wmed=16, netype=16, with_residual=with_residual)

        self.mp_conv2 = mp_conv_v2(nin=64, nout=128, win=16, wout=32, nedge_types=16)

        self.mp_residual2 = gconv_residual(128, 128, 32, 32, 16, with_residual=with_residual)

        self.att_decoding = attention_decoding(128)

    def forward(self, pts, pair_weight, nn_idx, last_current):
        # pts: [batch, input_feature_size, nnodes]
        # nn_idx: [batch, nnodes, knn_k]
        pts_knn = get_nn_node_feature(pts, nn_idx)  # pts_knn: [batch, input_feature_size, city_num, knn_k]
        # need check
        efeature = get_edge_feature(pts_knn, pts)  # efeature: [batch, input_feature_size * 2, city_num, knn_k]

        etype = self.etype_net(efeature, pair_weight)

        nfeature, wfeature = self.mp_conv1(
            pts.view(pts.shape[0], pts.shape[1], pts.shape[2], 1), pair_weight, nn_idx,
            etype)
        nfeature, wfeature = self.mp_residual1(nfeature, wfeature, etype, nn_idx)

        nfeature, wfeature = self.mp_conv2(nfeature, wfeature, nn_idx, etype)

        nfeature, wfeature = self.mp_residual2(nfeature, wfeature, etype, nn_idx)

        # decoding part
        probs = self.att_decoding(nfeature, last_current)
        return probs


class tsp_coder_simply(torch.nn.Module):
    def __init__(self, with_residual=True,
                 with_global=False,
                 nodeFeature=2,
                 weightFeature=1,
                 with_gnn_decode=False,
                 dropout=False):
        super(tsp_coder_simply, self).__init__()
        self.dropout = dropout

        self.with_global = with_global
        self.with_gnn_decode = with_gnn_decode
        self.etype_net = etype_net(16, 64, nodeFeature, wfeature_size=weightFeature)

        self.mp_conv1 = mp_conv_v2(nin=nodeFeature, nout=64, win=weightFeature, wout=16, nedge_types=16)

        self.mp_residual1 = gconv_residual(
            nin=64, nmed=64, win=16, wmed=16, netype=16, with_residual=with_residual)

        self.mp_residual2 = gconv_residual(64, 64, 16, 16, 16, with_residual=with_residual)

        if self.with_gnn_decode:
            self.mp_conv2 = mp_conv_v2(128, 128, 16, 16, 16)
            self.node_conv = torch.nn.Sequential(
                torch.nn.Conv2d(128, 64, 1),
                SyncBatchNorm(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 1, 1)
            )
        else:
            self.context_conv = torch.nn.Sequential(
                torch.nn.Conv2d(128, 64, 1),
                SyncBatchNorm(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 1, 1)
            )

    def forward(self, pts, pair_weight, nn_idx):
        # pts: [batch, input_feature_size, nnodes]
        # nn_idx: [batch, nnodes, knn_k]
        pts_knn = get_nn_node_feature(pts, nn_idx)  # pts_knn: [batch, input_feature_size, city_num, knn_k]
        # need check
        efeature = get_edge_feature(pts_knn, pts)  # efeature: [batch, input_feature_size * 2, city_num, knn_k]

        etype = self.etype_net(efeature, pair_weight)

        nfeature, wfeature = self.mp_conv1(
            pts.view(pts.shape[0], pts.shape[1], pts.shape[2], 1), pair_weight, nn_idx,
            etype)
        nfeature, wfeature = self.mp_residual1(nfeature, wfeature, etype, nn_idx)

        nfeature, wfeature = self.mp_residual2(nfeature, wfeature, etype, nn_idx)

        # decoding part
        batch_size = nfeature.size(0)
        global_feature, _ = nfeature.max(dim=2, keepdim=True)

        nfeature = torch.cat(
            [nfeature, global_feature.repeat(1, 1, nfeature.shape[2], 1)], dim=1)
        if self.with_gnn_decode:
            nfeature, wfeature = self.mp_conv2(nfeature, wfeature, nn_idx, etype)
            probs = self.node_conv(nfeature).squeeze()
        else:
            probs = self.context_conv(nfeature).squeeze()
        if batch_size == 1:
            probs = probs.unsqueeze(0)
        return probs


class tsp_coder(torch.nn.Module):
    def __init__(self, with_residual=True,
                 with_global=False,
                 nodeFeature=2,
                 weightFeature=1,
                 with_gnn_decode=False,
                 dropout=False):
        super(tsp_coder, self).__init__()
        self.dropout = dropout

        self.with_global = with_global
        self.with_gnn_decode = with_gnn_decode
        self.etype_net = etype_net(16, 64, nodeFeature, wfeature_size=weightFeature)

        self.mp_conv1 = mp_conv_v2(nin=nodeFeature, nout=64, win=weightFeature, wout=16, nedge_types=16)
        self.nconv1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 1), SyncBatchNorm(128),
            torch.nn.ReLU(inplace=True))
        self.wconv1 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 64, 1), SyncBatchNorm(64),
            torch.nn.ReLU(inplace=True))

        self.mp_residual1 = gconv_residual(
            nin=128, nmed=64, win=64, wmed=32, netype=16, with_residual=with_residual)

        self.nconv2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, 1), SyncBatchNorm(128),
            torch.nn.ReLU(inplace=True))
        self.wconv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 1), SyncBatchNorm(128),
            torch.nn.ReLU(inplace=True))
        self.mp_conv2 = mp_conv_v2(128, 128, win=128, wout=128, nedge_types=16)
        self.nconv3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 1), SyncBatchNorm(256),
            torch.nn.ReLU(inplace=True))
        self.wconv3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 1), SyncBatchNorm(256),
            torch.nn.ReLU(inplace=True))

        self.mp_residual2 = gconv_residual(
            256, 128, 256, 128, 16, with_residual=with_residual)

        if self.with_gnn_decode:
            self.context_conv = torch.nn.Sequential(
                torch.nn.Conv2d(256 * 2, 256, 1),
                SyncBatchNorm(256),
                torch.nn.ReLU(inplace=True))
            self.mp_residual3 = gconv_residual(
                256, 128, 256, 128, 16, with_residual=with_residual)
            self.node_conv = torch.nn.Sequential(
                torch.nn.Conv2d(256, 128, 1),
                SyncBatchNorm(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(128, 64, 1),
                SyncBatchNorm(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 1, 1)
            )
        else:
            self.context_conv = torch.nn.Sequential(
                torch.nn.Conv2d(256 * 2, 256, 1),
                SyncBatchNorm(256),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(256, 128, 1),
                SyncBatchNorm(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(128, 64, 1),
                SyncBatchNorm(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 1, 1)
            )

    def forward(self, pts, pair_weight, nn_idx):
        # pts: [batch, input_feature_size, nnodes]
        # nn_idx: [batch, nnodes, knn_k]
        pts_knn = get_nn_node_feature(pts, nn_idx)  # pts_knn: [batch, input_feature_size, city_num, knn_k]
        # need check
        efeature = get_edge_feature(pts_knn, pts)  # efeature: [batch, input_feature_size * 2, city_num, knn_k]

        etype = self.etype_net(efeature, pair_weight)

        nfeature, wfeature = self.mp_conv1(
            pts.view(pts.shape[0], pts.shape[1], pts.shape[2], 1), pair_weight, nn_idx,
            etype)
        nfeature = self.nconv1(nfeature)
        wfeature = self.wconv1(wfeature)

        nfeature, wfeature = self.mp_residual1(nfeature, wfeature, etype, nn_idx)

        nfeature = self.nconv2(nfeature)
        wfeature = self.wconv2(wfeature)
        # nfeature_knn = get_nn_node_feature(nfeature, nn_idx)
        nfeature, wfeature = self.mp_conv2(nfeature, wfeature, nn_idx, etype)
        nfeature = self.nconv3(nfeature)
        wfeature = self.wconv3(wfeature)

        nfeature, wfeature = self.mp_residual2(nfeature, wfeature, etype, nn_idx)

        # decoding part
        batch_size = nfeature.size(0)
        global_feature, _ = nfeature.max(dim=2, keepdim=True)

        nfeature = torch.cat(
            [nfeature, global_feature.repeat(1, 1, nfeature.shape[2], 1)], dim=1)
        if self.with_gnn_decode:
            nfeature = self.context_conv(nfeature)
            nfeature, wfeature = self.mp_residual3(nfeature, wfeature, etype, nn_idx)
            probs = self.node_conv(nfeature).squeeze()
        else:
            probs = self.context_conv(nfeature).squeeze()
        if batch_size == 1:
            probs = probs.unsqueeze(0)
        return probs

