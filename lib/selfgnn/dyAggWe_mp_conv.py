import torch
from enum import Enum
# try:
#     from encoding.nn import SyncBatchNorm
# except:
SyncBatchNorm = torch.nn.BatchNorm2d


class etype_net(torch.nn.Module):
    def __init__(self, nedge_type, nmid_filter=64, nfeature_size=3, wfeature_size=1, wmid_filter=16):
        super(etype_net, self).__init__()
        self.nodeop = torch.nn.Sequential(
            torch.nn.Conv2d(nfeature_size * 2, nmid_filter, 1, 1),
            torch.nn.BatchNorm2d(nmid_filter),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(nmid_filter, nedge_type, 1, 1, bias=False))
        self.weightop = torch.nn.Sequential(
            torch.nn.Conv2d(wfeature_size, wmid_filter, 1, 1),
            torch.nn.BatchNorm2d(wmid_filter),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(wmid_filter, wmid_filter * 2, 1, 1),
            torch.nn.BatchNorm2d(wmid_filter * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(wmid_filter * 2, nedge_type, 1, 1, bias=False))

    def forward(self, edge_feature, edge_weight):
        # edge_feature: [batch, 2*nfeature_size, nnodes, knn_k]]
        # wdge_weight: [batch, nnodes, knn_k]
        nfeature = self.nodeop(edge_feature)
        efeature = self.weightop(edge_weight)

        return nfeature * efeature


class mp_conv_type(Enum):
    NO_EXTENSION = 0
    ORIG_WITH_NEIGHBOR = 1
    ORIG_WITH_DIFF = 2


class weight_dy_graph(torch.nn.Module):
    def __init__(self,
                 nin,
                 med,
                 win,
                 wout):
        super(weight_dy_graph, self).__init__()

        self.nin = nin
        self.med = med
        self.win = win
        self.wout = wout

        self.nodeCnn = torch.nn.Sequential(
            torch.nn.Conv2d(2 * self.nin, self.med // 2, 1, 1),
            SyncBatchNorm(self.med // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.med // 2, self.med, 1, 1)
        )
        self.weightCnnPlus = torch.nn.Sequential(
            torch.nn.Conv2d(self.win, self.med // 2, 1, 1),
            SyncBatchNorm(self.med // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.med // 2, self.med, 1, 1)
        )
        self.weightCnnPure = torch.nn.Sequential(
            torch.nn.Conv2d(self.win, self.med // 2, 1, 1),
            SyncBatchNorm(self.med // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.med // 2, self.med, 1, 1)
        )
        self.weightMix = torch.nn.Sequential(
            torch.nn.Conv2d(self.med, self.wout, 1, 1),
            SyncBatchNorm(self.wout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.wout, self.wout, 1, 1)
        )
    def forward(self, pair_node_feature, pair_weight):
        nfeature = self.nodeCnn(pair_node_feature)
        wfeature_og = self.weightCnnPure(pair_weight)
        wfeature_plus = self.weightCnnPlus(pair_weight)
        wfeature = nfeature * wfeature_plus
        wfeature = self.weightMix(wfeature_og + wfeature)
        return wfeature



class mp_conv_v2(torch.nn.Module):
    def __init__(self,
                 nin,
                 nout,
                 win,
                 wout,
                 nedge_types,
                 bias=True,
                 bn=True,
                 extension=mp_conv_type.ORIG_WITH_DIFF,
                 activation_fn='relu',
                 aggregtor='max'):
        super(mp_conv_v2, self).__init__()

        self.nin = nin
        self.nout = nout
        self.win = win
        self.wout = wout
        self.nedge_types = nedge_types
        self.extension = extension

        self.wdyCnn = weight_dy_graph(nin, 3*nin, win, wout)

        self.nodeCnn = torch.nn.Sequential(
            torch.nn.Conv2d(2*self.nin, self.nout // 2, 1, 1),
            SyncBatchNorm(self.nout // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.nout // 2, self.nout, 1, 1)
        )
        self.weightCnn = torch.nn.Sequential(
            torch.nn.Conv2d(self.win, self.wout // 2, 1, 1),
            SyncBatchNorm(self.wout // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.wout // 2, self.wout, 1, 1)
        )
        self.wfilter_node = torch.nn.Parameter(torch.zeros(self.wout, self.nout, dtype=torch.float32))
        self.wfilter_node.data.uniform_(-0.01, 0.01)
        self.nfilter = torch.nn.Parameter(torch.zeros(self.nin, self.nout * self.nedge_types,
                                                      dtype=torch.float32))
        self.nfilter.data.uniform_(-0.01, 0.01)

        if self.extension == mp_conv_type.NO_EXTENSION:
            self.filters = torch.nn.Parameter(
                torch.zeros(nin, nout, nedge_types, dtype=torch.float32))
            self.filters.data.uniform_(-0.01, 0.01)

        elif self.extension == mp_conv_type.ORIG_WITH_DIFF or self.extension == mp_conv_type.ORIG_WITH_NEIGHBOR:
            self.filters1 = torch.nn.Parameter(
                torch.zeros(nout, nout, nedge_types, dtype=torch.float32))
            self.filters2 = torch.nn.Parameter(
                torch.zeros(nout, nout, nedge_types, dtype=torch.float32))
            self.filters = [self.filters1, self.filters2]
            for f in self.filters:
                f.data.uniform_(-0.01, 0.01)
        else:
            raise ValueError("extension must one of mp_conv_type")

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.nout))
            self.bias.data.uniform_(0, 0.05)
        else:
            self.bias = None

        if bn:
            self.bn = SyncBatchNorm(self.nout)
        else:
            self.bn = None
        if isinstance(activation_fn, torch.nn.Module):
            self.activation_fn = activation_fn
        elif activation_fn == 'relu':
            self.activation_fn = torch.nn.ReLU(inplace=True)
        else:
            self.activation_fn = None

        if isinstance(aggregtor, str):
            if aggregtor == 'max':

                def agg_max(node_feature):
                    res, *_ = torch.max(node_feature, dim=3, keepdim=True)
                    return res

                self.aggregtor = agg_max
            elif aggregtor == 'mean':
                self.aggregtor = lambda node_feature: torch.mean(node_feature, dim=3, keepdim=True)

        else:
            self.aggregtor = aggregtor

    def gather_pair_node_feature(self, node_feature, nn_idx):
        batch_size = node_feature.size(0)
        k = nn_idx.size(2)
        npts = nn_idx.view(batch_size, -1).unsqueeze(1).repeat(1, node_feature.size(1), 1)
        nknn = torch.gather(node_feature.squeeze(3), 2, npts)\
            .view(batch_size, node_feature.size(1), node_feature.size(2), -1)
        nself = node_feature.repeat(1, 1, 1, k)
        return torch.cat((nself, nknn), 1)  # [batch, 2*node_feature_size, city_num, k]

    def forward(self, node_feature, pair_weight, nn_idx, etype):
        # node_feature: [batch ,input_feature_size, city_num, 1]
        # nn_idx : [batch, ciy_num, k]
        # pair_weight :[ batch, 1, city_num, k]
        # etype :[ batch, 16, city_num, k]
        batch_size, nin, nnodes = node_feature.size(0), node_feature.size(1), node_feature.size(2)
        k = nn_idx.shape[2]
        paired_node_feature = self.gather_pair_node_feature(node_feature, nn_idx).view(batch_size, -1, nnodes, k)

        final_weight = self.wdyCnn(paired_node_feature, pair_weight.view(batch_size, -1, nnodes, k))

        paired_node_feature = self.nodeCnn(paired_node_feature)
        pair_weight = self.weightCnn(pair_weight.view(batch_size, -1, nnodes, k))
        mediate_weight = pair_weight.permute(0, 2, 3, 1).contiguous().view(batch_size * nnodes * k, -1)\
            .matmul(self.wfilter_node).view(batch_size, nnodes, k, -1).permute(0, 3, 1, 2)
        paired_node_feature = paired_node_feature * mediate_weight
        # paired_node_feature: [batch, nout, nnodes, k ]
        paired_node_feature = paired_node_feature.permute(0, 2, 3, 1).contiguous().view(
            batch_size * nnodes * k, self.nout)
        # node_feature: [batch, nnodes, k, nout * etype_size]

        nfeature = node_feature.permute(0, 2, 3, 1).contiguous().view(-1, self.nin)\
            .matmul(self.nfilter).view(batch_size, nnodes, 1, self.nout * self.nedge_types)
        efeature = paired_node_feature.matmul(self.filters[1].view(
            self.nout, self.nout * self.nedge_types)).view(
            batch_size, nnodes, k, self.nout * self.nedge_types)

        nedge_type = etype.permute(0, 2, 3, 1).contiguous().view(
            -1, self.nedge_types, 1)
        edge_feature = (efeature + nfeature).view(
            -1, self.nout, self.nedge_types).bmm(nedge_type).view(
            batch_size, nnodes, k, self.nout)

        nfeature = edge_feature.permute(0, 3, 1, 2).contiguous()
        if self.bias is not None:
            nfeature = nfeature + self.bias.view(1, self.nout, 1, 1)

        if self.aggregtor is not None:
            nfeature = self.aggregtor(nfeature)

        if self.bn is not None:
            nfeature = self.bn(nfeature)

        if self.activation_fn is not None:
            nfeature = self.activation_fn(nfeature)

        return nfeature, final_weight
