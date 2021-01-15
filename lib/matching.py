import torch
import numpy as np
from .layers import node_cnn_module, node_cnn_lstm_module, node_fc_layer, stack_model
from .utils import edge_feature_to_node_feature
from .layers import map_edge_feature_to_node, map_node_feature_to_edge, map_features
from .auction import auction_lap


class MatchingModel(torch.nn.Module):
    def __init__(self, op_type, nfeatures):
        super(MatchingModel, self).__init__()
        self.layers = []
        if op_type == 'fc':
            L = node_fc_layer
        else:
            def L(x, y, **args): return node_cnn_module(x, y, 3, **args)

        for i in range(1, len(nfeatures)):
            args = dict()
            if(i == len(nfeatures) - 1):
                args['activation'] = None
            if(i == len(nfeatures) - 2):
                args['use_batchnorm'] = True

            self.layers.append(L(nfeatures[i-1], nfeatures[i], **args))

        self.main = stack_model(*self.layers)

        # for k, v in self.main.named_parameters():
        #     self.register_parameter(k, v)

    def forward(self, feature, eadj):
        return self.main(feature, eadj)


class GMSolver(torch.nn.Module):
    def __init__(self, nfeature_input_mapping,
                 nfeature_lstm,
                 nfeature_post_process,
                 softmax=True):
        super(GMSolver, self).__init__()
        self.softmax = softmax
        self.feature_mapping_module = node_cnn_module(
            8, nfeature_input_mapping, 3, False)
        self.efeature_mapping_module = node_cnn_module(
            6, nfeature_input_mapping, 3, False)

        self.register_parameter_submodule(
            self.feature_mapping_module, 'feature_mapping_module')
        self.register_parameter_submodule(
            self.efeature_mapping_module, 'efeature_mapping_module')

        self.lstm_modules = []
        self.elstm_modules = []
        self.nfeature_lstm = nfeature_lstm

        for i in range(len(nfeature_lstm)):
            if i == 0:
                input_size = nfeature_input_mapping[-1] * 3
            else:
                input_size = nfeature_lstm[i - 1]

            # print(input_size, nfeature_lstm, [], 3)
            lstm_module = node_cnn_lstm_module(
                input_size, nfeature_lstm[i], [], 3)
            elstm_module = node_cnn_lstm_module(
                input_size, nfeature_lstm[i], [], 3)

            self.register_parameter_submodule(
                lstm_module, 'lstm_module_%d' % i)
            self.register_parameter_submodule(
                elstm_module, 'elstm_module_%d' % i)

            self.lstm_modules.append(lstm_module)
            self.elstm_modules.append(elstm_module)

        if(nfeature_post_process[-1] != 2):
            nfeature_post_process.append(2)

        nfeature_post_process[-1] = 1
        self.post_process_module = node_cnn_module(
            nfeature_lstm[-1] * 3, nfeature_post_process, 3, True)
        nfeature_post_process[-1] = 2
        self.epost_process_module = node_cnn_module(
            nfeature_lstm[-1] * 3, nfeature_post_process, 3, True)

        self.register_parameter_submodule(
            self.post_process_module, 'post_process_module')
        self.register_parameter_submodule(
            self.epost_process_module, 'epost_process_module')

    def register_parameter_submodule(self, submodule, prefix):
        for k, v in submodule.named_parameters():
            # print('%s/%s' % (prefix, k))
            self.register_parameter('%s/%s' % (prefix, k), v)

    def generate_init_state(self, nnode, nedges):
        nmems = []
        emems = []

        for i in range(len(self.nfeature_lstm)):
            nmem = torch.cuda.FloatTensor(
                np.zeros([nnode, self.nfeature_lstm[i] * 2, nnode]))
            emem = torch.cuda.FloatTensor(
                np.zeros([nedges, self.nfeature_lstm[i] * 2, nnode])
            )
            nmems.append(nmem)
            emems.append(emem)

        return nmems, emems

    def get_current_dual(self, nnode, bi, bij,  v, e_to_n, epsilon=20):
        sum_uv = torch.sum(v)

        if self.softmax:
            max_bi = torch.logsumexp(
                bi.view(-1, nnode) * epsilon, dim=1).sum() / epsilon
            max_bij = torch.logsumexp(
                bij.view(-1, nnode * nnode) * epsilon, dim=1).sum() / epsilon
        else:
            max_bi, _ = torch.max(bi.view(-1, nnode), dim=1)
            max_bij, _ = torch.max(bij.view(-1, nnode * nnode), dim=1)
            max_bi = max_bi.sum()
            max_bij = max_bij.sum()

        return sum_uv + max_bi + max_bij

    def get_dual(self, nnode, bi, bij, v, e_to_n):
        # sum_uv = torch.sum(u) + torch.sum(v)

        nbi = bi + v
        # print(nbi.shape)
        score, decoding, counter, v = auction_lap(nbi.view(nnode, nnode))

        v = v.view([1, 1, -1])

        # print(v.shape)
        # print(u.shape)
        nbi = nbi - v

        nbij = bij

        # print(nbi.view(nnode, nnode).max(dim=1))
        max_bi, _ = nbi.view(nnode, nnode).max(dim=1)
        sum_bi = max_bi.sum()
        max_bij, _ = nbij.view(-1, nnode * nnode).max(dim=1)
        sum_bij = max_bij.sum()

        res = v.sum() + sum_bi + sum_bij

        return res, decoding, v

    def get_init_msg(self, nnode, nedge):
        v = torch.cuda.FloatTensor(np.zeros([1, 1, nnode]))

        msgi = torch.cuda.FloatTensor(np.zeros([nedge, 1, nnode]))
        msgj = torch.cuda.FloatTensor(np.zeros([nedge, 1, nnode]))

        return v, msgi, msgj

    def primal(self, node_potentials, edge_potentials, decoding, n_to_e):
        node_score = torch.squeeze(node_potentials).gather(
            dim=1, index=decoding.view(-1, 1)).sum()
        nofnodes = node_potentials.shape[0]

        xi = map_features(decoding.view(-1, 1).float(), n_to_e[0]).long()
        xj = map_features(decoding.view(-1, 1).float(), n_to_e[1]).long()

        xij = int(nofnodes) * xi + xj

        ep = edge_potentials.view(edge_potentials.shape[0], -1)

        edge_score = ep.gather(dim=1, index=xij.view(-1, 1)).sum()

        return node_score + edge_score

    def partial_feature(self, nnodes, edge_potentials, decoding, n_to_e, e_to_n):
        decoding_mat = torch.cuda.FloatTensor(np.zeros([nnodes, nnodes]))
        decoding_mat.scatter_(1, decoding.view(-1, 1), 1)

        xi = map_features(decoding_mat, n_to_e[0]).view(-1, nnodes, 1)
        xj = map_features(decoding_mat, n_to_e[1]).view(-1, 1, nnodes)

        ep = edge_potentials.view(-1, nnodes, nnodes)

        p1 = (ep * xi).sum(dim=1)
        p2 = (ep * xj).sum(dim=2)

        f1 = map_features(p1, e_to_n[0])
        f2 = map_features(p2, e_to_n[1])

        f1 = f1.view(nnodes, 1, nnodes).detach()
        f2 = f2.view(nnodes, 1, nnodes).detach()

        return f1, f2

    def forward(self, bi, bij, msgi, msgj, v, nmems, emems, neadj, eeadj, e_to_n, n_to_e, decoding, epsilon=20):

        ntoefeature1, ntoefeature2 = map_node_feature_to_edge(
            bi, *n_to_e, True, True)

        if(self.softmax):
            cmsgi = torch.logsumexp(
                (ntoefeature2 + bij) * epsilon, dim=3) / epsilon
            cmsgj = torch.logsumexp(
                (ntoefeature1 + bij) * epsilon, dim=2) / epsilon

        else:
            cmsgi, _ = torch.max(ntoefeature1 + bij, dim=3)
            cmsgj, _ = torch.max(ntoefeature2 + bij, dim=2)

        ncmsgi = map_features(cmsgi, e_to_n[0])
        ncmsgj = map_features(cmsgj, e_to_n[1])

        nnmsgi = map_features(msgi, e_to_n[0])
        nnmsgj = map_features(msgj, e_to_n[1])

        f1, f2 = self.partial_feature(
            int(bi.shape[0]), bij, decoding, n_to_e, e_to_n)

        node_feature = torch.cat(
            [bi, bi + v, nnmsgi, nnmsgj, ncmsgi, ncmsgj, f1, f2], dim=1)

        nnode = int(bi.shape[0])

        edge_feature = torch.cat(
            [msgi, msgj, cmsgi, cmsgj, ntoefeature1.view(-1, 1, nnode), ntoefeature2.view(-1, 1, nnode)], dim=1
        )

        nfeature = self.feature_mapping_module(
            node_feature, neadj)
        efeature = self.efeature_mapping_module(
            edge_feature, eeadj)

        # ntoe = map_node_feature_to_edge(nfeature, *n_to_e)

        ntoe1, ntoe2 = map_features(
            nfeature, n_to_e[0]),  map_features(nfeature, n_to_e[1])
        eton1, eton2 = map_features(
            efeature, e_to_n[0]), map_features(efeature, e_to_n[1])

        # eton = map_edge_feature_to_node(efeature, *e_to_n, False)

        nfeature = torch.cat([nfeature, eton1, eton2], dim=1)
        efeature = torch.cat([efeature, ntoe1, ntoe2], dim=1)

        cnmems = []
        cemems = []
        for idx, m in enumerate(self.lstm_modules):
            nmem = nmems[idx]
            emem = emems[idx]

            em = self.elstm_modules[idx]

            nfeature_channels = self.nfeature_lstm[idx]

            # print(efeature.size())
            # print(emem.size())

            nfeature = m(
                nfeature, nmem, neadj)
            efeature = em(efeature, emem, eeadj)

            cnmems.append(nfeature)
            cemems.append(efeature)

            nfeature = torch.split(
                nfeature, [nfeature_channels, nfeature_channels], 1)

            nfeature = nfeature[1]

            efeature = torch.split(
                efeature, [nfeature_channels, nfeature_channels], 1)

            efeature = efeature[1]

        ntoe1, ntoe2 = map_features(
            nfeature, n_to_e[0]),  map_features(nfeature, n_to_e[1])
        eton1, eton2 = map_features(
            efeature, e_to_n[0]), map_features(efeature, e_to_n[1])

        # eton = map_edge_feature_to_node(efeature, *e_to_n, False)

        nfeature = torch.cat([nfeature, eton1, eton2], dim=1)
        efeature = torch.cat([efeature, ntoe1, ntoe2], dim=1)

        # ntoe = map_node_feature_to_edge(nfeature, *n_to_e)
        # eton = map_edge_feature_to_node(efeature, *e_to_n, False)

        # efeature = torch.cat([efeature, ntoe], dim=1)
        # nfeature = torch.cat([nfeature, eton], dim=1)

        efeature = self.epost_process_module(efeature, eeadj)
        nfeature = self.post_process_module(nfeature, neadj)

        # print(efeature.size())

        nmsgi, nmsgj = torch.split(efeature, [1, 1], dim=1)

        nmsgi += 0.5 * cmsgi - msgi
        nmsgj += 0.5 * cmsgj - msgj

        nv = nfeature.mean(dim=0, keepdim=True)

        v = v + nv

        #nmsgi += 0.5 * cmsgi
        #nmsgj += 0.5 * cmsgj

        bij = bij - nmsgi.view(-1, 1, int(bi.shape[0]), 1) - \
            nmsgj.view(-1, 1, 1, int(bi.shape[0]))

        nnmsgi = map_features(nmsgi, e_to_n[0])
        nnmsgj = map_features(nmsgj, e_to_n[1])

        bi = bi + nnmsgi + nnmsgj - nv

        msgi = msgi + nmsgi
        msgj = msgj + nmsgj

        cdual = self.get_current_dual(int(bi.shape[0]),
                                      bi,
                                      bij,
                                      v,
                                      e_to_n)

        return bi, bij, msgi, msgj, v, cdual, cnmems, cemems, f1, f2
