import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lib.data.get_arbi_data_with_neighbor_index import gendata
from tensorboardX import SummaryWriter
from lib.selfgnn.dyAggWe_tsp_coder import tsp_coder as tspmodel
from lib.validation import validation_arbitrary_success_ratio_gap
from lib.validation import validation_loss_tour
from lib.validation import mask_validate_one_step
from lib.validation import validation_mask_BS_one_step
from tqdm import tqdm
from options import get_options
import time

USE_CUDA = True
modelpath = os.path.join(os.getcwd(), "sym_model")
device = torch.device("cuda:0")
datapath = '../dataset/arbitrary_graph'

template_train_str = '{:<18} {:<8} {:<10} {:<10} {:<10} {:<8} {:<8}'
template_train_out = '{:<18} {:<8} {:<10} {:<10.5f} {:<10.5f} {:<8} {:<8}'
info_train = ['TRAIN', 'epoch', 'icnt', 'entropy', 'acc', 'city', 'edge']

class TrainModleTSP(nn.Module):
    def __init__(self, start_city_num, end_city_num,
                 load=False,
                 objective='min_sum_nonEuclid',
                 _modelpath=modelpath,
                 _device=device,
                 sparse=0.5):
        super(TrainModleTSP, self).__init__()

        self.alpha = 0
        self.load = load
        self.device = _device
        self.sparse = sparse
        self.model = tspmodel(nodeFeature=8,
                              weightFeature=1,
                              with_global=True,
                              with_gnn_decode=True,
                              dropout=False)
        self.model.to(self.device)
        self.icnt = 0
        self.val = 0
        self.valForTrain = 0
        self.val_loss_testset = 0
        self.objective = objective
        self.modelpath = _modelpath
        self.model_name = "rmix_neighbor_sp{}_{}_{}_obj_{}".format(self.sparse, start_city_num, end_city_num, self.objective)
        self.writer = SummaryWriter('mixruns/{}'.format(self.model_name))

        self.datapath = os.path.join(datapath, self.objective)
        self.modelfile = os.path.join(self.modelpath, '{}.pt'.format(self.model_name))
        self.loss = 0

        self.start_city_num = start_city_num
        self.end_city_num = end_city_num

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_func = nn.CrossEntropyLoss()

        self.epoch = 0
        self.city_num = self.start_city_num

        if load:
            print("loading model {}".format(self.modelfile))
            if os.path.exists(self.modelfile):
                print("loading model:{}".format(self.modelfile))
                checkpoint = torch.load(self.modelfile, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.icnt = checkpoint['icnt'] + 1
                self.val = checkpoint['ival']
                self.valForTrain = self.val
                self.val_loss_testset = self.val
                self.epoch = checkpoint['epoch']
                self.city_num = checkpoint['city_num']
                self.edge_num = checkpoint['edge_num']
                print("Model loaded")
                print("cnum = ", self.city_num)
                print("icnt = ", self.icnt)
            else:
                print("No Model loaded")

    def dataExisting(self, agent_num, city_num, edge_num):
        path_temp = os.path.join(self.datapath, 'agent{}/city{}/edge{}'.format(agent_num, city_num, edge_num))
        if os.path.exists(path_temp):
            return True
        else:
            return False

    def train(self, n_epoch=10, k=10, batch_size=32):
        for epoch in range(self.epoch, self.epoch + n_epoch):
            for city_num in range(self.city_num, self.end_city_num + 1):
                if 'nonsym' in self.objective:
                    edge_num = int(city_num * (city_num - 1) * self.sparse)
                else:
                    edge_num = int(city_num * (city_num - 1) * 0.5 * self.sparse)
                # path = self.dataExisting(1, city_num, edge_num)
                # if path is False:
                #     continue

                # real_k = min(k, int(city_num * self.sparse + 5))
                if self.sparse > 0.3:
                    print("strptimes = 1")
                    train_set = gendata(agent_num=1, city_num=city_num, edge_num=edge_num,
                                        k=k,
                                        activate='train',
                                        path=self.datapath,
                                        straptimes=8)
                else:
                    print("strptimes = 2")
                    train_set = gendata(agent_num=1, city_num=city_num, edge_num=edge_num,
                                        k=k,
                                        activate='train',
                                        path=self.datapath,
                                        straptimes=2)
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
                start_time = time.time()
                for batch_id, sample in enumerate(train_loader):
                    self.model.train()
                    relation, target, weight, knn = sample
                    relation, target, weight, knn = \
                        relation.to(self.device), target.to(self.device), \
                        weight.to(self.device), knn.to(self.device)

                    loss, acc = self.train_for_squence(relation, weight, target, knn)
                    self.loss = torch.mean(loss)
                    ratio = torch.mean(acc)
                    if self.icnt % 50 == 0 and self.icnt != 0:
                        self.model_name = "rmix_neighbor_sp{}_{}_{}_obj_{}".format(self.sparse, self.start_city_num,
                                                                          self.end_city_num,
                                                                          self.objective)
                        self.modelfile = os.path.join(self.modelpath, '{}.pt'.format(self.model_name))
                        torch.save({'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'icnt': self.icnt,
                                    'ival': self.val,
                                    'city_num': city_num,
                                    'edge_num': edge_num,
                                    'epoch': epoch
                                    }, self.modelfile)
                        print("------------------")
                        print("saved model: {}".format(self.modelfile))
                        print("------------------")

                    if self.icnt % 500 == 0:
                        beamvalue = 1
                        self.eval_sparse(city_num, city_num, max(1, batch_size // beamvalue), 10, sp=self.sparse, beam=beamvalue,
                                        gap_clip=10)
                    if self.icnt % 1000 == 0:
                        beamvalue = 5
                        self.eval_sparse(city_num, city_num, max(1, batch_size // beamvalue), 10, sp=self.sparse,
                                         beam=beamvalue,
                                         gap_clip=10)

                    with torch.no_grad():
                        self.writer.add_scalar('train/loss', self.loss.item(), self.icnt)
                        self.writer.add_scalar('train/acc', ratio, self.icnt)
                        self.writer.add_scalar('train/loss{}'.format(city_num), self.loss.item(), self.icnt)
                        self.writer.add_scalar('train/acc{}'.format(city_num), ratio, self.icnt)
                        # print train condition
                        if self.icnt % 30 == 0:
                            out = ['rmix_{}_s{}_{}_{}'.format(self.alpha, self.sparse, self.start_city_num,
                                                              self.end_city_num),
                                   epoch, self.icnt, self.loss, ratio, city_num, edge_num]
                            print(template_train_str.format(*info_train))
                            print(template_train_out.format(*out))
                        self.icnt += 1
                del train_set, train_loader
                gc.collect()
                print("time usage: ", time.time() - start_time)

    def train_for_squence(self, relation, weight, target, knn):
        # generate another direction target
        batch_size, city_num = relation.size(0), relation.size(1)
        target2dir = torch.zeros(batch_size, 2, city_num + 1).long().to(self.device)
        target2dir[:, 0, :] = target.long()
        target2dir[:, 1, :] = self.generate_another_direction_target(target.long()).long().to(self.device)
        direction = 2

        # add another 3 feature for in_features
        nodeFeature = self.clear_input(relation)

        # deal with depot specially:
        # for 3 other features: 0: last city; 1: current city; 2:depot
        nodeFeature[:, 0, 4] = 1.0
        nodeFeature[:, 0, 5] = 0.0
        # adding input to model
        # model parameter need to change to 5, now it is 2
        pred = self.model(nodeFeature.permute(0, 2, 1),
                          weight.unsqueeze(1),
                          knn)

        sumloss = torch.zeros(direction).to(self.device)
        acc = torch.zeros(direction).to(self.device)

        for d in range(direction):
            sumloss[d], acc[d] = self.compute_train_loss_acc(pred, target2dir[:, d],
                                                             looptimes=0, prev_tour=[target2dir[:, d, :1]])
        con_loss = self.compute_train_connected_loss(relation, target[:, 0], pred)
        sumloss += self.alpha * con_loss
        depot_loss = torch.sum(sumloss)
        self.optimizer.zero_grad()
        depot_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        # label current city and last city
        for cnt in range(direction):
            prev_tour = []
            last_city = target2dir[:, cnt, 0]
            current_city = target2dir[:, cnt, 1]
            prev_tour.append(last_city)
            prev_tour.append(current_city)
            for c in range(1, city_num):
                nodeFeature = self.clear_input(relation)
                nodeFeature[[bs for bs in range(batch_size)], last_city, 2] = 1.0
                nodeFeature[[bs for bs in range(batch_size)], last_city, 3] = 0.0
                nodeFeature[[bs for bs in range(batch_size)], current_city, 4] = 1.0
                nodeFeature[[bs for bs in range(batch_size)], current_city, 5] = 0.0
                for prev in range(c - 1):
                    previous_city = target2dir[:, cnt, prev]
                    nodeFeature[[bs for bs in range(batch_size)], previous_city, 6] = 1.0
                    nodeFeature[[bs for bs in range(batch_size)], previous_city, 7] = 0.0
                pred = self.model(nodeFeature.permute(0, 2, 1),
                                  weight.unsqueeze(1),
                                  knn)
                loss, acc_loop = self.compute_train_loss_acc(pred, target2dir[:, cnt], looptimes=c,
                                                             prev_tour=prev_tour)
                con_loss = self.compute_train_connected_loss(relation, current_city, pred)
                acc[cnt] += acc_loop
                sumloss[cnt] += (self.alpha * con_loss + loss)
                self.optimizer.zero_grad()
                (self.alpha * con_loss + loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
                last_city = target2dir[:, cnt, c]
                current_city = target2dir[:, cnt, c + 1]
                prev_tour.append(current_city)
        return sumloss / city_num, acc / (batch_size * city_num)

    def eval_sparse(self, min_n, max_n, batch_size=32, knn_k=10, sp=0.5, beam=1, gap_clip=10):
        with torch.no_grad():
            snum = []
            sgap = []
            ttnum = 0
            stime = []
            sextend = []
            for city_num in range(min_n, max_n+1, 1):
                print("testing...", city_num)
                enum = int(city_num * (city_num - 1) * 0.5 * sp)
                test_set = gendata(agent_num=1, city_num=city_num, edge_num=enum,
                                   k=knn_k,
                                   activate='test',
                                   path=self.datapath,
                                   straptimes=0)
                if test_set.size <= 0:
                    continue
                ttnum += test_set.size
                dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8)
                for batch_id, sample in enumerate(tqdm(dataloader)):
                    relation, target, weight, knn = sample
                    relation, target, weight, knn = \
                        relation.to(self.device), target.to(self.device), \
                        weight.to(self.device), knn.to(self.device)
                    with torch.no_grad():
                        start = time.time()
                        # (model, relation, weight, target, knn, beam, device='cuda:0'):
                        if beam > 1:
                            extend_num = 0
                            success_num, success_gap, total_gap = \
                                validation_mask_BS_one_step(self.model, relation, weight, target, knn, beam,
                                                            self.device,
                                                            gap_clip=gap_clip)
                        else:
                            # success_num, success_gap, total_gap = \
                            #     self.validation_for_squence(relation, weight, target, knn, mask=True)
                            success_num, success_gap, total_gap, extend_num = \
                                self.validation_for_squence(relation, weight, target, knn, mask=True, gap_clip=gap_clip)
                        stime.append(time.time() - start)
                        snum.append(success_num)
                        sgap.append(success_gap)
                        sextend.append(extend_num)
                del test_set, dataloader
                gc.collect()

            sgap = torch.tensor(sgap)
            snum = torch.tensor(snum)
            # stime = torch.tensor(stime)
            # sextend = torch.tensor(sextend)
            # filename = os.path.join(os.getcwd(), 'test_result/test_gnn_sparse_withBS.txt')
            # if not os.path.exists(filename):
            #     f = open(filename, 'w+')
            #     f.close()
            # f = open(filename, 'a+')
            # f.write("GNN complete model: {} \n".format(self.modelfile))
            # f.write("\ttesting on dataset from {} - {} with sparse = {}\n".format(min_n, max_n, sp))
            # f.write("\t******beam = {}\n".format(beam))
            # f.write("\t******gap_clip = {}\n".format(gap_clip))
            # f.write("\ttotal number : {}  success num : {},   with success ratio = {}%\n"
            #         .format(ttnum, torch.sum(snum), torch.sum(snum) * 100.0 / ttnum))
            # f.write('\taverage gap = {}\n'.format(torch.sum(sgap) / torch.sum(snum)))
            # f.write('\taverage time = {}\n'.format(torch.sum(stime) / ttnum))
            #
            # f.write('\taverage extend node num: {}/{}={}\n\n\n'
            #         .format(torch.sum(sextend), torch.sum(snum), torch.sum(sextend) / torch.sum(snum)))
            # f.close()
            print("testing on dataset from {} - {} on model {} with beam = {} finish"
                  .format(min_n, max_n, self.modelfile, beam))
            print("success/total num = {}/{}".format(torch.sum(snum), ttnum))
            print('average gap = {}'.format(torch.sum(sgap) / torch.sum(snum)))
            print("-------------------------------------------\n")

    def validation_for_squence(self, relation, weight, target, knn, mask=False, gap_clip=10):
        extend_num = 0
        batch_size, city_num = relation.size(0), relation.size(1)
        # add another 3 feature for in_features
        nodeFeature = self.clear_input(relation)

        # deal with depot specially:
        # for 3 other features: 0: last city; 1: current city; 2:depot
        nodeFeature[:, 0, 4] = 1.0
        nodeFeature[:, 0, 5] = 0.0
        pred_tour = []
        pred_tour.append(target[:, 0])
        pred = self.model(nodeFeature.permute(0, 2, 1),
                          weight.unsqueeze(1),
                          knn)

        if mask is False:
            tour = validation_loss_tour(pred, pred_tour, self.device)
        else:
            # succeed, relation, weight, target, knn, pred_tour, tour = \
            #     mask_validate_one_step(relation, weight, target, knn, pred_tour, pred, self.device)
            succeed, relation, weight, target, knn, pred_tour, tour, extend_num = \
                mask_validate_one_step(relation, weight, target, knn, pred_tour, pred, self.device,
                                       extend_num=extend_num)
            if succeed is False:
                return 0.0, 0.0, 0.0, 0.0
        pred_tour.append(tour)

        for c in range(1, city_num):
            batch_size = relation.size(0)
            last_city = pred_tour[len(pred_tour) - 2]
            current_city = pred_tour[len(pred_tour) - 1]
            nodeFeature = self.clear_input(relation)
            nodeFeature[[bs for bs in range(batch_size)], last_city, 2] = 1.0
            nodeFeature[[bs for bs in range(batch_size)], last_city, 3] = 0.0
            nodeFeature[[bs for bs in range(batch_size)], current_city, 4] = 1.0
            nodeFeature[[bs for bs in range(batch_size)], current_city, 5] = 0.0
            for prev in range(c - 1):
                previous_city = pred_tour[prev]
                nodeFeature[[bs for bs in range(batch_size)], previous_city, 6] = 1.0
                nodeFeature[[bs for bs in range(batch_size)], previous_city, 7] = 0.0
            pred = self.model(nodeFeature.permute(0, 2, 1),
                              weight.unsqueeze(1),
                              knn)

            if mask is False:
                tour = validation_loss_tour(pred, pred_tour, self.device)
            else:
                # succeed, relation, weight, target, knn, pred_tour, tour = \
                #     mask_validate_one_step(relation, weight, target, knn, pred_tour, pred, self.device)
                succeed, relation, weight, target, knn, pred_tour, tour, extend_num = \
                    mask_validate_one_step(relation, weight, target, knn, pred_tour, pred, self.device,
                                           extend_num=extend_num)
                if succeed is False:
                    return 0.0, 0.0, 0.0, 0.0
            pred_tour.append(tour)
        success_ratio, success_gap, total_gap = \
            validation_arbitrary_success_ratio_gap(relation, pred_tour, target,
                                                   mask=mask, device=self.device, gap_clip=gap_clip)
        return success_ratio, success_gap, total_gap, torch.sum(extend_num)

    def clear_input(self, relation):
        batch_size, city_num = relation.size(0), relation.size(1)
        # clear
        # 0: is depot, 2: is last, 4: is current, 6: is traveled
        inputs = torch.ones(batch_size, city_num, 8).to(self.device)
        # keep depot information
        inputs[:, :, 0] = 0.0  # not depot
        inputs[:, :, 1] = 1.0
        inputs[:, 0, 0] = 1.0  # 0 is depot
        inputs[:, 0, 1] = 0.0
        inputs[:, :, 2] = 0.0
        inputs[:, :, 3] = 1.0
        inputs[:, :, 4] = 0.0
        inputs[:, :, 5] = 1.0
        inputs[:, :, 6] = 0.0
        inputs[:, :, 7] = 1.0
        return inputs

    def generate_another_direction_target(self, target):
        city_num = target.size(1)
        another_target = target.detach().clone()
        fromend = -1
        for c in range(city_num):
            another_target[:, c] = target[:, fromend]
            fromend -= 1
        return another_target

    def normalize_pred(self, pred):
        norm = torch.sum(torch.mul(pred, pred), dim=-1).unsqueeze(2).sqrt().expand_as(pred).float()
        return pred.div(norm)

    def compute_train_connected_loss(self, relation, current, pred):
        batch_size, city_num = relation.size(0), relation.size(1)
        loss = torch.zeros(batch_size).to(self.device)
        for bs in range(batch_size):
            for c in range(city_num):
                if relation[bs, current[bs], c] < 1:
                    loss[bs] += self.loss_func(pred[bs].unsqueeze(0), torch.tensor(c).unsqueeze(0).to(self.device))
        return loss.mean()

    def compute_train_loss_acc(self, pred, target, looptimes, prev_tour=None):
        batch_size =pred.size(0)
        nnode = pred.size(1)
        if prev_tour is not None:
            prev_times = len(prev_tour)
            mask = torch.ones(batch_size, nnode).to(self.device)
            for old_idx in prev_tour:
                mask[[bs for bs in range(batch_size)], old_idx.view(-1)] = 0.0
            if prev_times >= nnode:
                mask[[bs for bs in range(batch_size)], 0] = 1.0
            pred = pred * mask
        loss = self.loss_func(pred, target[:, looptimes + 1])

        # compute acc
        _, next_city = torch.topk(pred, 1, dim=-1)
        acc = (target[:, looptimes + 1] == next_city.squeeze().long()).nonzero().size()[0]
        return loss, acc


if __name__ == '__main__':
    print(os.getcwd())
    opts = get_options()
    aim = opts.aim
    if opts.load is 'False':
        load = True
    else:
        load = False
    batch_size = opts.batch_size
    if opts.cuda == 0:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:1")

    model_start = opts.start_city_num
    model_end = opts.end_city_num
    model_sparse = opts.sparse
    if aim is 'train':
        object = 'min_sum_nonEuclid_sym'
        start_city_num = model_start
        end_city_num = model_end
        sp = model_sparse
        print(start_city_num, end_city_num, 'load={}, batch={}'.format(load, batch_size))
        tsp = TrainModleTSP(start_city_num=start_city_num,
                            end_city_num=end_city_num,
                            load=load,
                            _device=device,
                            objective=object,
                            sparse=sp)
        tsp.train(5, k=10, batch_size=batch_size)
    else:
        object = 'min_sum_nonEuclid_sym'
        start_city_num = model_start
        end_city_num = model_end
        modelsp = model_sparse

        min_n = opts.test_min_num
        max_n = opts.test_max_num
        testsp = opts.test_sparse
        beamvalue = opts.beamvalue
        gap_clip = opts.gap_clip
        print(start_city_num, end_city_num, 'load={}, batch={}'.format(load, batch_size))
        tsp = TrainModleTSP(start_city_num=start_city_num,
                            end_city_num=end_city_num,
                            load=load,
                            _device=device,
                            objective=object,
                            sparse=modelsp)
        tsp.eval_sparse(min_n, max_n, max(1, batch_size // beamvalue), 10, sp=testsp, beam=beamvalue, gap_clip=gap_clip)







