import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from math import fabs

maxDateValue = 15000
testDataSize = 1000

class gendata(Dataset):
    def __init__(self, agent_num, city_num, edge_num,
                 path,
                 activate='train',
                 k=10,
                 straptimes=7,
                 sampleNum=maxDateValue):
        super(gendata, self).__init__()
        setpath = os.path.join(path, "{}/agent{}/city{}/edge{}".format(activate, agent_num, city_num, edge_num))
        print(setpath)
        self.dataset = []
        if os.path.exists(setpath):
            self.activate = activate
            self.city_num = city_num
            self.agent_num = agent_num
            self.k = k
            self.augmentation = straptimes

            if activate == 'train':
                self.sampleNum = maxDateValue
            else:
                self.sampleNum = testDataSize
            for i in range(self.sampleNum):
                if activate == 'test' and i == 950:
                    print("loading 950 test data now, remaining 50 data now")
                filename = "agent{}_city{}_edge{}_num{}.pt".format(agent_num, city_num, edge_num, i)
                filename = os.path.join(setpath, filename)
                if os.path.isfile(filename):
                    data = torch.load(filename)
                    correct, weight, target = self.makeup_data_from_file(data)
                    if correct:
                        self.dataset.append([weight, target])
                    else:
                        continue
            self.data_augmentation()

        else:
            print("{} data doesn't exist!".format(setpath))
        self.size = len(self.dataset)
        print("datanum", self.size)

    def __len__(self):
        return self.size

    def data_augmentation(self):

        def find_index(tensor_row, x):
            # for c in range(self.city_num):
            #     if tensor_row[c] == x:
            #         return c
            ind = torch.eq(tensor_row, x).nonzero().squeeze()
            return ind

        data_num = len(self.dataset)
        for times in range(self.augmentation):
            print("starting augmentation {}/{}".format(times, self.augmentation))
            for ind in range(data_num):
                weight = self.dataset[ind][0]
                target = self.dataset[ind][1]
                aug_times = torch.randint(1, 10, (1,))
                temp_weight = weight.clone()
                temp_target = target.clone()
                for aug in range(aug_times):
                    x, y = torch.randint(0, self.city_num, (1,)), torch.randint(0, self.city_num, (1,))
                    # change row
                    flag = temp_weight[x].clone()
                    temp_weight[x] = temp_weight[y].clone()
                    temp_weight[y] = flag.clone()
                    # change column
                    flag = temp_weight[:, x].clone()
                    temp_weight[:, x] = temp_weight[:, y].clone()
                    temp_weight[:, y] = flag.clone()
                    # change target
                    flag = temp_target[:self.city_num].clone()
                    # find x index
                    index_x = find_index(flag, x)
                    index_y = find_index(flag, y)
                    flag[index_x] = y
                    flag[index_y] = x
                    temp_target[:self.city_num] = flag.clone()
                # rotate target
                flag = temp_target[:self.city_num]
                temp_target = torch.zeros(self.city_num + 1).long()
                temp_target[:self.city_num] = flag
                temp_target[self.city_num] = temp_target[0]
                index_0 = find_index(flag, 0)
                rotated_target = torch.zeros(self.city_num + 1).long()
                k = 0
                for c in range(index_0, self.city_num):
                    rotated_target[k] = flag[c]
                    k += 1
                for c in range(index_0):
                    rotated_target[k] = flag[c]
                    k += 1
                rotated_target[self.city_num] = rotated_target[0]
                weight = temp_weight.clone()
                # # check
                # length1 = self.target_length(self.dataset[ind][0], self.dataset[ind][1])
                # length2 = self.target_length(weight, rotated_target)
                # # length3 = self.target_length(weight, temp_target)
                # if fabs(length1 - length2) > 1e-5:
                #     print("error")
                random_coeff = torch.rand(1) * weight.lt(10).float() + (1-weight.lt(10).float())
                self.dataset.append([weight * random_coeff, rotated_target])

    def target_length(self, weight, target):
        length = 0
        for c in range(self.city_num):
            length += weight[target[c], target[c+1]]
        return length

    def makeup_data_from_file(self, filedata):
        if len(filedata) == 1:
            filedata = filedata[0]
        weight = filedata['weight']
        city_num = len(weight)
        matrix_dist = torch.zeros(city_num, city_num)
        # if weight.dtype == 'float32':
        #     matrix_dist = torch.from_numpy(weight)
        for c in range(city_num):
            for j in range(city_num):
                if weight[c][j] > 10:
                    matrix_dist[c, j] = 10
                else:
                    if weight[c][j].dtype == 'float32':
                        matrix_dist[c, j] = torch.from_numpy(weight[c][j])
                    else:
                        matrix_dist[c, j] = weight[c][j]
        weight = matrix_dist

        target = torch.tensor(filedata['path']).squeeze()
        correct = False
        if target.size()[0] == len(weight) + 1:
            correct = True
        return correct, weight, target

    def __getitem__(self, idx):
        while True:
            weight, target, dist, knn = self.extract(self.dataset[idx])
            if target.size()[0] == weight.size()[0] + 1:
                data_item = [weight, target, dist, knn]
                break
            else:
                idx = np.random.randint(0, self.size)
        return data_item

    def extract(self, sample):
        """
            now only for TSP
            sample(dict) -> input_features: [batch_size, city_num, input_feature_size] here is cities
                      target: [batch_size, city_num]
        """
        weight = sample[0]
        target = sample[1]
        weight, dist, knn = self.compute_knn(weight)
        return weight, target, dist, knn

    def compute_knn(self, weight):
        dist, knn = torch.sort(weight, dim=-1)
        knn = knn[:, :self.k]
        dist = dist[:, :self.k]
        nnode = dist.size(0)
        for c in range(nnode):
            for m in range(self.k):
                if dist[c, m] >= 10:
                    dist[c, m] = dist[c, 0]
                    knn[c, m] = knn[c, 0]
        return weight, dist, knn

if __name__ == '__main__':
    'only for tsp load data'
    # train = gendata(1, 30, activate='train')
    # test = gendata(1, 30, activate='test')
    path = '../../../dataset/arbitrary_graph/min_sum_nonEuclid'
    trainVal = gendata(1, 30, 62, activate='trainVal', path=path)
    loader = DataLoader(trainVal, batch_size=2, shuffle=True, num_workers=1)
    it = iter(loader)
    sample = next(it)
    sample = next(it)
    sample = next(it)
    sample = next(it)

