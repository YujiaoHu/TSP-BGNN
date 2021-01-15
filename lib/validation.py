import torch
import torch.nn.functional as F
import numpy as np


def validation_arbitrary_success_ratio_gap(relation, tour, target, mask=False, beam=1, device='cuda:0', Un0=False,
                                           gap_clip=10):
    batch_size, city_num = relation.size(0), relation.size(1)
    tour_length = 0.0
    success_batch = []
    for bs in range(batch_size):
        success = True
        visit = torch.zeros(city_num).to(device)
        for i in range(city_num):
            visit[tour[i][bs]] += 1
            temp = relation[bs, tour[i][bs], tour[i + 1][bs]]
            if Un0 is False:
                if temp < 10 and visit.ge(2).nonzero().size(0) <= 0:
                    # unconnected adjacency cities
                    continue
                else:
                    success = False
                    break
            else:
                if temp >= 10:
                    print("-----------   temp >= 5", temp, bs, tour[i][bs], tour[i + 1][bs])
                if 0 < temp and temp < 10 and visit.ge(2).nonzero().size(0) <= 0:
                    # unconnected adjacency cities
                    continue
                else:
                    success = False
                    break
        if success is True:
            success_batch.append(bs)

    success_num = len(success_batch)

    for i in range(city_num):
        tour_length += relation[[bs for bs in range(batch_size)], tour[i], tour[i + 1]]

        # test_tour = torch.zeros(batch_size).to(device)
        # for bs in range(batch_size):
        #     for i in range(city_num):
        #         test_tour[bs] += relation[bs, tour[i][bs], tour[i + 1][bs]]

    test_target = torch.zeros(batch_size).to(device)
    for bs in range(batch_size):
        for i in range(city_num):
            test_target[bs] += relation[bs, target[bs, i], target[bs, i + 1]]

    success_pred_tour = []
    correspond_target = []
    for bs in range(success_num):
        success_pred_tour.append(tour_length[success_batch[bs]])
        correspond_target.append(test_target[success_batch[bs]])

    if beam == 1:
        gap = torch.tensor(success_pred_tour) / torch.tensor(correspond_target)
        success_num = torch.sum(gap.lt(gap_clip).float())
        gap = gap.lt(gap_clip).float() * gap
        return success_num, torch.sum(gap) if success_num > 0 else 0, torch.sum(1.0/gap) if success_num > 0 else 0
    else:
        origin_batch_size = relation.size(0) // beam
        success_gap = (torch.ones(origin_batch_size, beam) * 10).to(device)
        s_batch = torch.zeros(origin_batch_size).to(device)
        for bs in success_batch:
            id = bs // beam
            offset = bs % beam
            s_batch[id] = 1
            success_gap[id, offset] = tour_length[bs] / test_target[bs]
            if tour_length[bs] / test_target[bs] < 1:
                # print(tour_length[bs], test_target[bs])
                print_tour = []
                for c in range(city_num+1):
                    print_tour.append(tour[c][bs])
                # print("tour : ", print_tour)
        Sgap, _ = torch.min(success_gap, dim=1)
        Snum = 0
        sum = 0
        for bs in range(origin_batch_size):
            if Sgap[bs] < gap_clip:
                Snum += 1.0
                sum += Sgap[bs]

        # ave = (sum / Snum) if Snum > 0 else 0
        # if ave < 1:
        #     print("sum = {}, Snum = {}, Sgap = {}, ave = {}".format(sum, Snum, Sgap, ave))
        return Snum, sum, torch.mean(tour_length / test_target)

def compute_tour_gap(relation, tour, target):
    batch_size, city_num = relation.size(0), relation.size(1)
    tour_length = 0.0
    for i in range(city_num):
        tour_length += relation[[bs for bs in range(batch_size)], tour[i], tour[i + 1]]

    test_tour = torch.zeros(batch_size)
    for bs in range(batch_size):
        for i in range(city_num):
            test_tour[bs] += relation[bs, tour[i][bs], tour[i + 1][bs]]

    # target_length = 0.0
    # for c in range(city_num):
    #     target_length += relation[[bs for bs in range(batch_size)], target[:, i], target[:, i+1]]

    test_target = torch.zeros(batch_size).to(device)
    for bs in range(batch_size):
        for i in range(city_num):
            test_target[bs] += relation[bs, target[bs, i], target[bs, i + 1]]
    gap = tour_length / test_target
    return torch.mean(gap)


def validation_loss_tour(pred, prev_tour, device='cuda:0'):
    looptime = len(prev_tour) - 1
    batch_size = pred.size(0)
    city_num = pred.size(1)

    similarity = F.softmax(pred, dim=1)

    # remove previous tour
    masks = torch.ones(batch_size, city_num).to(device)
    for old_idx in prev_tour:
        masks[[bs for bs in range(batch_size)], old_idx] = 0.0
    if looptime == city_num - 1:
        masks[:, 0] = 1.0
    similarity = similarity * masks

    correct = False
    candidate = similarity.clone()
    while not correct:
        _, next_city = torch.topk(candidate, 1, dim=-1)
        next_city = next_city.squeeze().long()
        if looptime == city_num - 1:
            # next_city cannot include city 0
            next_city = prev_tour[0]
            break
        for idxs in prev_tour:
            x = (next_city == idxs).nonzero().squeeze(1)
            x_size = x.size(0)
            if x_size > 0:
                for s in range(x_size):
                    bs = x[s]
                    # print(candidate[bs])
                    if candidate[bs].nonzero().squeeze(1).size(0) <= 0:
                        candidate[bs] = masks[bs]
                correct = False
                break
            else:
                correct = True
    tour = next_city.to(device)
    return tour


def mask_validate_one_step(relation, weight, target, knn, prev_tour, pred,
                           device='cuda:0',
                           BS=False,
                           Un0=False,
                           objective='sym',
                           statis_extend=True,
                           extend_num=0):
    succeed = True
    batch_size, city_num = pred.size(0), pred.size(1)
    looptime = len(prev_tour)
    # make masks
    masks = torch.ones(batch_size, city_num).to(device)
    for old_idx in prev_tour:
        masks[[bs for bs in range(batch_size)], old_idx] = 0.0
    if looptime == city_num:
        masks[:, 0] = 1.0
    if Un0 is False:
        mask = relation[[bs for bs in range(batch_size)], prev_tour[looptime-1]].lt(10)
    else:
        mask = relation[[bs for bs in range(batch_size)], prev_tour[looptime-1]].gt(0)
    masks = masks * mask.float()

    extend_num = extend_num + torch.sum(masks, dim=1)

    probs = F.softmax(pred, dim=1) * masks

    if BS is True:
        return masks, probs

    success = []
    tour = []
    for bs in range(batch_size):
        if masks[bs].nonzero().squeeze(1).size(0) > 0:
            success.append(bs)
            if probs[bs].nonzero().squeeze(1).size(0) > 0:
                _, action = torch.topk(probs[bs], 1, dim=-1)
            else:
                _, action = torch.topk(masks[bs], 1, dim=-1)
            tour.append(action)
    tour = torch.tensor(tour).to(device)
    if 0 < len(success) < batch_size:
        # exist a unsuccessful path
        success = torch.tensor(success).to(device)
        for prev_time in range(looptime):
            prev_tour[prev_time] = torch.gather(prev_tour[prev_time], 0, success)
        success = success.unsqueeze(1).unsqueeze(1)
        relation = torch.gather(relation, 0, success.repeat(1, relation.size(1), relation.size(2)))
        if 'nonsym' in objective:
            weight = torch.gather(weight, 0,
                                  success.unsqueeze(3).repeat(
                                      1, weight.size(1), weight.size(2), weight.size(3))
                                  )
        else:
            weight = torch.gather(weight, 0, success.repeat(1, weight.size(1), weight.size(2)))
        knn = torch.gather(knn, 0, success.repeat(1, knn.size(1), knn.size(2)))
        target = torch.gather(target, 0, success.squeeze(2).repeat(1, target.size(1)))
        extend_num = torch.gather(extend_num, 0, success.squeeze(2).squeeze(1))

        # temp_relation = relation.clone()
        # temp_weight = weight.clone()
        # temp_target = target.clone()
        # temp_knn = knn.clone()
        # m = 0
        # for bs in success:
        #     relation[m] = temp_relation[bs]
        #     weight[m] = temp_weight[bs]
        #     target[m] = temp_target[bs]
        #     knn[m] = temp_knn[bs]
        #     m += 1
    elif len(success) <= 0:
        succeed = False
    if statis_extend is True:
        return succeed, relation, weight, target, knn, prev_tour, tour, extend_num
    else:
        return succeed, relation, weight, target, knn, prev_tour, tour



def validation_mask_BS_one_step(model, relation, weight, target, knn, beam,
                                device='cuda:0',
                                Un0=False,
                                rollout=False,
                                att=False,
                                gap_clip=10,
                                objective='sym'):
    batch_size, city_num = relation.size(0), relation.size(1)
    beam_batch_size = batch_size * beam

    nodeFeature = clear_input(relation, device, rollout)

    # deal with depot specially:
    # for 3 other features: 0: last city; 1: current city; 2:depot
    nodeFeature[:, 0, 4] = 1.0
    nodeFeature[:, 0, 5] = 0.0
    prev_tour = []
    prev_tour.append(target[:, 0])

    last_current = torch.zeros(batch_size, 2).long().to(device)
    if att is False:
        if 'nonsym' in objective:
            pred = model(nodeFeature.permute(0, 2, 1),
                        weight,
                        knn)
        else:
            pred = model(nodeFeature.permute(0, 2, 1),
                         weight.unsqueeze(1),
                         knn)
    else:
        if 'nonsym' in objective:
            pred = model(nodeFeature.permute(0, 2, 1),
                         weight,
                         knn,
                         last_current)
        else:
            pred = model(nodeFeature.permute(0, 2, 1),
                         weight.unsqueeze(1),
                         knn,
                         last_current)

    masks, probs = mask_validate_one_step(relation, weight, target, knn, prev_tour, pred, device, BS=True, Un0=Un0)
    _, pred_actions = torch.topk(probs, beam, dim=-1)
    pred_masks = torch.gather(masks, 1, pred_actions)

    current_city = torch.zeros(batch_size).long().to(device)
    prev_tour[0] = current_city.repeat(beam)
    current_city = pred_actions.view(beam_batch_size)
    prev_tour.append(current_city)

    repeat_relation = torch.tensor(np.repeat(np.array(relation.cpu()), beam, axis=0)).to(device)
    repeat_knn = torch.tensor(np.repeat(np.array(knn.cpu()), beam, axis=0)).to(device)
    repeat_weight = torch.tensor(np.repeat(np.array(weight.cpu()), beam, axis=0)).to(device)
    repeat_target = torch.tensor(np.repeat(np.array(target.cpu()), beam, axis=0)).to(device)

    for c in range(1, city_num-1):
        # prepare
        succeed =True
        last_city = prev_tour[len(prev_tour) - 2]
        current_city = prev_tour[len(prev_tour) - 1]

        bsize = current_city.size(0)
        last_current = torch.zeros(bsize, 2).long().to(device)
        last_current[:, 0] = last_city
        last_current[:, 1] = current_city

        batch_size = current_city.size(0) // beam
        beam_batch_size = batch_size * beam

        nodeFeature = clear_input(repeat_relation, device)
        nodeFeature[[bs for bs in range(beam_batch_size)], last_city, 2] = 1.0
        nodeFeature[[bs for bs in range(beam_batch_size)], last_city, 3] = 0.0
        nodeFeature[[bs for bs in range(beam_batch_size)], current_city, 4] = 1.0
        nodeFeature[[bs for bs in range(beam_batch_size)], current_city, 5] = 0.0
        for prev in range(c - 1):
            previous_city = prev_tour[prev]
            nodeFeature[[bs for bs in range(beam_batch_size)], previous_city, 6] = 1.0
            nodeFeature[[bs for bs in range(beam_batch_size)], previous_city, 7] = 0.0
        if att is False:
            if 'nonsym' in objective:
                pred = model(nodeFeature.permute(0, 2, 1),
                                  repeat_weight,
                                  repeat_knn)
            else:
                pred = model(nodeFeature.permute(0, 2, 1),
                             repeat_weight.unsqueeze(1),
                             repeat_knn)
        else:
            pred = model(nodeFeature.permute(0, 2, 1),
                         repeat_weight.unsqueeze(1),
                         repeat_knn,
                         last_current)

        masks, probs = mask_validate_one_step(repeat_relation,
                                              repeat_weight, repeat_target, repeat_knn,
                                              prev_tour, pred, device, BS=True, Un0=Un0)
        priority = probs + (pred_masks.view(beam_batch_size).unsqueeze(1).repeat(1, probs.size(1))-1)*1e4

        priority = priority.view(batch_size, beam, city_num).view(batch_size, beam * city_num)
        res_masks = masks.view(batch_size, beam, city_num).view(batch_size, beam * city_num)

        _, pred_actions = torch.topk(priority, beam, dim=-1)  # pred_actions: batch, beam
        pred_masks = torch.gather(res_masks, 1, pred_actions)

        next_loop_last_city_beam = pred_actions // city_num  # batch, beam
        pred_step_actions = pred_actions % city_num  # batch, beam

        # modify prev_tour use next_loop_last_city , 必须回溯修改之前所有的路径
        for prev_time in range(c + 1):
            prev_tour[prev_time] = torch.gather(prev_tour[prev_time].view(batch_size, beam), 1,
                                                next_loop_last_city_beam).view(beam_batch_size)
        next_loop_current_city = pred_step_actions.view(beam_batch_size)
        pred_masks = pred_masks.view(beam_batch_size)

        prev_tour.append(next_loop_current_city.view(beam_batch_size))

        # if a batch is failed getting ang feasible path
        f = pred_masks.view(-1, beam)
        success = []
        for fb in range(batch_size):
            if f[fb].nonzero().size(0) > 0:
                for bm in range(beam):
                    success.append(fb*beam + bm)
        if 0 < len(success) < beam_batch_size:
            # exist a unsuccessful path
            success = torch.tensor(success).to(device)
            looptime = len(prev_tour)
            for prev_time in range(looptime):
                prev_tour[prev_time] = torch.gather(prev_tour[prev_time], 0, success)
            pred_masks = torch.gather(pred_masks.view(beam_batch_size), 0, success)
            success = success.unsqueeze(1).unsqueeze(1)
            repeat_relation = torch.gather(repeat_relation, 0, success.repeat(1, relation.size(1), relation.size(2)))
            if 'nonsym' in objective:
                repeat_weight = torch.gather(repeat_weight, 0,
                                      success.unsqueeze(3).repeat(
                                          1, weight.size(1), weight.size(2), weight.size(3))
                                      )
            else:
                repeat_weight = torch.gather(repeat_weight, 0, success.repeat(1, weight.size(1), weight.size(2)))

            # repeat_weight = torch.gather(repeat_weight, 0, success.repeat(1, weight.size(1), weight.size(2)))

            repeat_knn = torch.gather(repeat_knn, 0, success.repeat(1, knn.size(1), knn.size(2)))
            repeat_target = torch.gather(repeat_target, 0, success.squeeze(2).repeat(1, target.size(1)))
        elif len(success) <= 0:
            succeed = False
            break
    if succeed is True:
        prev_tour.append(prev_tour[0])
        Snum, Sgap, Tgap = validation_arbitrary_success_ratio_gap\
            (repeat_relation, prev_tour, repeat_target, mask=False, beam=beam, device=device, gap_clip=gap_clip)
        return Snum, Sgap, Tgap
    else:
        return 0.0, 0.0, 0.0


def clear_input(relation, device, rollout=False):
    if rollout is False:
        batch_size, city_num = relation.size(0), relation.size(1)
        # clear
        # 0: is depot, 2: is last, 4: is current, 6: is traveled
        inputs = torch.ones(batch_size, city_num, 8).to(device)
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
    else:
        batch_size, city_num = relation.size(0), relation.size(1)
        inputs = torch.ones(batch_size, city_num, 2).to(self.device)
        # visited
        inputs[:, :, 0] = 0.0
        inputs[:, :, 1] = 1.0
        inputs = inputs.repeat(1, city_num, 1)
    return inputs