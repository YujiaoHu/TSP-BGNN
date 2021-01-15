"""
    auction_lap.py
    From:
       https://github.com/bkj/auction-lap
    The original version didn't work.
    Modified by zhen to fitting the newest pytorch
    From
        https://dspace.mit.edu/bitstream/handle/1721.1/3265/P-2108-26912652.pdf;sequence=1
"""

from __future__ import print_function, division

import sys
import torch


def auction_lap(X, eps=None, compute_score=True):
    """
        X: n-by-n matrix w/ integer entries
        eps: "bid size" -- smaller values means higher accuracy w/ longer runtime
    """

    eps = 1 / X.shape[0] if eps is None else eps

    # --
    # Init

    cost = torch.zeros((1, X.shape[1]))
    curr_ass = torch.zeros(X.shape[0]).long() - 1
    bids = torch.zeros(X.shape)

    if X.is_cuda:
        X = X.cpu()
        # cost, curr_ass, bids = cost.cuda(), curr_ass.cuda(), bids.cuda()

    counter = 0
    while (curr_ass == -1).any():
        counter += 1

        # --
        # Bidding

        unassigned = (curr_ass == -1).nonzero().squeeze().view(-1)

        value = X[unassigned] - cost
        top_value, top_idx = value.topk(2, dim=1)

        first_idx = top_idx[:, 0]
        first_value, second_value = top_value[:, 0], top_value[:, 1]

        bid_increments = first_value - second_value + eps

        # print(unassigned)
        # print(bid_increments.size())
        # print(first_idx)
        # print(bid_increments)
        bids_ = bids[unassigned].view(-1, bids.shape[1])
        bids_.zero_()
        # print(bids_.size())
        bids_.scatter_(
            dim=1,
            index=first_idx.contiguous().view(-1, 1),
            src=bid_increments.view([-1, 1])
        )

        # --
        # Assignment

        have_bidder = (bids_ > 0).int().sum(dim=0).nonzero()
        high_bids, high_bidders = bids_[:, have_bidder].max(dim=0)

        high_bidders = unassigned[high_bidders.view(-1)]

        # print(high_bidders)

        cost[:, have_bidder] += high_bids
        # print(have_bidder)
        # print(unassigned)
        # print((curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1))
        curr_ass[(curr_ass.view(-1, 1) == have_bidder.view(1, -1)
                  ).sum(dim=1).nonzero()] = -1
        # print(curr_ass)
        curr_ass[high_bidders] = have_bidder.squeeze()
        # print(curr_ass)

    score = None
    if compute_score:
        score = X.gather(dim=1, index=curr_ass.view(-1, 1)).sum()

    return score.cuda(), curr_ass.cuda(), counter, cost.cuda()
