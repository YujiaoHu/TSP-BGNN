import argparse
import os
import sys

def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="model parameters")
    parser.add_argument('--load', default=False, help='load model ? ')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cuda', type=int, default=0, help='which cuda')
    parser.add_argument('--aim', default='train', help='train or test')

    parser.add_argument('--start_city_num', type=int, default=30, help="train model: start city number")
    parser.add_argument('--end_city_num', type=int, default=30, help="train model: end city number")
    parser.add_argument('--sparse', type=float, default=0.5, help='train model: sparse')

    parser.add_argument("--test_min_num", type=int, default=30, help="test dataset: from start_min_num")
    parser.add_argument("--test_max_num", type=int, default=30, help="test dataset: end until test_max_num")
    parser.add_argument("--test_sparse", type=int, default=0.5, help="test dataset: sparse")
    parser.add_argument("--beamvalue", type=int, default=1, help="test: beam value for beam search")
    parser.add_argument("--gap_clip", type=float, default=10, help="test: lower bound of the sub-optimal computed by the bgnn")

    opts = parser.parse_args(args)
    return opts
