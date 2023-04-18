import argparse
import os
import numpy as np

from src.handlers.evaluater import Evaluater


if __name__ == '__main__':
    ### Decoding arguments
    eval_parser = argparse.ArgumentParser(description='Arguments for training the system')
    eval_parser.add_argument('--path', type=str, help='path to experiment')
    eval_parser.add_argument('--dataset', default='wikibio', type=str, help='dataset to train the system on')
    eval_parser.add_argument('--mode', default='test', type=str, help='which data split to evaluate on')
    eval_parser.add_argument('--device', default='cuda', type=str, help='selecting device to use')
    eval_parser.add_argument('--lim', type=int, default=None, help='whether subset of data to be used') 
    eval_parser.add_argument('--template', type=str, default=None, help='prompt-template')
    
    args = eval_parser.parse_args()
    evaluater = Evaluater(args.path, args.device)
    metrics = evaluater.load_likelihood(args.dataset, args.mode, args.template)
    print(metrics)
