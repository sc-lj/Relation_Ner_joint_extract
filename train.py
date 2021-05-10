import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import config
import framework
import argparse
import models
import torch
import numpy as np
import random


# seed = 1234
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='globalpointer', help='name of the model')
parser.add_argument('--learning_rate', type=float, default=5e-6)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='baidu')
parser.add_argument('--optimizer', type=str, default='adamw',choices=['adamw',"adam","sgd"])
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--max_epoch', type=int, default=20)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--train_prefix', type=str, default='duie_train')
parser.add_argument('--dev_prefix', type=str, default='duie_dev')
parser.add_argument('--test_prefix', type=str, default='duie_test1')
parser.add_argument('--max_len', type=int, default=300)
parser.add_argument('--period', type=int, default=50)
parser.add_argument('--pretrain_path', type=str, default="../chinese_bert_large")
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument("--local_rank",default=-1,help="rank of current process")
parser.add_argument("--world_size",default=3,help="word size")
parser.add_argument("--backend",default="nccl")
parser.add_argument("--gradient_accumulation_steps",default=3)
args = parser.parse_args()

con = config.Config(args)

fw = framework.Framework(con)

model = {
    'casrel': models.Casrel,
    "globalpointer":models.GlobalPointerRel
}

fw.train(model[args.model_name])
