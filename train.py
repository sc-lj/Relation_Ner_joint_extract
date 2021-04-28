import config
import framework
import argparse
import models
import os
import torch
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Casrel', help='name of the model')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='baidu')
parser.add_argument('--optimizer', type=str, default='adamw',choices=['adamw',"adam","sgd"])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--train_prefix', type=str, default='duie_train')
parser.add_argument('--dev_prefix', type=str, default='duie_dev')
parser.add_argument('--test_prefix', type=str, default='duie_test1')
parser.add_argument('--max_len', type=int, default=300)
parser.add_argument('--rel_num', type=int, default=53)
parser.add_argument('--period', type=int, default=50)
parser.add_argument('--pretrain_path', type=str, default="../chinese_bert_base")
parser.add_argument('--debug', type=bool, default=False)
args = parser.parse_args()

con = config.Config(args)

fw = framework.Framework(con)

model = {
    'Casrel': models.Casrel
}

fw.train(model[args.model_name])
