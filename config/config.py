import time
import os
import json
import torch

class Config(object):
    def __init__(self, args):
        # train hyper parameter
        
        for name,value in args.__dict__.items():
            setattr(self,name,value)
        self.multi_gpu = args.multi_gpu
        self.learning_rate = args.learning_rate
        if self.gradient_accumulation_steps > 1:
            self.batch_size = int(self.batch_size/self.gradient_accumulation_steps)
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.optimizer = args.optimizer
        self.weight_decay = args.weight_decay
        self.model_name = args.model_name
        self.teacher_pro = 0.5  #teacher probability
        self.use_focal = True
        self.sub_threhold = 0.5
        self.dropout = 0.3
        self.attention = "None" # head2tail,None
        self.discr = True # 是否对bert分层设置学习率
        self.head_size = 64 # head size of globale pointer network
        self.warmup_proportion = 0.01
        self.fuse16 = False # 混合半精度训练
        self.add_layernormal = False
        self.identiy = True
        self.head_inter_tail = False #使用将训练时，主语的首尾之间的词用于预测谓语， 不能和identiy同时为True
        self.fusion = True # 使用subject与encoded的交互矩阵


        # dataset
        self.dataset = args.dataset
        self.pretrain_path = args.pretrain_path

        # path and name
        self.root = '.'
        self.data_path = self.root + '/data/' + self.dataset
        self.checkpoint_dir = self.root + '/checkpoint/' + self.dataset
        self.log_dir = self.root + '/log/' + self.dataset
        self.result_dir = self.root + '/result/' + self.dataset
        self.train_prefix = args.train_prefix
        self.dev_prefix = args.dev_prefix
        self.test_prefix = args.test_prefix
        self.time_postfix = time.strftime("%m%d%H%M",time.localtime(time.time()))
        self.model_save_name = args.model_name + '_DATASET_' + self.dataset +"_T_"+self.time_postfix
        self.log_save_name = 'LOG_' + args.model_name + '_DATASET_' + self.dataset +"_T_"+self.time_postfix
        self.result_save_name = 'RESULT_' + args.model_name + '_DATASET_' + self.dataset +"_T_"+self.time_postfix+ ".json"
        self.config_file_name = "CONFIG_" + args.model_name + '_DATASET_' + self.dataset +"_T_"+self.time_postfix+ ".json"

        self.rel_num = len(json.load(open(os.path.join(self.data_path, 'rel2id.json'),'r',encoding="utf-8")))
        
        # log setting
        self.period = args.period
        self.test_epoch = args.test_epoch

        # debug
        self.debug = args.debug
        if self.debug:
            self.dev_prefix = self.train_prefix
            self.test_prefix = self.train_prefix

    def dump_to(self):
        with open(self.checkpoint_dir+"/"+self.config_file_name, 'w', encoding='utf-8') as fout:
            json.dump(self.__dict__, fout,ensure_ascii=False,indent=2)


