import torch.optim as optim
from torch import nn
import os
import data_loader
import torch.nn.functional as F
import torch.distributed as dist
from apex import amp
from transformers import get_polynomial_decay_schedule_with_warmup,get_cosine_schedule_with_warmup
import torch
import numpy as np
import json
import time


class Framework(object):
    def __init__(self, con):
        self.config = con
        if int(self.config.local_rank) == -1:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cuda",int(self.config.local_rank))
            torch.distributed.init_process_group(backend=self.backend,world_size=int(self.config.world_size))
        if int(self.config.local_rank)!=-1:
            self.set_environment_variables_for_nccl_backend()
    
    def set_environment_variables_for_nccl_backend(self):
        os.environ['RANK'] = str(self.config.local_rank)
        os.environ['WORLD_SIZE'] = str(self.config.world_size)
        # os.environ['MASTER_ADDR'] = str(addr)
        # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
        # os.environ['MASTER_PORT'] = str(master_port)

        # TODO make this parameterizable
        os.environ['NCCL_SOCKET_IFNAME'] = '^docker0,lo'

    def logging(self, s, print_=True, log_=True):
        if int(self.config.local_rank) == -1 or(int(self.config.local_rank) > -1 and dist.get_rank()==0):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(self.config.log_dir, self.config.log_save_name), 'a+') as f_log:
                    f_log.write(s + '\n')

    def train(self, model_pattern):
        # initialize the model
        ori_model = model_pattern(self.config)
        ori_model.to(self.device)
        params = ori_model.parameters()
        if self.config.optimizer == 'sgd':
            optimizer = optim.SGD(params, self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer  == 'adam':
            optimizer = optim.Adam(params, self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer  == 'adamw':  # Optimizer for BERT
            from transformers import AdamW
            params = list(ori_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            if self.config.discr:
                # group1=['layer.0.','layer.1.','layer.2.','layer.3.']
                # group2=['layer.4.','layer.5.','layer.6.','layer.7.']
                # group3=['layer.8.','layer.9.','layer.10.','layer.11.']
                # group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
                # grouped_params = [
                #     {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.01},
                #     {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.01, 'lr': self.config.learning_rate/1.6},
                #     {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.01, 'lr': self.config.learning_rate},
                #     {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.01, 'lr': self.config.learning_rate*1.6},
                #     {'params': [p for n, p in params if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.0},
                #     {'params': [p for n, p in params if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.0, 'lr': self.config.learning_rate/1.6},
                #     {'params': [p for n, p in params if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.0, 'lr': self.config.learning_rate},
                #     {'params': [p for n, p in params if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.0, 'lr': self.config.learning_rate*1.6},
                # ]
                bert_group = ["bert"]
                grouped_params = [
                    {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and any(nd in n for nd in bert_group)], 'weight_decay': 0.01, 'lr': self.config.learning_rate},
                    {'params': [p for n, p in params if any(nd in n for nd in no_decay) and any(nd in n for nd in bert_group)], 'weight_decay': 0.0, 'lr': self.config.learning_rate},
                    {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_group)],'weight_decay': 0.01, 'lr': self.config.learning_rate*20},
                    {'params': [p for n, p in params if any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_group)],'weight_decay': 0.0, 'lr': self.config.learning_rate*20},
                    ]
            else:
                grouped_params = [
                    {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                    ]
            optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # define the optimizer
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, ori_model.parameters()), lr=self.config.learning_rate)

        # 如果要使用分布式混合半精度训练，混合半精度要在分布式前面进行声明
        if self.config.fuse16:
            model, optimizer = amp.initialize(ori_model, optimizer, opt_level='O1')

        # whether use multi GPU
        if self.config.multi_gpu:
            if self.config.fuse16:
                from apex.parallel import DistributedDataParallel as DDP
                # DDP模块同时也计算整体的平均梯度, 这样我们就不需要在训练步骤计算平均梯度。
                model = DDP(ori_model, delay_allreduce=True)
            else:
                # model = nn.DataParallel(ori_model)
                model = nn.parallel.DistributedDataParallel(ori_model,device_ids=[int(self.config.local_rank)],find_unused_parameters=True)
        else:
            model = ori_model
        

        # check the checkpoint dir
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        # check the log dir
        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)

        train_dataset = data_loader.dataLoader[self.config.dataset](self.config, self.config.train_prefix, is_test=False)
        dev_dataset = data_loader.dataLoader[self.config.dataset](self.config, self.config.dev_prefix, is_test=True)
        # get the data loader
        dev_data_loader = data_loader.get_loader(dev_dataset, self.config, is_test=True,num_workers=5)
        num_training_steps = int(len(train_dataset)/self.config.batch_size*self.config.max_epoch)
        num_warmup_steps = self.config.warmup_proportion * num_training_steps
        schedule = get_polynomial_decay_schedule_with_warmup(optimizer,num_warmup_steps=num_warmup_steps,num_training_steps=num_training_steps)

        # other
        model.train()
        global_step = 0
        loss_sum = 0
        sub_loss_sum = 0
        rel_loss_sum = 0

        best_f1_score = 0
        best_precision = 0
        best_recall = 0

        best_epoch = 0
        init_time = time.time()
        start_time = time.time()

        # the training loop
        for epoch in range(self.config.max_epoch):
            train_data_loader = data_loader.get_loader(train_dataset, self.config, num_workers=5, epochs = epoch)
            train_data_prefetcher = data_loader.DataPreFetcher(train_data_loader)
            data = train_data_prefetcher.next()
            step = 0
            while data is not None:
            # for step,data in enumerate(train_data_loader):
                # for k, v in data.items():
                #     if isinstance(v, torch.Tensor):
                #         data[k] = data[k].cuda()
                if self.config.model_name == "casrel":
                    sub_heads_loss,sub_tails_loss,obj_heads_loss,obj_tails_loss = model(data)
                    sub_loss = sub_heads_loss + sub_tails_loss
                    obj_loss = obj_heads_loss + obj_tails_loss
                    total_loss = 12*sub_loss + obj_loss
                elif self.config.model_name == "globalpointer":
                    sub_loss,obj_loss = model(data)
                    total_loss = 8*sub_loss+1.*obj_loss
                elif self.config.model_name == "casglobal":
                    sub_loss,obj_loss = model(data)
                    total_loss = 8*sub_loss+1.*obj_loss
                else:
                    raise ValueError(f"{self.config.model_name} not in [casrel,globalpointer]")

                if self.config.gradient_accumulation_steps > 1:
                    total_loss /= self.config.gradient_accumulation_steps
                    sub_loss /= self.config.gradient_accumulation_steps
                    obj_loss /= self.config.gradient_accumulation_steps

                if self.config.fuse16:
                    with amp.scale_loss(total_loss,optimizer) as scale_loss:
                        scale_loss.backward()
                else:
                    total_loss.backward()

                if (step+1)%self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    schedule.step()
                    optimizer.zero_grad()


                global_step += 1
                loss_sum += total_loss.item()
                sub_loss_sum += sub_loss.item()
                rel_loss_sum += obj_loss.item()

                if (step+1) % (self.config.period*self.config.gradient_accumulation_steps) == 0:
                    cur_loss = loss_sum / self.config.period
                    cur_sub_loss = sub_loss_sum / self.config.period
                    cur_rel_loss = rel_loss_sum / self.config.period
                    elapsed = time.time() - start_time
                    self.logging("epoch: {:3d}, global_step: {:4d}, step: {:4d}, speed: {:5.2f}ms/b, train loss: {:5.3f},subject loss:{:5.3f},rel_object_loss:{:5.3f}".
                                 format(epoch, global_step, step+1, elapsed * 1000 / self.config.period, cur_loss,cur_sub_loss,cur_rel_loss))
                    loss_sum = 0
                    sub_loss_sum = 0
                    rel_loss_sum = 0
                    start_time = time.time()

                data = train_data_prefetcher.next()
                step += 1
            
            if (epoch + 1) % self.config.test_epoch == 0:
                eval_start_time = time.time()
                model.eval()
                # call the test function
                precision, recall, f1_score,sub_precision,sub_recall,sub_f1_score = self.test(dev_data_loader, model)
                model.train()
                self.logging('epoch {:3d}, eval time: {:5.2f}s, f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}, sub_f1: {:4.2f}, sub_precision: {:4.2f}, sub_recall: {:4.2f}'.
                             format(epoch, time.time() - eval_start_time, f1_score, precision, recall,sub_f1_score,sub_precision,sub_recall))

                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_epoch = epoch
                    best_precision = precision
                    best_recall = recall
                    self.logging("saving the model, epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}".
                                 format(best_epoch, best_f1_score, precision, recall))
                    # save the best model
                    path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
                    if not self.config.debug and ((int(self.config.local_rank) > -1 and dist.get_rank()==0) or int(self.config.local_rank) == -1):
                        if int(self.config.local_rank) == -1:
                            torch.save(model.state_dict(), path)
                        else:
                            torch.save(model.modules.state_dict(), path)
                        self.config.dump_to()

            # manually release the unused cache
            torch.cuda.empty_cache()

        self.logging("finish training")
        self.logging("best epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2}, total time: {:5.2f}s".
                     format(best_epoch, best_f1_score, best_precision, best_recall, time.time() - init_time))

    def test(self, test_data_loader, model, output=True):

        if output:
            # check the result dir
            if not os.path.exists(self.config.result_dir):
                os.makedirs(self.config.result_dir)

            path = os.path.join(self.config.result_dir, self.config.result_save_name)

            fw = open(path, 'w')

        orders = ['subject', 'relation', 'object']

        def to_tup(triple_list):
            ret = []
            for triple in triple_list:
                ret.append(tuple(triple))
            return ret

        test_data_prefetcher = data_loader.DataPreFetcher(test_data_loader)
        data = test_data_prefetcher.next()
        rel2id = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))
        id2rel = {str(v):k for k,v in rel2id.items()}
        correct_num, predict_num, gold_num = 0, 0, 0
        correct_sub_num,predict_sub_num,gold_sub_num = 0,0,0
        while data is not None:
            with torch.no_grad():
                token_ids = data['token_ids']
                tokens = data['tokens'][0]
                mask = data['mask']
                encoded_text = model.get_encoded_text(token_ids, mask)
                if self.config.model_name == "casrel":
                    pred_list,pred_sub_list = self.casrel_test(model,encoded_text,tokens,id2rel,h_bar=0.5, t_bar=0.5)
                elif self.config.model_name == 'globalpointer':
                    pred_list,pred_sub_list = self.globalpointer_test(model,encoded_text,tokens,id2rel,mask,threshold=0)
                elif self.config.model_name == "casglobal":
                    pred_list,pred_sub_list = self.casglobalpointer_test(model,encoded_text,tokens,id2rel,mask,threshold=0,h_bar=0.5, t_bar=0.5)
                else:
                    raise ValueError(f"{self.config.model_name} not in [casrel,globalpointer,casglobal]")
                pred_triples = set(pred_list)
                pred_sub_list = set(pred_sub_list)
                gold_triples = set(to_tup(data['triples'][0]))
                gold_sub = set([line[0] for line in data['triples'][0]])

                correct_num += len(pred_triples & gold_triples)
                predict_num += len(pred_triples)
                gold_num += len(gold_triples)

                correct_sub_num += len(pred_sub_list & gold_sub)
                predict_sub_num += len(pred_sub_list)
                gold_sub_num += len(gold_sub)

                if output:
                    result = json.dumps({
                        'text': ' '.join(''.join([i.lstrip("##") for i in tokens]).split('[unused1]')).replace("[CLS]","").replace("[SEP]",""),
                        'triple_list_gold': [
                            dict(zip(orders, triple)) for triple in gold_triples
                        ],
                        'triple_list_pred': [
                            dict(zip(orders, triple)) for triple in pred_triples
                        ],
                        'new': [
                            dict(zip(orders, triple)) for triple in pred_triples - gold_triples
                        ],
                        'lack': [
                            dict(zip(orders, triple)) for triple in gold_triples - pred_triples
                        ]
                    }, ensure_ascii=False)
                    fw.write(result + '\n')

                data = test_data_prefetcher.next()

        self.logging("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))
        self.logging("correct_sub_num: {:3d}, predict_sub_num: {:3d}, gold_sub_num: {:3d}".format(correct_sub_num, predict_sub_num, gold_sub_num))

        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        
        sub_precision = correct_sub_num / (predict_sub_num + 1e-10)
        sub_recall = correct_sub_num / (gold_sub_num + 1e-10)
        sub_f1_score = 2 * sub_precision * sub_recall / (sub_precision + sub_recall + 1e-10)
        return precision, recall, f1_score,sub_precision,sub_recall,sub_f1_score

    def testall(self, model_pattern):
        model = model_pattern(self.config)
        path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
        model.load_state_dict(torch.load(path))
        model.cuda()
        model.eval()
        test_data_loader = data_loader.get_loader(self.config, prefix=self.config.test_prefix, is_test=True)
        precision, recall, f1_score,sub_precision,sub_recall,sub_f1_score = self.test(test_data_loader, model, True)
        print("f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}".format(f1_score, precision, recall))
        print("sub_f1: {:4.2f}, sub_precision: {:4.2f}, sub_recall: {:4.2f}".format(sub_f1_score, sub_precision,sub_recall))


    def casrel_test(self,model,encoded_text,tokens,id2rel,h_bar=0.5,t_bar=0.5):
        pred_sub_heads, pred_sub_tails = model.get_subs(encoded_text)
        sub_heads, sub_tails = np.where(pred_sub_heads.cpu()[0] > h_bar)[0], np.where(pred_sub_tails.cpu()[0] > t_bar)[0]
        subjects = []
        for sub_head in sub_heads:
            sub_tail = sub_tails[sub_tails >= sub_head]
            if len(sub_tail) > 0:
                sub_tail = sub_tail[0]
                subject = tokens[sub_head: sub_tail]
                subjects.append((subject, sub_head, sub_tail))
        if subjects:
            triple_list = []
            sub_list= []
            # [subject_num, seq_len, bert_dim]
            repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
            # [subject_num, 1, seq_len],每个主语构建一个样本
            sub_head_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
            sub_tail_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
            for subject_idx, subject in enumerate(subjects):
                sub_head_mapping[subject_idx][0][subject[1]] = 1
                sub_tail_mapping[subject_idx][0][subject[2]] = 1
            sub_tail_mapping = sub_tail_mapping.to(repeated_encoded_text)
            sub_head_mapping = sub_head_mapping.to(repeated_encoded_text)
            pred_obj_heads, pred_obj_tails = model.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, repeated_encoded_text)
            for subject_idx, subject in enumerate(subjects):
                sub = subject[0]
                sub = ''.join([i.lstrip("##") for i in sub])
                sub = ' '.join(sub.split('[unused1]'))
                sub_list.append(sub)
                obj_heads, obj_tails = np.where(pred_obj_heads.cpu()[subject_idx] > h_bar), np.where(pred_obj_tails.cpu()[subject_idx] > t_bar)
                for obj_head, rel_head in zip(*obj_heads):
                    for obj_tail, rel_tail in zip(*obj_tails):
                        if obj_head <= obj_tail and rel_head == rel_tail:
                            rel = id2rel[str(int(rel_head))]
                            obj = tokens[obj_head: obj_tail]
                            obj = ''.join([i.lstrip("##") for i in obj])
                            obj = ' '.join(obj.split('[unused1]'))
                            triple_list.append((sub, rel, obj))
                            break
            triple_set = set()
            for s, r, o in triple_list:
                triple_set.add((s.strip(), r.strip(), o.strip()))
            pred_list = list(triple_set)
            sub_list = list(set(sub_list))
        else:
            pred_list = []
            sub_list = []
        return pred_list,sub_list

    def globalpointer_test(self,model,encoded_text,tokens,id2rel,mask,threshold=0):
        pred_subs = model.get_subs(encoded_text,mask)
        pred_subs[:, [0, -1]] -= np.inf
        pred_subs[:, :, [0, -1]] -= np.inf
        sub_head_tail = np.where(pred_subs.cpu()[0] > 0)
        subjects = []
        for sub_head, sub_tail in zip(*np.where(pred_subs.cpu()[0] > threshold)):
            subject = tokens[sub_head: sub_tail]
            subjects.append((subject, sub_head, sub_tail))

        if subjects:
            triple_list = []
            sub_list = []
            # [subject_num, seq_len, bert_dim]
            repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
            # [subject_num, 1, seq_len],每个主语构建一个样本
            sub_head_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
            sub_tail_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
            for subject_idx, subject in enumerate(subjects):
                sub_head_mapping[subject_idx][0][subject[1]] = 1
                sub_tail_mapping[subject_idx][0][subject[2]] = 1
            sub_tail_mapping = sub_tail_mapping.to(repeated_encoded_text)
            sub_head_mapping = sub_head_mapping.to(repeated_encoded_text)
            pred_objs = model.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, repeated_encoded_text, mask)
            for subject_idx, subject in enumerate(subjects):
                sub = subject[0]
                sub = ''.join([i.lstrip("##") for i in sub])
                sub = ' '.join(sub.split('[unused1]'))
                sub_list.append(sub)
                for rel_id, obj_head, obj_tail in zip(*np.where(pred_objs.cpu()[subject_idx] > threshold)):
                    rel = id2rel[str(int(rel_id))]
                    obj = tokens[obj_head: obj_tail]
                    obj = ''.join([i.lstrip("##") for i in obj])
                    obj = ' '.join(obj.split('[unused1]'))
                    triple_list.append((sub, rel, obj))

            triple_set = set()
            for s, r, o in triple_list:
                triple_set.add((s.strip(), r.strip(), o.strip()))
            pred_list = list(triple_set)
            sub_list = list(set(sub_list))
        else:
            pred_list = []
            sub_list = []
        return pred_list,sub_list


    def casglobalpointer_test(self,model,encoded_text,tokens,id2rel,mask,threshold=0,h_bar=0.5,t_bar=0.5):
        pred_sub_heads,pred_sub_tails = model.get_subs(encoded_text)
        sub_heads, sub_tails = np.where(pred_sub_heads.cpu()[0] > h_bar)[0], np.where(pred_sub_tails.cpu()[0] > t_bar)[0]
        subjects = []
        for sub_head in sub_heads:
            sub_tail = sub_tails[sub_tails >= sub_head]
            if len(sub_tail) > 0:
                sub_tail = sub_tail[0]
                subject = tokens[sub_head: sub_tail]
                subjects.append((subject, sub_head, sub_tail))

        if subjects:
            triple_list = []
            sub_list = []
            # [subject_num, seq_len, bert_dim]
            repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
            # [subject_num, 1, seq_len],每个主语构建一个样本
            sub_head_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
            sub_tail_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
            for subject_idx, subject in enumerate(subjects):
                sub_head_mapping[subject_idx][0][subject[1]] = 1
                sub_tail_mapping[subject_idx][0][subject[2]] = 1
            sub_tail_mapping = sub_tail_mapping.to(repeated_encoded_text)
            sub_head_mapping = sub_head_mapping.to(repeated_encoded_text)
            pred_objs = model.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, repeated_encoded_text, mask)
            for subject_idx, subject in enumerate(subjects):
                sub = subject[0]
                sub = ''.join([i.lstrip("##") for i in sub])
                sub = ' '.join(sub.split('[unused1]'))
                sub_list.append(sub)
                for rel_id, obj_head, obj_tail in zip(*np.where(pred_objs.cpu()[subject_idx] > threshold)):
                    rel = id2rel[str(int(rel_id))]
                    obj = tokens[obj_head: obj_tail]
                    obj = ''.join([i.lstrip("##") for i in obj])
                    obj = ' '.join(obj.split('[unused1]'))
                    triple_list.append((sub, rel, obj))

            triple_set = set()
            for s, r, o in triple_list:
                triple_set.add((s.strip(), r.strip(), o.strip()))
            pred_list = list(triple_set)
            sub_list = list(set(sub_list))
        else:
            pred_list = []
            sub_list = []
        return pred_list,sub_list





