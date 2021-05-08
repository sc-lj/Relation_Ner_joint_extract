import torch.optim as optim
from torch import nn
import os
import data_loader
import torch.nn.functional as F
from .Loss import FocalLoss,GlobalCrossEntropy
import torch
import numpy as np
import json
import time


class Framework(object):
    def __init__(self, con):
        self.config = con
        self.focal_loss = FocalLoss()

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(self.config.log_dir, self.config.log_save_name), 'a+') as f_log:
                f_log.write(s + '\n')

    def train(self, model_pattern):
        # initialize the model
        ori_model = model_pattern(self.config)
        ori_model.cuda()
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
                group1=['layer.0.','layer.1.','layer.2.','layer.3.']
                group2=['layer.4.','layer.5.','layer.6.','layer.7.']
                group3=['layer.8.','layer.9.','layer.10.','layer.11.']
                group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
                grouped_params = [
                    {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.01},
                    {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.01, 'lr': self.config.learning_rate/1.6},
                    {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.01, 'lr': self.config.learning_rate},
                    {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.01, 'lr': self.config.learning_rate*1.6},
                    {'params': [p for n, p in params if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.0},
                    {'params': [p for n, p in params if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.0, 'lr': self.config.learning_rate/1.6},
                    {'params': [p for n, p in params if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.0, 'lr': self.config.learning_rate},
                    {'params': [p for n, p in params if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.0, 'lr': self.config.learning_rate*1.6},
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

        # whether use multi GPU
        if self.config.multi_gpu:
            model = nn.DataParallel(ori_model)
        else:
            model = ori_model

        # define the loss function
        def loss(gold, pred, mask,use_focal=False):
            pred = pred.squeeze(-1)
            los = F.binary_cross_entropy(pred, gold, reduction='none')
            if los.shape != mask.shape:
                mask = mask.unsqueeze(-1)
            los = torch.sum(los * mask) / torch.sum(mask)
            if self.config.use_focal and use_focal:
                los += self.focal_loss(pred,gold,None)
            return los
        
        def pointer_loss(gold,pred,threshold=0):
            loss_func = GlobalCrossEntropy()
            los = loss_func(gold,pred,threshold)
            return los
        
        def pointer_sub_loss(gold,pred,use_focal=False):
            los = F.binary_cross_entropy(pred, gold)
            if self.config.use_focal and use_focal:
                los += self.focal_loss(pred,gold)
            return los

        # check the checkpoint dir
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        # check the log dir
        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)

        # get the data loader
        train_data_loader = data_loader.get_loader(self.config, prefix=self.config.train_prefix,num_workers=5)
        dev_data_loader = data_loader.get_loader(self.config, prefix=self.config.dev_prefix, is_test=True,num_workers=5)

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
            train_data_prefetcher = data_loader.DataPreFetcher(train_data_loader)
            data = train_data_prefetcher.next()
            while data is not None:
                if self.config.model_name == "casrel":
                    pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails = model(data)

                    sub_heads_loss = loss(data['sub_heads'], pred_sub_heads, data['mask'],True)
                    sub_tails_loss = loss(data['sub_tails'], pred_sub_tails, data['mask'],True)
                    obj_heads_loss = loss(data['obj_heads'], pred_obj_heads, data['mask'])
                    obj_tails_loss = loss(data['obj_tails'], pred_obj_tails, data['mask'])
                    total_loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)
                elif self.config.model_name == "globalpointer":
                    pred_subs, pred_objs = model(data)
                    sub_loss = pointer_loss(data['pointer_sub'], pred_subs)
                    # pred_subs = torch.sigmoid(pred_subs)
                    # sub_loss = pointer_sub_loss(data['pointer_sub'], pred_subs,True)
                    obj_loss = pointer_loss(data['pointer_obj'], pred_objs)
                    total_loss = 1.2*sub_loss+1.*obj_loss
                else:
                    raise ValueError(f"{self.config.model_name} not in [casrel,globalpointer]")


                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                global_step += 1
                loss_sum += total_loss.item()
                sub_loss_sum += sub_loss.item()
                rel_loss_sum += obj_loss.item()

                if global_step % self.config.period == 0:
                    cur_loss = loss_sum / self.config.period
                    cur_sub_loss = sub_loss_sum / self.config.period
                    cur_rel_loss = rel_loss_sum / self.config.period
                    elapsed = time.time() - start_time
                    self.logging("epoch: {:3d}, step: {:4d}, speed: {:5.2f}ms/b, train loss: {:5.3f},subject loss:{:5.3f},rel_object_loss:{:5.3f}".
                                 format(epoch, global_step, elapsed * 1000 / self.config.period, cur_loss,cur_sub_loss,cur_rel_loss))
                    loss_sum = 0
                    sub_loss_sum = 0
                    rel_loss_sum = 0
                    start_time = time.time()

                data = train_data_prefetcher.next()

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
                    if not self.config.debug:
                        torch.save(ori_model.state_dict(), path)
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
                else:
                    raise ValueError(f"{self.config.model_name} not in [casrel,globalpointer]")
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

        print("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))
        print("correct_sub_num: {:3d}, predict_sub_num: {:3d}, gold_sub_num: {:3d}".format(correct_sub_num, predict_sub_num, gold_sub_num))

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