from torch.utils.data import DataLoader, Dataset
import json
import os
from transformers import BertTokenizer
import torch
from utils import HBTokenizer,strQ2B
import numpy as np
from random import choice
import ahocorasick
import copy
from utils.registry import register
from functools import partial
import re
import pickle
BERT_MAX_LEN = 512

dataLoader = {}
register = partial(register,registry=dataLoader)
def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


@register("cmed")
class CMEDDataset(Dataset):
    def __init__(self, config, prefix, is_test):
        self.config = config
        self.prefix = prefix
        self.is_test = is_test
        self.tokenizer = HBTokenizer.from_pretrained(config.pretrain_path)

        if self.config.debug:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))[:500]
        else:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))
        self.rel2id = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))[1]

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        ins_json_data = self.json_data[idx]
        text = ins_json_data['text']
        text = ' '.join(text.split()[:self.config.max_len])
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > BERT_MAX_LEN:
            tokens = tokens[: BERT_MAX_LEN]
        text_len = len(tokens)

        if not self.is_test:
            s2ro_map = {}
            for triple in ins_json_data['triple_list']:
                triple = (self.tokenizer.tokenize(triple[0])[1:-1], triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    s2ro_map[sub].append((obj_head_idx, obj_head_idx + len(triple[2]) - 1, self.rel2id[triple[1]]))

            if s2ro_map:
                # token_ids, segment_ids = self.tokenizer.encode(first=text)
                outputs = self.tokenizer.encode_plus(text,return_token_type_ids=True)
                token_ids, segment_ids = outputs['input_ids'],outputs['token_type_ids']

                masks = segment_ids
                if len(token_ids) > text_len:
                    token_ids = token_ids[:text_len]
                    masks = masks[:text_len]
                token_ids = np.array(token_ids)
                masks = np.array(masks) + 1
                sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
                # 所有 subject 的头尾index
                for s in s2ro_map:
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                # 随机选择一个subject
                sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
                sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1
                # 选择的某个subject 对应的所有 object的头尾index以及对应的关系
                obj_heads, obj_tails = np.zeros((text_len, self.config.rel_num)), np.zeros((text_len, self.config.rel_num))
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1
                return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, ins_json_data['triple_list'], tokens
            else:
                return None
        else:
            # token_ids, segment_ids = self.tokenizer.encode(first=text)
            outputs = self.tokenizer.encode_plus(text,return_token_type_ids=True)
            token_ids, segment_ids = outputs['input_ids'],outputs['token_type_ids']
            masks = segment_ids
            if len(token_ids) > text_len:
                token_ids = token_ids[:text_len]
                masks = masks[:text_len]
            token_ids = np.array(token_ids)
            masks = np.array(masks) + 1
            sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
            sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
            obj_heads, obj_tails = np.zeros((text_len, self.config.rel_num)), np.zeros((text_len, self.config.rel_num))
            return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, ins_json_data['triple_list'], tokens


def cmed_collate_fn(batch,rel_num):
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[2], reverse=True)
    token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, tokens = zip(*batch)
    cur_batch = len(batch)
    max_text_len = max(text_len)
    batch_token_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_masks = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_sub_heads = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_tails = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_head = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_tail = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_obj_heads = torch.Tensor(cur_batch, max_text_len, rel_num).zero_() #数字表示是关系数量
    batch_obj_tails = torch.Tensor(cur_batch, max_text_len, rel_num).zero_()

    for i in range(cur_batch):
        batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
        batch_sub_heads[i, :text_len[i]].copy_(torch.from_numpy(sub_heads[i]))
        batch_sub_tails[i, :text_len[i]].copy_(torch.from_numpy(sub_tails[i]))
        batch_sub_head[i, :text_len[i]].copy_(torch.from_numpy(sub_head[i]))
        batch_sub_tail[i, :text_len[i]].copy_(torch.from_numpy(sub_tail[i]))
        batch_obj_heads[i, :text_len[i], :].copy_(torch.from_numpy(obj_heads[i]))
        batch_obj_tails[i, :text_len[i], :].copy_(torch.from_numpy(obj_tails[i]))

    return {'token_ids': batch_token_ids,
            'mask': batch_masks,
            'sub_heads': batch_sub_heads,
            'sub_tails': batch_sub_tails,
            'sub_head': batch_sub_head,
            'sub_tail': batch_sub_tail,
            'obj_heads': batch_obj_heads,
            'obj_tails': batch_obj_tails,
            'triples': triples,
            'tokens': tokens}

@register("baidu")
class BaiduDataset(Dataset):
    def __init__(self,config, prefix, is_test):
        self.config = config
        self.prefix = prefix
        self.is_test = is_test
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrain_path)
        if self.config.debug:
            with open(os.path.join(self.config.data_path, prefix + '.json'),'r',encoding="utf-8") as f:
                self.json_data = f.readlines()[:500]
        else:
            with open(os.path.join(self.config.data_path, prefix + '.json'),'r') as f:
                self.json_data = f.readlines()
        self.rel2id = json.load(open(os.path.join(self.config.data_path, 'rel2id.json'),'r',encoding="utf-8"))
        if not os.path.exists(os.path.join(self.config.data_path,prefix+".pkl")) or self.config.debug:
            self.load_dataset()
            with open(os.path.join(self.config.data_path,prefix+".pkl"),'wb') as f:
                pickle.dump(self.json_data,f)
        else:
            with open(os.path.join(self.config.data_path,prefix+".pkl"),'rb') as f:
                self.json_data = pickle.load(f)

    def load_dataset(self):
        for index,line in enumerate(self.json_data):
            line = json.loads(line)
            line = self.parse_single_line(line)
            self.json_data[index] = line

    def parse_single_line(self, line):
        text = line['text']
        text = strQ2B(text)
        line['text'] = text
        drop_index = []
        new_spo = []
        for i, spo in enumerate(line['spo_list']):
            tree = ahocorasick.Automaton()
            index = 0
            relation = spo['predicate']
            relid = self.rel2id[relation]
            object_value = strQ2B(spo['object']['@value']) #主语
            spo_, new = self.get_complex(spo, text)
            if new:
                new_spo.append(spo_)
            spo['object']['@value'] = object_value
            subject_value = strQ2B(spo['subject']) #谓语
            spo['subject'] = subject_value
            if len(object_value.strip()) == 0 or len(subject_value.strip()) == 0:
                drop_index.insert(0, i)
                continue
            tree.add_word(object_value, (index, object_value))
            index += 1
            tree.add_word(subject_value, (index, subject_value))
            index += 1
            tree.dump()
            object_type = spo['object_type']["@value"]
            subject_type = spo['subject_type']
            tree.make_automaton()

            if object_value == subject_value:
                if relation == "改编自":
                    objects, subjects = self.match_adapt(text,object_value)
                    spo['h'] = subjects
                    spo['t'] = objects
                elif relation == "所属专辑":
                    objects, subjects = self.match_album(text, object_value)
                    spo['h'] = subjects
                    spo['t'] = objects
                elif relation == "主题曲":
                    objects, subjects =self.match_theme(text, object_value)
                    spo['h'] = subjects
                    spo['t'] = objects
                else:
                    word_index = [index for index, _ in list(tree.iter(text))]
                    objects = {"name": object_value, "pos": [word_index[0] - len(object_value) + 1, word_index[0] + 1]}
                    subjects = {"name": objects, "pos": [word_index[-1] - len(objects) + 1, word_index[-1] + 1]}
                    spo['h'] = subjects
                    spo['t'] = objects
            else:
                for j, word in tree.iter(text):
                    if word[1] == subject_value:
                        if 'h' not in spo:
                            spo['h'] = {"name": subject_value, "pos": [j-len(subject_value)+1, j+1]}
                    elif word[1] == object_value:
                        if "t" not in spo:
                            spo['t'] = {"name": object_value, "pos": [j-len(object_value)+1, j+1]}
            if "t" not in spo or "h" not in spo:
                print(spo,text)

        if len(drop_index):
            spo_list = line['spo_list']
            for ind in drop_index:
                del spo_list[ind]
            if len(spo_list):
                line['spo_list'] = spo_list
        if len(new_spo):
            line['spo_list'].extend(new_spo)
        return line

    def match_adapt(self, text, word):
        """匹配‘改编自’关系中，obeject和suject相同的情况"""
        compiles1 = re.compile("(%s).*?改编自.*?(%s)" % (word, word))
        compiles2 = re.compile(".*?(%s).{,3}改编" % (word))
        compiles3 = re.compile("剧.{,3}(%s)" % (word))
        tree = ahocorasick.Automaton()
        tree.add_word(word, (0, word))
        tree.make_automaton()

        for a in compiles1.finditer(text):
            object_start = a.start(2)
            object_end = a.end(2)
            subject_start = a.start(1)
            subject_end = a.end(1)
            objects = {"name": word, "pos": [object_start, object_end]}
            subjects = {"name": word, "pos": [subject_start, subject_end]}
            return objects,subjects
        for a in compiles2.finditer(text):
            object_start = a.start(1)
            object_end = a.end(1)
            objects = {"name": word, "pos": [object_start, object_end]}

            word_index = [(index, abs(object_end-index)) for index,_ in list(tree.iter(text)) if index+1 != object_end]
            word_index = sorted(word_index,key=lambda x:x[1])
            word_index = word_index[0][0]
            subjects = {"name": word, "pos": [word_index-len(word)+1, word_index+1]}
            return objects, subjects

        for a in compiles3.finditer(text):
            subject_start = a.start(1)
            subject_end = a.end(1)
            subjects = {"name": word, "pos": [subject_start, subject_end]}

            word_index = [(index, abs(subject_end-index)) for index,_ in list(tree.iter(text)) if index+1 != subject_end]
            word_index = sorted(word_index,key=lambda x:x[1])
            word_index = word_index[0][0]
            objects = {"name": word, "pos": [word_index-len(word)+1, word_index+1]}
            return objects, subjects

        word_index = [index for index, _ in list(tree.iter(text))]
        subjects = {"name": word, "pos": [word_index[0] - len(word) + 1, word_index[0] + 1]}
        objects = {"name": word, "pos": [word_index[-1] - len(word) + 1, word_index[-1] + 1]}
        return objects, subjects

    def match_album(self, text, word):
        compiles1 = re.compile("[曲歌].{,5}(%s)" % (word))
        compiles2 = re.compile("(%s).{,5}专辑" % (word))
        compiles3 = re.compile("专辑.{,3}(%s)" % (word))
        tree = ahocorasick.Automaton()
        tree.add_word(word, (0, word))
        tree.make_automaton()
        for a in compiles3.finditer(text):
            object_start = a.start(1)
            object_end = a.end(1)
            objects = {"name": word, "pos": [object_start, object_end]}

            word_index = [(index, abs(object_end-index)) for index,_ in list(tree.iter(text)) if index+1 != object_end]
            word_index = sorted(word_index, key=lambda x: x[1])
            word_index = word_index[0][0]
            subjects = {"name": word, "pos": [word_index-len(word)+1, word_index+1]}
            return objects, subjects

        for a in compiles2.finditer(text):
            object_start = a.start(1)
            object_end = a.end(1)
            objects = {"name": word, "pos": [object_start, object_end]}

            word_index = [(index, abs(object_end-index)) for index,_ in list(tree.iter(text)) if index+1 != object_end]
            word_index = sorted(word_index, key=lambda x: x[1])
            word_index = word_index[0][0]
            subjects = {"name": word, "pos": [word_index-len(word)+1, word_index+1]}
            return objects, subjects

        for a in compiles1.finditer(text):
            subject_start = a.start(1)
            subject_end = a.end(1)
            subjects = {"name": word, "pos": [subject_start, subject_end]}

            word_index = [(index, abs(subject_end-index)) for index,_ in list(tree.iter(text)) if index+1 != subject_end]
            if len(word_index) == 0:
                word_index = subject_end-1
            else:
                word_index = sorted(word_index, key=lambda x: x[1])
                word_index = word_index[0][0]
            objects = {"name": word, "pos": [word_index-len(word)+1, word_index+1]}
            return objects, subjects

        word_index = [index for index, _ in list(tree.iter(text))]
        objects = {"name": word, "pos": [word_index[0] - len(word) + 1, word_index[0] + 1]}
        subjects = {"name": word, "pos": [word_index[-1] - len(word) + 1, word_index[-1] + 1]}
        return objects, subjects

    def match_theme(self, text, word):
        compiles1 = re.compile("[剧影目].{,5}(%s)" % (word))
        tree = ahocorasick.Automaton()
        tree.add_word(word, (0, word))
        tree.make_automaton()
        for a in compiles1.finditer(text):
            subject_start = a.start(1)
            subject_end = a.end(1)
            subjects = {"name": word, "pos": [subject_start, subject_end]}

            word_index = [(index, abs(subject_end-index)) for index,_ in list(tree.iter(text)) if index+1 != subject_end]
            word_index = sorted(word_index, key=lambda x: x[1])
            word_index = word_index[0][0]
            objects = {"name": word, "pos": [word_index-len(word)+1, word_index+1]}
            return objects, subjects

        word_index = [index for index, _ in list(tree.iter(text))]
        objects = {"name": word, "pos": [word_index[0] - len(word) + 1, word_index[0] + 1]}
        subjects = {"name": word, "pos": [word_index[-1] - len(word) + 1, word_index[-1] + 1]}
        return objects, subjects

    def get_complex(self, spo,  text):
        """添加复杂关系"""
        spo_ = copy.deepcopy(spo)
        new = False
        del spo_['object_type']['@value']
        if len(spo_['object_type']) == 1:
            tree = ahocorasick.Automaton()
            index = 0
            items = copy.deepcopy(spo_['object_type'])
            for k, v in items.items():
                predicate = k.lower()+"_"+v.lower()
                object_type = v
                object_value = strQ2B(spo_['object'][k])
                subject_value = strQ2B(spo_['subject'])
                tree.add_word(object_value, (index, object_value))
                index += 1
                tree.add_word(subject_value, (index, subject_value))
                tree.make_automaton()
                if object_value == subject_value:
                    word_index = [index for index, _ in list(tree.iter(text))]
                    objects = {"name": object_value, "pos": [word_index[0] - len(object_value) + 1, word_index[0] + 1]}
                    subjects = {"name": subject_value, "pos": [word_index[-1] - len(subject_value) + 1, word_index[-1] + 1]}
                    spo_['h'] = subjects
                    spo_['t'] = objects
                else:
                    for index, word in tree.iter(text):
                        if word[1] == subject_value:
                            spo_['h'] = {"name": subject_value, "pos": [index - len(subject_value) + 1, index + 1]}
                        elif word[1] == object_value:
                            spo_['t'] = {"name": object_value, "pos": [index - len(object_value) + 1, index + 1]}
                    if "t" not in spo_ or "h" not in spo_:
                        print(spo_,text)

                spo_['object_type']["@value"] = object_type
                spo_['object']["@value"] = object_value
                spo_['predicate'] = predicate
                new = True
        return spo_, new

    def __len__(self):
        return len(self.json_data)
    
    def tokenize_(self,text):
        tokens = text.split(" ")
        re_tokens = ['[CLS]']
        cur_pos = 0
        new_index = [-1]
        start = 0
        for token in tokens:
            root_tokens = self.tokenizer.basic_tokenizer.tokenize(token, never_split=self.tokenizer.all_special_tokens)
            for t in root_tokens:
                split_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(t)
                re_tokens += split_tokens
                new_index.extend([start]*len(split_tokens))
                start += len(t)
            start += 1 # 表示有空格
        start -= 1 # 最后一个list表示文本结束，后面不应有空格
        re_tokens.append('[SEP]')
        new_index.append(start)
        return re_tokens,new_index
    
    def tokenize(self,text):
        tokens = text.split(" ")
        re_tokens = ['[CLS]']
        cur_pos = 0
        new_index = [-1]
        start = 0
        for token in tokens:
            root_tokens = self.tokenizer.basic_tokenizer.tokenize(token, never_split=self.tokenizer.all_special_tokens)
            for t in root_tokens:
                split_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(t)
                re_tokens += split_tokens
                if len(split_tokens)==1:
                        new_index.append(start)
                        start += len(t.replace("##",""))
                else:
                    if "[UNK]" in split_tokens:
                        print("[UNK] in ",split_tokens)
                    for t1 in split_tokens:
                            new_index.append(start)
                            start += len(t1.replace("##",""))
            re_tokens.append("[unused1]")
            new_index.append(start)
            start += 1 # 表示有空格
        start -= 1 # 最后一个list表示文本结束，后面不应有空格
        del new_index[-1]
        del re_tokens[-1]
        re_tokens.append('[SEP]')
        new_index.append(start)
        return re_tokens,new_index


    def check(self,pos_head,pos_tail,tokens,sub,text):
        substring = tokens[pos_head:pos_tail]
        substring = [a.replace("##","") for a in substring if a!="[unused1]"]
        subtoken = self.tokenizer.tokenize(sub,add_special_tokens=False)
        subtoken = [a.replace("##","") for a in subtoken]
        if subtoken != substring:
            print(substring,subtoken,text)
            print()

    def get_index(self,pos,new_index):
            new_pos = [-1,-1]
            for i in range(len(new_index)-1):
                if new_index[i]<=pos[0]<new_index[i+1]:
                    value = new_index[i]
                    new_pos[0] = new_index.index(value)
                
                if new_index[i]==pos[1]:
                    value = new_index[i]
                    new_pos[1] = new_index.index(value)
                elif new_index[i]<pos[1]<new_index[i+1]:
                    value = new_index[i+1]
                    new_pos[1] = new_index.index(value)
            if new_index[-1]==pos[1]:
                value = new_index[-1]
                new_pos[1] = new_index.index(value)
            assert -1 not in new_pos
            if -1 in new_pos or new_pos[0]>new_pos[1]:
                print(new_pos)
            return new_pos

    def __getitem__(self,index):
        ins_json_data = self.json_data[index]
        text = ins_json_data['text']
        tokens,new_index = self.tokenize(text)
        if len(tokens) > BERT_MAX_LEN:
            tokens = tokens[:BERT_MAX_LEN]
        text_len = len(tokens)

        if not self.is_test:
            s2ro_map = {}
            triples = []
            for triple in ins_json_data['spo_list']:
                pos_head = triple["h"]['pos']
                pos_tail = triple['t']['pos']
                # triple = (self.tokenizer.tokenize(triple['subject'])[1:-1], triple[1], self.tokenizer.tokenize(triple["object"]['@value'])[1:-1])
                # sub_head_idx = find_head_idx(tokens, triple[0])
                # obj_head_idx = find_head_idx(tokens, triple[2])
                triple = (triple['subject'], triple['predicate'], triple["object"]['@value'])
                triples.append(triple)
                sub_head_idx,sub_tail_idx = self.get_index(pos_head,new_index)
                self.check(sub_head_idx,sub_tail_idx,tokens,triple[0],text)
                obj_head_idx,obj_tail_idx = self.get_index(pos_tail,new_index)
                self.check(obj_head_idx,obj_tail_idx,tokens, triple[2],text)
                if sub_head_idx != -1 and obj_head_idx != -1:
                    # sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    sub = (sub_head_idx, sub_tail_idx)
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    s2ro_map[sub].append((obj_head_idx, obj_tail_idx, self.rel2id[triple[1]]))

            if s2ro_map:
                outputs = self.tokenizer.encode_plus(text,max_length=self.config.max_len,pad_to_max_length=True,return_attention_mask=True)
                token_ids, masks = np.array(outputs['input_ids']),np.array(outputs['attention_mask'])
                if len(token_ids) > text_len:
                    token_ids = token_ids[:text_len]
                    masks = masks[:text_len]
                token_ids = np.array(token_ids)
                sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
                # 所有 subject 的头尾index
                for s in s2ro_map:
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                # 随机选择一个subject
                sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
                sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1
                # 选择的某个subject 对应的所有 object的头尾index以及对应的关系
                obj_heads, obj_tails = np.zeros((text_len, self.config.rel_num)), np.zeros((text_len, self.config.rel_num))
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1
                return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, tokens
            else:
                return None
        else:
            triples = [(triple['subject'], triple['predicate'], triple["object"]['@value']) for triple in ins_json_data['spo_list']]
            outputs = self.tokenizer.encode_plus(text,max_length=self.config.max_len,pad_to_max_length=True,return_attention_mask=True)
            token_ids, masks = np.array(outputs['input_ids']),np.array(outputs['attention_mask'])
            if len(token_ids) > text_len:
                token_ids = token_ids[:text_len]
                masks = masks[:text_len]
            token_ids = np.array(token_ids)
            sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
            sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
            obj_heads, obj_tails = np.zeros((text_len, self.config.rel_num)), np.zeros((text_len, self.config.rel_num))
            return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, tokens


def get_loader(config, prefix, is_test=False, num_workers=0):
    # dataset = CMEDDataset(config, prefix, is_test)
    dataset = dataLoader[config.dataset](config, prefix, is_test)
    collate_fn = lambda x: cmed_collate_fn(x,config.rel_num)
    if not is_test:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return data_loader


class DataPreFetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = self.next_data[k].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data
