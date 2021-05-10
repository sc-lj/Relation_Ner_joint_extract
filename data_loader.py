from torch.utils.data import DataLoader, Dataset, distributed
import json
import os
from transformers import BertTokenizer
import torch
from utils import HBTokenizer,strQ2B,BDTokenizer
import numpy as np
from random import choice
import ahocorasick
import copy
from utils.registry import register
from functools import partial
import re
import itertools
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


@register("CMED")
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
        pointer_sub, pointer_obj= [],[]
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
                return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails,pointer_sub, pointer_obj, ins_json_data['triple_list'], tokens
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
            return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails,pointer_sub, pointer_obj, ins_json_data['triple_list'], tokens


@register("baidu")
class BaiduDataset(Dataset):
    def __init__(self,config, prefix, is_test):
        self.config = config
        self.prefix = prefix
        self.is_test = is_test
        self.tokenizer = BDTokenizer.from_pretrained(config.pretrain_path)
        self.vocab = self.tokenizer.vocab
        self.rel2id = json.load(open(os.path.join(self.config.data_path, 'rel2id.json'),'r',encoding="utf-8"))
        # if self.config.debug:
        #     with open(os.path.join(self.config.data_path, prefix + '.json'),'r',encoding="utf-8") as f:
        #         self.json_data = f.readlines()[:500]
        #         self.load_dataset()
        # else:
        #     if not os.path.exists(os.path.join(self.config.data_path,prefix+".pkl")):
        #         with open(os.path.join(self.config.data_path, prefix + '.json'),'r') as f:
        #             self.json_data = f.readlines()
        #         self.load_dataset()
        #         with open(os.path.join(self.config.data_path,prefix+".pkl"),'wb') as f:
        #             pickle.dump(self.json_data,f)
        #     else:
        #         with open(os.path.join(self.config.data_path,prefix+".pkl"),'rb') as f:
        #             self.json_data = pickle.load(f)
        with open(os.path.join(self.config.data_path, prefix + '.json'),'r') as f:
            self.json_data = f.readlines()

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
            if len(spo['object'])==2:
                spo_1 = self.get_complex_1(spo,text)
                spo_2 = self.get_complex_2(spo,text)
                new_spo.extend(spo_1)
                new_spo.extend(spo_2)
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
                    subjects = {"name": subject_value, "pos": [word_index[-1] - len(subject_value) + 1, word_index[-1] + 1]}
                    spo['h'] = subjects
                    spo['t'] = objects
            else:
                word_index = [(index,w[1]) for index, w in list(tree.iter(text))]
                subjects,objects = self.choice_subject_object(object_value,subject_value,word_index,text)
                spo['h'] = subjects
                spo['t'] = objects
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
            # 《左耳》是2015年由陈慧翎执导的青春电视剧,改编自饶雪漫同名小说《左耳》
            object_start = a.start(2)
            object_end = a.end(2)
            subject_start = a.start(1)
            subject_end = a.end(1)
            objects = {"name": word, "pos": [object_start, object_end]}
            subjects = {"name": word, "pos": [subject_start, subject_end]}
            return objects,subjects
        for a in compiles2.finditer(text):
            # 《历史的天空》,是根据同名长篇小说《历史的天空》改编,高希希执导,张丰毅、1李雪健、林永健、殷桃等主演的军事历史题材电视剧
            object_start = a.start(1)
            object_end = a.end(1)
            objects = {"name": word, "pos": [object_start, object_end]}

            word_index = [(index, abs(object_end-index)) for index,_ in list(tree.iter(text)) if index+1 != object_end]
            word_index = sorted(word_index,key=lambda x:x[1])
            word_index = word_index[0][0]
            subjects = {"name": word, "pos": [word_index-len(word)+1, word_index+1]}
            return objects, subjects

        for a in compiles3.finditer(text):# 《将夜》（objects）是起点白金作家猫腻的第五部作品,现由杨阳导演改编同名电视剧《将夜》（subjects）杀青了,预告已经在腾讯视频播出
            subject_start = a.start(1)
            subject_end = a.end(1)
            subjects = {"name": word, "pos": [subject_start, subject_end]}

            word_index = [(index, abs(subject_end-index)) for index,_ in list(tree.iter(text)) if index+1 != subject_end]
            word_index = sorted(word_index,key=lambda x:x[1])
            word_index = word_index[0][0]
            objects = {"name": word, "pos": [word_index-len(word)+1, word_index+1]}
            return objects, subjects
        # 其他情况默认主语在前，谓语在后
        word_index = [index for index, _ in list(tree.iter(text))]
        subjects = {"name": word, "pos": [word_index[0] - len(word) + 1, word_index[0] + 1]}
        objects = {"name": word, "pos": [word_index[-1] - len(word) + 1, word_index[-1] + 1]}
        return objects, subjects

    def match_album(self, text, word): 
        """匹配‘所属专辑’关系中，obeject和suject相同的情况"""
        compiles1 = re.compile("[曲歌].{,5}(%s)" % (word))
        compiles2 = re.compile("(%s).{,5}专辑" % (word))
        compiles3 = re.compile("专辑.{,3}(%s)" % (word))
        tree = ahocorasick.Automaton()
        tree.add_word(word, (0, word))
        tree.make_automaton()
        for a in compiles3.finditer(text):
            # 专辑名称《这般发生》（objects）专辑背景这是苏有朋1993年出的第二张专辑,其中专辑同名作《这般发生》
            object_start = a.start(1)
            object_end = a.end(1)
            objects = {"name": word, "pos": [object_start, object_end]}

            word_index = [(index, abs(object_end-index)) for index,_ in list(tree.iter(text)) if index+1 != object_end]
            word_index = sorted(word_index, key=lambda x: x[1])
            word_index = word_index[0][0]
            subjects = {"name": word, "pos": [word_index-len(word)+1, word_index+1]}
            return objects, subjects

        for a in compiles2.finditer(text):
            # 出自《一场游戏一场梦》（objects）专辑的有五首:《故事的角色》(《一场游戏一场梦》)
            object_start = a.start(1)
            object_end = a.end(1)
            objects = {"name": word, "pos": [object_start, object_end]}

            word_index = [(index, abs(object_end-index)) for index,_ in list(tree.iter(text)) if index+1 != object_end]
            word_index = sorted(word_index, key=lambda x: x[1])
            word_index = word_index[0][0]
            subjects = {"name": word, "pos": [word_index-len(word)+1, word_index+1]}
            return objects, subjects

        for a in compiles1.finditer(text):
            # 《我们的爱情》是李承铉所演唱的一张专辑,该专辑由华音鼎天(北京)音乐文化有限公司于2012年发行,该专辑共收录三首歌曲,分别为《我们的爱情》《妮可与国王》
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

    def match_theme(self, text, word): # 主题曲
        compiles1 = re.compile("[剧影目].{,5}(%s)" % (word))
        tree = ahocorasick.Automaton()
        tree.add_word(word, (0, word))
        tree.make_automaton()
        for a in compiles1.finditer(text):
            # 电影《绝世高手》（subjects）曝同名主题曲赵英俊作词作曲又献唱电影明晚开启全国万场点映由卢正雨自导自演的暑期爆笑喜剧电影《绝世高手》今日曝光同名主题曲《绝世高手》该首主题曲由赵英俊作词作曲并演唱
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

    def get_complex_1(self,spo,text):
        # 构建复杂关系中{"predicate": "获奖", "object": {"inWork": "线", "@value": "十大金曲"}, "subject": "刘惜君"} 构建(刘惜君,获奖-inWork,线)三元组
        spo_ = copy.deepcopy(spo)
        spos = []
        old_predicate = spo_['predicate']
        items = copy.deepcopy(spo_['object_type'])
        value_ = items.pop("@value")
        subject_value = strQ2B(spo_['subject'])
        for k,v in items.items():
            tree = ahocorasick.Automaton()
            index = 0
            predicate = old_predicate+"_"+k.lower()
            object_type = v
            object_value = strQ2B(spo_['object'][k])
            tree.add_word(object_value, (index, object_value))
            index += 1
            tree.add_word(subject_value, (index, subject_value))
            tree.make_automaton()
            word_index = [(index,w[1]) for index, w in list(tree.iter(text))]
            subjects,objects = self.choice_subject_object(object_value,subject_value,word_index,text)
            spo_['h'] = subjects
            spo_['t'] = objects

            spo_['object_type']["@value"] = object_type
            spo_['object']["@value"] = object_value
            spo_['predicate'] = predicate
            spos.append(spo_)
        return spos

    def get_complex_2(self,spo,text):
        # 获取复杂关系中，{"inWork": "线", "@value": "十大金曲"} 构建(十大金曲,inWork,线)三元组
        spo_ = copy.deepcopy(spo)
        items = copy.deepcopy(spo_['object_type'])
        spos = []
        value_ = items.pop("@value")
        subject_value = strQ2B(spo_['object']["@value"])
        for k,v in items.items():
            tree = ahocorasick.Automaton()
            index = 0
            predicate = k.lower()
            object_type = v
            object_value = strQ2B(spo_['object'][k])
            tree.add_word(object_value, (index, object_value))
            index += 1
            tree.add_word(subject_value, (index, subject_value))
            tree.make_automaton()
            word_index = [(index,w[1]) for index, w in list(tree.iter(text))]
            subjects,objects = self.choice_subject_object(object_value,subject_value,word_index,text)
            spo_['h'] = subjects
            spo_['t'] = objects

            spo_['object_type']["@value"] = object_type
            spo_['object']["@value"] = object_value
            spo_['predicate'] = predicate
            spo_['subject'] = subject_value
            spos.append(spo_)
        return spos

    def choice_subject_object(self,object_value,subject_value,word_index,text):
        if  object_value == subject_value:
            word_index = [index for index, _ in word_index]
            if len(word_index)==1:
                objects = {"name": object_value, "pos": [word_index[-1] - len(object_value) + 1, word_index[-1] + 1]}
                subjects = {"name": subject_value, "pos": [word_index[0] - len(subject_value) + 1, word_index[0] + 1]}
            else:
                # 主语在前，谓语在后
                subject_index = choice(list(range(len(word_index)-1)))
                object_index = subject_index+1
                objects = {"name": object_value, "pos": [word_index[object_index] - len(object_value) + 1, word_index[object_index] + 1]}
                subjects = {"name": subject_value, "pos": [word_index[subject_index] - len(subject_value) + 1, word_index[subject_index] + 1]}
        else:
            is_nest = False
            if subject_value in object_value or object_value in subject_value:
                is_nest = True
                subjects,objects = self.drop_nest_words(word_index,subject_value,object_value)
            else:
                subjects = [(index,w)for index,w in word_index if w == subject_value]
                objects = [(index,w)for index,w in word_index if w == object_value]
            sub_obj = list(itertools.product(subjects, objects))
            
            sub_obj = sorted(sub_obj,key=lambda x:abs(x[0][0]-x[1][0]))
            subjects,objects = choice(sub_obj)
            subjects = {"name": subject_value, "pos": [subjects[0] - len(subject_value) + 1, subjects[0] + 1]}
            objects = {"name": object_value, "pos": [objects[0] - len(object_value) + 1, objects[0] + 1]}
        return subjects,objects

    def drop_nest_words(self,word_index,subject_value,object_value):
        old_word_index = copy.deepcopy(word_index)
        nest_word_index = set()
        number = len(word_index)
        for i in range(number-1):
            for j in range(i+1,number):
                if self.is_nest(word_index[i],word_index[j]):
                    nest_word_index.add(word_index[i] if len(word_index[i][1])<len(word_index[j][1]) else word_index[j])
        
        for nest in nest_word_index:
            word_index.remove(nest)
        
        subjects = [(index,w)for index,w in word_index if w == subject_value]
        objects = [(index,w)for index,w in word_index if w == object_value]
        if len(subjects)==0:
            subjects = []
            new_objects = []
            for nest in nest_word_index:
                if nest[1] == subject_value:
                    for o in objects:
                        if self.is_nest(nest,o):
                            subjects.append(nest)
                            new_objects.append(o)
            objects = new_objects
        elif len(objects) == 0:
            objects  = []
            new_subjects = []
            for nest in nest_word_index:
                if nest[1] == object_value:
                    for s in subjects:
                        if self.is_nest(nest,s):
                            objects.append(nest)
                            new_subjects.append(s)
            subjects = new_subjects

        return subjects,objects

    def is_nest(self,s,o):
        if (s[0]-len(s[1]))<=o[0]<=s[0] or (o[0]-len(o[1]))<=s[0]<=o[0]:
            return True
        return False

    def __len__(self):
        return len(self.json_data)
    
    def check(self,pos_head,pos_tail,tokens,sub,text):
        substring = tokens[pos_head:pos_tail]
        substring = [a.replace("##","") if a!="[unused1]" else " " for a in substring]
        # subtoken = self.tokenizer.tokenize(strQ2B(sub))
        subtoken = list(strQ2B(sub))
        subtoken = [a.replace("##","") for a in subtoken]
        if subtoken != substring:
            print(substring,subtoken,text)
            print()
        if pos_head <=0 or pos_tail<1:
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

    def pad_to_max_length(self,token_ids,max_length):
        if len(token_ids)>max_length:
            token_ids = token_ids[:max_length]
            mask = [1]*max_length
        else:
            mask = [1]*len(token_ids) + [0]*(max_length-len(token_ids))
            token_ids += [0]*(max_length-len(token_ids))
        return np.array(token_ids),np.array(mask)

    def __getitem__(self,index):
        ins_json_data = self.json_data[index]
        line = json.loads(ins_json_data)
        ins_json_data = self.parse_single_line(line)
        text = ins_json_data['text']
        tokens,new_index = self.tokenizer.tokenize(text)
        if len(tokens) > BERT_MAX_LEN:
            tokens = tokens[:BERT_MAX_LEN]
        text_len = len(tokens)

        if not self.is_test:
            s2ro_map = {}
            triples = []
            for triple in ins_json_data['spo_list']:
                pos_head = triple["h"]['pos']
                pos_tail = triple['t']['pos']
                # pos_head = triple["t"]['pos'] # 这是object
                # pos_tail = triple['h']['pos'] # 这是subject
                # triple = (self.tokenizer.tokenize(triple['subject'])[1:-1], triple[1], self.tokenizer.tokenize(triple["object"]['@value'])[1:-1])
                # sub_head_idx = find_head_idx(tokens, triple[0])
                # obj_head_idx = find_head_idx(tokens, triple[2])
                triple = (triple['subject'], triple['predicate'], triple["object"]['@value'])
                # triple = (triple["object"]['@value'], triple['predicate'], triple['subject']) #百度的主语是用object表示的
                triples.append(triple)
                # sub_head_idx,sub_tail_idx = self.get_index(pos_head,new_index) #按照tokenizer自行分词进行预测
                sub_head_idx,sub_tail_idx = pos_head[0]+1,pos_head[1]+1 # 以char级别进行预测，加1是因为前面有[CLS]字符
                self.check(sub_head_idx,sub_tail_idx,tokens,triple[0],text)
                # obj_head_idx,obj_tail_idx = self.get_index(pos_tail,new_index) # 按照tokenizer自行分词进行预测
                obj_head_idx,obj_tail_idx = pos_tail[0]+1,pos_tail[1]+1 # 以char级别进行预测,加1是因为前面有[CLS]字符
                self.check(obj_head_idx,obj_tail_idx,tokens, triple[2],text)
                if sub_head_idx != -1 and obj_head_idx != -1:
                    # sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    sub = (sub_head_idx, sub_tail_idx)
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    s2ro_map[sub].append((obj_head_idx, obj_tail_idx, self.rel2id[triple[1]]))

            if s2ro_map:
                # outputs = self.tokenizer.encode_plus(text,max_length=self.config.max_len,pad_to_max_length=True,return_attention_mask=True)
                # token_ids, masks = np.array(outputs['input_ids']),np.array(outputs['attention_mask'])
                token_ids = np.array(self.tokenizer.convert_tokens_to_ids(tokens))
                # token_ids,masks = self.pad_to_max_length(input_ids,self.config.max_len)
                masks = np.array([1]*len(token_ids))
                if len(token_ids) > text_len:
                    token_ids = token_ids[:text_len]
                    masks = masks[:text_len]
                sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
                pointer_sub = np.zeros((text_len,text_len)) # for global pointer network
                # 所有 subject 的头尾index
                for s in s2ro_map:
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                    pointer_sub[s[0]][s[1]]=1
                # 随机选择一个subject
                sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
                sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1
                # 选择的某个subject 对应的所有 object的头尾index以及对应的关系
                obj_heads, obj_tails = np.zeros((text_len, self.config.rel_num)), np.zeros((text_len, self.config.rel_num))
                pointer_obj = np.zeros((self.config.rel_num,text_len,text_len)) # for global pointer network
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1
                    pointer_obj[ro[2]][ro[0]][ro[1]]=1
                return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, pointer_sub, pointer_obj, triples, tokens
            else:
                return None
        else:
            triples = [(triple['subject'], triple['predicate'], triple["object"]['@value']) for triple in ins_json_data['spo_list']]
            # 理由同上
            # triples = [(triple["object"]['@value'], triple['predicate'], triple['subject']) for triple in ins_json_data['spo_list']]

            # outputs = self.tokenizer.encode_plus(text,max_length=self.config.max_len,pad_to_max_length=True,return_attention_mask=True)
            # token_ids, masks = np.array(outputs['input_ids']),np.array(outputs['attention_mask'])
            token_ids = np.array(self.tokenizer.convert_tokens_to_ids(tokens))
            # token_ids,masks = self.pad_to_max_length(input_ids,self.config.max_len)
            masks = np.array([1]*len(token_ids))
            if len(token_ids) > text_len:
                token_ids = token_ids[:text_len]
                masks = masks[:text_len]
            sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
            pointer_sub = np.zeros((text_len,text_len)) # for global pointer network
            sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
            pointer_obj = np.zeros((self.config.rel_num,text_len,text_len)) # for global pointer network
            obj_heads, obj_tails = np.zeros((text_len, self.config.rel_num)), np.zeros((text_len, self.config.rel_num))
            return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, pointer_sub, pointer_obj, triples, tokens


def get_loader(dataset, config, is_test=False, num_workers=5,epochs = 0):
    # dataset = CMEDDataset(config, prefix, is_test)
    # dataset = dataLoader[config.dataset](config, prefix, is_test)
    # collate_fn = lambda x: casrel_collate_fn(x,config.rel_num)
    collate_fn = lambda x: collate_fn_register[config.model_name](x,config.rel_num)
    train_sampler = None
    if int(config.local_rank) !=-1:
        train_sampler = distributed.DistributedSampler(dataset)
        train_sampler.set_epoch(epochs) #是在DDP模式下shuffle数据集的方式；
    if not is_test:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True if train_sampler is None else None,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 sampler=train_sampler)
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
                    # self.next_data[k] = self.next_data[k]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data


collate_fn_register = {}
collate_register = partial(register,registry=collate_fn_register)


@collate_register("casrel")
def casrel_collate_fn(batch,rel_num):
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[2], reverse=True)
    token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, pointer_sub, pointer_obj, triples, tokens = zip(*batch)
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


@collate_register("globalpointer")
def global_pointer_collate_fn(batch,rel_num):
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[2], reverse=True)
    token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, pointer_sub, pointer_obj, triples, tokens = zip(*batch)
    cur_batch = len(batch)
    max_text_len = max(text_len)
    batch_token_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_masks = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_pointer_sub = torch.Tensor(cur_batch, max_text_len, max_text_len).zero_()
    batch_sub_head = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_tail = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_pointer_obj = torch.Tensor(cur_batch, rel_num, max_text_len, max_text_len).zero_() #数字表示是关系数量

    for i in range(cur_batch):
        batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
        batch_pointer_sub[i, :text_len[i],:text_len[i]].copy_(torch.from_numpy(pointer_sub[i]))
        batch_sub_head[i, :text_len[i]].copy_(torch.from_numpy(sub_head[i]))
        batch_sub_tail[i, :text_len[i]].copy_(torch.from_numpy(sub_tail[i]))
        batch_pointer_obj[i, :, :text_len[i], :text_len[i]].copy_(torch.from_numpy(pointer_obj[i]))

    return {'token_ids': batch_token_ids,
            'mask': batch_masks,
            'pointer_sub': batch_pointer_sub,
            'sub_head': batch_sub_head,
            'sub_tail': batch_sub_tail,
            'pointer_obj': batch_pointer_obj,
            'triples': triples,
            'tokens': tokens}
