from transformers import BertTokenizer
import codecs
import unicodedata


class HBTokenizer(BertTokenizer):
    def _tokenize(self, text):
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        text = text.lower()
        tokens = []
        for word in text.strip().split():
            tokens += self.wordpiece_tokenizer.tokenize(word)
            tokens.append('[unused1]') # 将空格用一个特殊符号代替了
        return tokens


def get_tokenizer(vocab_path):
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return HBTokenizer(token_dict, cased=True)

class BDTokenizer(BertTokenizer):
    def tokenize(self,text):
        """以char级别进行预测"""
        tokens = text.split(" ")
        re_tokens = ['[CLS]']
        new_index = [-1]
        start = 0

        for token in tokens:
            for t in list(token):
                if t not in self.vocab:
                    if "##"+t in self.vocab:
                        re_tokens.append("##"+t)
                    else:
                        re_tokens.append("[UNK]")
                else:
                    re_tokens.append(t)
                new_index.append(start)
                start += 1
            re_tokens.append("[unused1]")
            new_index.append(start)
            start += 1
        del new_index[-1]
        del re_tokens[-1]
        start -= 1 # 最后一个list表示文本结束，后面不应有空格
        re_tokens.append('[SEP]')
        new_index.append(start)
        return re_tokens,new_index


    def tokenize_(self,text):
        """按照tokenizer自行分词进行预测"""
        tokens = text.split(" ")
        re_tokens = ['[CLS]']
        cur_pos = 0
        new_index = [-1]
        start = 0
        for token in tokens:
            root_tokens = self.basic_tokenizer.tokenize(token, never_split=self.all_special_tokens)
            for t in root_tokens:
                split_tokens = self.wordpiece_tokenizer.tokenize(t)
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

