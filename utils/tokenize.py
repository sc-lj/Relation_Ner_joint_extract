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
