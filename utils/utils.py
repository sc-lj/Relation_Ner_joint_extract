
import collections
import unicodedata
from html import unescape
import six
import os
import re


# html and http
# HTML_HTTP_COMPILE = re.compile("(https?|ftp|file)://?[-A-Za-z0-9+&@#￥₴€₰¢£₤$/%?False:,.;=~_|!\s]+[-A-Za-z0-9+&@#￥₴€₰£¢₤$/%=~_|]")
HTML_HTTP_COMPILE = re.compile("(https?|ftp|file)://?[-A-Za-z0-9+&@#￥₴€₰¢£₤$/%?False:,.;=~_|!\s]+[-+html&@#.cn￥₴€₰£¢₤$/%=~_|]")

def keep_english_space_(text):
    """"
    处理多余的空格
    """
    match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
    should_replace_list = match_regex.findall(text)
    order_replace_list = sorted(should_replace_list,key=lambda i:len(i),reverse=True)
    for i in order_replace_list:
        if i == u' ':
            continue
        new_i = i.strip()
        text = text.replace(i,new_i)
    return text

def keep_english_space(text):
    """"
    处理多余的空格
    """
    space_pattern1 = re.compile(r"([\u4e00-\u9fa5。\-\.,，:：/_《》、\|><\(\)（）])\s+([\u4e00-\u9fa5。\-\.,，\|:：/_《》><、\(\)（）])")
    space_pattern2 = re.compile(r"([\u4e00-\u9fa5。\.,，:：/_《》、><\(\)（）])\s+([a-zA-Z0-9])")
    space_pattern3 = re.compile(r"([a-zA-Z0-9])\s+([\u4e00-\u9fa5。\.,，:：/_《》、><\(\)（）])")
    text = space_pattern1.sub(r'\1\2',text)
    text = space_pattern1.sub(r'\1\2',text)
    text = space_pattern2.sub(r'\1\2',text)
    text = space_pattern3.sub(r'\1\2',text)

    return text


def strQ2B(ustring):
    """中文特殊符号转英文特殊符号"""
    ustring=convert_to_unicode(ustring)
    ustring = unicodedata.normalize("NFD", ustring)
    ustring = unescape(ustring)
    ustring = HTML_HTTP_COMPILE.sub("", ustring)
    # 对有中文特殊符号的文本进行符号替换
    table = {ord(s):ord(e) for s,e in zip("，：“”【】？；（）‘’』『「」﹃﹄〔〕—",',:""[]?;()\'\'[][][]{}-')}
    ustring = ustring.translate(table)

    ustring=ustring.replace("\ue40c",'')

    """全角转半角"""
    # 转换说明：
    # 全角字符unicode编码从65281~65374 （十六进制 0xFF01 ~ 0xFF5E）
    # 半角字符unicode编码从33~126 （十六进制 0x21~ 0x7E）
    # 空格比较特殊，全角为 12288（0x3000），半角为 32（0x20）
    # 除空格外，全角/半角按unicode编码排序在顺序上是对应的（半角 + 0x7e= 全角）,所以可以直接通过用+-法来处理非空格数据，对空格单独处理。
    rstring = ""
    for uchar in ustring:
        # 返回赋予Unicode字符uchar的字符串型通用分类。
        inside_code = ord(uchar)

        if inside_code == 0 or inside_code == 0xfffd:
            continue
        cat = unicodedata.category(uchar)
        if cat == "Mn" or cat.startswith("C") or cat=="So":
            continue
        if inside_code in [12288,32,160,58853,8195,60633]:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    rstring = keep_english_space(rstring)
    rstring = re.sub("\s+"," ",rstring).strip()
    return rstring.lower()

def is_whitespace(char):
    """    Checks whether `chars` is a whitespace character.
        \t, \n, and \r are technically contorl characters but we treat them
        as whitespace since they are generally considered as such.
    """
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def is_control(char):
    """    Checks whether `chars` is a control character.
        These are technically control characters but we count them as whitespace characters.
    """
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def is_punctuation(char):
    """ Checks whether `chars` is a punctuation character.
        We treat all non-letter/number ASCII as punctuation. Characters such as "^", "$", and "`" are not in the Unicode.
        Punctuation class but we treat them as punctuation anyways, for consistency.
    """
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def is_chinese_char(cp):
    """    Checks whether CP is the codepoint of a CJK character.
        This defines a "chinese character" as anything in the CJK Unicode block:
        https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        despite its name. The modern Korean Hangul alphabet is a different block,
        as is Japanese Hiragana and Katakana. Those alphabets are used to write
        space-separated words, so they are not treated specially and handled
        like the all of the other languages.
    """
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or
        (cp >= 0x3400 and cp <= 0x4DBF) or
        (cp >= 0x20000 and cp <= 0x2A6DF) or
        (cp >= 0x2A700 and cp <= 0x2B73F) or
        (cp >= 0x2B740 and cp <= 0x2B81F) or
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or
        (cp >= 0x2F800 and cp <= 0x2FA1F)):
        return True
    return False

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def clean_text(text):
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        if is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def split_on_whitespace(text):
    """ Runs basic whitespace cleaning and splitting on a peice of text.
    e.g, 'a b c' -> ['a', 'b', 'c']
    """
    text = text.strip()
    if not text:
        return []
    return text.split()

def split_on_punctuation(text):
    """Splits punctuation on a piece of text."""
    start_new_word = True
    output = []
    for char in text:
        if is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
    return ["".join(x) for x in output]

def tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def convert_by_vocab(vocab, items, max_seq_length = None, blank_id = 0, unk_id = 1, uncased = False):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        if uncased:
            item = item.lower()
        if item in vocab:
            output.append(vocab[item])
        else:
            output.append(unk_id)
    if max_seq_length != None:
        if len(output) > max_seq_length:
            output = output[:max_seq_length]
        else:
            while len(output) < max_seq_length:
                output.append(blank_id)
    return output


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def add_token(tokens_a, tokens_b = None):
    assert len(tokens_a) >= 1
    
    tokens = []
    segment_ids = []
    
    tokens.append("[CLS]")
    segment_ids.append(0)
    
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b != None:
        assert len(tokens_b) >= 1

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)

        tokens.append("[SEP]")
        segment_ids.append(1)

    return tokens, segment_ids


def is_normal(epl):
    entities = set()
    for e in epl:
        entities.add(e[0])
        entities.add(e[1])
    return len(entities) == (len(epl) * 2)


def is_multi_label(epl):
    if is_normal(epl):
        return False
    entities_pair = []
    for i, e in enumerate(epl):
        entities_pair.append(tuple([e[0], e[1]]))
    return len(entities_pair) != len(set(entities_pair))


def is_over_lapping(epl):
    if is_normal(epl):
        return False

    entities_pair = []
    for i, e in enumerate(epl):
        entities_pair.append(tuple([e[0], e[1]]))

    entities_pair = set(entities_pair)
    entities = []
    for pair in entities_pair:
        entities.extend(pair)
    entities = set(entities)
    return len(entities) != (2 * len(entities_pair))

if __name__ == '__main__':
    text="<爱情不辛苦 >(我爱河东狮尾曲)唱/词:佟湘君.曲:刘鹏都说爱情盲目, / 大家好啊,我是DELLY我说难得糊涂, / 上传<爱情不辛苦>的歌词总要换个笑语, / 原版声音,希望大家喜欢 "
    text = '雷顿教授与魔神之笛》官网 http://www.layton.jp/majin/index.html2009年11月26日 剧场版《雷顿教授与永远的歌姬》官网 http://www.layton-movie.jp/12月19日上映、预定每年冬季出一部剧场版'
    text = "《约瑟芬·铁伊推理全集7:排队的人》讲述伦敦沃芬顿剧院门口大排长龙"
    text = strQ2B(text)
    print(text)
    text = keep_english_space(text)
    print(text)
    print(len(text.strip()))
