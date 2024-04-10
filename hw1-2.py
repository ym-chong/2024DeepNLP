import jieba
import jieba.analyse
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import numpy as np

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 中文字体为宋体
mpl.rcParams['axes.unicode_minus'] = False

def preprocess_text(text):
    # 删除所有的隐藏符号
    cleaned_text = ''.join(char for char in text if char.isprintable())
    # 删除所有的非中文字符
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', cleaned_text)
    # 删除所有停用词
    with open('cn_stopwords.txt', "r", encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
        cleaned_text = ''.join(char for char in cleaned_text if char not in stopwords)
    # 删除所有的标点符号
    punctuation_pattern = re.compile(r'[\s!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+')
    cleaned_text = punctuation_pattern.sub('', cleaned_text)
    return cleaned_text

class ChineseDataSet:
    def __init__(self, name):
        self.data = None
        self.name = name
        # 字
        self.word = []  # 单个字列表
        self.word_len = 0  # 单个字总字数
        # 词
        self.split_word = []  # 单个词列表
        self.split_word_len = 0  # 单个词总数
        with open("cn_stopwords.txt", "r", encoding='utf-8') as f:
            self.stop_word = f.read().split('\n')
            f.close()

    def read_file(self, filename=""):
        # 如果未指定名称，则默认为类名
        if filename == "":
            filename = self.name
        target = "jyxstxtqj_downcc.com/" + filename + ".txt"
        with open(target, "r", encoding='gbk', errors='ignore') as f:
            self.data = f.read()
            f.close()
        # 分词
        for words in jieba.cut(self.data):
            if (words not in self.stop_word) and (not words.isspace()):
                self.split_word.append(words)
                self.split_word_len += 1
        # 统计字数
        for word in self.data:
            if (word not in self.stop_word) and (not word.isspace()):
                # if not word.isspace():
                self.word.append(word)
                self.word_len += 1

    def get_unigram_tf(self, word):
        # 得到单个词的词频表
        unigram_tf = {}
        for w in word:
            unigram_tf[w] = unigram_tf.get(w, 0) + 1
        return unigram_tf

    def get_bigram_tf(self, word):
        # 得到二元词的词频表
        bigram_tf = {}
        for i in range(len(word) - 1):
            bigram_tf[(word[i], word[i + 1])] = bigram_tf.get(
                (word[i], word[i + 1]), 0) + 1
        return bigram_tf

    def get_trigram_tf(self, word):
        # 得到三元词的词频表
        trigram_tf = {}
        for i in range(len(word) - 2):
            trigram_tf[(word[i], word[i + 1], word[i + 2])] = trigram_tf.get(
                (word[i], word[i + 1], word[i + 2]), 0) + 1
        return trigram_tf

    def calc_entropy_unigram(self, word, is_ci=0):
        # 计算一元模型的信息熵
        word_tf = self.get_unigram_tf(word)
        word_len = sum([item[1] for item in word_tf.items()])
        entropy = sum(
            [-(word[1] / word_len) * math.log(word[1] / word_len, 2) for word in
             word_tf.items()])
        if is_ci:
            print("<{}>基于词的一元模型的中文信息熵为：{}比特/词".format(self.name, entropy))
        else:
            print("<{}>基于字的一元模型的中文信息熵为：{}比特/字".format(self.name, entropy))
        return entropy

    def calc_entropy_bigram(self, word, is_ci=0):
        # 计算二元模型的信息熵
        word_tf = self.get_bigram_tf(word)
        last_word_tf = self.get_unigram_tf(word)
        bigram_len = sum([item[1] for item in word_tf.items()])
        entropy = []
        for bigram in word_tf.items():
            p_xy = bigram[1] / bigram_len  # 联合概率p(xy)
            p_x_y = bigram[1] / last_word_tf[bigram[0][0]]  # 条件概率p(x|y)
            entropy.append(-p_xy * math.log(p_x_y, 2))
        entropy = sum(entropy)
        if is_ci:
            print("<{}>基于词的二元模型的中文信息熵为：{}比特/词".format(self.name, entropy))
        else:
            print("<{}>基于字的二元模型的中文信息熵为：{}比特/字".format(self.name, entropy))
        return entropy

    def calc_entropy_trigram(self, word, is_ci):
        # 计算三元模型的信息熵
        word_tf = self.get_trigram_tf(word)
        last_word_tf = self.get_bigram_tf(word)
        trigram_len = sum([item[1] for item in word_tf.items()])
        entropy = []
        for trigram in word_tf.items():
            p_xy = trigram[1] / trigram_len  # 联合概率p(xy)
            p_x_y = trigram[1] / last_word_tf[(trigram[0][0], trigram[0][1])]  # 条件概率p(x|y)
            entropy.append(-p_xy * math.log(p_x_y, 2))
        entropy = sum(entropy)
        if is_ci:
            print("<{}>基于词的三元模型的中文信息熵为：{}比特/词".format(self.name, entropy))
        else:
            print("<{}>基于字的三元模型的中文信息熵为：{}比特/字".format(self.name, entropy))
        return entropy


def my_plot(X, Y1, Y2, Y3, num):
    # 标签位置
    x = range(0, len(X))
    # 设置图片大小、绘制折线图
    plt.figure(figsize=(19.2, 10.8))
    plt.plot(x, Y1, color="r", marker='o', label="一元模型")
    plt.plot(x, Y2, color="b", marker='o', label="二元模型")
    plt.plot(x, Y3, color="g", marker='o', label="三元模型")
    # 设置x轴
    plt.xticks(x, X, rotation=40, fontsize=18)
    plt.xlabel('数据库', fontsize=18)
    # 设置y轴
    plt.ylabel('信息熵', fontsize=18)
    plt.ylim(0, max(max(Y1), max(Y2), max(Y3)) + 2)
    # 标题
    if (num == 1):
        plt.title("以字为单位的信息熵", fontsize=18)
    elif num == 2:
        plt.title("以词为单位的信息熵", fontsize=18)

    plt.legend()
    plt.savefig('chinese' + str(num) + '.png')
    plt.show()


def autolabel(x, y):
    for a, b in zip(x, y):
        plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=18)

if __name__ == "__main__":
    data_set_list = []
    # 每次运行程序将总内容文件清空
    with open("log.txt", "w") as f:
        f.close()
    # 读取小说名字
    with open("inf.txt", "r") as f:
        txt_list = f.read().split(',')
        i = 0
        for name in txt_list:
            locals()[f'set{i}'] = ChineseDataSet(name)
            data_set_list.append(locals()[f'set{i}'])
            i += 1
        f.close()
    # 分别针对每本小说进行操作
    word_unigram_entropy, word_bigram_entropy, word_trigram_entropy, words_unigram_entropy, words_bigram_entropy, words_trigram_entropy = [], [], [], [], [], []
    for set in data_set_list:
        set.read_file()
        # 字为单位
        word_unigram_entropy.append(set.calc_entropy_unigram(set.word, 0))
        word_bigram_entropy.append(set.calc_entropy_bigram(set.word, 0))
        word_trigram_entropy.append(set.calc_entropy_trigram(set.word, 0))
        # 词为单位
        words_unigram_entropy.append(set.calc_entropy_unigram(set.split_word, 1))
        words_bigram_entropy.append(set.calc_entropy_bigram(set.split_word, 1))
        words_trigram_entropy.append(set.calc_entropy_trigram(set.split_word, 1))
        with open("log.txt", "a") as f:
            f.write("{:<10} 字数：{:10d} 词数：{:10d} 信息熵：{:.4f}、{:.4f}、{:.4f}、{:.4f}、{:.4f}、{:.4f}\n".format(set.name,set.word_len,set.split_word_len,word_unigram_entropy[-1],word_bigram_entropy[-1], word_trigram_entropy[-1],words_unigram_entropy[-1],words_bigram_entropy[-1],words_trigram_entropy[-1]))
            f.close()

    # 绘图
    x_label = [i.name for i in data_set_list]

    my_plot(x_label, word_unigram_entropy, word_bigram_entropy, word_trigram_entropy, 1)
    my_plot(x_label, words_unigram_entropy, words_bigram_entropy, words_trigram_entropy, 2)