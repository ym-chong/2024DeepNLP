import jieba
import os
import matplotlib.pyplot as plt
import re
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为中文字体，这里以宋体为例
plt.figure(figsize=(10, 6))
# 预处理函数：删除隐藏符号、非中文字符和标点符号
def preprocess_text(text):
    # 删除所有的隐藏符号
    cleaned_text = ''.join(char for char in text if char.isprintable())
    # 删除所有的非中文字符
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', cleaned_text)
    # 删除停用词
    with open('cn_stopwords.txt', "r", encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
        cleaned_text = ''.join(char for char in cleaned_text if char not in stopwords)
    # 删除所有的标点符号
    punctuation_pattern = re.compile(r'[\s!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+')
    cleaned_text = punctuation_pattern.sub('', cleaned_text)
    return cleaned_text

folder_path = r"D:\研究僧\必修课\深度学习与自然语言处理\作业1\jyxstxtqj_downcc.com"

jieba.load_userdict("cn_stopwords.txt")
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    print("处理文件",file_path)
    # 读取文本内容并进行预处理和分词
    with open(file_path, "r", encoding='ansi') as file:
        text = file.read()
        preprocessed_text = preprocess_text(text)
        words = jieba.lcut(preprocessed_text)
    # 统计词频
    counts = {}

    for word in words:
        counts[word] = counts.get(word, 0) + 1  # 遍历所有词语，每出现一次其对应的值加 1

    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)  # 根据词语出现的次数进行从大到小排序
    sort_list = sorted(counts.items(), reverse=True)

    # 将词频数据写入文件
    with open('data.txt', mode='w', encoding='utf-8') as file:
        for word, count in counts.items():
            new_context = f"{word:<10}{count:>5}\n"
            file.write(new_context)

    file.close()

    # 绘制曲线
    freq_list = sorted(counts.values(), reverse=True)  # 词频列表
    x = range(1, len(freq_list) + 1)  # 词频排名列表
    plt.plot(x, freq_list, label=file_name)  # 绘制词频分布图

# 设置图例

# 设置标题和标签
plt.title('Zipf-Law for All Files', fontsize=18)  # 标题
plt.xlabel('rank', fontsize=18)     # 排名
plt.ylabel('freq', fontsize=18)     # 频度
plt.yscale('log')                   # 设置纵坐标的缩放
plt.xscale('log')                   # 设置横坐标的缩放

ax = plt.gca()

# 获取当前坐标轴的位置
box = ax.get_position()

# 将坐标轴的位置上移10%
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# 将图例置于当前坐标轴下
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

# 显示图表
plt.show()
