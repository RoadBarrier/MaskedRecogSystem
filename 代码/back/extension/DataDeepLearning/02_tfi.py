# -*-coding:utf-8-*-
"""
@Time：2022/7/16 19:29
@Auth：黄祎
@File：02_tfi.py
@IDE：PyCharm
@Type：
@Remark：
@History：
"""
import os
import time
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from matplotlib.font_manager import FontProperties

"""
TF-IDF（Term Frequency-InversDocument Frequency）是一种常用于信息处理和数据挖掘的加权技术。该技术采用一种统计方法，
根据字词的在文本中出现的次数和在整个语料中出现的文档频率来计算一个字词在整个语料中的重要程度。它的优点是能过滤掉一些常见
的却无关紧要本的词语，同时保留影响整个文本的重要字词。
"""
# ------------------------------------中文分词------------------------------------
cut_words = ""
for line in open('C-class.txt', encoding='utf-8'):
    line.strip('\n')
    seg_list = jieba.cut(line, cut_all=False)
    # print(" ".join(seg_list))
    cut_words += (" ".join(seg_list))

# jieba.load_userdict("userdict.txt")              # 自定义词典
# jieba.analyse.set_stop_words('stop_words.txt')   # 停用词词典

# 提取主题词
keywords = jieba.analyse.extract_tags(cut_words, topK=50, withWeight=True,
                                      allowPOS=('a', 'e', 'n', 'nr', 'ns', 'v'))  # 词性 形容词 叹词 名词 动词

print(keywords)

pd.DataFrame(keywords, columns=['词语', '重要性']).to_excel('TF_IDF关键词前50.xlsx')

ss = pd.DataFrame(keywords, columns=['词语', '重要性'])
# print(ss)

# ------------------------------------数据可视化------------------------------------
plt.figure(figsize=(10, 6))
plt.title('TF-IDF Ranking', color='#264653')
fig = plt.axes()
# 横向的柱状图
plt.barh(range(len(ss.重要性[:25][::-1])), ss.重要性[:25][::-1], color='#e76f51')
fig.set_yticks(np.arange(len(ss.重要性[:25][::-1])))
font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc')
fig.set_yticklabels(ss.词语[:25][::-1], fontproperties=font, color='#264653')
fig.set_xlabel('Importance', color='#264653')
plt.show()
