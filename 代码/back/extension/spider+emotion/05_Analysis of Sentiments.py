# -*-coding:utf-8-*-
"""
@Time：2022/7/16 07:29
@Auth：黄祎
@File：05_Analysis of Sentiments.py
@IDE：PyCharm
@Type：
@Remark：
@History：
"""
from snownlp import SnowNLP
import codecs
import os

source = open("data.txt", "r", encoding='utf-8')
line = source.readlines()
sentimentslist = []
for i in line:
    s = SnowNLP(i)
    print(s.sentiments)
    sentimentslist.append(s.sentiments)

import matplotlib.pyplot as plt
import numpy as np

plt.hist(sentimentslist, bins=np.arange(0, 1, 0.01), facecolor='#f08080')
plt.xlabel('Sentiments Probability')
plt.ylabel('Quantity')
plt.title('Analysis of Sentiments')
plt.show()
