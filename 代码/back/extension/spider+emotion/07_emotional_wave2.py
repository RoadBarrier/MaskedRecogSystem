# -*-coding:utf-8-*-
"""
@Time：2022/7/16 07:29
@Auth：黄祎
@File：07_emotional_wave2.py
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

result = []
i = 0
while i < len(sentimentslist):
    result.append(sentimentslist[i] - 0.5)
    i = i + 1

# 画图
import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.arange(0, 356, 1), result, 'k-',color="#0081a7")
plt.xlabel('Number')
plt.ylabel('Sentiment')
plt.title('Analysis of Sentiments')
plt.show()
