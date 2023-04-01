# -*-coding:utf-8-*-
"""
@Time：2022/7/16 07:29
@Auth：黄祎
@File：01_spider.py
@IDE：PyCharm
@Type：
@Remark：
@History：
"""
import jieba
import re
import sys
import time
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ------------------------------------中文分词------------------------------------
cut_words = ""
all_words = ""
f = open('data-fenci.txt', 'w')
for line in open('data.txt', encoding='utf-8'):
    line.strip('\n')
    seg_list = jieba.cut(line, cut_all=False)
    # print(" ".join(seg_list))
    cut_words = (" ".join(seg_list))
    f.write(cut_words)
    all_words += cut_words
else:
    f.close()

# 输出结果
all_words = all_words.split()
print(all_words)

c = Counter()
for x in all_words:
    if len(x) > 1 and x != '\r\n':
        c[x] += 1

print('\n词频统计结果：')
for (k, v) in c.most_common(10):
    print("%s:%d" % (k, v))

name = time.strftime("%Y-%m-%d") + "-fc.csv"
fw = open(name, 'w', encoding='utf-8')
i = 1
for (k, v) in c.most_common(len(c)):
    fw.write(str(i) + ',' + str(k) + ',' + str(v) + '\n')
    i = i + 1
else:
    print("Over write file!")
    fw.close()

# ------------------------------------词云分析------------------------------------
text = open('data.txt', "r", encoding='utf-8').read()

wordlist = jieba.cut(text, cut_all=False)

wl_space_split = " ".join(wordlist)
# print(wl_space_split)

my_wordcloud = WordCloud(font_path='HKWW.ttc', background_color='white', width=1800, height=1200).generate(
    wl_space_split)

plt.imshow(my_wordcloud)
plt.axis("off")
plt.show()
