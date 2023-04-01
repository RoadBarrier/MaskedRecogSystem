# -*-coding:utf-8-*-
"""
@Time：2022/7/16 19:29
@Auth：黄祎
@File：01_hy_analysis.py
@IDE：PyCharm
@Type：
@Remark：
@History：
"""
import jieba
import re
import time
from collections import Counter

"""
1.将新闻正文文本“C-class.txt”数据进行中文分词，每行代表一条新闻，并生成对应的内容
2.生成高频特征词，并保存至CSV文件中。
"""
# ------------------------------------中文分词------------------------------------
cut_words = ""
all_words = ""
f = open('C-class-fenci.txt', 'w')
for line in open('C-class.txt', encoding='utf-8'):
    line.strip('\n')
    seg_list = jieba.cut(line, cut_all=False)
    # print(" ".join(seg_list))
    cut_words = (" ".join(seg_list))
    f.write(cut_words)
    all_words += cut_words
else:
    f.close()

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
from pyecharts import options as opts
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType

words = []
for (k, v) in c.most_common(1000):
    # print(k, v)
    words.append((k, v))

# 渲染图
def wordcloud_base() -> WordCloud:
    c = (
        WordCloud()
            .add("", words, word_size_range=[20, 100], shape=SymbolType.DIAMOND)
            .set_global_opts(title_opts=opts.TitleOpts(title='第一组疫情词云图展示'))
    )
    return c


wordcloud_base().render('疫情词云图.html')
