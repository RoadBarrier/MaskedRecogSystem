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
from snownlp import SnowNLP
s1 = SnowNLP(u"这本书质量真不太好！")
print("SnowNLP:")
print(" ".join(s1.words))

import jieba
s2 = jieba.cut(u"这本书质量真不太好！", cut_all=False)
print("jieba:")
print(" ".join(s2))
