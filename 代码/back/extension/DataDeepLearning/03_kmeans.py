# -*-coding:utf-8-*-
"""
@Time：2022/7/16 19:29
@Auth：黄祎
@File：03_kmeans.py
@IDE：PyCharm
@Type：
@Remark：
@History：
"""
# KNN文本聚类
import time
import re
import os
import sys
import codecs
import shutil
import numpy as np
import matplotlib
import scipy
import matplotlib.pyplot as plt
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

if __name__ == "__main__":

    # 计算TFIDF

    corpus = []

    for line in open('C-class-fenci.txt', 'r').readlines():
        corpus.append(line.strip())

    vectorizer = CountVectorizer()

    transformer = TfidfTransformer()

    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()

    # 打印特征向量文本内容
    print('Features length: ' + str(len(word)))

    """
    # 输出单词
    for j in range(len(word)):
        print(word[j] + ' ')
        
    for i in range(len(weight)):
        print u"-------这里输出第", i, u"类文本的词语tf-idf权重------"  
        for j in range(len(word)):
            print weight[i][j],
    """

    # 聚类Kmeans

    print('Start Kmeans:')
    from sklearn.cluster import KMeans

    clf = KMeans(n_clusters=2)
    print(clf)
    pre = clf.fit_predict(weight)
    print(pre)

    # 中心点
    print(clf.cluster_centers_)
    print(clf.inertia_)

    # 图形输出 降维

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    newData = pca.fit_transform(weight)
    print(newData)

    x = [n[0] for n in newData]
    y = [n[1] for n in newData]

    plt.scatter(x, y, c=pre, s=100)
    plt.legend()
    plt.title("Cluster with Text Mining")
    plt.show()
